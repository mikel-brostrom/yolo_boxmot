import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import math

class SingleScaleModel(pl.LightningModule):
    def __init__(self, num_classes=80, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters({'model_type': self.__class__.__name__})  # Saves all arguments passed to the constructor as hyperparameters

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_type = "SingleScaleModel"

        # Load resnet50 backbone and remove the fully connected layer
        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: [batch_size, 2048, H', W']

        # Define anchors (3 scales)
        self.anchors = torch.tensor([
            [0.1, 0.1], [0.2, 0.2], [0.4, 0.4],  # Small scale
            [0.1, 0.2], [0.2, 0.4], [0.4, 0.8],  # Medium scale
            [0.2, 0.1], [0.4, 0.2], [0.8, 0.4],  # Large scale
        ])  # Shape: [9, 2] -> 9 anchors defined for different scales

        self.num_anchors = self.anchors.size(0)

        # Detection head: predicts bounding boxes, objectness score, and class probabilities
        self.detection_head = nn.Conv2d(in_channels=512, out_channels=self.num_anchors * (5 + num_classes), kernel_size=1)
        # Output channels: (4 bbox coordinates + 1 objectness score + num_classes class scores) * num_anchors
        
        
    def decode_predictions(self, predictions):
        """
        Decode predictions to bounding boxes, objectness scores, and class scores.
        Args:
            predictions: Tensor of shape [batch_size, num_predictions, 5 + num_classes]
        Returns:
            decoded_boxes: Bounding boxes in original image coordinates
            decoded_scores: Objectness scores
            decoded_labels: Class labels
        """
        batch_size, num_predictions, _ = predictions.shape
        device = predictions.device

        H = W = int(math.sqrt(num_predictions / self.num_anchors))

        # Extract components
        pred_bbox = predictions[:, :, :4]
        pred_objectness = torch.sigmoid(predictions[:, :, 4])
        pred_class_scores = F.softmax(predictions[:, :, 5:], dim=-1)

        decoded_boxes = []
        for b in range(batch_size):
            boxes = []
            for n in range(num_predictions):
                # Get anchor box
                anchor_idx = n % self.num_anchors
                anchor = self.anchors[anchor_idx].to(device)

                # Get grid location
                grid_x = (n // self.num_anchors) % W
                grid_y = (n // self.num_anchors) // W

                # Extract offsets and sizes
                x_offset, y_offset, w_offset, h_offset = pred_bbox[b, n]

                # Decode center
                x_offset = torch.sigmoid(x_offset)
                y_offset = torch.sigmoid(y_offset)
                center_x = (grid_x + x_offset) / W
                center_y = (grid_y + y_offset) / H

                # Decode width and height
                width = anchor[0] * torch.exp(w_offset)
                height = anchor[1] * torch.exp(h_offset)

                # Convert to (x1, y1, x2, y2)
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2

                # Append box
                boxes.append([x1, y1, x2, y2])

            decoded_boxes.append(torch.tensor(boxes, device=device))

        # Stack the boxes for the whole batch
        decoded_boxes = torch.stack(decoded_boxes)

        # Get objectness and class labels
        decoded_scores = pred_objectness
        decoded_labels = torch.argmax(pred_class_scores, dim=-1)

        return decoded_boxes, decoded_scores, decoded_labels


    def forward(self, x, decode=False):
        features = self.backbone(x)
        predictions = self.detection_head(features)
        batch_size, _, H, W = predictions.shape

        # Reshape predictions to [batch_size, H * W * num_anchors, 5 + num_classes]
        predictions = predictions.view(batch_size, self.num_anchors, (5 + self.num_classes), H, W)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous().reshape(batch_size, H * W * self.num_anchors, 5 + self.num_classes)

        if decode:
            decoded_boxes, decoded_scores, decoded_labels = self.decode_predictions(predictions)
            return decoded_boxes, decoded_scores, decoded_labels

        return predictions

    def training_step(self, batch, batch_idx):
        images, target_cls, boxes, obj_labels = batch
        batch_size = images.size(0)
        device = images.device

        predictions = self.forward(images)
        batch_size, num_predictions, _ = predictions.shape

        # Split predictions
        pred_bbox = predictions[:, :, :4]
        pred_objectness = predictions[:, :, 4]
        pred_class_scores = predictions[:, :, 5:]

        # Initialize target tensors
        H_W = int(math.sqrt(num_predictions // self.num_anchors))
        target_objectness = torch.zeros(batch_size, H_W * H_W * self.num_anchors, device=device)
        target_bbox = torch.zeros(batch_size, H_W * H_W * self.num_anchors, 4, device=device)
        target_class = torch.zeros((batch_size, H_W * H_W * self.num_anchors), dtype=torch.long, device=device)

        # Process targets
        epsilon = 1e-6
        for b in range(batch_size):
            valid_mask = target_cls[b] != -1
            valid_target_cls = target_cls[b][valid_mask]
            valid_boxes = boxes[b][valid_mask]

            for t in range(valid_target_cls.size(0)):
                cls = valid_target_cls[t].long()
                box = valid_boxes[t, :]

                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]

                # Adjust for edge cases using torch.clamp
                x_center = torch.clamp(x_center, min=0, max=1 - epsilon)
                y_center = torch.clamp(y_center, min=0, max=1 - epsilon)

                # Compute grid cell indices
                i = min(int(x_center * H_W), H_W - 1)
                j = min(int(y_center * H_W), H_W - 1)

                # Find the best anchor index based on IoU with the target box
                anchor_idx = self.get_best_anchor(width, height)
                index = (j * H_W + i) * self.num_anchors + anchor_idx

                # Offsets within the grid cell
                x_offset = x_center * H_W - i
                y_offset = y_center * H_W - j

                # Clamp offsets to [0, 1]
                x_offset = torch.clamp(x_offset, min=0, max=1)
                y_offset = torch.clamp(y_offset, min=0, max=1)

                target_objectness[b, index] = 1
                target_bbox[b, index, 0] = x_offset
                target_bbox[b, index, 1] = y_offset
                target_bbox[b, index, 2] = width * H_W
                target_bbox[b, index, 3] = height * H_W
                target_class[b, index] = cls

        # Clamp target_objectness to ensure values are in [0, 1]
        target_objectness = torch.clamp(target_objectness, min=0, max=1)

        # Compute losses
        pred_objectness = pred_objectness.reshape(-1)
        target_objectness = target_objectness.reshape(-1)

        objectness_loss = F.binary_cross_entropy_with_logits(pred_objectness, target_objectness, reduction='mean')

        # Bounding box loss
        obj_mask = target_objectness == 1
        if obj_mask.any():
            pred_bbox = pred_bbox.reshape(-1, 4)
            target_bbox = target_bbox.reshape(-1, 4)

            bbox_loss = F.mse_loss(pred_bbox[obj_mask], target_bbox[obj_mask], reduction='mean')

            # Class loss with ignore_index
            pred_class_scores = pred_class_scores.reshape(-1, self.num_classes)
            target_class = target_class.view(-1)
            class_loss = F.cross_entropy(pred_class_scores[obj_mask], target_class[obj_mask], ignore_index=-1, reduction='mean')
        else:
            bbox_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)

        train_loss = objectness_loss + bbox_loss + class_loss

        # Logging
        self.log('train_loss', train_loss.detach(), on_step=True, on_epoch=True)
        self.log('obj_loss', objectness_loss.detach(), on_step=True, on_epoch=True)
        self.log('bbox_loss', bbox_loss.detach(), on_step=True, on_epoch=True)
        self.log('class_loss', class_loss.detach(), on_step=True, on_epoch=True)

        return train_loss

    def get_best_anchor(self, width, height):
        anchors = self.anchors.to(width.device)
        ious = self.compute_iou(anchors, torch.tensor([0, 0, width, height], device=width.device).unsqueeze(0))
        return torch.argmax(ious).item()

    def compute_iou(self, anchors, box):
        anchors_w, anchors_h = anchors[:, 0], anchors[:, 1]
        box_w, box_h = box[0, 2] - box[0, 0], box[0, 3] - box[0, 1]
        inter_area = torch.min(anchors_w, box_w) * torch.min(anchors_h, box_h)
        anchors_area = anchors_w * anchors_h
        box_area = box_w * box_h
        union_area = anchors_area + box_area - inter_area
        return inter_area / (union_area + 1e-6)
    
    def on_train_epoch_end(self):
        # Access the average metrics for the epoch
        avg_total_loss = self.trainer.callback_metrics['train_loss'].item()
        avg_bbox_loss = self.trainer.callback_metrics['bbox_loss'].item()
        avg_obj_loss = self.trainer.callback_metrics['obj_loss'].item()
        avg_class_loss = self.trainer.callback_metrics['class_loss'].item()

        # Print the metrics
        print(f"Epoch {self.current_epoch} - Avg Total Loss: {avg_total_loss:.4f}, "
              f"BBox Loss: {avg_bbox_loss:.4f}, Obj Loss: {avg_obj_loss:.4f}, Class Loss: {avg_class_loss:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]
