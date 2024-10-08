import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import math


class SingleScaleModel(pl.LightningModule):
    def __init__(self, backbone='resnet18', num_classes=80, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(
            {
                'model_type': self.__class__.__name__,
                'backbone': backbone
            }
        )

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Load the appropriate ResNet backbone based on the input parameter
        if backbone == 'resnet18':
            resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            in_channels = 512  # Final feature map has 512 channels for ResNet18
        elif backbone == 'resnet34':
            resnet = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
            in_channels = 512  # Final feature map has 512 channels for ResNet34
        elif backbone == 'resnet50':
            resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            in_channels = 2048  # Final feature map has 2048 channels for ResNet50
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: [batch_size, 2048, H', W']

        # Define anchors (3 scales)
        self.anchors = torch.tensor([
            [0.1, 0.1], [0.2, 0.2], [0.4, 0.4],  # Small scale
            [0.1, 0.2], [0.2, 0.4], [0.4, 0.8],  # Medium scale
            [0.2, 0.1], [0.4, 0.2], [0.8, 0.4],  # Large scale
        ])  # Shape: [9, 2] -> 9 anchors defined for different scales

        self.num_anchors = self.anchors.size(0)

        # Enhanced Detection head: multiple convolutional layers with BatchNorm and ReLU
        self.detection_head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=self.num_anchors * (5 + num_classes), kernel_size=1)
        )

    def decode_predictions(self, predictions):
        batch_size, num_predictions, _ = predictions.shape
        num_anchors = self.num_anchors
        num_classes = self.num_classes
        H_W = int(math.sqrt(num_predictions // num_anchors))
        H, W = H_W, H_W

        device = predictions.device
        predictions = predictions.view(batch_size, H, W, num_anchors, 5 + num_classes)

        # Get x_offset, y_offset, w_offset, h_offset, objectness, class_scores
        x_offset = torch.sigmoid(predictions[..., 0])
        y_offset = torch.sigmoid(predictions[..., 1])
        w_offset = predictions[..., 2]
        h_offset = predictions[..., 3]
        objectness = torch.sigmoid(predictions[..., 4])

        # Apply softmax to class scores
        class_scores = F.softmax(predictions[..., 5:], dim=-1)

        # Compute x_center, y_center
        grid_x = torch.arange(W, device=device).view(1, 1, W, 1).repeat(1, H, 1, num_anchors)
        grid_y = torch.arange(H, device=device).view(1, H, 1, 1).repeat(1, 1, W, num_anchors)

        x_center = (grid_x + x_offset) / W
        y_center = (grid_y + y_offset) / H

        # Compute width and height
        anchors = self.anchors.to(device)  # [num_anchors, 2]
        anchor_w = anchors[:, 0].view(1, 1, 1, num_anchors)
        anchor_h = anchors[:, 1].view(1, 1, 1, num_anchors)

        width = anchor_w * torch.exp(w_offset)
        height = anchor_h * torch.exp(h_offset)

        # Convert to x1, y1, x2, y2
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # Shape: [batch_size, H, W, num_anchors, 4]
        
        # Clamp the values between 0 and 1
        boxes = torch.clamp(boxes, min=0.0, max=1.0)

        # Reshape to [batch_size, num_predictions, ...]
        boxes = boxes.view(batch_size, -1, 4)
        objectness = objectness.view(batch_size, -1)
        class_scores = class_scores.view(batch_size, -1, num_classes)

        # Multiply objectness with class scores
        scores = objectness.unsqueeze(-1) * class_scores  # [batch_size, num_predictions, num_classes]

        # For each prediction, get the max class score and corresponding label
        scores_max, labels = torch.max(scores, dim=-1)  # [batch_size, num_predictions]

        return boxes, scores_max, labels



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
                
                # Clamp the box coordinates to be within the image bounds
                box[0] = torch.clamp(box[0], min=0, max=1)
                box[1] = torch.clamp(box[1], min=0, max=1)
                box[2] = torch.clamp(box[2], min=0, max=1)
                box[3] = torch.clamp(box[3], min=0, max=1)

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

                # Get the corresponding anchor for this prediction
                anchor = self.anchors[anchor_idx].to(device)

                # Compute width and height ratios
                width_ratio = width / (anchor[0] + epsilon)
                height_ratio = height / (anchor[1] + epsilon)

                # Compute w_offset and h_offset
                w_offset = torch.log(width_ratio + epsilon)
                h_offset = torch.log(height_ratio + epsilon)

                target_objectness[b, index] = 1
                target_bbox[b, index, 0] = x_offset
                target_bbox[b, index, 1] = y_offset
                target_bbox[b, index, 2] = w_offset
                target_bbox[b, index, 3] = h_offset
                target_class[b, index] = cls

        # Clamp target_objectness to ensure values are in [0, 1]
        target_objectness = torch.clamp(target_objectness, min=0, max=1)

        # Compute losses
        pred_objectness = pred_objectness.reshape(-1)
        target_objectness = target_objectness.reshape(-1)

        alpha, gamma = 0.25, 2
        BCE_loss = F.binary_cross_entropy_with_logits(pred_objectness, target_objectness, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss
        objectness_loss = F_loss.mean()

        # Bounding box loss
        obj_mask = target_objectness == 1
        if obj_mask.any():
            pred_bbox = pred_bbox.reshape(-1, 4)
            target_bbox = target_bbox.reshape(-1, 4)

            bbox_loss = F.smooth_l1_loss(pred_bbox[obj_mask], target_bbox[obj_mask], reduction='mean')

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
