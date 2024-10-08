import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import math

class MultiScaleModel(pl.LightningModule):
    def __init__(self, backbone='resnet50', num_classes=80, learning_rate=1e-3):
        super().__init__()
        
        self.save_hyperparameters(
            {
                'model_type': self.__class__.__name__,
                'backbone': 'resnet50'
            }
        )

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Load resnet50 backbone
        resnet = torchvision.models.resnet50(pretrained=True)
        
        # Extract layers up to 'layer4' to get feature maps at different scales
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # Output stride: 4
        self.layer1 = resnet.layer1  # Output stride: 4
        self.layer2 = resnet.layer2  # Output stride: 8
        self.layer3 = resnet.layer3  # Output stride: 16
        self.layer4 = resnet.layer4  # Output stride: 32

        # Define anchors per scale
        self.num_anchors_per_scale = 3
        self.anchors = {
            'small': torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]]),   # Scale 1
            'medium': torch.tensor([[0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]),  # Scale 2
            'large': torch.tensor([[0.2, 0.1], [0.4, 0.2], [0.8, 0.4]])    # Scale 3
        }

        # Detection heads for each scale
        self.head_small = nn.Conv2d(in_channels=512, out_channels=self.num_anchors_per_scale * (5 + num_classes), kernel_size=1)
        self.head_medium = nn.Conv2d(in_channels=1024, out_channels=self.num_anchors_per_scale * (5 + num_classes), kernel_size=1)
        self.head_large = nn.Conv2d(in_channels=2048, out_channels=self.num_anchors_per_scale * (5 + num_classes), kernel_size=1)

    def forward(self, x, decode=False):
        # Backbone forward pass
        x = self.layer0(x)
        x1 = self.layer1(x)  # Not used for detection
        x2 = self.layer2(x1)  # Small scale feature map
        x3 = self.layer3(x2)  # Medium scale feature map
        x4 = self.layer4(x3)  # Large scale feature map

        predictions = {}
        feature_maps = {'small': x2, 'medium': x3, 'large': x4}
        detection_heads = {'small': self.head_small, 'medium': self.head_medium, 'large': self.head_large}

        for scale in ['small', 'medium', 'large']:
            feature_map = feature_maps[scale]
            head = detection_heads[scale]

            pred = head(feature_map)
            batch_size, _, H, W = pred.shape
            pred = pred.view(batch_size, self.num_anchors_per_scale, (5 + self.num_classes), H, W)
            pred = pred.permute(0, 3, 4, 1, 2).contiguous()
            pred = pred.view(batch_size, H * W * self.num_anchors_per_scale, 5 + self.num_classes)

            predictions[scale] = {'pred': pred, 'H': H, 'W': W}

        if decode:
            decoded_boxes, decoded_scores, decoded_labels = self.decode_predictions(predictions)
            return decoded_boxes, decoded_scores, decoded_labels

        return predictions

    def decode_predictions(self, predictions):
        """
        Decode predictions from all scales to bounding boxes, objectness scores, and class scores.
        Args:
            predictions: Dictionary containing predictions per scale
        Returns:
            decoded_boxes: Bounding boxes in original image coordinates
            decoded_scores: Objectness scores
            decoded_labels: Class labels
        """
        device = next(self.parameters()).device
        batch_size = list(predictions.values())[0]['pred'].size(0)
        decoded_boxes = []
        decoded_scores = []
        decoded_labels = []

        for scale in ['small', 'medium', 'large']:
            pred = predictions[scale]['pred']
            H, W = predictions[scale]['H'], predictions[scale]['W']
            num_anchors = self.num_anchors_per_scale
            anchors = self.anchors[scale].to(device)

            # Extract components
            pred_bbox = pred[:, :, :4]
            pred_objectness = torch.sigmoid(pred[:, :, 4])
            pred_class_scores = F.softmax(pred[:, :, 5:], dim=-1)

            for b in range(batch_size):
                boxes = []
                scores = []
                labels = []

                for n in range(pred.shape[1]):
                    # Get anchor box
                    anchor_idx = n % num_anchors
                    anchor = anchors[anchor_idx]

                    # Get grid location
                    grid_cell = n // num_anchors
                    i = grid_cell % W
                    j = grid_cell // W

                    # Extract offsets and sizes
                    x_offset, y_offset, w_offset, h_offset = pred_bbox[b, n]

                    # Decode center
                    x_offset = torch.sigmoid(x_offset)
                    y_offset = torch.sigmoid(y_offset)
                    center_x = (i + x_offset) / W
                    center_y = (j + y_offset) / H

                    # Decode width and height
                    width = anchor[0] * torch.exp(w_offset) / W
                    height = anchor[1] * torch.exp(h_offset) / H

                    # Convert to (x1, y1, x2, y2)
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2

                    # Append box, score, and label
                    boxes.append([x1, y1, x2, y2])
                    scores.append(pred_objectness[b, n])
                    labels.append(torch.argmax(pred_class_scores[b, n]))

                decoded_boxes.append(torch.stack(boxes))
                decoded_scores.append(torch.stack(scores))
                decoded_labels.append(torch.stack(labels))

        # Concatenate results from all scales
        decoded_boxes = torch.cat(decoded_boxes, dim=0)
        decoded_scores = torch.cat(decoded_scores, dim=0)
        decoded_labels = torch.cat(decoded_labels, dim=0)

        return decoded_boxes, decoded_scores, decoded_labels

    def training_step(self, batch, batch_idx):
        images, target_cls, boxes, obj_labels = batch
        batch_size = images.size(0)
        device = images.device

        predictions = self.forward(images)
        total_objectness_loss = 0
        total_bbox_loss = 0
        total_class_loss = 0

        epsilon = 1e-6

        # Initialize target tensors per scale
        target_objectness = {}
        target_bbox = {}
        target_class = {}

        for scale in ['small', 'medium', 'large']:
            pred = predictions[scale]['pred']
            H, W = predictions[scale]['H'], predictions[scale]['W']
            num_predictions = pred.size(1)
            target_objectness[scale] = torch.zeros(batch_size, num_predictions, device=device)
            target_bbox[scale] = torch.zeros(batch_size, num_predictions, 4, device=device)
            target_class[scale] = torch.zeros(batch_size, num_predictions, dtype=torch.long, device=device)

        # Process targets
        for b in range(batch_size):
            valid_mask = target_cls[b] != -1
            valid_target_cls = target_cls[b][valid_mask]
            valid_boxes = boxes[b][valid_mask]

            for t in range(valid_target_cls.size(0)):
                cls = valid_target_cls[t].long()
                box = valid_boxes[t, :]  # Normalized box coordinates [x1, y1, x2, y2]

                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]

                # Clamp centers
                x_center = torch.clamp(x_center, min=0, max=1 - epsilon)
                y_center = torch.clamp(y_center, min=0, max=1 - epsilon)

                # Compute IoU with all anchors at all scales
                ious = []
                anchor_indices = []
                scales = []
                for scale in ['small', 'medium', 'large']:
                    anchors = self.anchors[scale].to(device)
                    iou = self.compute_iou(anchors, torch.tensor([0, 0, width, height], device=device).unsqueeze(0))
                    ious.append(iou)
                    anchor_indices.append(torch.arange(self.num_anchors_per_scale))
                    scales.extend([scale] * self.num_anchors_per_scale)

                ious = torch.cat(ious)
                anchor_indices = torch.cat(anchor_indices)
                scales = scales

                # Find the best matching anchor
                best_iou, best_idx = ious.max(0)
                best_anchor_idx = anchor_indices[best_idx]
                best_scale = scales[best_idx]

                # Get H and W of the best scale
                H, W = predictions[best_scale]['H'], predictions[best_scale]['W']

                # Compute grid cell indices
                i = min(int(x_center * W), W - 1)
                j = min(int(y_center * H), H - 1)

                # Index of the prediction
                index = (j * W + i) * self.num_anchors_per_scale + best_anchor_idx

                # Offsets within the grid cell
                x_offset = x_center * W - i
                y_offset = y_center * H - j

                # Clamp offsets to [0, 1]
                x_offset = torch.clamp(x_offset, min=0, max=1)
                y_offset = torch.clamp(y_offset, min=0, max=1)

                # Assign targets
                target_objectness[best_scale][b, index] = 1
                target_bbox[best_scale][b, index, 0] = x_offset
                target_bbox[best_scale][b, index, 1] = y_offset
                target_bbox[best_scale][b, index, 2] = torch.log(width * W / self.anchors[best_scale][best_anchor_idx][0] + epsilon)
                target_bbox[best_scale][b, index, 3] = torch.log(height * H / self.anchors[best_scale][best_anchor_idx][1] + epsilon)
                target_class[best_scale][b, index] = cls

        # Compute losses per scale
        for scale in ['small', 'medium', 'large']:
            pred = predictions[scale]['pred']
            batch_size, num_predictions, _ = pred.shape

            # Split predictions
            pred_bbox = pred[:, :, :4]
            pred_objectness = pred[:, :, 4]
            pred_class_scores = pred[:, :, 5:]

            # Get targets
            target_obj = target_objectness[scale]
            target_bbox_scale = target_bbox[scale]
            target_class_scale = target_class[scale]

            # Objectness loss
            pred_objectness_flat = pred_objectness.reshape(-1)
            target_obj_flat = target_obj.reshape(-1)
            objectness_loss = F.binary_cross_entropy_with_logits(pred_objectness_flat, target_obj_flat, reduction='mean')

            # Bounding box loss
            obj_mask = target_obj == 1
            if obj_mask.any():
                pred_bbox_flat = pred_bbox.reshape(-1, 4)
                target_bbox_flat = target_bbox_scale.reshape(-1, 4)
                bbox_loss = F.mse_loss(pred_bbox_flat[obj_mask.view(-1)], target_bbox_flat[obj_mask.view(-1)], reduction='mean')

                # Class loss
                pred_class_scores_flat = pred_class_scores.reshape(-1, self.num_classes)
                target_class_flat = target_class_scale.view(-1)
                class_loss = F.cross_entropy(pred_class_scores_flat[obj_mask.view(-1)], target_class_flat[obj_mask.view(-1)], ignore_index=-1, reduction='mean')
            else:
                bbox_loss = torch.tensor(0.0, device=device)
                class_loss = torch.tensor(0.0, device=device)

            total_objectness_loss += objectness_loss
            total_bbox_loss += bbox_loss
            total_class_loss += class_loss

        train_loss = total_objectness_loss + total_bbox_loss + total_class_loss

        # Logging
        self.log('train_loss', train_loss.detach(), on_step=True, on_epoch=True)
        self.log('obj_loss', total_objectness_loss.detach(), on_step=True, on_epoch=True)
        self.log('bbox_loss', total_bbox_loss.detach(), on_step=True, on_epoch=True)
        self.log('class_loss', total_class_loss.detach(), on_step=True, on_epoch=True)

        return train_loss

    def compute_iou(self, anchors, box):
        """
        Compute IoU between anchors and target box.
        Args:
            anchors: Tensor of shape [num_anchors, 2] (widths, heights)
            box: Tensor of shape [1, 4] (x1, y1, x2, y2)
        Returns:
            iou: Tensor of shape [num_anchors]
        """
        anchors_w, anchors_h = anchors[:, 0], anchors[:, 1]
        box_w, box_h = box[0, 2], box[0, 3]

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