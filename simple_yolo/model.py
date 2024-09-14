import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from simple_yolo.assignment import compute_ciou  # Ensure you have this function
from scipy.optimize import linear_sum_assignment  # For optimal assignment
import torch.nn.functional as F


class SimpleObjectDetector(pl.LightningModule):
    def __init__(self, resnet_version='resnet18', num_boxes=10, num_classes=80, learning_rate=1e-4, ciou_threshold=0.5):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.learning_rate = learning_rate
        self.train_losses = []
        self.ciou_threshold = ciou_threshold

        # Define anchors (width, height) ratios for 3 different scales and 3 aspect ratios
        self.anchors = torch.tensor([
            [0.1, 0.1], [0.2, 0.2], [0.4, 0.4],  # Small scale
            [0.1, 0.2], [0.2, 0.4], [0.4, 0.8],  # Medium scale
            [0.2, 0.1], [0.4, 0.2], [0.8, 0.4],  # Large scale
        ])  # Shape: (num_anchors, 2)

        # Choose ResNet backbone based on the version specified
        if resnet_version == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            in_channels = 512
        elif resnet_version == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            in_channels = 512
        elif resnet_version == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_channels = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}. Choose 'resnet18', 'resnet34', or 'resnet50'.")

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)  # Increase dropout rate
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers

        # Advanced bbox regressor: additional layers for more complex learning
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),  # Add BatchNorm after ReLU
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),  # Add BatchNorm after ReLU
            nn.Conv2d(512, 4 * num_boxes * len(self.anchors), kernel_size=1)  # Adjust for anchors
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the heads."""
        for m in self.bbox_head:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)
        features = self.dropout(features)

        # Predict bounding boxes (dx, dy, dw, dh relative to anchors)
        bbox_pred = self.bbox_head(features).permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(x.size(0), -1, 4)

        # Apply sigmoid to dx, dy and exponential to dw, dh for stability
        bbox_pred[..., :2] = torch.sigmoid(bbox_pred[..., :2])  # dx, dy
        bbox_pred[..., 2:] = torch.exp(bbox_pred[..., 2:])  # dw, dh

        return bbox_pred

    def decode_boxes(self, bbox_pred, anchors, feature_map_size, img_size):
        """Convert predicted offsets to bounding box coordinates."""
        # Calculate grid size
        grid_h, grid_w = feature_map_size
        stride_h, stride_w = img_size[0] / grid_h, img_size[1] / grid_w

        # Decode each predicted box
        boxes = []
        for i in range(grid_h):
            for j in range(grid_w):
                for anchor in anchors:
                    dx, dy, dw, dh = bbox_pred[:, i, j, :4].T
                    x_center = (j + dx) * stride_w
                    y_center = (i + dy) * stride_h
                    w = anchor[0] * dw * img_size[1]
                    h = anchor[1] * dh * img_size[0]

                    # Convert (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2

                    boxes.append([x_min, y_min, x_max, y_max])

        return torch.stack(boxes, dim=1)

    @staticmethod
    def match_predictions_to_targets(pred_bbox, target_bbox, ciou_threshold=0.7):
        """
        Match predicted bounding boxes to ground truth boxes and compute CIOU loss and L1 bbox loss.

        Args:
            pred_bbox (torch.Tensor): Predicted bounding boxes decoded from the model (batch_size, num_boxes, 4).
            target_bbox (torch.Tensor): Ground truth bounding boxes (batch_size, max_num_boxes, 4).
            ciou_threshold (float): The threshold for considering a valid match based on CIOU.

        Returns:
            total_loss (torch.Tensor): Combined loss of CIOU and L1 loss.
            all_pred_indices (list): List of matched prediction indices for each sample in the batch.
            all_target_indices (list): List of matched target indices for each sample in the batch.
        """
        batch_size = pred_bbox.shape[0]
        valid_mask = (target_bbox.sum(dim=-1) != 0)  # True if bbox is valid, False if bbox is padded

        total_ciou_loss = torch.tensor(0.0, device=pred_bbox.device, requires_grad=True)  # Initialize loss tensors
        total_l1_loss = torch.tensor(0.0, device=pred_bbox.device, requires_grad=True)
        all_pred_indices = []
        all_target_indices = []

        for i in range(batch_size):
            valid_target_bbox_i = target_bbox[i][valid_mask[i]]  # Filter valid boxes
            if len(valid_target_bbox_i) == 0:
                continue

            # Compute CIOU between predicted and target boxes
            ciou_matrix = compute_ciou(pred_bbox[i].unsqueeze(0), valid_target_bbox_i.unsqueeze(0))[0]  # (num_preds, num_targets)

            # Convert the CIOU matrix to a cost matrix for the Hungarian algorithm
            cost_matrix = 1 - ciou_matrix  # Lower cost is better, hence (1 - CIOU)

            # Use Hungarian algorithm to find optimal one-to-one matches
            pred_indices, target_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())  # Move to CPU for numpy

            # Convert back to tensors
            pred_indices = torch.tensor(pred_indices, device=pred_bbox.device)
            target_indices = torch.tensor(target_indices, device=pred_bbox.device)

            # Filter matches by the CIOU threshold
            valid_matches = ciou_matrix[pred_indices, target_indices] >= ciou_threshold
            pred_indices = pred_indices[valid_matches]
            target_indices = target_indices[valid_matches]

            # Compute CIOU and L1 Loss for matched pairs
            for pred_idx, target_idx in zip(pred_indices, target_indices):
                pred_box = pred_bbox[i, pred_idx]  # Predicted bounding box coordinates (decoded)
                target_box = valid_target_bbox_i[target_idx]  # Ground truth bounding box coordinates

                # Compute the CIOU loss
                print(pred_box.unsqueeze(0).shape, target_box.unsqueeze(0).shape)
                ciou_loss = 1 - compute_ciou(pred_box.unsqueeze(0).unsqueeze(0), target_box.unsqueeze(0).unsqueeze(0)).squeeze()
                total_ciou_loss = total_ciou_loss + ciou_loss

                # Compute the L1 Loss (bbox regression loss) for matched pairs
                l1_loss = F.l1_loss(pred_box, target_box, reduction='mean')  # L1 Loss for bbox regression
                total_l1_loss = total_l1_loss + l1_loss

            # Store indices for further validation
            all_pred_indices.append(pred_indices.tolist())
            all_target_indices.append(target_indices.tolist())

        # Return total loss (CIOU + BBox Loss) and indices for validation
        total_loss = (total_ciou_loss + total_l1_loss) / batch_size
        return total_loss, all_pred_indices, all_target_indices

    def training_step(self, batch, batch_idx):
        images, _, target_bbox, _ = batch
        pred_bbox = self(images)

        # Calculate bbox loss
        total_loss, all_pred_indices, all_target_indices = self.match_predictions_to_targets(pred_bbox, target_bbox)

        # Log bbox loss
        self.log('train_bbox_loss', total_loss, prog_bar=True, on_epoch=True)

        # Return loss for optimization
        return total_loss

    def on_train_epoch_end(self):
        # The average loss will be automatically calculated by Lightning when using self.log with on_epoch=True.
        print(f"Epoch {self.current_epoch} - Avg BBox Loss: {self.trainer.callback_metrics['train_bbox_loss']:.4f}")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_bbox_loss", "clip_grad": {"clip_val": 1.0}}
