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
    def __init__(self, resnet_version='resnet34', num_boxes=10, num_classes=80, learning_rate=1e-5, ciou_threshold=0.5):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.learning_rate = learning_rate
        self.train_losses = []
        self.ciou_threshold = ciou_threshold

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
        self.dropout = nn.Dropout(0.3)  # Increase dropout rate
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
        
        # Advanced bbox regressor: additional layers for more complex learning
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),  # Add BatchNorm after ReLU
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),  # Add BatchNorm after ReLU
            nn.Conv2d(512, 4 * num_boxes, kernel_size=1)
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

        # Predict bounding boxes
        bbox_pred = self.bbox_head(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        # Normalize bounding box predictions (x_min, y_min, x_max, y_max between 0 and 1)
        bbox_pred = torch.sigmoid(bbox_pred)
        return bbox_pred


    @staticmethod
    def match_predictions_to_targets(pred_bbox, target_bbox, ciou_threshold=0.5):
        """Match predicted bounding boxes to ground truth boxes and compute CIOU loss and L1 bbox loss."""
        batch_size = pred_bbox.shape[0]
        valid_mask = (target_bbox.sum(dim=-1) != 0)  # True if bbox is valid, False if bbox is padded

        total_bbox_loss = torch.tensor(0.0, device=pred_bbox.device, requires_grad=True)  # Start as a tensor
        all_pred_indices = []
        all_target_indices = []

        for i in range(batch_size):
            valid_target_bbox_i = target_bbox[i][valid_mask[i]]
            if len(valid_target_bbox_i) == 0:
                continue

            # Compute CIOU between predicted and target boxes
            ciou_matrix = compute_ciou(pred_bbox[i].unsqueeze(0), valid_target_bbox_i.unsqueeze(0))[0]

            # Apply a threshold to the CIoU matrix to determine valid matches
            valid_matches = (ciou_matrix >= ciou_threshold)  # Boolean matrix for valid CIoU matches

            pred_indices = []
            target_indices = []

            # Iterate over each target box
            for target_idx in range(valid_target_bbox_i.size(0)):
                # Get predicted boxes that match this target box
                matching_pred_indices = torch.where(valid_matches[:, target_idx])[0]

                if len(matching_pred_indices) > 0:
                    for pred_idx in matching_pred_indices:
                        pred_indices.append(pred_idx.item())
                        target_indices.append(target_idx)

                        # Compute the L1 Loss (bbox regression loss) for matched pairs
                        pred_box = pred_bbox[i, pred_idx]  # Predicted bounding box coordinates
                        target_box = valid_target_bbox_i[target_idx]  # Ground truth bounding box coordinates
                        bbox_loss = F.l1_loss(pred_box, target_box, reduction='sum')  # L1 Loss for bbox regression
                        total_bbox_loss = total_bbox_loss + bbox_loss

            # Store indices for further validation
            all_pred_indices.append(pred_indices)
            all_target_indices.append(target_indices)

        # Return total loss (CIOU + BBox Loss) and indices for validation
        total_loss = total_bbox_loss / batch_size
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
