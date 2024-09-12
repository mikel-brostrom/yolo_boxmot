import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from simple_yolo.assignment import compute_ciou  # Ensure you have this function
from scipy.optimize import linear_sum_assignment  # For optimal assignment

class SimpleObjectDetector(pl.LightningModule):
    def __init__(self, resnet_version='resnet18', num_boxes=100, num_classes=80, learning_rate=1e-4):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.learning_rate = learning_rate
        self.train_losses = []

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
        self.dropout = nn.Dropout(0.05)  # Increase dropout rate
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
        self.bbox_regressor = nn.Conv2d(in_channels, 4 * num_boxes, kernel_size=1)

        # Initialize weights
        nn.init.normal_(self.bbox_regressor.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bbox_regressor.bias, 0)


    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)
        features = self.dropout(features)

        # Predict bounding boxes
        bbox_pred = self.bbox_regressor(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        # Normalize bounding box predictions (x_min, y_min, x_max, y_max between 0 and 1)
        bbox_pred = torch.sigmoid(bbox_pred)
        return bbox_pred

    def match_predictions_to_targets(self, pred_bbox, target_bbox):
        """Match predicted bounding boxes to ground truth boxes and compute CIOU loss."""
        batch_size = pred_bbox.shape[0]
        valid_mask = (target_bbox.sum(dim=-1) != 0)  # True if bbox is valid, False if bbox is padded

        total_ciou_loss = 0.0

        for i in range(batch_size):
            valid_target_bbox_i = target_bbox[i][valid_mask[i]]
            if len(valid_target_bbox_i) == 0:
                continue

            # Compute CIOU between predicted and target boxes
            ciou_matrix = compute_ciou(pred_bbox[i].unsqueeze(0), valid_target_bbox_i.unsqueeze(0))[0]

            # Convert ciou_matrix to NumPy for linear_sum_assignment
            ciou_matrix_np = ciou_matrix.cpu().detach().numpy()

            # Optimal assignment using Hungarian algorithm
            pred_indices, target_indices = linear_sum_assignment(-ciou_matrix_np)

            # Convert back to PyTorch tensors
            ciou_loss = 1 - ciou_matrix[pred_indices, target_indices]

            # Sum the losses for valid matches
            total_ciou_loss += ciou_loss.sum()

        # Average CIOU loss over the batch and ensure it's a tensor
        total_ciou_loss = total_ciou_loss / batch_size
        return total_ciou_loss

    def training_step(self, batch, batch_idx):
        images, _, target_bbox, _ = batch
        pred_bbox = self(images)

        # Calculate bbox loss
        total_ciou_loss = self.match_predictions_to_targets(pred_bbox, target_bbox)

        # Log bbox loss
        self.log('train_bbox_loss', total_ciou_loss, prog_bar=True, on_epoch=True)

        # Return loss for optimization
        return total_ciou_loss

    def on_train_epoch_end(self):
        # The average loss will be automatically calculated by Lightning when using self.log with on_epoch=True.
        print(f"Epoch {self.current_epoch} - Avg BBox Loss: {self.trainer.callback_metrics['train_bbox_loss']:.4f}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_bbox_loss", "clip_grad": {"clip_val": 1.0}}
