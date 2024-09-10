import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.ops import box_iou
from simple_yolo.assignment import match_and_compute_loss, compute_ciou
import torch.nn.functional as F


class SimpleObjectDetector(pl.LightningModule):
    def __init__(self, resnet_version='resnet18', num_classes=80, num_boxes=10, learning_rate=1e-3):
        """
        Initialize the SimpleObjectDetector with different ResNet backbones.
        
        Args:
            resnet_version (str): ResNet version ('resnet18', 'resnet34', 'resnet50').
            num_classes (int): Number of object classes.
            num_boxes (int): Number of bounding boxes to predict.
            learning_rate (float): Learning rate for optimizer.
        """
        super(SimpleObjectDetector, self).__init__()
        self.num_classes = num_classes
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
        self.dropout = nn.Dropout(0.3)
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers

        self.classifier = nn.Conv2d(in_channels, num_classes * num_boxes, kernel_size=1)
        self.bbox_regressor = nn.Conv2d(in_channels, 4 * num_boxes, kernel_size=1)
        self.objectness = nn.Conv2d(in_channels, num_boxes, kernel_size=1)
        self.learning_rate = learning_rate

    def forward(self, x):
        # (batch_size, 512, 7, 7) (for resnet18 or resnet34 with input size 224x224)
        # (batch_size, 512, 16, 16) (for resnet18 or resnet34 with input size 224x224)
        features = self.backbone(x)
        features = self.dropout(features)
        # (batch_size, (height//32 * width//32 * num_boxes), num_classes)
        # (batch_size, 7*7*num_boxes, num_classes)
        cls_pred = self.classifier(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        # (batch_size, (height//32 * width//32 * num_boxes), 4)
        # (batch_size, 7*7*num_boxes, 4)
        bbox_pred = self.bbox_regressor(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        # (batch_size, (height//32 * width//32 * num_boxes), 1)
        # (batch_size, 7*7*num_boxes, 1)
        obj_pred = self.objectness(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        # Normalize bounding box predictions (x_min, y_min, x_max, y_max between 0 and 1)
        bbox_pred = torch.sigmoid(bbox_pred)
        return cls_pred, bbox_pred, torch.sigmoid(obj_pred)


    def match_predictions_to_targets(self, pred_cls, pred_bbox, pred_obj, target_cls, target_bbox, target_obj):
        """Match predicted bounding boxes to ground truth boxes and compute losses."""
        batch_size = pred_bbox.shape[0]
        valid_mask = (target_bbox.sum(dim=-1) != 0)  # True if bbox is valid, False if bbox is padded

        total_ciou_loss = 0.0
        matched_pred_cls, matched_target_cls = [], []
        matched_pred_obj, matched_target_obj = [], []

        for i in range(batch_size):
            valid_target_bbox_i = target_bbox[i][valid_mask[i]]
            if len(valid_target_bbox_i) == 0:
                continue

            ciou_matrix = compute_ciou(pred_bbox[i].unsqueeze(0), valid_target_bbox_i.unsqueeze(0))[0]
            max_cious, best_pred_indices = torch.max(ciou_matrix, dim=0)
            valid_matches = max_cious >= 0.5

            if valid_matches.sum() == 0:
                continue

            ciou_loss = 1 - max_cious[valid_matches]
            total_ciou_loss += ciou_loss.sum()

            matched_pred_cls.append(pred_cls[i][best_pred_indices[valid_matches]])
            matched_target_cls.append(target_cls[i][valid_mask[i]][valid_matches])
            matched_pred_obj.append(pred_obj[i][best_pred_indices[valid_matches]])
            matched_target_obj.append(target_obj[i][valid_mask[i]][valid_matches])

        if matched_pred_cls:
            matched_pred_cls = torch.cat(matched_pred_cls, dim=0)
            matched_target_cls = torch.cat(matched_target_cls, dim=0)
            matched_pred_obj = torch.cat(matched_pred_obj, dim=0).squeeze(-1)
            matched_target_obj = torch.cat(matched_target_obj, dim=0)

            # Compute Focal Loss for classification
            loss_cls = F.cross_entropy(matched_pred_cls, matched_target_cls, reduction='mean')

            # Compute objectness loss
            loss_obj = F.binary_cross_entropy(matched_pred_obj, matched_target_obj, reduction='mean')

            return total_ciou_loss / batch_size, loss_cls, loss_obj
        else:
            loss_cls = torch.tensor(0.0, device=pred_cls.device)
            loss_obj = torch.tensor(0.0, device=pred_cls.device)
            return total_ciou_loss / batch_size, loss_cls, loss_obj

    def training_step(self, batch, batch_idx):
        images, target_cls, target_bbox, target_obj = batch
        pred_cls, pred_bbox, pred_obj = self(images)

        # Use the matching function to calculate losses
        total_ciou_loss, loss_cls, loss_obj = self.match_predictions_to_targets(
            pred_cls, pred_bbox, pred_obj, target_cls, target_bbox, target_obj
        )

        # Total loss
        total_loss = total_ciou_loss + loss_cls + loss_obj
        # Logging losses
        self.log('train_loss', total_loss, prog_bar=True)

        # Append loss details
        self.train_losses.append({
            'loss': total_loss.detach(),
            'loss_cls': loss_cls.detach(),
            'loss_bbox': (total_ciou_loss).detach(),
            'loss_obj': loss_obj.detach()
        })


    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.train_losses]).mean()
        avg_loss_cls = torch.stack([x['loss_cls'] for x in self.train_losses]).mean()
        avg_loss_bbox = torch.stack([x['loss_bbox'] for x in self.train_losses]).mean()
        avg_loss_obj = torch.stack([x['loss_obj'] for x in self.train_losses]).mean()

        self.log('avg_train_loss', avg_loss, prog_bar=True)
        print(f"Epoch {self.current_epoch} - Loss: {avg_loss:.4f}, Class Loss: {avg_loss_cls:.4f}, BBox Loss: {avg_loss_bbox:.4f}, Objectness Loss: {avg_loss_obj:.4f}")
        self.train_losses.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}