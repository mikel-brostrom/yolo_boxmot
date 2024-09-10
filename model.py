import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.ops import box_iou
from assignment import match_and_compute_loss, compute_ciou
import torch.nn.functional as F


class SimpleObjectDetector(pl.LightningModule):
    def __init__(self, num_classes=80, num_boxes=100, learning_rate=1e-4):
        super(SimpleObjectDetector, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers

        self.classifier = nn.Conv2d(2048, num_classes * num_boxes, kernel_size=1)
        self.bbox_regressor = nn.Conv2d(2048, 4 * num_boxes, kernel_size=1)
        self.objectness = nn.Conv2d(2048, num_boxes, kernel_size=1)

        self.learning_rate = learning_rate
        self.train_losses = []

    def forward(self, x):
        features = self.backbone(x)
        cls_pred = self.classifier(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        bbox_pred = self.bbox_regressor(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        obj_pred = self.objectness(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        # Normalize bounding box predictions (x_min, y_min, x_max, y_max between 0 and 1)
        bbox_pred = torch.sigmoid(bbox_pred)
        return cls_pred, bbox_pred, torch.sigmoid(obj_pred)

    def training_step(self, batch, batch_idx):
        images, target_cls, target_bbox, target_obj = batch
        pred_cls, pred_bbox, pred_obj = self(images)
        
        # print('target_cls', target_cls.shape)
        # print('target_bbox', target_bbox.shape)
        # print('target_obj', target_obj.shape)
        # print('pred_cls', pred_cls.shape)
        # print('pred_bbox', pred_bbox.shape)
        # print('pred_obj', pred_obj.shape)

        # Compute CIoU and get the matched indices
        batch_size, num_preds, _ = pred_bbox.shape
        _, num_gts, _ = target_bbox.shape
        
        # Compute CIoU matrix and matched indices for each batch
        ciou_matrix = compute_ciou(pred_bbox, target_bbox)
        
        total_ciou_loss = 0.0
        matched_pred_bbox = []
        matched_target_bbox = []
        matched_pred_cls = []
        matched_target_cls = []
        matched_pred_obj = []
        matched_target_obj = []

        for i in range(batch_size):
            # Find the best matches for each ground truth box
            max_cious, best_pred_indices = torch.max(ciou_matrix[i], dim=0)
            # print('best_pred_indices', best_pred_indices.shape)

            # Filter matches based on IoU threshold
            valid_matches = max_cious >= 0.5
            
            # Calculate CIoU loss for matched pairs
            ciou_loss = 1 - max_cious[valid_matches]
            total_ciou_loss += ciou_loss.sum()

            # Extract matched predictions for classification and objectness loss
            #if valid_matches.sum() > 0:
            matched_pred_cls.append(pred_cls[i][best_pred_indices].unsqueeze(0))
            matched_target_cls.append(target_cls[i])
            matched_pred_obj.append(pred_obj[i][best_pred_indices].unsqueeze(0))
            matched_target_obj.append(target_obj[i])


        # Compute classification loss and objectness loss using matched pairs
        # Compute classification loss and objectness loss using matched pairs
        if matched_pred_cls:
            matched_pred_cls = torch.cat(matched_pred_cls, dim=0)
            matched_target_cls = torch.cat(matched_target_cls, dim=0)
            matched_pred_obj = torch.cat(matched_pred_obj, dim=0).squeeze(-1)
            matched_target_obj = torch.cat(matched_target_obj, dim=0)

            # Corrected classification loss calculation
            loss_cls = F.cross_entropy(matched_pred_cls, target_cls, reduction='mean')
            loss_obj = F.binary_cross_entropy(matched_pred_obj, target_obj, reduction='mean')

            # Total loss
            total_loss = total_ciou_loss / batch_size + loss_cls + loss_obj
        else:
            # If no matches found, set losses to 0
            loss_cls = torch.tensor(0.0, device=images.device)
            loss_obj = torch.tensor(0.0, device=images.device)
            total_loss = total_ciou_loss / batch_size

        self.log('train_loss', total_loss, prog_bar=True)
        
        # Append as tensors instead of floats
        self.train_losses.append({
            'loss': total_loss, 
            'loss_cls': loss_cls, 
            'loss_bbox': total_ciou_loss / batch_size, 
            'loss_obj': loss_obj
        })

        return total_loss


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