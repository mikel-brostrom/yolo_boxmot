import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from simple_yolo.assignment import compute_ciou, compute_iou
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

class SimpleObjectDetector(pl.LightningModule):
    def __init__(self, resnet_version='resnet18', num_boxes=1, num_classes=80, learning_rate=1e-5):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes  # Define the number of classes
        self.learning_rate = learning_rate

        # Define anchor box sizes for three scales
        self.anchors = torch.tensor([
            [0.1, 0.1], [0.2, 0.2], [0.4, 0.4],  # Small scale
            [0.1, 0.2], [0.2, 0.4], [0.4, 0.8],  # Medium scale
            [0.2, 0.1], [0.4, 0.2], [0.8, 0.4]   # Large scale
        ])  # (num_anchors, 2)

        # Initialize ResNet backbone based on specified version
        self.backbone = self._init_backbone(resnet_version)
        in_channels = self._get_in_channels(resnet_version)

        # Define dropout and advanced bbox regressor layers
        self.dropout = nn.Dropout(0.1)
        self.bbox_head = self._build_bbox_head(in_channels)

        # Define classification head
        #self.cls_head = self._build_cls_head(in_channels)
        
        #self.obj_head = self._build_obj_head(in_channels)

        # Initialize weights
        self._initialize_weights()
        
        self.cls_loss_function = nn.CrossEntropyLoss()  # Initialize the CrossEntropy loss
        self.obj_loss_function = nn.BCEWithLogitsLoss()


    def _init_backbone(self, resnet_version):
        """Initialize the ResNet backbone."""
        resnet_versions = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50
        }
        if resnet_version not in resnet_versions:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}. Choose from 'resnet18', 'resnet34', or 'resnet50'.")
        backbone = resnet_versions[resnet_version](pretrained=True)
        return nn.Sequential(*list(backbone.children())[:-2])  # Remove fully connected layers

    def _get_in_channels(self, resnet_version):
        """Return the number of input channels based on ResNet version."""
        return 512 if resnet_version in ['resnet18', 'resnet34'] else 2048

    def _build_bbox_head(self, in_channels):
        """Build the bounding box regressor head."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 4 * self.num_boxes * len(self.anchors), kernel_size=1)
        )

    def _build_cls_head(self, in_channels):
        """Build the classification head."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, self.num_classes * self.num_boxes * len(self.anchors), kernel_size=1)
        )
        
    def _build_obj_head(self, in_channels):
        """Build the classification head."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, self.num_boxes * len(self.anchors), kernel_size=1)
        )

    def _initialize_weights(self):
        """Initialize the weights of the heads."""
        for head in [self.bbox_head]:
            for m in head:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.constant_(m.bias, 0)
                    

    def forward(self, x):
        """Forward pass through the network."""
        features = self.backbone(x)
        features = self.dropout(features)
        
        # print(features.shape)

        # Predict bounding boxes
        bbox_pred = self.bbox_head(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        # # Predict classification scores
        # cls_pred = self.cls_head(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        
        # # Predict objectness scores
        # obj_pred = self.obj_head(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        # Apply sigmoid to dx, dy and exponential to dw, dh for stability in bbox predictions
        #obj_pred = torch.sigmoid(obj_pred)
        bbox_pred[..., :2] = torch.sigmoid(bbox_pred[..., :2])
        bbox_pred[..., 2:] = torch.exp(bbox_pred[..., 2:])
        
        return bbox_pred#, cls_pred, obj_pred


    def decode_anchors_to_image_space(self, anchors, feature_map_size, img_size):
        """
        Decode anchors from feature map space to image space.
        """
        num_anchors = anchors.shape[0]
        
        img_height, img_width = img_size
        stride_h, stride_w = 32, 32
        feature_map_height, feature_map_width = img_height // stride_h, img_width // stride_w

        # Prepare tensors to hold the decoded anchors
        decoded_anchors = torch.zeros((2304, 4), device=anchors.device)

        # Iterate over the feature map grid
        idx = 0
        for i in range(feature_map_height):
            for j in range(feature_map_width):
                # Convert anchor box ratios to image size
                for anchor in anchors:
                    anchor_width = anchor[0] * img_width
                    anchor_height = anchor[1] * img_height

                    # Calculate center of the anchor in image space
                    x_center = (j + 0.5) * stride_w  # Anchor centered in the grid cell
                    y_center = (i + 0.5) * stride_h  # Anchor centered in the grid cell

                    # Store decoded anchor (x_center, y_center, width, height)
                    decoded_anchors[idx, :] = torch.tensor([x_center, y_center, anchor_width, anchor_height])
                    idx += 1

        return decoded_anchors


    @staticmethod
    def match_anchors_to_ground_truth(anchors, ground_truth_boxes, iou_threshold=0.5):
        # Ensure anchors and ground_truth_boxes are 3D for IoU computation
        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(0)
        if ground_truth_boxes.dim() == 2:
            ground_truth_boxes = ground_truth_boxes.unsqueeze(0)

        # Move ground truth boxes to the same device as anchors
        ground_truth_boxes = ground_truth_boxes.to(anchors.device)

        # Compute IoU between each anchor and each ground truth box
        iou_matrix = compute_iou(anchors, ground_truth_boxes)  # Shape: (batch_size, num_anchors, num_gt_boxes)

        # Precompute max IoU and best-matching ground truth for each anchor
        max_iou_per_anchor, best_gt_per_anchor = torch.max(iou_matrix, dim=2)  # Shape: (batch_size, nr predictions)

        # Find positive anchors based on IoU threshold high
        positive_anchors_mask = (max_iou_per_anchor >= iou_threshold)  # Shape: (batch_size, nr predictions)

        # Find negative anchors based on IoU threshold low
        negative_anchors_mask = (max_iou_per_anchor < iou_threshold)

        # Prepare the assignment of anchors to ground truths
        anchor_to_gt_assignment = torch.where(positive_anchors_mask, best_gt_per_anchor, -1 * torch.ones_like(best_gt_per_anchor))
        
        # Return masks for positive, neutral, and negative anchors
        return positive_anchors_mask, negative_anchors_mask, anchor_to_gt_assignment  # Shape: (batch_size, nr predictions)


    def training_step(self, batch, batch_idx):
        images, targets_cls, targets_bbox, targets_obj = batch

        # Forward pass through the model to get predictions
        pred_bbox = self(images)

        # Assume pred_bbox is of shape (batch_size, num_anchors, 4) and represents predicted bounding boxes.
        # pred_cls is of shape (batch_size, num_anchors, num_classes) for class predictions.
        # pred_obj is of shape (batch_size, num_anchors) representing objectness scores.
        
        batch_size = images.size(0)
        num_anchors = pred_bbox.size(1)

        # Compute the ground truth matching for anchors
        positive_mask, negative_mask, anchor_to_gt_assignment = self.match_anchors_to_ground_truth(
            pred_bbox, targets_bbox, iou_threshold=0.5
        )

        # Initialize losses
        loss_bbox = 0
        loss_cls = 0
        loss_obj = 0

        for b in range(batch_size):
            # Extract the relevant ground truth for the current batch item
            gt_cls = targets_cls[b]  # Shape: (num_gt_boxes, num_classes)
            gt_bbox = targets_bbox[b]  # Shape: (num_gt_boxes, 4)
            gt_obj = targets_obj[b]  # Shape: (num_gt_boxes)

            # Retrieve the matched ground truth boxes for positive anchors
            pos_anchors = positive_mask[b]
            matched_gt_indices = anchor_to_gt_assignment[b][pos_anchors]

            if matched_gt_indices.numel() > 0:
                # Bounding Box Loss (e.g., Smooth L1 or GIoU Loss)
                matched_pred_bbox = pred_bbox[b][pos_anchors]
                matched_gt_bbox = gt_bbox[matched_gt_indices]
                
                loss_bbox += F.smooth_l1_loss(matched_pred_bbox, matched_gt_bbox)

        total_loss = loss_bbox
        
        # Log the losses
        # self.log('cls_loss', cls_loss_weight * loss_cls, prog_bar=True, on_step=True, on_epoch=True)
        # self.log('bbox_loss', reg_loss_weight * loss_bbox, prog_bar=True, on_step=True, on_epoch=True)
        # self.log('obj_loss', obj_loss_weight * loss_obj, prog_bar=True, on_step=True, on_epoch=True)
        self.log('total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Log or return the loss
        return total_loss

    def on_train_epoch_end(self):
        # Access the average metrics for the epoch
        avg_total_loss = self.trainer.callback_metrics['total_loss'].item()
        # avg_bbox_loss = self.trainer.callback_metrics['bbox_loss'].item()
        # avg_cls_loss = self.trainer.callback_metrics['cls_loss'].item()
        # avg_obj_loss = self.trainer.callback_metrics['obj_loss'].item()

        # Print the metrics
        print(f"Epoch {self.current_epoch} - Avg Total Loss: {avg_total_loss:.4f}")
        # print(f"Epoch {self.current_epoch} - Avg Reg Loss: {avg_bbox_loss:.4f}")
        # print(f"Epoch {self.current_epoch} - Avg Cls Loss: {avg_cls_loss:.4f}")
        # print(f"Epoch {self.current_epoch} - Avg Obj Loss: {avg_obj_loss:.4f}")


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss", "clip_grad": {"clip_val": 1.0}}
