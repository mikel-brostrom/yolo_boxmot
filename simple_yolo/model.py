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
    def __init__(self, resnet_version='resnet18', num_boxes=1, num_classes=80, learning_rate=1e-3):
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
        self.dropout = nn.Dropout(0.5)
        self.bbox_head = self._build_bbox_head(in_channels)

        # Define classification head
        self.cls_head = self._build_cls_head(in_channels)
        
        self.obj_head = self._build_obj_head(in_channels)

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
        for head in [self.bbox_head, self.cls_head, self.obj_head]:
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

        # Predict classification scores
        cls_pred = self.cls_head(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        
        # Predict objectness scores
        obj_pred = self.obj_head(features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        # Apply sigmoid to dx, dy and exponential to dw, dh for stability in bbox predictions
        obj_pred = torch.sigmoid(obj_pred)
        bbox_pred[..., :2] = torch.sigmoid(bbox_pred[..., :2])
        bbox_pred[..., 2:] = torch.exp(bbox_pred[..., 2:])
        
        return bbox_pred, cls_pred, obj_pred


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
    def match_anchors_to_ground_truth(anchors, ground_truth_boxes, iou_threshold_high=0.5, iou_threshold_low=0.3):

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
        max_iou_per_anchor, best_gt_per_anchor = torch.max(iou_matrix, dim=2)  # Shape: (batch_size, num_anchors)
        
        positive_anchors = [set() for _ in range(len(anchors))]
        negative_anchors = [set(range(len(anchors[0]))) for _ in range(len(anchors))]
        anchor_to_gt_assignment = [{} for _ in range(len(anchors))]

        # Assign each ground truth to the best-matching anchor if IoU is above high threshold
        for batch_idx in range(len(anchors)):
            # For each ground truth, find the best matching anchor
            max_iou_per_gt, best_anchor_per_gt = torch.max(iou_matrix[batch_idx], dim=0)
            
            for gt_idx in range(len(ground_truth_boxes[batch_idx])):
                best_anchor = best_anchor_per_gt[gt_idx].item()
                if max_iou_per_gt[gt_idx].item() >= iou_threshold_high:
                    positive_anchors[batch_idx].add(best_anchor)
                    anchor_to_gt_assignment[batch_idx][best_anchor] = gt_idx

            # Assign additional positive anchors and determine negatives
            for anchor_idx in range(len(anchors[batch_idx])):
                max_iou = max_iou_per_anchor[batch_idx, anchor_idx].item()
                gt_idx = best_gt_per_anchor[batch_idx, anchor_idx].item()

                # Assign positive anchor if IoU exceeds the high threshold
                if max_iou >= iou_threshold_high:
                    positive_anchors[batch_idx].add(anchor_idx)
                    anchor_to_gt_assignment[batch_idx][anchor_idx] = gt_idx

                # Assign negative anchor if IoU is below the low threshold
                elif max_iou < iou_threshold_low:
                    negative_anchors[batch_idx].add(anchor_idx)

            # Remove positive anchors from negative set
            negative_anchors[batch_idx] -= positive_anchors[batch_idx]

        return positive_anchors, negative_anchors, anchor_to_gt_assignment


    def training_step(self, batch, batch_idx):
        images, targets_cls, targets_bbox, targets_obj = batch
        
        # print(targets_cls[0][0])

        # Forward pass through the model to get predictions
        pred_bbox, pred_cls, pred_obj = self(images)
        
        # print(pred_cls[0][0])

        # Match anchors (predicted boxes) to ground truth boxes
        positive_anchors, negative_anchors, anchor_to_gt_assignment = self.match_anchors_to_ground_truth(pred_bbox, targets_bbox)

        # Create masks for positive and negative anchors
        positive_anchor_mask = torch.zeros(pred_bbox.shape[:2], dtype=torch.bool, device=pred_bbox.device)
        # negative_anchor_mask = torch.zeros(pred_bbox.shape[:2], dtype=torch.bool, device=pred_bbox.device)

        # Populate the masks for positive and negative anchors
        for b_idx in range(len(positive_anchors)):
            for anchor_idx in positive_anchors[b_idx]:
                positive_anchor_mask[b_idx, anchor_idx] = True
            # for anchor_idx in negative_anchors[b_idx]:
            #     negative_anchor_mask[b_idx, anchor_idx] = True

        # Filter predictions using the positive anchor mask
        positive_pred_bbox = pred_bbox[positive_anchor_mask]
        positive_pred_cls = pred_cls[positive_anchor_mask]
        positive_pred_obj = pred_obj[positive_anchor_mask]
        
        # print(positive_pred_bbox.shape)
        # print(positive_pred_cls.shape)
        # print(positive_pred_obj.shape)
        
        # Initialize the ground truth bbox tensor with the correct size
        num_pos_anchors = positive_anchor_mask.sum().item()
        positive_gt_bbox = torch.zeros((num_pos_anchors, 4), dtype=targets_bbox.dtype, device=targets_bbox.device)
        positive_gt_cls = torch.zeros((num_pos_anchors, pred_cls.shape[-1]), dtype=pred_cls.dtype, device=pred_cls.device)
        positive_gt_obj = torch.zeros((num_pos_anchors, 1), dtype=pred_cls.dtype, device=pred_cls.device)

        pos_anchor_counter = 0  # Counter to track the index in positive_gt_bbox
        for b_idx, anchor_gt_map in enumerate(anchor_to_gt_assignment):
            for anchor_idx, gt_idx in anchor_gt_map.items():
                # Assign the ground truth bbox to the positive_gt_bbox
                positive_gt_bbox[pos_anchor_counter] = targets_bbox[b_idx][gt_idx]
                positive_gt_cls[pos_anchor_counter] = targets_cls[b_idx, gt_idx]
                positive_gt_obj[pos_anchor_counter] = targets_obj[b_idx, gt_idx]
                pos_anchor_counter += 1
                          
        # print(positive_gt_bbox.shape)
        # print(positive_gt_cls.shape)
        # print(positive_gt_obj.shape)
        
        # Convert one-hot ground truth to class indices
        positive_gt_cls = torch.argmax(positive_gt_cls, dim=-1).long()

        # Compute classification loss
        cls_loss = self.cls_loss_function(positive_pred_cls, positive_gt_cls)

        # Bounding Box Loss: Compute for positive anchors
        bbox_loss = F.smooth_l1_loss(positive_pred_bbox, positive_gt_bbox, reduction='mean')  # L1 Loss for bbox regression

        # Objectness Loss: Compute for both positive and negative anchors
        obj_loss = self.obj_loss_function(positive_pred_obj, positive_gt_obj)

        # Combine losses (you can adjust the weights as needed)
        cls_loss_weight = 0.2  # Scale down classification loss
        reg_loss_weight = 100  # Scale up regression loss
        obj_loss_weight = 1.5  # Slightly increase objectness loss

        total_loss = (
            cls_loss_weight * cls_loss +
            reg_loss_weight * bbox_loss +
            obj_loss_weight * obj_loss
        )

        # Log the losses
        self.log('cls_loss', cls_loss_weight * cls_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('bbox_loss', reg_loss_weight * bbox_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('obj_loss', obj_loss_weight * obj_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        # Access the average metrics for the epoch
        avg_total_loss = self.trainer.callback_metrics['total_loss'].item()
        avg_bbox_loss = self.trainer.callback_metrics['bbox_loss'].item()
        avg_cls_loss = self.trainer.callback_metrics['cls_loss'].item()
        avg_obj_loss = self.trainer.callback_metrics['obj_loss'].item()

        # Print the metrics
        print(f"Epoch {self.current_epoch} - Avg Total Loss: {avg_total_loss:.4f}")
        print(f"Epoch {self.current_epoch} - Avg Reg Loss: {avg_bbox_loss:.4f}")
        print(f"Epoch {self.current_epoch} - Avg Cls Loss: {avg_cls_loss:.4f}")
        print(f"Epoch {self.current_epoch} - Avg Obj Loss: {avg_obj_loss:.4f}")


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss", "clip_grad": {"clip_val": 1.0}}
