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
    def __init__(self, resnet_version='resnet18', num_boxes=1, num_classes=80, learning_rate=1e-3, input_size=512):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input_size = input_size
        
        # Define widths and heights
        self.widths = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32, device='mps')
        self.heights = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32, device='mps')

        # Compute the number of anchor sizes
        self.num_anchor_sizes = len(self.widths) * len(self.heights)

        # Initialize ResNet backbone based on specified version
        self.backbone = self._init_backbone(resnet_version)
        in_channels = self._get_in_channels(resnet_version)

        # Compute expected feature map size
        self.grid_height, self.grid_width = self._compute_feature_map_size(self.input_size)
        
        # Generate anchors once and register as buffer
        self.anchors = self.generate_anchors(self.grid_height, self.grid_width)

        # Define dropout and advanced bbox regressor layers
        self.dropout = nn.Dropout(0.1)
        self.bbox_head = self._build_bbox_head(in_channels)

        # Initialize weights
        self._initialize_weights()

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
            nn.Conv2d(in_channels, 1024, kernel_size=1),  # Adjusted input channels
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 4 * self.num_boxes * self.num_anchor_sizes, kernel_size=1)
        )

    def _initialize_weights(self):
        """Initialize the weights of the heads."""
        for head in [self.bbox_head]:
            for m in head:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.constant_(m.bias, 0)

    def _compute_feature_map_size(self, input_size):
        """Compute the feature map size given the input size."""
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        with torch.no_grad():
            dummy_features = self.backbone(dummy_input)
        grid_height, grid_width = dummy_features.shape[2], dummy_features.shape[3]
        return grid_height, grid_width

    def forward(self, x, decode=True):
        # Pass input through the backbone to get feature maps
        features = self.backbone(x)
        
        # Pass features through the bounding box head
        bbox_pred = self.bbox_head(features)  # Shape: (batch_size, 4 * num_anchor_sizes, grid_height, grid_width)
        
        batch_size = x.size(0)
        grid_height = features.size(2)
        grid_width = features.size(3)
        
        # Reshape bbox_pred to (batch_size, num_anchor_sizes, 4, grid_height, grid_width)
        bbox_pred = bbox_pred.view(batch_size, self.num_anchor_sizes, 4, grid_height, grid_width)
        
        # Permute dimensions to (batch_size, grid_height, grid_width, num_anchor_sizes, 4)
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2)  # Shape: (batch_size, grid_height, grid_width, num_anchor_sizes, 4)
        
        # Flatten spatial dimensions and anchor sizes to get (batch_size, num_anchors, 4)
        bbox_pred = bbox_pred.reshape(batch_size, -1, 4)
        
        if decode:
            # Move anchors to the same device as input
            anchors = self.anchors.to(x.device)
            # Expand anchors to match batch size
            anchors_expanded = anchors.unsqueeze(0).expand(batch_size, -1, -1)
            # Decode the predicted offsets to get final bounding boxes
            decoded_boxes = self.decode_boxes(bbox_pred, anchors_expanded)
            return decoded_boxes
        else:
            return bbox_pred


    def generate_anchors(self, grid_height, grid_width):
        """
        Generates anchors based on the feature map size.
        """
        device = torch.device('mps')
        # Generate normalized grid of anchor center positions
        center_x = (torch.arange(grid_width, dtype=torch.float32, device=device) + 0.5) / grid_width
        center_y = (torch.arange(grid_height, dtype=torch.float32, device=device) + 0.5) / grid_height
        center_y, center_x = torch.meshgrid(center_y, center_x, indexing='ij')
        anchor_centers = torch.stack([center_x, center_y], dim=-1).reshape(-1, 2)  # Shape: (num_anchor_centers, 2)

        # Anchor sizes in normalized coordinates
        anchor_sizes = torch.stack(torch.meshgrid(self.widths, self.heights, indexing='ij'), dim=-1).reshape(-1, 2)  # Shape: (num_anchor_sizes, 2)

        # Get all combinations of centers and sizes
        num_anchor_centers = anchor_centers.size(0)
        num_anchor_sizes = anchor_sizes.size(0)
        anchor_centers = anchor_centers.repeat_interleave(num_anchor_sizes, dim=0)
        anchor_sizes = anchor_sizes.repeat(num_anchor_centers, 1)

        # Combine centers and sizes to get anchors
        anchors = torch.cat([anchor_centers, anchor_sizes], dim=1)  # Shape: (num_anchors, 4)
        return anchors


    def encode_boxes(self, gt_boxes, anchors):
        """
        Encodes the ground truth boxes relative to the anchors.
        gt_boxes: Tensor of shape (num_anchors, 4)
        anchors: Tensor of shape (num_anchors, 4)
        """
        # Ground truth box parameters
        gt_x_min = gt_boxes[..., 0]
        gt_y_min = gt_boxes[..., 1]
        gt_x_max = gt_boxes[..., 2]
        gt_y_max = gt_boxes[..., 3]
        gt_center_x = (gt_x_min + gt_x_max) / 2.0
        gt_center_y = (gt_y_min + gt_y_max) / 2.0
        gt_width = gt_x_max - gt_x_min
        gt_height = gt_y_max - gt_y_min

        # Anchor box parameters
        anchor_center_x = anchors[..., 0]
        anchor_center_y = anchors[..., 1]
        anchor_width = anchors[..., 2]
        anchor_height = anchors[..., 3]

        # Encoding
        t_x = (gt_center_x - anchor_center_x) / anchor_width
        t_y = (gt_center_y - anchor_center_y) / anchor_height
        t_w = torch.log(gt_width / anchor_width)
        t_h = torch.log(gt_height / anchor_height)

        encoded_boxes = torch.stack([t_x, t_y, t_w, t_h], dim=-1)
        return encoded_boxes

    def decode_boxes(self, bbox_pred, anchors):
        """
        Decodes the predicted bounding boxes relative to the provided anchors.
        bbox_pred: Tensor of shape (batch_size, num_anchors, 4)
        anchors: Tensor of shape (batch_size, num_anchors, 4)
        """
        # Predicted offsets
        t_x = bbox_pred[..., 0]
        t_y = bbox_pred[..., 1]
        t_w = bbox_pred[..., 2]
        t_h = bbox_pred[..., 3]

        # Anchor box parameters
        anchor_center_x = anchors[..., 0]
        anchor_center_y = anchors[..., 1]
        anchor_width = anchors[..., 2]
        anchor_height = anchors[..., 3]

        # Decoding
        decoded_center_x = t_x * anchor_width + anchor_center_x
        decoded_center_y = t_y * anchor_height + anchor_center_y
        decoded_width = torch.exp(t_w) * anchor_width
        decoded_height = torch.exp(t_h) * anchor_height

        # Convert to (x_min, y_min, x_max, y_max)
        x_min = decoded_center_x - 0.5 * decoded_width
        y_min = decoded_center_y - 0.5 * decoded_height
        x_max = decoded_center_x + 0.5 * decoded_width
        y_max = decoded_center_y + 0.5 * decoded_height

        decoded_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        return decoded_boxes

    @staticmethod
    def convert_anchors_to_corners(anchors):
        """
        Converts anchors from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max).
        """
        cx = anchors[..., 0]
        cy = anchors[..., 1]
        w = anchors[..., 2]
        h = anchors[..., 3]
        x_min = cx - 0.5 * w
        y_min = cy - 0.5 * h
        x_max = cx + 0.5 * w
        y_max = cy + 0.5 * h
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    @staticmethod
    def match_anchors_to_ground_truth(anchors, ground_truth_boxes, iou_threshold=0.5):
        batch_size = anchors.size(0)
        num_anchors = anchors.size(1)
        positive_masks = []
        negative_masks = []
        anchor_to_gt_assignments = []

        for i in range(batch_size):
            anchors_per_image = anchors[i]  # Shape: (num_anchors, 4)
            gt_boxes_per_image = ground_truth_boxes[i]  # Shape: (num_gt_boxes_i, 4)
            num_gt_boxes = gt_boxes_per_image.size(0)

            # Convert anchors to (x_min, y_min, x_max, y_max)
            anchors_corners = SimpleObjectDetector.convert_anchors_to_corners(anchors_per_image)

            if num_gt_boxes == 0:
                # No ground truth boxes in this image
                positive_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
                negative_mask = torch.ones(num_anchors, dtype=torch.bool, device=anchors.device)
                anchor_to_gt_assignment = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)
            else:
                iou_matrix = compute_iou(anchors_corners.unsqueeze(0), gt_boxes_per_image.unsqueeze(0))[0]
                max_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)

                positive_mask = max_iou_per_anchor >= iou_threshold
                negative_mask = max_iou_per_anchor < iou_threshold
                anchor_to_gt_assignment = torch.where(
                    positive_mask, best_gt_per_anchor, torch.full_like(best_gt_per_anchor, -1)
                )

            positive_masks.append(positive_mask)
            negative_masks.append(negative_mask)
            anchor_to_gt_assignments.append(anchor_to_gt_assignment)

        positive_mask = torch.stack(positive_masks)
        negative_mask = torch.stack(negative_masks)
        anchor_to_gt_assignment = torch.stack(anchor_to_gt_assignments)

        return positive_mask, negative_mask, anchor_to_gt_assignment

    def training_step(self, batch, batch_idx):
        images, targets_cls, targets_bbox, targets_obj = batch
        device = images.device

        # Forward pass
        pred_bbox = self(images, decode=False)

        batch_size = images.size(0)
        num_anchors = pred_bbox.size(1)

        # Use anchors from the model
        anchors_expanded = self.anchors.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        # Match anchors to ground truth boxes
        positive_mask, negative_mask, anchor_to_gt_assignment = self.match_anchors_to_ground_truth(
            anchors_expanded, targets_bbox, iou_threshold=0.5
        )

        # Ensure anchor_to_gt_assignment has valid indices
        anchor_to_gt_assignment = anchor_to_gt_assignment.long()

        # Compute loss only for positive anchors
        if positive_mask.any():
            # Get indices of positive anchors
            positive_anchor_indices = positive_mask.nonzero(as_tuple=True)

            # Get the matched ground truth indices
            matched_gt_indices = anchor_to_gt_assignment[positive_anchor_indices]

            # Gather the matched ground truth boxes
            matched_gt_boxes = targets_bbox[positive_anchor_indices[0], matched_gt_indices]

            # Get anchors for positive anchors
            anchors_positive = anchors_expanded[positive_anchor_indices]

            # Encode ground truth boxes relative to anchors
            target_reg = self.encode_boxes(matched_gt_boxes, anchors_positive)

            # Get predicted offsets for positive anchors
            pred_reg = pred_bbox[positive_anchor_indices]

            # Compute loss
            loss = F.smooth_l1_loss(pred_reg, target_reg)
        else:
            # No positive samples, set loss to zero
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Log and return loss
        self.log('total_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Access the average metrics for the epoch
        avg_total_loss = self.trainer.callback_metrics['total_loss'].item()

        # Print the metrics
        print(f"Epoch {self.current_epoch} - Avg Total Loss: {avg_total_loss:.4f}")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss", "clip_grad": {"clip_val": 1.0}}
