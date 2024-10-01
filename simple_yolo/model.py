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
        
        print(self.anchors.shape)

        # Define dropout and advanced bbox regressor layers
        self.dropout = nn.Dropout(0.1)
        self.bbox_head = self._build_bbox_head(in_channels)
        self.objectness_head = self._build_objectness_head(in_channels)
        self.class_head = self._build_class_head(in_channels)

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
            nn.Conv2d(512, self.num_anchor_sizes * 4, kernel_size=1)
        )

    def _build_objectness_head(self, in_channels):
        """Build the objectness head."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, self.num_anchor_sizes * 1, kernel_size=1)
        )

    def _build_class_head(self, in_channels):
        """Build the class prediction head."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, self.num_anchor_sizes * self.num_classes, kernel_size=1)
        )

    def _initialize_weights(self):
        """Initialize the weights of the heads."""
        for head in [self.bbox_head, self.objectness_head, self.class_head]:
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


    def forward(self, x, decode=False):
        # Pass input through the backbone to get feature maps
        features = self.backbone(x)
        
        # print('features.shape', features.shape)
        
        # Pass features through the heads
        bbox_pred = self.bbox_head(features)  # Shape: (batch_size, 4 * num_anchor_sizes, grid_height, grid_width)
        obj_pred = self.objectness_head(features)  # Shape: (batch_size, num_anchor_sizes, grid_height, grid_width)
        class_pred = self.class_head(features)  # Shape: (batch_size, num_classes * num_anchor_sizes, grid_height, grid_width)
        
        batch_size = x.size(0)
        grid_height = features.size(2)
        grid_width = features.size(3)

        # Reshape bbox_pred to separate grid dimensions and anchor dimension
        bbox_pred = bbox_pred.view(batch_size, self.num_anchor_sizes, 4, grid_height, grid_width)
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2)  # Shape: (batch_size, grid_height, grid_width, num_anchor_sizes, 4)
        bbox_pred = bbox_pred.contiguous()  # Ensure contiguous memory layout

        # Reshape obj_pred to separate grid dimensions and anchor dimension
        obj_pred = obj_pred.view(batch_size, self.num_anchor_sizes, 1, grid_height, grid_width)
        obj_pred = obj_pred.permute(0, 3, 4, 1, 2)  # Shape: (batch_size, grid_height, grid_width, num_anchor_sizes, 1)
        obj_pred = obj_pred.contiguous()

        # Reshape class_pred to separate grid dimensions and anchor dimension
        class_pred = class_pred.view(batch_size, self.num_anchor_sizes, self.num_classes, grid_height, grid_width)
        class_pred = class_pred.permute(0, 3, 4, 1, 2)  # Shape: (batch_size, grid_height, grid_width, num_anchor_sizes, num_classes)
        class_pred = class_pred.contiguous()

        # Reshape to get final shape as (batch_size, grid_height, grid_width, ...)
        bbox_pred = bbox_pred.view(batch_size, grid_height, grid_width, self.num_anchor_sizes, 4)
        obj_pred = obj_pred.view(batch_size, grid_height, grid_width, self.num_anchor_sizes, 1)
        class_pred = class_pred.view(batch_size, grid_height, grid_width, self.num_anchor_sizes, self.num_classes)
        
        # print('bbox_pred.shape', bbox_pred.shape)
        # print('obj_pred.shape', obj_pred.shape)
        # print('class_pred.shape', class_pred.shape)

        # Optional decode stage
        if decode:
            # Move anchors to the same device as input
            anchors = self.anchors.to(x.device)
            
            # Decode bounding box predictions
            decoded_boxes = self.decode_boxes(bbox_pred, anchors)  # Shape: (batch_size, grid_height, grid_width, num_anchor_sizes, 4)
            return decoded_boxes, obj_pred, class_pred
        else:
            return bbox_pred, obj_pred, class_pred


    def generate_anchors(self, grid_height, grid_width, device='mps'):
        """
        Generates anchors based on the feature map size.
        """
        # Generate normalized grid of anchor center positions
        center_x = (torch.arange(grid_width, dtype=torch.float32, device=device) + 0.5) / grid_width
        center_y = (torch.arange(grid_height, dtype=torch.float32, device=device) + 0.5) / grid_height
        center_y, center_x = torch.meshgrid(center_y, center_x, indexing='ij')
        anchor_centers = torch.stack([center_x, center_y], dim=-1)  # Shape: (grid_height, grid_width, 2)

        # Anchor sizes in normalized coordinates
        widths, heights = torch.meshgrid(self.widths, self.heights, indexing='ij')
        anchor_sizes = torch.stack([widths, heights], dim=-1).reshape(-1, 2)  # Shape: (num_anchor_sizes, 2)

        # Expand dimensions to match grid and anchor sizes
        anchor_centers = anchor_centers.unsqueeze(2).expand(-1, -1, self.num_anchor_sizes, -1)  # Shape: (grid_height, grid_width, num_anchor_sizes, 2)
        anchor_sizes = anchor_sizes.unsqueeze(0).unsqueeze(0).expand(grid_height, grid_width, -1, -1)  # Shape: (grid_height, grid_width, num_anchor_sizes, 2)

        # Combine centers and sizes to get anchors
        anchors = torch.cat([anchor_centers, anchor_sizes], dim=-1)  # Shape: (grid_height, grid_width, num_anchor_sizes, 4)
        return anchors


    def decode_boxes(self, bbox_pred, anchors):
        """
        Decodes the predicted bounding boxes relative to the provided anchors.
        bbox_pred: Tensor of shape (batch_size, grid_h, grid_w, num_anchor_sizes, 4)
        anchors: Tensor of shape (grid_h, grid_w, num_anchor_sizes, 4)
        """
        # Predicted offsets
        t_x = torch.sigmoid(bbox_pred[..., 0])
        t_y = torch.sigmoid(bbox_pred[..., 1])
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


    def convert_anchors_to_corners(self, anchors):
        # anchors: (num_anchor_sizes, 4) -> [center_x, center_y, width, height]
        # Convert to [x_min, y_min, x_max, y_max]
        anchors_corners = torch.zeros_like(anchors)
        anchors_corners[:, 0] = anchors[:, 0] - anchors[:, 2] / 2  # x_min
        anchors_corners[:, 1] = anchors[:, 1] - anchors[:, 3] / 2  # y_min
        anchors_corners[:, 2] = anchors[:, 0] + anchors[:, 2] / 2  # x_max
        anchors_corners[:, 3] = anchors[:, 1] + anchors[:, 3] / 2  # y_max
        return anchors_corners


    def training_step(self, batch, batch_idx):
        images, targets_cls, targets_bbox, _ = batch  # bbox in normalized [0, 1] x1, y1, x2, y2 format
        device = images.device
        batch_size = images.size(0)

        # Forward pass
        pred_bbox, pred_obj, pred_class = self(images, decode=False)  # Shapes as specified

        # Get shapes
        grid_height = pred_bbox.size(1)
        grid_width = pred_bbox.size(2)
        num_anchor_sizes = pred_bbox.size(3)
        num_classes = pred_class.size(-1)

        # Get anchors
        anchors = self.anchors.to(device)  # Shape: (grid_height, grid_width, num_anchor_sizes, 4)

        # Initialize target tensors
        target_obj = torch.zeros_like(pred_obj, device=device)  # Shape: (batch_size, grid_h, grid_w, num_anchor_sizes, 1)
        target_bbox = torch.zeros_like(pred_bbox, device=device)  # Shape: (batch_size, grid_h, grid_w, num_anchor_sizes, 4)
        target_class = torch.zeros_like(pred_class, device=device)  # Shape: (batch_size, grid_h, grid_w, num_anchor_sizes, num_classes)

        # For each image in the batch
        for b in range(batch_size):
            gt_boxes = targets_bbox[b].to(device)  # Shape: (num_gt_boxes, 4)
            gt_labels = targets_cls[b].to(device).long()  # Shape: (num_gt_boxes,)
            num_gt_boxes = gt_boxes.size(0)

            for idx in range(num_gt_boxes):
                gt_box = gt_boxes[idx]  # (x_min, y_min, x_max, y_max)
                gt_label = gt_labels[idx]  # Scalar

                # Calculate center coordinates and size of the ground truth box
                gt_x = (gt_box[0] + gt_box[2]) / 2.0
                gt_y = (gt_box[1] + gt_box[3]) / 2.0
                gt_w = gt_box[2] - gt_box[0]
                gt_h = gt_box[3] - gt_box[1]

                # Determine which grid cell the center falls into
                grid_x = int(gt_x * grid_width)
                grid_y = int(gt_y * grid_height)
                grid_x = min(grid_width - 1, max(0, grid_x))
                grid_y = min(grid_height - 1, max(0, grid_y))

                # Get anchors at the grid cell
                anchors_at_cell = anchors[grid_y, grid_x]  # Shape: (num_anchor_sizes, 4)

                # Convert anchors to corners for IoU calculation
                anchors_corners = self.convert_anchors_to_corners(anchors_at_cell)  # Should return (num_anchor_sizes, 4)

                # Ground truth box in corner format
                gt_box_corners = gt_box.unsqueeze(0)  # Shape: (1, 4)

                # Compute IoU between ground truth box and anchors
                iou_scores = compute_iou(gt_box_corners, anchors_corners)  # Shape: (1, num_anchor_sizes)

                # Ensure iou_scores is a 1D tensor
                iou_scores = iou_scores.squeeze(0).squeeze(0)  # Shape: (num_anchor_sizes,)

                # Find the best matching anchor
                best_anchor_idx = torch.argmax(iou_scores).item()  # Convert to scalar integer

                # Assign positive example
                target_obj[b, grid_y, grid_x, best_anchor_idx, 0] = 1.0

                # Compute bbox regression targets
                tx = gt_x * grid_width - grid_x
                ty = gt_y * grid_height - grid_y
                tw = torch.log(gt_w / anchors_at_cell[best_anchor_idx, 2] + 1e-16)
                th = torch.log(gt_h / anchors_at_cell[best_anchor_idx, 3] + 1e-16)

                target_bbox[b, grid_y, grid_x, best_anchor_idx] = torch.tensor([tx, ty, tw, th], device=device)

                # Set target class (one-hot encoding)
                target_class[b, grid_y, grid_x, best_anchor_idx, gt_label] = 1.0

                # Ignore other anchors with high IoU
                ignore_iou_thresh = 0.5
                for anchor_idx in range(num_anchor_sizes):
                    if (anchor_idx != best_anchor_idx) and (iou_scores[anchor_idx].item() > ignore_iou_thresh):
                        target_obj[b, grid_y, grid_x, anchor_idx, 0] = -1.0  # Ignore this anchor

        # Compute objectness loss
        obj_mask = target_obj != -1  # Boolean mask where target_obj is not -1 (i.e., not ignored)
        obj_pred_filtered = pred_obj[obj_mask].view(-1)
        target_obj_filtered = target_obj[obj_mask].view(-1)
        obj_loss = F.binary_cross_entropy_with_logits(obj_pred_filtered, target_obj_filtered, reduction='mean')

        # Compute bounding box loss
        pos_mask = (target_obj == 1).squeeze(-1)  # Shape: [batch_size, grid_h, grid_w, num_anchor_sizes]
        num_pos = pos_mask.sum()

        if num_pos > 0:
            # Extract positive predictions and targets
            pred_bbox_pos = pred_bbox[pos_mask].view(-1, 4)
            target_bbox_pos = target_bbox[pos_mask].view(-1, 4)

            # Bounding box regression loss (e.g., Smooth L1 loss)
            bbox_loss = F.smooth_l1_loss(pred_bbox_pos, target_bbox_pos, reduction='mean')

            # Classification loss
            pred_class_pos = pred_class[pos_mask].view(-1, self.num_classes)
            target_class_pos = target_class[pos_mask].view(-1, self.num_classes)

            # Convert target classes from one-hot encoding to class indices
            target_class_indices = target_class_pos.argmax(dim=1)

            # Compute classification loss
            class_loss = F.cross_entropy(pred_class_pos, target_class_indices, reduction='mean')
        else:
            bbox_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = obj_loss + bbox_loss + class_loss

        # Log losses
        self.log('total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('bbox_loss', bbox_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('obj_loss', obj_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('class_loss', class_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss



        

    def on_train_epoch_end(self):
        # Access the average metrics for the epoch
        avg_total_loss = self.trainer.callback_metrics['total_loss'].item()
        avg_bbox_loss = self.trainer.callback_metrics['bbox_loss'].item()
        avg_obj_loss = self.trainer.callback_metrics['obj_loss'].item()
        avg_class_loss = self.trainer.callback_metrics['class_loss'].item()

        # Print the metrics
        print(f"Epoch {self.current_epoch} - Avg Total Loss: {avg_total_loss:.4f}, "
              f"BBox Loss: {avg_bbox_loss:.4f}, Obj Loss: {avg_obj_loss:.4f}, Class Loss: {avg_class_loss:.4f}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss", "clip_grad": {"clip_val": 1.0}}
