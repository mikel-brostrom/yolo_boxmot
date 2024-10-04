import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from simple_yolo.assignment import compute_iou
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_assignments(image, gt_boxes, target_obj, grid_width, grid_height, num_anchor_sizes, batch_size):
    """
    Visualizes the assignment of ground-truth boxes to anchor grid cells on an image.
    
    Args:
        image (Tensor): The input image tensor of shape (C, H, W).
        gt_boxes (Tensor): Ground-truth bounding boxes.
        target_obj (Tensor): Target object presence tensor.
        grid_width (int): Width of the feature grid.
        grid_height (int): Height of the feature grid.
        num_anchor_sizes (int): Number of anchor sizes.
        batch_size (int): Batch size of the input.
    
    """
    fig, ax = plt.subplots(1)
    normalized_image = image.permute(1, 2, 0).cpu().numpy()
    denormalized_image = (normalized_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

    ax.imshow(denormalized_image)

    image_height, image_width = 512, 512

    for b in range(batch_size):
        for i in range(grid_height):
            for j in range(grid_width):
                for anchor_idx in range(num_anchor_sizes):
                    if target_obj[b, i, j, anchor_idx, 0] == 1:
                        # Convert normalized grid coordinates to pixel values
                        x = j * (image_width / grid_width)
                        y = i * (image_height / grid_height)
                        width = image_width / grid_width
                        height = image_height / grid_height
                        
                        # Draw the rectangle
                        rect = patches.Rectangle(
                            (x, y),
                            width, height,
                            linewidth=1, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)

    plt.show()

class SimpleObjectDetector(pl.LightningModule):
    """
    A simple object detection model based on a modified ResNet backbone.
    
    Args:
        resnet_version (str): The version of ResNet to use ('resnet18', 'resnet34', or 'resnet50').
        num_boxes (int): Number of boxes per cell.
        num_classes (int): Number of classes for classification.
        learning_rate (float): Learning rate for optimizer.
        input_size (int): Input image size.
    """
    
    def __init__(self, resnet_version='resnet18', num_boxes=1, num_classes=80, learning_rate=1e-3, input_size=512):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input_size = input_size
        
        # Define widths and heights for anchors
        self.widths = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32, device='mps')
        self.heights = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32, device='mps')

        # Compute the number of anchor sizes
        self.num_anchor_sizes = len(self.widths) * len(self.heights)

        # Initialize ResNet backbone based on specified version
        self.backbone = self._init_backbone(resnet_version)
        in_channels = self._get_in_channels(resnet_version)

        # Compute expected feature map size
        self.grid_height, self.grid_width = self._compute_feature_map_size(self.input_size)
        
        # Generate anchors once and register as buffer
        self.anchors = self.generate_anchors(self.grid_height, self.grid_width)  # (grid_height, grid_width, num_anchor_sizes, 4)
                
        # Define dropout and advanced bbox regressor layers
        self.dropout = nn.Dropout(0.1)
        self.bbox_head = self._build_bbox_head(in_channels)
        self.objectness_head = self._build_objectness_head(in_channels)
        self.class_head = self._build_class_head(in_channels)

        # Initialize weights
        self._initialize_weights()

    def _init_backbone(self, resnet_version):
        """
        Initialize the ResNet backbone.
        
        Args:
            resnet_version (str): ResNet version ('resnet18', 'resnet34', 'resnet50').

        Returns:
            nn.Module: ResNet backbone without fully connected layers.
        """
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
        """
        Return the number of input channels based on ResNet version.
        
        Args:
            resnet_version (str): ResNet version ('resnet18', 'resnet34', 'resnet50').

        Returns:
            int: Number of output channels from ResNet backbone.
        """
        return 512 if resnet_version in ['resnet18', 'resnet34'] else 2048

    def _build_bbox_head(self, in_channels):
        """
        Build the bounding box regressor head.

        Args:
            in_channels (int): Number of input channels for the bbox head.

        Returns:
            nn.Sequential: Bounding box regression head.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, self.num_anchor_sizes * 4, kernel_size=1)
        )

    def _build_objectness_head(self, in_channels):
        """
        Build the objectness head.

        Args:
            in_channels (int): Number of input channels for the objectness head.

        Returns:
            nn.Sequential: Objectness head.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, self.num_anchor_sizes * 1, kernel_size=1)
        )

    def _build_class_head(self, in_channels):
        """
        Build the class prediction head.

        Args:
            in_channels (int): Number of input channels for the class head.

        Returns:
            nn.Sequential: Class prediction head.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, self.num_anchor_sizes * self.num_classes, kernel_size=1)
        )

    def _initialize_weights(self):
        """
        Initialize the weights of the model's layers.
        """
        for head in [self.bbox_head, self.objectness_head, self.class_head]:
            for m in head:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if head is self.objectness_head:
                        nn.init.constant_(m.bias, -4.0)  # Example negative bias
                    else:
                        nn.init.constant_(m.bias, 0)

    def _compute_feature_map_size(self, input_size):
        """
        Compute the feature map size given the input size.

        Args:
            input_size (int): Input image size.

        Returns:
            tuple: Height and width of the feature map.
        """
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        with torch.no_grad():
            dummy_features = self.backbone(dummy_input)
        grid_height, grid_width = dummy_features.shape[2], dummy_features.shape[3]
        return grid_height, grid_width

    def forward(self, x, decode=False):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, H, W).
            decode (bool): Whether to decode the bounding boxes.

        Returns:
            Tuple of Tensors: (bbox_pred, obj_pred, class_pred) if decode is False.
                              (decoded_boxes, obj_pred, class_pred) if decode is True.
        """
        # Pass input through the backbone to get feature maps
        features = self.backbone(x)

        # Pass features through the heads
        bbox_pred = self.bbox_head(features)  # Bounding box prediction
        obj_pred = self.objectness_head(features)  # Objectness prediction
        class_pred = self.class_head(features)  # Class prediction

        batch_size = x.size(0)
        grid_height = features.size(2)
        grid_width = features.size(3)

        # Reshape bbox_pred to separate grid dimensions and anchor dimension
        bbox_pred = bbox_pred.view(batch_size, self.num_anchor_sizes, 4, grid_height, grid_width)
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2).contiguous()  # Shape: (batch_size, grid_height, grid_width, num_anchor_sizes, 4)

        # Reshape obj_pred and class_pred similarly
        obj_pred = obj_pred.view(batch_size, self.num_anchor_sizes, 1, grid_height, grid_width)
        obj_pred = obj_pred.permute(0, 3, 4, 1, 2).contiguous()

        class_pred = class_pred.view(batch_size, self.num_anchor_sizes, self.num_classes, grid_height, grid_width)
        class_pred = class_pred.permute(0, 3, 4, 1, 2).contiguous()

        return bbox_pred, obj_pred, class_pred
    
    
    def decode_bbox_predictions(self, bbox_pred, anchors):
        """
        Decode bbox predictions using anchor boxes.

        Args:
            bbox_pred (Tensor): Predicted bbox offsets.
            anchors (Tensor): Anchor boxes.

        Returns:
            Tensor: Decoded bbox predictions in (x1, y1, x2, y2) format.
        """
        pred_bbox = torch.zeros_like(bbox_pred, device=bbox_pred.device)
        pred_bbox[..., 0] = bbox_pred[..., 0] * anchors[..., 2] + anchors[..., 0]  # x_center
        pred_bbox[..., 1] = bbox_pred[..., 1] * anchors[..., 3] + anchors[..., 1]  # y_center
        pred_bbox[..., 2] = torch.exp(bbox_pred[..., 2]) * anchors[..., 2]  # width
        pred_bbox[..., 3] = torch.exp(bbox_pred[..., 3]) * anchors[..., 3]  # height

        # Convert center coordinates to corner coordinates
        pred_bbox_corners = torch.zeros_like(pred_bbox)
        pred_bbox_corners[..., 0] = pred_bbox[..., 0] - pred_bbox[..., 2] / 2  # x_min
        pred_bbox_corners[..., 1] = pred_bbox[..., 1] - pred_bbox[..., 3] / 2  # y_min
        pred_bbox_corners[..., 2] = pred_bbox[..., 0] + pred_bbox[..., 2] / 2  # x_max
        pred_bbox_corners[..., 3] = pred_bbox[..., 1] + pred_bbox[..., 3] / 2  # y_max

        return pred_bbox_corners


    def generate_anchors(self, grid_height, grid_width, device='mps'):
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
        # print(anchors[0][0])
        # print(anchors[15][15])
        return anchors


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
        images, targets_cls, targets_bbox, _ = batch
        device = images.device
        batch_size = images.size(0)

        # Forward pass
        pred_bbox, pred_obj, pred_class = self(images, decode=False)

        # Get anchors
        anchors = self.anchors.to(device)

        # Initialize target tensors
        target_obj = torch.zeros_like(pred_obj, device=device)
        target_bbox = torch.zeros_like(pred_bbox, device=device)
        target_class = torch.zeros(batch_size, self.grid_height, self.grid_width, self.num_anchor_sizes, dtype=torch.long, device=device)

        for b in range(batch_size):
            gt_boxes = targets_bbox[b]  # Ground truth boxes for image b, shape (num_gt_boxes, 4)
            gt_classes = targets_cls[b]  # Ground truth classes for image b, shape (num_gt_boxes,)
            
            num_gt_boxes = gt_boxes.shape[0]
            if num_gt_boxes == 0:
                continue  # Skip images without annotations

            # Flatten anchors to (num_anchors, 4)
            grid_h, grid_w, num_anchor_sizes = anchors.shape[:3]
            num_anchors = grid_h * grid_w * num_anchor_sizes
            anchors_flat = anchors.view(grid_h, grid_w, num_anchor_sizes, 4).reshape(-1, 4)
            
            # Convert anchors to corner format
            anchors_corners = self.convert_anchors_to_corners(anchors_flat)  # (num_anchors, 4)

            # Compute IoU between gt_boxes and anchors
            ious = compute_iou(gt_boxes, anchors_corners).squeeze(0)  # (num_gt_boxes, num_anchors)
            
            # Assign multiple anchors to each ground truth if IoU exceeds threshold
            iou_threshold = 0.5
            for gt_idx in range(num_gt_boxes):
                high_iou_indices = torch.where(ious[gt_idx] > iou_threshold)[0]
                for anchor_flat_idx in high_iou_indices:
                    anchor_flat_idx = int(anchor_flat_idx)
                    # Convert flat anchor index to grid cell and anchor size indices
                    grid_cell_idx = anchor_flat_idx // num_anchor_sizes
                    anchor_size_idx = anchor_flat_idx % num_anchor_sizes
                    grid_y = grid_cell_idx // grid_w
                    grid_x = grid_cell_idx % grid_w

                    # Ensure indices are within bounds
                    grid_y = min(grid_y, grid_h - 1)
                    grid_x = min(grid_x, grid_w - 1)
                    
                    # Update target tensors
                    target_obj[b, grid_y, grid_x, anchor_size_idx, 0] = 1
                    target_class[b, grid_y, grid_x, anchor_size_idx] = gt_classes[gt_idx]

                    # Get anchor parameters
                    anchor = anchors[grid_y, grid_x, anchor_size_idx]
                    anchor_cx = anchor[0]
                    anchor_cy = anchor[1]
                    anchor_w = anchor[2]
                    anchor_h = anchor[3]

                    # Compute target offsets
                    gt_box = gt_boxes[gt_idx]
                    gt_cx = (gt_box[0] + gt_box[2]) / 2
                    gt_cy = (gt_box[1] + gt_box[3]) / 2
                    gt_w = gt_box[2] - gt_box[0]
                    gt_h = gt_box[3] - gt_box[1]

                    t_x = (gt_cx - anchor_cx) / (anchor_w + 1e-6)
                    t_y = (gt_cy - anchor_cy) / (anchor_h + 1e-6)
                    t_w = torch.log(gt_w / (anchor_w + 1e-6))
                    t_h = torch.log(gt_h / (anchor_h + 1e-6))
                    
                    target_bbox[b, grid_y, grid_x, anchor_size_idx, :] = torch.tensor([t_x, t_y, t_w, t_h], device=device)
                    
        # Flatten tensors
        pred_bbox = pred_bbox.view(batch_size, -1, 4)  # (batch_size, num_preds, 4)
        pred_obj = pred_obj.view(batch_size, -1, 1)  # (batch_size, num_preds, 1)
        pred_class = pred_class.view(batch_size, -1, self.num_classes)  # (batch_size, num_preds, num_classes)

        target_bbox = target_bbox.view(batch_size, -1, 4)
        target_obj = target_obj.view(batch_size, -1, 1)
        target_class = target_class.view(batch_size, -1)

        # Objectness loss (Focal Loss for handling class imbalance)
        alpha = 0.25
        gamma = 2.0
        bce_loss = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - p_t) ** gamma * bce_loss
        obj_loss = focal_loss.sum() / batch_size

        # Bbox loss (e.g., CIoU loss)
        mask = target_obj.squeeze(-1) > 0  # (batch_size, num_preds)
        if mask.sum() > 0:
            pred_bbox_pos = pred_bbox[mask]
            target_bbox_pos = target_bbox[mask]
            bbox_loss = F.smooth_l1_loss(pred_bbox_pos, target_bbox_pos).mean()  # Using CIoU loss for better localization
        else:
            bbox_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Class loss (Cross Entropy)
        if mask.sum() > 0:
            pred_class_pos = pred_class[mask]
            target_class_pos = target_class[mask]
            class_loss = F.cross_entropy(pred_class_pos, target_class_pos, reduction='sum') / batch_size
        else:
            class_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Total loss
        total_loss = 2 * bbox_loss + obj_loss + class_loss

        # Log losses
        self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('bbox_loss', bbox_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('obj_loss', obj_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('class_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True)

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
