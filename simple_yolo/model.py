import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from simple_yolo.assignment import compute_ciou, compute_iou
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
    
    def __init__(self, resnet_version='resnet18', num_boxes=1, num_classes=80, learning_rate=1e-5, input_size=512):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input_size = input_size
        
        # Define widths and heights for anchors
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
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2)  # Shape: (batch_size, grid_height, grid_width, num_anchor_sizes, 4)
        bbox_pred = bbox_pred.contiguous()

        # Reshape obj_pred and class_pred similarly
        obj_pred = obj_pred.view(batch_size, self.num_anchor_sizes, 1, grid_height, grid_width)
        obj_pred = obj_pred.permute(0, 3, 4, 1, 2).contiguous()

        class_pred = class_pred.view(batch_size, self.num_anchor_sizes, self.num_classes, grid_height, grid_width)
        class_pred = class_pred.permute(0, 3, 4, 1, 2).contiguous()

        if decode:
            anchors = self.anchors.to(x.device)
            decoded_boxes = self.decode_boxes(bbox_pred, anchors)
            return decoded_boxes, obj_pred, class_pred
        else:
            return bbox_pred, obj_pred, class_pred


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
        images, targets_cls, targets_bbox, _ = batch
        device = images.device
        batch_size = images.size(0)

        # Forward pass
        pred_bbox, pred_obj, pred_class = self(images, decode=False)

        # Get anchors
        anchors = self.anchors.to(device)

        # Initialize target tensors
        device = pred_obj.device
        target_obj = torch.zeros_like(pred_obj, device=device)
        target_bbox = torch.zeros_like(pred_bbox, device=device)
        target_class = torch.zeros_like(pred_class, device=device)

        # Process batch
        self.assign_targets_to_anchors(
            batch_size, targets_bbox, targets_cls, target_obj, target_bbox, target_class, anchors
        )

        # Compute losses
        obj_loss, bbox_loss, class_loss = self.compute_losses(pred_bbox, pred_obj, pred_class, target_obj, target_bbox, target_class)

        # Total loss
        total_loss = sum(w * l for w, l in zip([2.0, 3.0, 1.0], [obj_loss, bbox_loss, class_loss]))

        # Log losses
        self.log('total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('bbox_loss', bbox_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('obj_loss', obj_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('class_loss', class_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss


    def assign_targets_to_anchors(
        self, 
        batch_size, 
        targets_bbox, 
        targets_cls, 
        target_obj, 
        target_bbox, 
        target_class, 
        anchors
    ):
        """
        Assigns ground-truth bounding boxes and classes to the best matching anchors.

        Args:
            batch_size (int): Number of images in the batch.
            targets_bbox (list[Tensor]): List of ground-truth bounding boxes for each image.
                                        Each Tensor is of shape (num_gt, 4) with [x_min, y_min, x_max, y_max] in normalized coordinates.
            targets_cls (list[Tensor]): List of class indices for each ground-truth box in each image.
                                        Each Tensor is of shape (num_gt,).
            target_obj (Tensor): Tensor to be filled with objectness targets. Shape: (batch_size, grid_h, grid_w, num_anchor_sizes, 1)
            target_bbox (Tensor): Tensor to be filled with bounding box targets. Shape: (batch_size, grid_h, grid_w, num_anchor_sizes, 4)
            target_class (Tensor): Tensor to be filled with class targets. Shape: (batch_size, grid_h, grid_w, num_anchor_sizes, num_classes)
            anchors (Tensor): Anchor boxes in center format. Shape: (grid_h, grid_w, num_anchor_sizes, 4)
        """
        device = anchors.device
        anchors_corners = self.convert_anchors_to_corners(anchors)  # (grid_h, grid_w, num_anchor_sizes, 4)
        grid_h, grid_w, num_anchor_sizes, _ = anchors_corners.shape
        num_anchors = grid_h * grid_w * num_anchor_sizes

        # Flatten anchors for easier computation: (num_anchors, 4)
        anchors_corners_flat = anchors_corners.view(-1, 4)

        for b in range(batch_size):
            gt_boxes = targets_bbox[b]  # Tensor of shape (num_gt, 4)
            gt_classes = targets_cls[b]  # Tensor of shape (num_gt,)
            num_gt = gt_boxes.size(0)

            if num_gt == 0:
                continue  # No ground-truth boxes for this image

            # Compute IoU between all ground-truth boxes and all anchors
            iou_matrix = compute_iou(gt_boxes, anchors_corners_flat)  # (num_gt, num_anchors (16 * 16 * 9))

            # For each ground-truth box, find the anchor with the highest IoU
            best_anchor_per_gt, best_iou_per_gt = iou_matrix.max(dim=1)  # Both are (num_gt,)
            best_anchor_per_gt = best_anchor_per_gt.squeeze(0)
            best_iou_per_gt = best_iou_per_gt.squeeze(0)

            for gt_idx in range(num_gt):
                anchor_idx = best_anchor_per_gt[gt_idx].item()
                iou = best_iou_per_gt[gt_idx].item()
                gt_box = gt_boxes[gt_idx]
                gt_class = gt_classes[gt_idx].item()

                # Define an IoU threshold to consider an anchor as positive
                IoU_THRESHOLD = 0.5

                if iou < IoU_THRESHOLD:
                    continue  # Skip anchors with IoU below the threshold

                # Convert the flat anchor index back to grid coordinates and anchor size index
                grid_i = int(anchor_idx // (grid_w * num_anchor_sizes))
                rem = anchor_idx % (grid_w * num_anchor_sizes)
                grid_j = int(rem // num_anchor_sizes)
                anchor_size_idx = int(rem % num_anchor_sizes)

                # Assign the ground-truth box and class to the selected anchor
                target_obj[b, grid_i, grid_j, anchor_size_idx, 0] = 1  # Objectness target
                target_bbox[b, grid_i, grid_j, anchor_size_idx] = gt_box  # Bounding box target

                # One-hot encode the class
                target_class[b, grid_i, grid_j, anchor_size_idx, gt_class] = 1



    def compute_losses(self, pred_bbox, pred_obj, pred_class, target_obj, target_bbox, target_class):
        device = target_obj.device

        obj_mask = target_obj != -1
        obj_pred_filtered = pred_obj[obj_mask].view(-1)
        target_obj_filtered = target_obj[obj_mask].view(-1)
        obj_loss = F.binary_cross_entropy_with_logits(obj_pred_filtered, target_obj_filtered, reduction='mean')

        pos_mask = (target_obj == 1).squeeze(-1)
        num_pos = pos_mask.sum()

        if num_pos > 0:
            pred_bbox_pos = pred_bbox[pos_mask].view(-1, 4)
            target_bbox_pos = target_bbox[pos_mask].view(-1, 4)
            bbox_loss = F.smooth_l1_loss(pred_bbox_pos, target_bbox_pos, reduction='mean')

            print('pred_class.shape', pred_class.shape)
            print('target_class.shape', target_class.shape)
            pred_class_pos = pred_class[pos_mask].view(-1, self.num_classes)
            target_class_pos = target_class[pos_mask].view(-1, self.num_classes)
            target_class_indices = target_class_pos.argmax(dim=1)
            
            class_loss = F.cross_entropy(pred_class_pos, target_class_indices.long(), reduction='mean')
        else:
            bbox_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)

        return obj_loss, bbox_loss, class_loss


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
