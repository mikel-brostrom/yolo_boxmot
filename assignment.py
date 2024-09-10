import torch

def compute_iou(box1, box2):
    """Calculate IoU of two 2D bounding boxes in a batch-wise manner."""
    # Compute intersection coordinates
    inter_xmin = torch.max(box1[:, :, None, 0], box2[:, None, :, 0])
    inter_ymin = torch.max(box1[:, :, None, 1], box2[:, None, :, 1])
    inter_xmax = torch.min(box1[:, :, None, 2], box2[:, None, :, 2])
    inter_ymax = torch.min(box1[:, :, None, 3], box2[:, None, :, 3])

    # Compute intersection area
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    intersection = inter_width * inter_height

    # Compute areas of boxes
    box1_area = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1])
    box2_area = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1])

    # Compute union area
    union = box1_area[:, :, None] + box2_area[:, None, :] - intersection

    # IoU
    iou = intersection / torch.clamp(union, min=1e-6)
    return iou

def compute_ciou(box1, box2):
    """Calculate CIoU for two 2D bounding boxes in a batch-wise manner."""
    iou = compute_iou(box1, box2)

    # Calculate center coordinates
    center1_x = (box1[:, :, 0] + box1[:, :, 2]) / 2
    center1_y = (box1[:, :, 1] + box1[:, :, 3]) / 2
    center2_x = (box2[:, :, 0] + box2[:, :, 2]) / 2
    center2_y = (box2[:, :, 1] + box2[:, :, 3]) / 2

    # Compute center distance and diagonal length of enclosing box
    rho2 = (center1_x[:, :, None] - center2_x[:, None, :]) ** 2 + (center1_y[:, :, None] - center2_y[:, None, :]) ** 2
    c_xmin = torch.min(box1[:, :, None, 0], box2[:, None, :, 0])
    c_ymin = torch.min(box1[:, :, None, 1], box2[:, None, :, 1])
    c_xmax = torch.max(box1[:, :, None, 2], box2[:, None, :, 2])
    c_ymax = torch.max(box1[:, :, None, 3], box2[:, None, :, 3])
    c2 = (c_xmax - c_xmin) ** 2 + (c_ymax - c_ymin) ** 2

    # Aspect ratio consistency term (v)
    w1 = box1[:, :, 2] - box1[:, :, 0]
    h1 = box1[:, :, 3] - box1[:, :, 1]
    w2 = box2[:, :, 2] - box2[:, :, 0]
    h2 = box2[:, :, 3] - box2[:, :, 1]
    v = (4 / (torch.pi ** 2)) * (torch.atan(w1[:, :, None] / torch.clamp(h1[:, :, None], min=1e-6)) - torch.atan(w2[:, None, :] / torch.clamp(h2[:, None, :], min=1e-6))) ** 2

    # Trade-off parameter (alpha)
    alpha = v / (1 - iou + v + 1e-6)

    # CIoU calculation
    ciou = iou - (rho2 / torch.clamp(c2, min=1e-6)) - (alpha * v)
    return ciou

def match_and_compute_loss(predictions, ground_truths, iou_threshold=0.5):
    """Match ground truth boxes with predicted boxes and compute CIoU loss in a vectorized way."""
    batch_size, num_preds, _ = predictions.shape
    _, num_gts, _ = ground_truths.shape
    
    # Compute CIoU matrix for each batch
    ciou_matrix = compute_ciou(predictions, ground_truths)
    
    losses = torch.zeros(batch_size, device=predictions.device)
    
    for i in range(batch_size):
        # Find the best matches for each ground truth box
        max_cious, best_pred_indices = torch.max(ciou_matrix[i], dim=0)

        # Filter matches based on IoU threshold
        valid_matches = max_cious >= iou_threshold
        
        # Calculate CIoU loss for matched pairs
        loss = 1 - max_cious[valid_matches]
        losses[i] = loss.sum()

    return losses.mean()

# Example tensors for predictions and ground truths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictions = torch.rand(8, 16, 4, device=device) * 100  # [batch_size, num_predictions, 4]
ground_truths = torch.rand(8, 8, 4, device=device) * 100  # [batch_size, num_ground_truths, 4]

# Calculate CIoU loss
ciou_loss = match_and_compute_loss(predictions, ground_truths)
print("Average CIoU Loss for the batch:", ciou_loss.item())
