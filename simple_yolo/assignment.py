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
    # Compute IoU between box1 and box2
    iou = compute_iou(box1, box2)  # Assume this is implemented correctly

    # Ensure IoU is in the range [0, 1]
    iou = iou.clamp(min=0, max=1)

    # Calculate center coordinates for box1 and box2
    center1_x = (box1[:, :, 0] + box1[:, :, 2]) / 2
    center1_y = (box1[:, :, 1] + box1[:, :, 3]) / 2
    center2_x = (box2[:, :, 0] + box2[:, :, 2]) / 2
    center2_y = (box2[:, :, 1] + box2[:, :, 3]) / 2

    # Compute the squared distance between centers
    rho2 = (center1_x[:, :, None] - center2_x[:, None, :]) ** 2 + (center1_y[:, :, None] - center2_y[:, None, :]) ** 2

    # Compute the diagonal length squared of the smallest enclosing box
    c_xmin = torch.min(box1[:, :, None, 0], box2[:, None, :, 0])
    c_ymin = torch.min(box1[:, :, None, 1], box2[:, None, :, 1])
    c_xmax = torch.max(box1[:, :, None, 2], box2[:, None, :, 2])
    c_ymax = torch.max(box1[:, :, None, 3], box2[:, None, :, 3])
    c2 = (c_xmax - c_xmin).clamp(min=1e-6) ** 2 + (c_ymax - c_ymin).clamp(min=1e-6) ** 2

    # Ensure all widths and heights are positive
    w1 = (box1[:, :, 2] - box1[:, :, 0]).clamp(min=1e-6)
    h1 = (box1[:, :, 3] - box1[:, :, 1]).clamp(min=1e-6)
    w2 = (box2[:, :, 2] - box2[:, :, 0]).clamp(min=1e-6)
    h2 = (box2[:, :, 3] - box2[:, :, 1]).clamp(min=1e-6)

    # Aspect ratio consistency term (v)
    v = (4 / (torch.pi ** 2)) * (torch.atan(w1[:, :, None] / h1[:, :, None]) - 
                                  torch.atan(w2[:, None, :] / h2[:, None, :])) ** 2

    # Trade-off parameter (alpha) with improved numerical stability
    alpha = v / ((1 - iou + v).clamp(min=1e-6))

    # Complete IoU (CIoU) calculation
    ciou = iou - (rho2 / c2.clamp(min=1e-6)) - (alpha * v)

    # Ensure CIoU is in the range [-1, 1]
    ciou = ciou.clamp(min=-1.0, max=1.0)

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


# Case 1: Identical boxes
box1 = torch.tensor([[[1, 1, 4, 4]]])  # Shape (1, 1, 4)
box2 = torch.tensor([[[1, 1, 4, 4]]])  # Shape (1, 1, 4)
expected_ciou = torch.tensor([[[1.0]]])  # CIoU = 1.0
ciou = compute_ciou(box1, box2)
assert torch.allclose(ciou, expected_ciou, atol=1e-6)

# Case 2: Partially overlapping boxes
box1 = torch.tensor([[[1, 1, 4, 4]]])  # Shape (1, 1, 4)
box2 = torch.tensor([[[2, 2, 5, 5]]])  # Shape (1, 1, 4)
ciou = compute_ciou(box1, box2)
assert ciou.shape == (1, 1, 1)
assert ciou.item() < 1.0  # CIoU < 1 for partial overlap