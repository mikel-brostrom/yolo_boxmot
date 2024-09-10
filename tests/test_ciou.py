import torch
import pytest
from simple_yolo.assignment import compute_iou, compute_ciou

# Define test cases for compute_iou
def test_compute_iou():
    # Case 1: Identical boxes
    box1 = torch.tensor([[[1, 1, 4, 4]]])  # Shape (1, 1, 4)
    box2 = torch.tensor([[[1, 1, 4, 4]]])  # Shape (1, 1, 4)
    expected_iou = torch.tensor([[[1.0]]])  # IoU = 1.0
    iou = compute_iou(box1, box2)
    assert torch.allclose(iou, expected_iou, atol=1e-6)

    # Case 2: Partially overlapping boxes
    # box1 = torch.tensor([[[1, 1, 4, 4]]])  # Shape (1, 1, 4)
    # box2 = torch.tensor([[[2, 2, 5, 5]]])  # Shape (1, 1, 4)
    # expected_iou = torch.tensor([[[0.2857]]])  # IoU = 1/7 = 0.142857
    # iou = compute_iou(box1, box2)
    # assert torch.allclose(iou, expected_iou, atol=1e-6)

    # Case 3: Non-overlapping boxes
    box1 = torch.tensor([[[1, 1, 2, 2]]])  # Shape (1, 1, 4)
    box2 = torch.tensor([[[3, 3, 4, 4]]])  # Shape (1, 1, 4)
    expected_iou = torch.tensor([[[0.0]]])  # IoU = 0.0
    iou = compute_iou(box1, box2)
    assert torch.allclose(iou, expected_iou, atol=1e-6)

    # Case 4: Edge-aligned boxes
    box1 = torch.tensor([[[1, 1, 2, 2]]])  # Shape (1, 1, 4)
    box2 = torch.tensor([[[2, 2, 3, 3]]])  # Shape (1, 1, 4)
    expected_iou = torch.tensor([[[0.0]]])  # IoU = 0.0 (touching but no overlap)
    iou = compute_iou(box1, box2)
    assert torch.allclose(iou, expected_iou, atol=1e-6)

# Define test cases for compute_ciou
def test_compute_ciou():
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

    # Case 3: Non-overlapping boxes
    box1 = torch.tensor([[[1, 1, 2, 2]]])  # Shape (1, 1, 4)
    box2 = torch.tensor([[[3, 3, 4, 4]]])  # Shape (1, 1, 4)
    ciou = compute_ciou(box1, box2)
    assert ciou.shape == (1, 1, 1)
    assert ciou.item() < 0.0  # CIoU is less than 0 for non-overlapping boxes

    # Case 4: Different aspect ratios
    box1 = torch.tensor([[[1, 1, 4, 4]]])  # Shape (1, 1, 4)
    box2 = torch.tensor([[[1, 1, 3, 5]]])  # Shape (1, 1, 4)
    ciou = compute_ciou(box1, box2)
    assert ciou.shape == (1, 1, 1)
    assert ciou.item() < 1.0  # CIoU < 1 for different aspect ratios

if __name__ == "__main__":
    pytest.main()