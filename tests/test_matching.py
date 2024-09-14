import pytest
import sys
sys.path.append('/Users/mikel.brostrom/yolo_boxmot')
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from simple_yolo.assignment import compute_iou, compute_ciou
from simple_yolo.model import SimpleObjectDetector


def validate_indices(pred_bbox, target_bbox, pred_indices, target_indices):
    """
    Validate if the indices from linear_sum_assignment correctly match the highest CIOU pairs.
    """
    # Compute CIOU matrix
    ciou_matrix = compute_ciou(pred_bbox.unsqueeze(0), target_bbox.unsqueeze(0))[0]
    
    # For each predicted index, check if it maps to the highest CIOU value in its row
    for i, (pred_idx, target_idx) in enumerate(zip(pred_indices, target_indices)):
        row = ciou_matrix[pred_idx]
        max_target_idx = row.argmax().item()
        assert target_idx == max_target_idx, f"Mismatch: Expected {max_target_idx} but got {target_idx} for prediction index {pred_idx}"

def validate_bbox_matches(pred_bbox, target_bbox, pred_indices, target_indices):
    """
    Validate if the matched bounding boxes based on indices are indeed correct matches.
    """
    for pred_idx, target_idx in zip(pred_indices, target_indices):
        pred_box = pred_bbox[pred_idx]
        target_box = target_bbox[target_idx]

        assert torch.allclose(pred_box, target_box, atol=1e-4), f"Bounding boxes do not match: Pred {pred_box} vs Target {target_box}"


@pytest.fixture
def bbox_data_1():
    # Define the bounding box data for tests
    pred_bbox = torch.tensor([
        [[0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.9, 0.9]],  # Predictions for batch 1
    ])
    target_bbox = torch.tensor([
        [[0.4, 0.4, 0.8, 0.8], [0.5, 0.5, 0.9, 0.9]],   # Ground truth for batch 2, second box is invalid
    ])
    return pred_bbox, target_bbox


@pytest.fixture
def bbox_data_2():
    # Define the bounding box data for tests
    pred_bbox = torch.tensor([
        [[0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.9, 0.9]],  # Predictions for batch 1
    ])
    target_bbox = torch.tensor([
        [[0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.9, 0.9]],   # Ground truth for batch 2, second box is invalid
    ])
    return pred_bbox, target_bbox


@pytest.fixture
def bbox_data_3():
    # Define the bounding box data for tests
    pred_bbox = torch.tensor([
        [[0.1, 0.1, 0.3, 0.3], [0.3, 0.3, 0.7, 0.7]],  # Predictions for batch 1
    ])
    target_bbox = torch.tensor([
        [[0.2, 0.2, 0.4, 0.4], [0.4, 0.4, 0.8, 0.8]],   # Ground truth for batch 2, second box is invalid
    ])
    return pred_bbox, target_bbox


@pytest.fixture
def bbox_data_3():
    # Define the bounding box data for tests
    pred_bbox = torch.tensor([
        [[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]],  # Predictions for batch 1
    ])
    target_bbox = torch.tensor([
        [[0.05, 0.05, 0.55, 0.55], [0.25, 0.25, 0.75, 0.75]],   # Ground truth for batch 2, second box is invalid
    ])
    return pred_bbox, target_bbox


# @pytest.fixture
# def bbox_data_mismatched_1():
#     # Define the bounding box data for tests where targets are more than predictions
#     pred_bbox = torch.tensor([
#         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # Only 1 Prediction for batch 1
#     ])
#     target_bbox = torch.tensor([
#         [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9], [0.2, 0.2, 0.3, 0.3]]  # 3 Ground truths for batch 1
#     ])
#     return pred_bbox, target_bbox


# @pytest.fixture
# def bbox_data_mismatched_2():
#     # Define the bounding box data for tests where targets are more than predictions
#     pred_bbox = torch.tensor([
#         [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.8, 0.8], [0.25, 0.25, 0.4, 0.4]]  # Only 1 Prediction for batch 1
#     ])
#     target_bbox = torch.tensor([
#         [[0.1, 0.1, 0.4, 0.4]]  # 3 Ground truths for batch 1
#     ])
#     return pred_bbox, target_bbox


def test_zero_loss(bbox_data_2):
    pred_bbox, target_bbox = bbox_data_2

    # Perform the matching
    output_loss, all_pred_indices, all_target_indices = total_ciou_loss = SimpleObjectDetector.match_predictions_to_targets(pred_bbox, target_bbox)

    assert output_loss.item() == 0, "Output loss should be zero for perfectly aligned bboxes."
    print(f"Test Passed: Output loss = {output_loss.item()}")
    
    
def test_gt_zero_loss(bbox_data_3):
    pred_bbox, target_bbox = bbox_data_3

    # Perform the matching
    output_loss, all_pred_indices, all_target_indices = total_ciou_loss = SimpleObjectDetector.match_predictions_to_targets(pred_bbox, target_bbox)

    assert output_loss.item() > 0, "Output loss should be zero for perfectly aligned bboxes."
    print(f"Test Passed: Output loss = {output_loss.item()}")


# def test_match_predictions_to_targets(bbox_data_1):
#     pred_bbox, target_bbox = bbox_data_1

#     # Perform the matching
#     output_loss, all_pred_indices, all_target_indices = total_ciou_loss = SimpleObjectDetector.match_predictions_to_targets(pred_bbox, target_bbox)
#     print('output loss\n', output_loss)
#     print('all_pred_indices\n', all_pred_indices)
#     print('all_target_indices\n', all_target_indices)

#     # Validate the indices and bounding box matches for each batch
#     for i in range(len(all_pred_indices)):
#         validate_indices(pred_bbox[i], target_bbox[i][(target_bbox[i].sum(dim=-1) != 0)], all_pred_indices[i], all_target_indices[i])
#         validate_bbox_matches(pred_bbox[i], target_bbox[i][(target_bbox[i].sum(dim=-1) != 0)], all_pred_indices[i], all_target_indices[i])

#     # Ensure that the total loss is a tensor and has the expected range
#     assert isinstance(output_loss, torch.Tensor), "Output should be a tensor."
#     assert output_loss.item() >= 0, "Output loss should be non-negative."
#     print(f"Test Passed: Output loss = {output_loss.item()}")
    
    
# def test_match_predictions_to_targets_mismatched_1(bbox_data_mismatched_1):
#     pred_bbox, target_bbox = bbox_data_mismatched_1

#     # Perform the matching
#     output_loss, all_pred_indices, all_target_indices = SimpleObjectDetector.match_predictions_to_targets(pred_bbox, target_bbox, ciou_threshold=0.7)

#     # There should only be one match for one prediction (ignoring padded boxes)
#     assert len(all_pred_indices[0]) == 2, "There should be only one predicted box matched."
#     assert len(all_target_indices[0]) == 2, "There should be only one target box matched."

#     # Ensure that the total loss is a tensor and has the expected range
#     assert isinstance(output_loss, torch.Tensor), "Output should be a tensor."
#     assert output_loss.item() >= 0, "Output loss should be non-negative."
#     print(f"Test Passed: Output loss = {output_loss.item()}")
    
    
# def test_match_predictions_to_targets_mismatched_2(bbox_data_mismatched_2):
#     pred_bbox, target_bbox = bbox_data_mismatched_2

#     # Perform the matching
#     output_loss, all_pred_indices, all_target_indices = SimpleObjectDetector.match_predictions_to_targets(pred_bbox, target_bbox, ciou_threshold=0.7)

#     # There should only be one match for one prediction (ignoring padded boxes)
#     assert len(all_pred_indices[0]) == 1, "There should be only one predicted box matched."
#     assert len(all_target_indices[0]) == 1, "There should be only one target box matched."

#     # Ensure that the total loss is a tensor and has the expected range
#     assert isinstance(output_loss, torch.Tensor), "Output should be a tensor."
#     assert output_loss.item() >= 0, "Output loss should be non-negative."
#     print(f"Test Passed: Output loss = {output_loss.item()}")


if __name__ == "__main__":
    pytest.main()