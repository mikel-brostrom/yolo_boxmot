import sys
import pytest
import torch

sys.path.append('/Users/mikel.brostrom/yolo_boxmot')

from simple_yolo.model import SimpleObjectDetector

# Assuming compute_iou is already defined elsewhere

class TestAnchorMatching:
    
    def test_match_anchors_to_ground_truth(self):
        # Define anchors and ground truth boxes
        anchors = torch.tensor([
            [[0.1, 0.1, 0.17, 0.17], [0.2, 0.2, 0.25, 0.25], [0.4, 0.4, 0.45, 0.45]]
        ])
        ground_truth_boxes = torch.tensor([[[0.1, 0.1, 0.15, 0.15], [0.4, 0.4, 0.45, 0.45]]])

        # Instantiate the class if needed or run function directly
        matcher = SimpleObjectDetector.match_anchors_to_ground_truth  # Replace with the appropriate class name
        
        # Call the function with default IoU thresholds
        positive_anchors, negative_anchors, anchor_to_gt_assignment = matcher(
            anchors, ground_truth_boxes
        )
        
        # Assert the expected output for positive anchors
        assert positive_anchors == [{0, 2}], "Expected positive anchors: {0, 2}"
        
        # Assert the expected output for negative anchors
        assert negative_anchors == [{1}], "Expected negative anchors: {1}"
        
        # Assert the expected output for anchor-to-ground-truth assignments
        assert anchor_to_gt_assignment == [{0: 0, 2: 1}], "Expected anchor-to-ground-truth assignment: {0: 0, 2: 1}"

    def test_match_with_custom_thresholds(self):
        # Define anchors and ground truth boxes
        anchors = torch.tensor([
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]
        ])
        ground_truth_boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]]])

        matcher = SimpleObjectDetector.match_anchors_to_ground_truth  # Replace with the appropriate class name

        # Call the function with custom IoU thresholds
        positive_anchors, negative_anchors, anchor_to_gt_assignment = matcher(
            anchors, ground_truth_boxes, iou_threshold_high=0.7, iou_threshold_low=0.1
        )
        
        # Assert the expected output for positive anchors
        assert positive_anchors == [{0, 2}], "Expected positive anchors: {0, 2}"

        # Assert the expected output for negative anchors
        assert negative_anchors == [{1}], "Expected negative anchors: {1}"

        # Assert the expected output for anchor-to-ground-truth assignments
        assert anchor_to_gt_assignment == [{0: 0, 2: 1}], "Expected anchor-to-ground-truth assignment: {0: 0, 2: 1}"

    def test_no_anchors_matched(self):
        # Define anchors and ground truth boxes with no overlap
        anchors = torch.tensor([
            [[0.7, 0.7, 0.8, 0.8], [0.9, 0.9, 1.0, 1.0]]  # Anchors that do not overlap with ground truth
        ])
        ground_truth_boxes = torch.tensor([[[0.5, 0.5, 0.6, 0.6]]])  # Ground truth box

        matcher = SimpleObjectDetector.match_anchors_to_ground_truth  # Replace with the appropriate class name

        # Call the function
        positive_anchors, negative_anchors, anchor_to_gt_assignment = matcher(
            anchors, ground_truth_boxes, iou_threshold_high=0.5, iou_threshold_low=0.1
        )
        
        print(anchor_to_gt_assignment)
        
        # Assert that there are no positive anchors
        assert positive_anchors == [set()], "Expected no positive anchors"
        
        # Assert that all anchors are negative
        assert negative_anchors == [{0, 1}], "Expected all anchors to be negative"
        
        # Assert that there are no anchor-to-ground-truth assignments
        assert anchor_to_gt_assignment == [{}], "Expected no anchor-to-ground-truth assignments"