import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import os


# Import the YOLODataset and collate_fn
from simple_yolo.dataset import YOLODataset, collate_fn  # Adjust import as needed

# Helper function to create temporary files and directories for testing
@pytest.fixture
def setup_test_environment(tmp_path):
    # Create directories for images and labels
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    # Create a dummy image
    img = Image.new('RGB', (640, 480), color='white')
    img.save(image_dir / "image_1.jpg")

    # Create a dummy label file
    with open(label_dir / "image_1.txt", 'w') as f:
        f.write("0 0.5 0.5 0.4 0.4\n")  # Class 0, centered at (0.5, 0.5), width and height 0.4

    return image_dir, label_dir

def test_yolodataset_length(setup_test_environment):
    image_dir, label_dir = setup_test_environment

    # Create dataset
    dataset = YOLODataset(image_dir=image_dir, label_dir=label_dir, num_classes=80)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Test length
    assert len(dataset) == 1, "Dataset length should match the number of images."

def test_yolodataset_image_loading_and_transformation(setup_test_environment):
    image_dir, label_dir = setup_test_environment

    # Create dataset
    dataset = YOLODataset(image_dir=image_dir, label_dir=label_dir, num_classes=80)

    # Load the first item
    img, target_cls, boxes, obj_labels = dataset[0]

    # Check image size and type
    assert isinstance(img, torch.Tensor), "Image should be a tensor."
    assert img.shape[1:] == (512, 512), "Image should be resized to 512x512."

    # Check bounding box processing
    assert boxes.shape[1] == 4, "Bounding boxes should have 4 coordinates."
    assert (boxes >= 0).all() and (boxes <= 1).all(), "Bounding boxes should be normalized."

def test_yolodataset_collate_fn(setup_test_environment):
    image_dir, label_dir = setup_test_environment

    # Create dataset and dataloader
    dataset = YOLODataset(image_dir=image_dir, label_dir=label_dir, num_classes=80)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Fetch a batch
    for batch in dataloader:
        images, target_cls, boxes, obj_labels = batch

        # Check batch dimensions
        assert images.shape[0] == 1, "Batch size should be 1 (since we have only one sample)."
        assert target_cls.shape[0] == 1, "Batch size should be 1 for target_cls."
        assert boxes.shape[0] == 1, "Batch size should be 1 for boxes."
        assert obj_labels.shape[0] == 1, "Batch size should be 1 for obj_labels."
        
        # Check for correct padding
        assert target_cls.shape[1] == max(1, len(dataset[0][1])), "Target class padding should match max_boxes."
        assert boxes.shape[1] == max(1, len(dataset[0][2])), "Box padding should match max_boxes."
        assert obj_labels.shape[1] == max(1, len(dataset[0][3])), "Object label padding should match max_boxes."

def test_yolodataset_no_label_file(setup_test_environment):
    image_dir, label_dir = setup_test_environment

    # Remove label file to test empty label scenario
    os.remove(label_dir / "image_1.txt")

    # Create dataset
    dataset = YOLODataset(image_dir=image_dir, label_dir=label_dir, num_classes=80)

    # Load the first item
    img, target_cls, boxes, obj_labels = dataset[0]

    # Check if boxes and labels are empty
    assert boxes.size(0) == 0, "Boxes should be empty when no label file is present."
    assert target_cls.size(0) == 0, "Target class should be empty when no label file is present."
    assert obj_labels.size(0) == 0, "Object labels should be empty when no label file is present."

if __name__ == "__main__":
    pytest.main()
