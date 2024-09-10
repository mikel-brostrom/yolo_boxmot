# test_simple_object_detector.py

import pytest
import torch
from simple_yolo.model import SimpleObjectDetector  # Importing SimpleObjectDetector from model.py

# ----------------------------------------
# Global Variables
# ----------------------------------------

IMAGE_HEIGHT = 224  # Height of the input image
IMAGE_WIDTH = 224   # Width of the input image
FEATURE_MAP_SIZE = (IMAGE_HEIGHT // 32) * (IMAGE_WIDTH // 32) # Output size of the feature map (7x7 grid) for ResNet backbones

# ----------------------------------------
# Tests for SimpleObjectDetector
# ----------------------------------------

def test_initialization():
    # Test supported ResNet versions
    for version in ['resnet18', 'resnet34', 'resnet50']:
        model = SimpleObjectDetector(resnet_version=version)
        assert model.num_classes == 80, "Default number of classes should be 80"
        assert model.num_boxes == 10, "Default number of boxes should be 10"
        assert isinstance(model.backbone, torch.nn.Sequential), "Backbone should be a Sequential model"
    
    # Test unsupported ResNet version
    with pytest.raises(ValueError):
        SimpleObjectDetector(resnet_version='resnet101')

    # Test different number of classes and boxes
    model = SimpleObjectDetector(num_classes=5, num_boxes=20)
    assert model.num_classes == 5, "Number of classes should be set to 5"
    assert model.num_boxes == 20, "Number of boxes should be set to 20"

def test_forward_pass():
    model = SimpleObjectDetector(resnet_version='resnet18')
    model.eval()
    batch_size = 2
    
    # Create a dummy input tensor of size (batch_size, channels, height, width)
    x = torch.randn(batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    # Perform forward pass
    cls_pred, bbox_pred, obj_pred = model(x)
    
    # Check output shapes
    assert cls_pred.shape == (batch_size, FEATURE_MAP_SIZE * model.num_boxes, model.num_classes), "Class predictions shape mismatch"
    assert bbox_pred.shape == (batch_size, FEATURE_MAP_SIZE * model.num_boxes, 4), "Bounding box predictions shape mismatch"
    assert obj_pred.shape == (batch_size, FEATURE_MAP_SIZE * model.num_boxes, 1), "Objectness predictions shape mismatch"

def test_loss_computation():
    model = SimpleObjectDetector(resnet_version='resnet18')
    batch_size = 2
    
    # Dummy predictions and targets
    pred_cls = torch.randn(batch_size, FEATURE_MAP_SIZE * model.num_boxes, model.num_classes)
    pred_bbox = torch.rand(batch_size, FEATURE_MAP_SIZE * model.num_boxes, 4)  # between 0 and 1 for sigmoid
    pred_obj = torch.sigmoid(torch.randn(batch_size, FEATURE_MAP_SIZE * model.num_boxes, 1))
    
    target_cls = torch.randint(0, model.num_classes, (batch_size, FEATURE_MAP_SIZE * model.num_boxes))
    target_bbox = torch.rand(batch_size, FEATURE_MAP_SIZE * model.num_boxes, 4)
    target_obj = torch.randint(0, 2, (batch_size, FEATURE_MAP_SIZE * model.num_boxes)).float()  # Fix shape to match pred_obj

    # Calculate loss
    ciou_loss, cls_loss, obj_loss = model.match_predictions_to_targets(pred_cls, pred_bbox, pred_obj, target_cls, target_bbox, target_obj)
    
    # Loss should be a non-negative tensor
    assert ciou_loss >= 0, "CIOU loss should be non-negative"
    assert cls_loss >= 0, "Classification loss should be non-negative"
    assert obj_loss >= 0, "Objectness loss should be non-negative"

def test_training_step():
    model = SimpleObjectDetector(resnet_version='resnet18')
    batch_size = 2
    
    # Dummy batch (images, target_cls, target_bbox, target_obj)
    images = torch.randn(batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    target_cls = torch.randint(0, model.num_classes, (batch_size, FEATURE_MAP_SIZE * model.num_boxes))
    target_bbox = torch.rand(batch_size, FEATURE_MAP_SIZE * model.num_boxes, 4)
    target_obj = torch.randint(0, 2, (batch_size, FEATURE_MAP_SIZE * model.num_boxes)).float()  # Fix shape to match pred_obj
    
    batch = (images, target_cls, target_bbox, target_obj)
    
    # Run training step
    model.training_step(batch, 0)
    
    # Check if train_losses are appended
    assert len(model.train_losses) == 1, "Training losses should have been appended."
    assert 'loss' in model.train_losses[0], "Loss should be logged in train_losses."

def test_configure_optimizers():
    model = SimpleObjectDetector()
    optimizers_and_schedulers = model.configure_optimizers()
    
    # Extract optimizer and scheduler from the return value
    optimizer = optimizers_and_schedulers["optimizer"]
    scheduler_dict = optimizers_and_schedulers["lr_scheduler"]
    
    # Check if the optimizer is correctly instantiated
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam."


def test_on_train_epoch_end():
    model = SimpleObjectDetector()
    
    # Manually set train_losses for testing
    model.train_losses = [
        {'loss': torch.tensor(1.0), 'loss_cls': torch.tensor(0.5), 'loss_bbox': torch.tensor(0.3), 'loss_obj': torch.tensor(0.2)},
        {'loss': torch.tensor(2.0), 'loss_cls': torch.tensor(1.0), 'loss_bbox': torch.tensor(0.7), 'loss_obj': torch.tensor(0.3)},
    ]
    
    # Run end of epoch
    model.on_train_epoch_end()
    
    # Check if train_losses were cleared
    assert len(model.train_losses) == 0, "train_losses should be cleared after epoch end."

def test_no_valid_bounding_boxes():
    model = SimpleObjectDetector()
    batch_size = 2
    
    # Dummy predictions
    pred_cls = torch.randn(batch_size, FEATURE_MAP_SIZE * model.num_boxes, model.num_classes)
    pred_bbox = torch.rand(batch_size, FEATURE_MAP_SIZE * model.num_boxes, 4)
    pred_obj = torch.sigmoid(torch.randn(batch_size, FEATURE_MAP_SIZE * model.num_boxes, 1))
    
    # Empty target bounding boxes
    target_cls = torch.randint(0, model.num_classes, (batch_size, FEATURE_MAP_SIZE * model.num_boxes))
    target_bbox = torch.zeros(batch_size, FEATURE_MAP_SIZE * model.num_boxes, 4)  # All zeros indicate no valid boxes
    target_obj = torch.zeros(batch_size, FEATURE_MAP_SIZE * model.num_boxes).float()  # Fix shape to match pred_obj

    # Ensure it handles the situation gracefully
    ciou_loss, cls_loss, obj_loss = model.match_predictions_to_targets(pred_cls, pred_bbox, pred_obj, target_cls, target_bbox, target_obj)
    
    assert ciou_loss == 0, "CIOU loss should be zero if there are no valid bounding boxes"
    assert cls_loss == 0, "Classification loss should be zero if there are no valid bounding boxes"
    assert obj_loss == 0, "Objectness loss should be zero if there are no valid bounding boxes"

# Run tests
if __name__ == "__main__":
    pytest.main()
