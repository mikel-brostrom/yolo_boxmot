import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import pytorch_lightning as pl
from albumentations import (
    Compose, Resize, HorizontalFlip, RandomBrightnessContrast, Normalize, BboxParams
)
from albumentations.pytorch import ToTensorV2
import argparse


# COCO class names
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'TV',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}



def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes in a batch.
    
    Args:
        batch (list): A list of tuples, each containing (image, target_cls, boxes, obj_labels).
    
    Returns:
        tuple: Collated batch containing stacked images, padded target_cls, padded boxes, and padded obj_labels.
    """
    images, target_cls_list, boxes_list, obj_labels_list = [], [], [], []
    max_boxes = max(len(boxes) for _, _, boxes, _ in batch)

    # Iterate through each sample in the batch
    for img, target_cls, boxes, obj_labels in batch:
        images.append(img)
        
        # Pad target classes to max_boxes size
        target_cls_padded = torch.full((max_boxes,), -1, dtype=torch.int64)  # (max_boxes,)
        target_cls_padded[:len(target_cls)] = target_cls
        target_cls_list.append(target_cls_padded)

        # Pad bounding boxes to max_boxes size with -1 to indicate invalid boxes
        boxes_padded = torch.full((max_boxes, 4), -1, dtype=torch.float32)  # (max_boxes, 4)
        boxes_padded[:len(boxes), :] = boxes
        boxes_list.append(boxes_padded)

        # Pad object labels to max_boxes size
        obj_labels_padded = torch.full((max_boxes,), -1, dtype=torch.float32)  # (max_boxes,)
        obj_labels_padded[:len(obj_labels)] = obj_labels
        obj_labels_list.append(obj_labels_padded)

    # Stack all images, target classes, bounding boxes, and object labels
    images = torch.stack(images)
    target_cls = torch.stack(target_cls_list)
    boxes = torch.stack(boxes_list)
    obj_labels = torch.stack(obj_labels_list)
    
    # Check that bounding box coordinates are between 0 and 1, ignoring padded (-1) values
    if torch.any((boxes != -1) & ((boxes < 0) | (boxes > 1))):
        raise ValueError(f"Bounding box coordinates should be normalized between 0 and 1, or -1 for padding.")
    
    # Check that objectness is between 0 and 1, ignoring padded (-1) values
    if torch.any((obj_labels != -1) & ((obj_labels < 0) | (obj_labels > 1))):
        raise ValueError(f"Objectness should be normalized between 0 and 1, or -1 for padding.")
    
    # Check that target_cls contains valid class indices or padding values (-1)
    if not torch.all((target_cls == -1) | ((target_cls >= 0) & (target_cls < 80))):
        raise ValueError(f"Class indices should be between 0 and {80 - 1}, or -1 for padding.")
    
    # Check that boxes have shape [batch_size, max_boxes, 4]
    if boxes.dim() != 3 or boxes.size(2) != 4:
        raise ValueError(f"boxes should have shape [batch_size, max_boxes, 4], but got {boxes.shape}.")

    return images, target_cls, boxes, obj_labels


class YOLODataset(Dataset):
    """
    Custom Dataset class for loading images and bounding box labels for YOLO object detection.
    """
    def __init__(self, image_dir, label_dir, visualize=False, num_classes=80):
        """
        Args:
            image_dir (str): Directory containing image files.
            label_dir (str): Directory containing corresponding label files.
            num_classes (int): Number of object classes.
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_classes = num_classes
        self.viz = visualize
        self.images = list(self.image_dir.glob("*.jpg"))  # List of image paths
        self.transformed_size = 512  # Since Resize is set to (512, 512)

        
        # Define data augmentation and normalization transforms
        self.transforms = Compose([
            Resize(self.transformed_size, self.transformed_size),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=BboxParams(format='pascal_voc', label_fields=['category_ids']))  # Pascal VOC format: [x_min, y_min, x_max, y_max]

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Loads an image and its corresponding bounding boxes and labels, applies transformations, and returns them.
        
        Args:
            index (int): Index of the image to be loaded.
        
        Returns:
            tuple: Transformed image, one-hot encoded target classes, normalized bounding boxes, and object labels.
        """
        # Load image
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        original_width, original_height = img.size
        
        # Load label
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # Parse YOLO label format (class, x_center, y_center, width, height)
                    cls, x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append(int(cls))
                    
                    # Convert to Pascal VOC format: (xmin, ymin, xmax, ymax)
                    xmin = (x_center - width / 2) * original_width
                    ymin = (y_center - height / 2) * original_height
                    xmax = (x_center + width / 2) * original_width
                    ymax = (y_center + height / 2) * original_height
                    
                    # Clamp bounding boxes to be within the image dimensions
                    xmin = max(0, min(original_width, xmin))
                    ymin = max(0, min(original_height, ymin))
                    xmax = max(0, min(original_width, xmax))
                    ymax = max(0, min(original_height, ymax))
                    
                    boxes.append([xmin, ymin, xmax, ymax])

        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transforms(image=np.array(img), bboxes=boxes, category_ids=labels)
            img = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['category_ids'], dtype=torch.int64)
        else:
            # Handle cases with no bounding boxes
            transformed = self.transforms(image=np.array(img), bboxes=[], category_ids=[])
            img = transformed['image']
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            
        # Visualize the image with bounding boxes (for debugging/verification)
        if self.viz:
            self.visualize(img, boxes, labels)
            
        # Normalize bounding boxes
        if boxes.size(0) > 0:  # Ensure there are boxes to normalize
            boxes[:, 0] /= self.transformed_size   # Normalize xmin
            boxes[:, 1] /= self.transformed_size  # Normalize ymin
            boxes[:, 2] /= self.transformed_size   # Normalize xmax
            boxes[:, 3] /= self.transformed_size  # Normalize ymax
            boxes = boxes.clamp(0, 1)

        # Create object labels (used as confidence scores in some models)
        obj_labels = torch.ones((len(labels),), dtype=torch.float32)
        
        # Return target class labels directly for single-class detection
        target_cls = labels

        return img, target_cls, boxes, obj_labels
    
    def visualize(self, img, boxes, labels):
        """
        Visualize the image and its bounding boxes using OpenCV.
        
        Args:
            img (Tensor): Transformed image tensor.
            boxes (Tensor): Bounding boxes tensor.
        """
        # Convert image tensor back to numpy for visualization
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to numpy format (H, W, C)

        # Denormalize the image (reverse the normalization applied during transformations)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean  # Reverse normalization
        img_np = np.clip(img_np, 0, 1)  # Ensure pixel values are within [0, 1]
        img_np = (img_np * 255).astype(np.uint8)  # Scale to [0, 255]

        # Convert from RGB to BGR for OpenCV
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes and labels
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            img_np = cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green color
            
            # Add label text above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(str(COCO_CLASSES[int(label) + 1]), font, font_scale, thickness)
            text_x, text_y = xmin, ymin - 5
            # Draw a filled rectangle behind the text to improve visibility
            cv2.rectangle(img_np, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 255, 0), cv2.FILLED)
            # Put the label text on the image
            img_np = cv2.putText(img_np, str(COCO_CLASSES[int(label) + 1]), (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

            # Display the image using OpenCV
            cv2.imshow('Image with Bounding Boxes', img_np)
            cv2.waitKey(0)  # Wait for a key press to continue
            cv2.destroyAllWindows()  # Close the window


def main():
    """
    Main function for testing the YOLODataset and DataLoader with custom collate function.
    """
    
    parser = argparse.ArgumentParser(description="YOLO Dataset Loader")
    parser.add_argument('--viz', type=bool, default=False, help='Visualize images in batch')
    args = parser.parse_args()
    
    # Directories containing images and labels
    image_dir = 'coco128/coco128/images/train2017'  # Update with your actual image directory path
    label_dir = 'coco128/coco128/labels/train2017'  # Update with your actual label directory path
    
    # Create dataset and dataloader
    dataset = YOLODataset(image_dir, label_dir, visualize=args.viz)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # Iterate through the dataloader
    for batch_idx, (images, target_cls, boxes, obj_labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Target classes shape: {target_cls.shape}")
        print(f"Boxes shape: {boxes.shape}")
        print(f"Object labels shape: {obj_labels.shape}")
        break  # Remove break to iterate through all batches

if __name__ == "__main__":
    main()