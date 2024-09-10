import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import pytorch_lightning as pl
from albumentations import (
    Compose, Resize, HorizontalFlip, RandomBrightnessContrast, Normalize
)
from albumentations.pytorch import ToTensorV2


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

    for img, target_cls, boxes, obj_labels in batch:
        images.append(img)
        target_cls_padded = torch.zeros((max_boxes, target_cls.size(1)))  # (max_boxes, num_classes)
        target_cls_padded[:len(target_cls), :] = target_cls
        target_cls_list.append(target_cls_padded)

        boxes_padded = torch.zeros((max_boxes, 4))  # (max_boxes, 4)
        boxes_padded[:len(boxes), :] = boxes
        boxes_list.append(boxes_padded)

        obj_labels_padded = torch.zeros((max_boxes,))  # (max_boxes,)
        obj_labels_padded[:len(obj_labels)] = obj_labels
        obj_labels_list.append(obj_labels_padded)

    images = torch.stack(images)
    target_cls = torch.stack(target_cls_list)
    boxes = torch.stack(boxes_list)
    obj_labels = torch.stack(obj_labels_list)

    return images, target_cls, boxes, obj_labels


class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, num_classes=80):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_classes = num_classes
        self.images = list(self.image_dir.glob("*.jpg"))
        self.transforms = Compose([
            Resize(512, 512),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['category_ids']})  # pasacal voc format: [x_min, y_min, x_max, y_max]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
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
                    cls, x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append(int(cls))
                    
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
            transformed = self.transforms(image=np.array(img), bboxes=[], category_ids=[])
            img = transformed['image']
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            
        # Visualize the image with bounding boxes
        #self.visualize(img, boxes)
            
        # Normalize bounding boxes (tensor)
        if boxes.size(0) > 0:  # Ensure there are boxes to normalize
            boxes[:, 0] /= original_width   # Normalize xmin
            boxes[:, 1] /= original_height  # Normalize ymin
            boxes[:, 2] /= original_width   # Normalize xmax
            boxes[:, 3] /= original_height  # Normalize ymax

        obj_labels = torch.ones((len(labels),), dtype=torch.float32)
        
        # Convert labels to one-hot encoding
        target_cls = torch.zeros(len(labels), self.num_classes)
        if len(labels) > 0:
            target_cls[torch.arange(len(labels)), labels] = 1

        return img, target_cls, boxes, obj_labels
    
    def visualize(self, img, boxes):
        """
        Visualize the image and its bounding boxes using OpenCV.
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
        
        # Draw bounding boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            img_np = cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green color

        # Display the image using OpenCV
        cv2.imshow('Image with Bounding Boxes', img_np)
        cv2.waitKey(0)  # Wait for a key press to continue
        cv2.destroyAllWindows()  # Close the window