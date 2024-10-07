import os
from pathlib import Path
from PIL import Image
import zipfile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import pytorch_lightning as pl
from torchvision import models
from torchvision.ops import box_iou
from albumentations import (
    Compose, Resize, HorizontalFlip, RandomBrightnessContrast, Normalize
)
from albumentations.pytorch import ToTensorV2
from simple_yolo.utils import download_coco128  # Utility for downloading dataset
from simple_yolo.models.single_scale import SingleScaleModel
from simple_yolo.models.multi_scale import MultiScaleModel
from simple_yolo.models.multi_scale_fpn import MultiScaleFPN
from simple_yolo.dataset import collate_fn, YOLODataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def main():
    download_coco128()  # Ensure the dataset is downloaded and extracted
    image_dir = 'coco128/coco128/images/train2017'
    label_dir = 'coco128/coco128/labels/train2017'

    model = SingleScaleModel(num_classes=80)
    dataset = YOLODataset(image_dir, label_dir)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename='yolo-{epoch:02d}-{avg_train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    early_stopping_callback = EarlyStopping(monitor='train_loss', patience=30)

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
