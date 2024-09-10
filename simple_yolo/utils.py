import os
import requests
from pathlib import Path


# Download COCO128 dataset
def download_coco128(data_dir="coco128"):
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    dataset_path = Path(data_dir)
    zip_path = dataset_path / "coco128.zip"
    
    if not dataset_path.exists():
        os.makedirs(dataset_path)

    # Download dataset if not already downloaded
    if not zip_path.exists():
        print("Downloading COCO128 dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        with open(zip_path, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)

        print("Download completed!")

    # Extract dataset if not already extracted
    if not (dataset_path / 'coco128').exists():
        print("Extracting COCO128 dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
        print("Extraction completed!")