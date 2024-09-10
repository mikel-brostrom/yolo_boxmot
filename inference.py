import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from pathlib import Path
from model import SimpleObjectDetector



def load_model(checkpoint_path, num_classes=80, num_boxes=100):
    model = SimpleObjectDetector(num_classes=num_classes, num_boxes=num_boxes)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()  # Set model to evaluation mode
    return model


def detect_objects(model, image_path, num_classes=80, conf_threshold=0.3, iou_threshold=0.6):
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_width, original_height = img.size  # Save original image dimensions
    transform = T.Compose([
        T.Resize((512, 512)),  # Resizing for model input
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Ensure these match training
    ])
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        pred_cls, pred_bbox, pred_obj = model(input_tensor)

    # Debugging: Check model output shapes
    print(f"Model outputs - pred_cls shape: {pred_cls.shape}, pred_bbox shape: {pred_bbox.shape}, pred_obj shape: {pred_obj.shape}")

    # Reshape predictions for processing
    batch_size, num_predictions, _ = pred_cls.shape
    pred_cls = torch.sigmoid(pred_cls)  # Apply sigmoid to class scores if needed
    pred_obj = torch.sigmoid(pred_obj)  # Apply sigmoid to objectness scores if needed

    pred_cls = pred_cls.view(-1, num_classes)  # Shape: (num_predictions, num_classes)
    pred_bbox = pred_bbox.view(-1, 4)  # Shape: (num_predictions, 4)
    pred_obj = pred_obj.view(-1)  # Shape: (num_predictions,)

    # Filter predictions by confidence threshold
    conf_mask = pred_obj >= conf_threshold
    pred_cls = pred_cls[conf_mask]
    pred_bbox = pred_bbox[conf_mask]
    pred_obj = pred_obj[conf_mask]

    print(f"Filtered {conf_mask.sum()} predictions above confidence threshold {conf_threshold}")

    # Apply Non-Maximum Suppression (NMS)
    selected_boxes = []
    selected_scores = []
    selected_labels = []
    for cls_idx in range(num_classes):
        class_scores = pred_cls[:, cls_idx]
        mask = class_scores >= conf_threshold
        if mask.sum() == 0:
            continue

        boxes = pred_bbox[mask]
        scores = class_scores[mask]
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

        selected_boxes.append(boxes[keep_indices])
        selected_scores.append(scores[keep_indices])
        selected_labels.extend([cls_idx] * len(keep_indices))

    # Combine results
    if len(selected_boxes) > 0:
        selected_boxes = torch.cat(selected_boxes, dim=0)
        selected_scores = torch.cat(selected_scores, dim=0)
    else:
        selected_boxes = torch.empty((0, 4))
        selected_scores = torch.empty(0)
        selected_labels = []

    print(f"Detected {selected_boxes.shape[0]} objects after NMS")
    
    # If in normalized format, scale boxes back to resized image dimensions (512, 512)
    selected_boxes[:, [0, 2]] *= 512  # Width scaling
    selected_boxes[:, [1, 3]] *= 512  # Height scaling

    # Clamp values to ensure they are within image boundaries after scaling
    selected_boxes[:, [0, 2]] = torch.clamp(selected_boxes[:, [0, 2]], min=0, max=512)
    selected_boxes[:, [1, 3]] = torch.clamp(selected_boxes[:, [1, 3]], min=0, max=512)

    print(f"Scaled and Clamped Bounding Boxes (first 5): {selected_boxes[:5]}")

    return selected_boxes, selected_scores, selected_labels, img


# Visualization function
def visualize_predictions(image, boxes, scores, labels, label_names=None):
    # Convert PIL image to OpenCV format (numpy array)
    image = np.array(image.resize((512, 512)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB (PIL) to BGR (OpenCV)

    # Draw boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i].cpu().numpy().astype(int)  # Convert to numpy and ensure integer coordinates
        score = scores[i].item()
        label = labels[i]
        color = (0, 0, 255)  # Red color in BGR format

        # Draw the bounding box
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Prepare label text
        label_text = f"{label_names[label] if label_names else label}: {score:.2f}"
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (box[0], box[1] - text_height - baseline), (box[0] + text_width, box[1]), color, thickness=cv2.FILLED)

        # Put label text on image
        cv2.putText(image, label_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the image with OpenCV
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()  # Close the window

def main():
    # Paths
    checkpoint_path = "lightning_logs/version_118/checkpoints/yolo-epoch=05-avg_train_loss=3.15.ckpt"  # Path to the trained weights
    image_path = "coco128/coco128/images/train2017/000000000025.jpg"  # Example image from COCO128
    
    # Load the trained model
    model = load_model(checkpoint_path)
    
    # Perform object detection
    boxes, scores, labels, img = detect_objects(model, image_path)
    
    # Define COCO labels (optional, for display purposes)
    label_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

    
    # Visualize results
    visualize_predictions(img.resize((512, 512)), boxes, scores, labels, label_names=label_names)

if __name__ == "__main__":
    main()
