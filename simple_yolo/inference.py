import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path
from simple_yolo.model import SimpleObjectDetector

def load_model(checkpoint_path, num_boxes=100):
    model = SimpleObjectDetector(num_boxes=num_boxes)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()  # Set model to evaluation mode
    return model

def detect_objects(model, image_path, conf_threshold=1, iou_threshold=0.8):
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
        bbox_pred, conf_pred = model(input_tensor)  # Updated model output

    # Reshape predictions for processing
    bbox_pred = bbox_pred.view(-1, 4)  # Shape: (num_predictions, 4)
    conf_pred = conf_pred.view(-1)  # Shape: (num_predictions,)
    

    # Filter predictions by confidence threshold
    conf_mask = conf_pred >= conf_threshold
    filtered_bbox = bbox_pred[conf_mask]
    filtered_scores = conf_pred[conf_mask]

    print(f"Filtered {conf_mask.sum()} predictions above confidence threshold {conf_threshold}")

    # Apply Non-Maximum Suppression (NMS)
    keep_indices = torchvision.ops.nms(filtered_bbox, filtered_scores, iou_threshold)
    selected_boxes = filtered_bbox[keep_indices]
    selected_scores = filtered_scores[keep_indices]

    print(f"Detected {selected_boxes.shape[0]} objects after NMS")

    # Scale boxes back to original image dimensions
    selected_boxes[:, [0, 2]] *= original_width
    selected_boxes[:, [1, 3]] *= original_height

    print(f"Scaled and Clamped Bounding Boxes (first 5): {selected_boxes[:5]} {selected_scores[:5]}")

    return selected_boxes, selected_scores, img

# Visualization function
def visualize_predictions(image, boxes, scores):
    # Convert PIL image to OpenCV format (numpy array)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB (PIL) to BGR (OpenCV)

    # Draw boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i].cpu().numpy().astype(int)  # Convert to numpy and ensure integer coordinates
        score = scores[i].item()
        color = (0, 0, 255)  # Red color in BGR format

        # Draw the bounding box
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Prepare label text
        label_text = f"Score: {score:.2f}"
        
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
    checkpoint_path = "lightning_logs/version_366/checkpoints/yolo-epoch=17-avg_train_loss=0.00.ckpt"  # Path to the trained weights
    image_path = "coco128/coco128/images/train2017/000000000025.jpg"  # Example image from COCO128
    
    # Load the trained model
    model = load_model(checkpoint_path)
    
    # Perform object detection
    boxes, scores, img = detect_objects(model, image_path)
    
    # Visualize results
    visualize_predictions(img.resize((512, 512)), boxes, scores)

if __name__ == "__main__":
    main()
