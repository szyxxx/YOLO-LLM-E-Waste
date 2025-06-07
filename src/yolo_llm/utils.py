import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def read_label_file(label_path: str) -> list:
    """Read YOLO format label file and return list of (class_id, bbox) tuples."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 5:
                class_id = int(values[0])
                bbox = [float(x) for x in values[1:]]
                labels.append((class_id, bbox))
    return labels

def convert_yolo_to_xyxy(bbox, img_width, img_height):
    """Convert YOLO format bbox to (x1, y1, x2, y2) format."""
    x_center, y_center, width, height = bbox
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return [x1, y1, x2, y2]

def draw_annotated_image(image_path, detections, save_path, classes):
    """Draw bounding boxes and labels on image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        bbox = det['bbox']
        class_id = det['class']
        conf = det['confidence']
        label = f"{classes[class_id]} {conf:.1f}"
        color = (0, 255, 255)  # Cyan color
        
        # Draw rectangle
        draw.rectangle(bbox, outline=color, width=3)
        
        # Calculate text size
        text_size = draw.textbbox((bbox[0], bbox[1]), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        
        # Draw filled rectangle for text background
        draw.rectangle([bbox[0], bbox[1], bbox[0] + text_width, bbox[1] + text_height], fill=color)
        
        # Draw text (black for contrast)
        draw.text((bbox[0], bbox[1]), label, fill=(0, 0, 0), font=font)
    
    image.save(save_path)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0 