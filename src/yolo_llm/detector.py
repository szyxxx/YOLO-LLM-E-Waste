import os
from PIL import Image
from ultralytics import YOLO
from .config import YOLO_MODEL_PATH, GEMINI_CLASSES
from .utils import convert_yolo_to_xyxy, draw_annotated_image
from .gemini import validate_detection

class YOLOLLMDetector:
    """Combined YOLO and Gemini LLM detector for electronic devices."""
    
    def __init__(self):
        """Initialize the detector with YOLO model."""
        self.model = YOLO(YOLO_MODEL_PATH)
    
    def process_image(self, image_path: str, use_gemini: bool = True) -> list:
        """
        Process an image and return detections.
        
        Args:
            image_path: Path to the input image
            use_gemini: Whether to use Gemini for validation
            
        Returns:
            list: List of detections with class, confidence, and bbox
        """
        image = Image.open(image_path)
        img_width, img_height = image.size
        detections = []
        
        # Get YOLO predictions
        results = self.model.predict(source=image_path, verbose=False)
        
        for box in results[0].boxes:
            xyxy = [int(x) for x in box.xyxy[0].tolist()]
            class_id = int(box.cls)
            
            if class_id >= len(GEMINI_CLASSES):
                continue
            
            if use_gemini:
                # Crop the detected object
                crop = image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
                yolo_class = GEMINI_CLASSES[class_id]
                
                # Validate with Gemini
                corrected_class = validate_detection(crop, yolo_class)
                if corrected_class == "ABSTAIN":
                    continue
                class_id = GEMINI_CLASSES.index(corrected_class) if corrected_class in GEMINI_CLASSES else class_id
            
            detections.append({
                'class': class_id,
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })
        
        return detections
    
    def process_dataset(self, images_dir: str, labels_dir: str, annotated_dir: str, use_gemini: bool = True):
        """
        Process a dataset of images and generate annotated results.
        
        Args:
            images_dir: Directory containing test images
            labels_dir: Directory containing ground truth labels
            annotated_dir: Directory to save annotated images
            use_gemini: Whether to use Gemini for validation
        """
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            detections = self.process_image(image_path, use_gemini)
            
            # Save annotated image
            annotated_path = os.path.join(annotated_dir, image_file)
            draw_annotated_image(image_path, detections, annotated_path, GEMINI_CLASSES) 