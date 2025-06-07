import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from .config import (
    TEST_IMAGES_DIR, TEST_LABELS_DIR, YOLO_ANNOTATED_DIR,
    GEMINI_ANNOTATED_DIR, METRICS_DIR, GEMINI_CLASSES
)
from .detector import YOLOLLMDetector
from .utils import read_label_file, convert_yolo_to_xyxy
from .metrics import calculate_metrics, plot_curves, save_results_csv

def main():
    """Main function to run the detector and generate evaluation metrics."""
    # Initialize detector
    detector = YOLOLLMDetector()
    conf_thresholds = np.linspace(0, 1, 101)

    # Process test dataset with YOLO
    print("Processing test dataset with YOLO...")
    yolo_detections, yolo_ground_truths, yolo_y_true, yolo_y_pred = process_dataset(
        detector, TEST_IMAGES_DIR, TEST_LABELS_DIR, YOLO_ANNOTATED_DIR, use_gemini=False
    )
    
    # Process test dataset with YOLO+Gemini
    print("Processing test dataset with YOLO+Gemini...")
    gemini_detections, gemini_ground_truths, gemini_y_true, gemini_y_pred = process_dataset(
        detector, TEST_IMAGES_DIR, TEST_LABELS_DIR, GEMINI_ANNOTATED_DIR, use_gemini=True
    )

    # Calculate metrics
    print("Calculating metrics...")
    yolo_metrics = calculate_metrics(yolo_detections, yolo_ground_truths, conf_thresholds)
    gemini_metrics = calculate_metrics(gemini_detections, gemini_ground_truths, conf_thresholds)

    # Generate plots
    print("Generating plots...")
    plot_curves(yolo_metrics, conf_thresholds, "yolo", METRICS_DIR)
    plot_curves(gemini_metrics, conf_thresholds, "yolo+gemini", METRICS_DIR)

    # Save results
    print("Saving results...")
    save_results_csv(yolo_metrics, gemini_metrics, conf_thresholds, METRICS_DIR)

    # Generate confusion matrices
    print("Generating confusion matrices...")
    for model_name, y_true, y_pred in [("yolo", yolo_y_true, yolo_y_pred), 
                                      ("yolo+gemini", gemini_y_true, gemini_y_pred)]:
        class_labels = GEMINI_CLASSES + ['background']
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name.capitalize()}')
        plt.colorbar()
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=90, fontsize=6)
        plt.yticks(tick_marks, class_labels, fontsize=6)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_DIR, f'confusion_matrix_{model_name}.png'))
        plt.close()

    print(f"All results saved in {METRICS_DIR}")
    print(f"YOLO results saved in {YOLO_ANNOTATED_DIR}")
    print(f"YOLO+Gemini results saved in {GEMINI_ANNOTATED_DIR}")

def process_dataset(detector, images_dir, labels_dir, annotated_dir, use_gemini=False):
    """Process dataset and return detections and ground truths."""
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    all_detections, all_ground_truths = [], []
    y_true_cm, y_pred_cm = [], []
    background_idx = len(GEMINI_CLASSES)

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
        
        # Get ground truth
        image = Image.open(image_path)
        img_width, img_height = image.size
        gt = read_label_file(label_path)
        gt_xyxy = [(c, convert_yolo_to_xyxy(b, img_width, img_height)) for c, b in gt]
        all_ground_truths.append(gt_xyxy)
        
        # Get detections
        dets = detector.process_image(image_path, use_gemini)
        all_detections.append(dets)
        
        # For confusion matrix
        gt_matched, det_matched = set(), set()
        for i, (gt_c, gt_box) in enumerate(gt_xyxy):
            best_iou, best_j = 0, -1
            for j, det in enumerate(dets):
                if j in det_matched or det['class'] != gt_c:
                    continue
                iou = calculate_iou(det['bbox'], gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou > 0.5 and best_j != -1:
                y_true_cm.append(gt_c)
                y_pred_cm.append(dets[best_j]['class'])
                gt_matched.add(i)
                det_matched.add(best_j)
            else:
                y_true_cm.append(gt_c)
                y_pred_cm.append(background_idx)
        for j, det in enumerate(dets):
            if j not in det_matched:
                y_true_cm.append(background_idx)
                y_pred_cm.append(det['class'])

    return all_detections, all_ground_truths, y_true_cm, y_pred_cm

if __name__ == "__main__":
    main() 