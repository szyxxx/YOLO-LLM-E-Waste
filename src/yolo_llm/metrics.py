import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
from .utils import calculate_iou

def calculate_metrics(all_detections, all_ground_truths, conf_thresholds):
    """Calculate various metrics for object detection evaluation."""
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'mAP50': [],
        'mAP50_95': []
    }
    
    for conf_threshold in conf_thresholds:
        tp, fp, fn = 0, 0, 0
        all_ap50 = []
        all_ap50_95 = []
        
        for dets, gts in zip(all_detections, all_ground_truths):
            # Filter detections by confidence threshold
            filtered_dets = [d for d in dets if d['confidence'] >= conf_threshold]
            
            # Calculate TP, FP, FN
            gt_matched = set()
            for det in filtered_dets:
                matched = False
                for i, (gt_c, gt_box) in enumerate(gts):
                    if i in gt_matched or det['class'] != gt_c:
                        continue
                    iou = calculate_iou(det['bbox'], gt_box)
                    if iou > 0.5:
                        tp += 1
                        gt_matched.add(i)
                        matched = True
                        break
                if not matched:
                    fp += 1
            
            fn += len(gts) - len(gt_matched)
            
            # Calculate AP50 and AP50-95
            for gt_c, gt_box in gts:
                ap50 = 0
                ap50_95 = 0
                for det in filtered_dets:
                    if det['class'] == gt_c:
                        iou = calculate_iou(det['bbox'], gt_box)
                        if iou > 0.5:
                            ap50 = 1
                        if iou > 0.95:
                            ap50_95 = 1
                all_ap50.append(ap50)
                all_ap50_95.append(ap50_95)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mAP50 = np.mean(all_ap50) if all_ap50 else 0
        mAP50_95 = np.mean(all_ap50_95) if all_ap50_95 else 0
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['mAP50'].append(mAP50)
        metrics['mAP50_95'].append(mAP50_95)
    
    return metrics

def plot_curves(metrics, conf_thresholds, model_name, save_dir):
    """Plot and save various evaluation curves."""
    # F1 Curve
    plt.figure(figsize=(10, 6))
    plt.plot(conf_thresholds, metrics['f1'], 'b-', label='F1 Score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score vs Confidence Threshold - {model_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'f1_curve_{model_name}.png'))
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['recall'], metrics['precision'], 'r-', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'pr_curve_{model_name}.png'))
    plt.close()
    
    # Precision Curve
    plt.figure(figsize=(10, 6))
    plt.plot(conf_thresholds, metrics['precision'], 'g-', label='Precision')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title(f'Precision vs Confidence Threshold - {model_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'precision_curve_{model_name}.png'))
    plt.close()
    
    # Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(conf_thresholds, metrics['recall'], 'y-', label='Recall')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.title(f'Recall vs Confidence Threshold - {model_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'recall_curve_{model_name}.png'))
    plt.close()

def save_results_csv(yolo_metrics, gemini_metrics, conf_thresholds, save_dir):
    """Save evaluation results to CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f'results_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'epoch', 'time', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
                        'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
                        'lr/pg0', 'lr/pg1', 'lr/pg2'])
        
        # For each confidence threshold, write a row for both models
        for i, conf in enumerate(conf_thresholds):
            # YOLO model
            writer.writerow([
                'yolo',  # model
                i,  # epoch
                datetime.now().timestamp(),  # time
                0,  # train/box_loss
                0,  # train/cls_loss
                0,  # train/dfl_loss
                yolo_metrics['precision'][i],  # metrics/precision(B)
                yolo_metrics['recall'][i],  # metrics/recall(B)
                yolo_metrics['mAP50'][i],  # metrics/mAP50(B)
                yolo_metrics['mAP50_95'][i],  # metrics/mAP50-95(B)
                0,  # val/box_loss
                0,  # val/cls_loss
                0,  # val/dfl_loss
                0,  # lr/pg0
                0,  # lr/pg1
                0   # lr/pg2
            ])
            
            # YOLO+Gemini model
            writer.writerow([
                'yolo+gemini',  # model
                i,  # epoch
                datetime.now().timestamp(),  # time
                0,  # train/box_loss
                0,  # train/cls_loss
                0,  # train/dfl_loss
                gemini_metrics['precision'][i],  # metrics/precision(B)
                gemini_metrics['recall'][i],  # metrics/recall(B)
                gemini_metrics['mAP50'][i],  # metrics/mAP50(B)
                gemini_metrics['mAP50_95'][i],  # metrics/mAP50-95(B)
                0,  # val/box_loss
                0,  # val/cls_loss
                0,  # val/dfl_loss
                0,  # lr/pg0
                0,  # lr/pg1
                0   # lr/pg2
            ]) 