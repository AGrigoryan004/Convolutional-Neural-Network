"""
YOLOv8 Object Detection Pipeline for Pascal VOC Dataset

This script handles:
1. Dataset preparation and conversion to YOLO format
2. Model training and validation
3. Performance evaluation and visualization
"""

import torch
from pathlib import Path
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from ultralytics import YOLO
from torchvision.datasets import VOCDetection
from typing import List, Dict, Tuple, Any

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in use: {DEVICE}")

# Class labels for Pascal VOC dataset
CLASS_LABELS = [
    'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataHandler:
    """Handles Pascal VOC dataset preparation and conversion to YOLO format"""
    
    def __init__(self):
        """Initialize with base dataset directory"""
        self.base_dir = Path("datasets/VOC")
        
    def prepare_dataset(self) -> VOCDetection:
        """
        Prepare Pascal VOC dataset by:
        1. Creating directory structure
        2. Downloading dataset
        3. Converting annotations to YOLO format
        4. Creating YAML configuration file
        
        Returns:
            VOCDetection: Validation dataset for evaluation
        """
        print("Initializing Pascal VOC dataset preparation...")
        
        # Create required directories
        for data_split in ['train', 'validation']:
            (self.base_dir/data_split/'images').mkdir(parents=True, exist_ok=True)
            (self.base_dir/data_split/'labels').mkdir(parents=True, exist_ok=True)

        # Download datasets
        train_data = VOCDetection(root='./datasets', year='2012', 
                                image_set='train', download=True)
        val_data = VOCDetection(root='./datasets', year='2012', 
                               image_set='val', download=True)

        # Process both splits
        self._process_data(train_data, 'train')
        self._process_data(val_data, 'validation')

        # Create YAML configuration file
        config = {
            'train': str(self.base_dir/'train'),
            'val': str(self.base_dir/'validation'),
            'nc': len(CLASS_LABELS),
            'names': CLASS_LABELS
        }

        with open('voc_setup.yaml', 'w') as cfg_file:
            yaml.dump(config, cfg_file)

        print(f"Dataset prepared - Training: {len(train_data)}, Validation: {len(val_data)}")
        return val_data

    def _process_data(self, dataset: VOCDetection, split_name: str) -> None:
        """
        Convert VOC annotations to YOLO format and save images
        
        Args:
            dataset: VOC dataset to process
            split_name: Name of dataset split ('train' or 'validation')
        """
        output_path = self.base_dir/split_name
        
        for i, (image, annot) in enumerate(dataset):
            # Save image
            img_id = annot['annotation']['filename'].split('.')[0]
            image.save(output_path/f"images/{img_id}.jpg")

            # Get image dimensions
            img_w, img_h = image.size
            
            # Prepare label file
            label_file = output_path/f"labels/{img_id}.txt"
            
            with label_file.open('w') as lbl_file:
                if 'object' in annot['annotation']:
                    objs = annot['annotation']['object']
                    objs = [objs] if not isinstance(objs, list) else objs
                    
                    # Process each object in the image
                    for obj in objs:
                        if obj['name'] in CLASS_LABELS:
                            # Convert VOC bbox to YOLO format (normalized cx, cy, w, h)
                            cls_idx = CLASS_LABELS.index(obj['name'])
                            box = obj['bndbox']
                            
                            x_c = (float(box['xmin']) + float(box['xmax'])) / (2 * img_w)
                            y_c = (float(box['ymin']) + float(box['ymax'])) / (2 * img_h)
                            w = (float(box['xmax']) - float(box['xmin'])) / img_w
                            h = (float(box['ymax']) - float(box['ymin'])) / img_h
                            
                            lbl_file.write(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            # Progress reporting
            if i % 1000 == 0:
                print(f"Converted {i}/{len(dataset)} samples")

class ModelEvaluator:
    """Handles model evaluation and metric calculations"""
    
    @staticmethod
    def compute_overlap(box_a: List[float], box_b: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box_a: First bounding box [x1, y1, x2, y2]
            box_b: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])

        # Check for no overlap
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        # Calculate areas
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        
        return intersection / (area_a + area_b - intersection) if (area_a + area_b - intersection) > 0 else 0.0

    @classmethod
    def assess_model(cls, model: YOLO, dataset: VOCDetection, sample_count: int = 200) -> Dict[str, float]:
        """
        Evaluate model performance on dataset
        
        Args:
            model: Trained YOLO model
            dataset: Validation dataset
            sample_count: Number of samples to evaluate
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print(f"Model assessment on {sample_count} samples initiated...")
        
        # Initialize counters
        true_pos, false_pos, false_neg = 0, 0, 0
        iou_values = []
        
        # Randomly select samples
        indices = np.random.choice(len(dataset), min(sample_count, len(dataset)), False)

        for idx in indices:
            img, annotation = dataset[idx]
            
            # Get ground truth data
            gt_boxes, gt_classes = cls._extract_ground_truth(annotation)
            
            # Get model predictions
            detections = model.predict(img, verbose=False, conf=0.5)
            pred_boxes, pred_classes = cls._extract_predictions(detections)
            
            # Match predictions to ground truth
            matched = set()
            for pred_box, pred_cls in zip(pred_boxes, pred_classes):
                best_match = cls._find_best_match(pred_box, pred_cls, gt_boxes, gt_classes, matched)
                
                # Count true/false positives based on IoU threshold
                if best_match['iou'] > 0.5:
                    true_pos += 1
                    iou_values.append(best_match['iou'])
                    matched.add(best_match['index'])
                else:
                    false_pos += 1

            # Count false negatives (undetected ground truths)
            false_neg += len(gt_boxes) - len(matched)

        # Calculate final metrics
        metrics = cls._calculate_metrics(true_pos, false_pos, false_neg, iou_values)
        return metrics

    @staticmethod
    def _extract_ground_truth(annotation: Dict) -> Tuple[List, List]:
        """Extract ground truth boxes and classes from VOC annotation"""
        boxes, classes = [], []
        if 'object' in annotation['annotation']:
            objects = annotation['annotation']['object']
            objects = [objects] if not isinstance(objects, list) else objects
            
            for obj in objects:
                if obj['name'] in CLASS_LABELS:
                    bbox = obj['bndbox']
                    boxes.append([float(bbox['xmin']), float(bbox['ymin']),
                                float(bbox['xmax']), float(bbox['ymax'])])
                    classes.append(CLASS_LABELS.index(obj['name']))
        return boxes, classes

    @staticmethod
    def _extract_predictions(results: List) -> Tuple[List, List]:
        """Extract predicted boxes and classes from model results"""
        boxes, classes = [], []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
        return boxes, classes

    @classmethod
    def _find_best_match(cls, pred_box: List, pred_cls: int, 
                        gt_boxes: List, gt_classes: List, matched: set) -> Dict[str, Any]:
        """
        Find best matching ground truth box for a prediction
        
        Args:
            pred_box: Predicted bounding box
            pred_cls: Predicted class
            gt_boxes: List of ground truth boxes
            gt_classes: List of ground truth classes
            matched: Set of already matched ground truth indices
            
        Returns:
            dict: Best match information with 'iou' and 'index'
        """
        best = {'iou': 0, 'index': -1}
        
        for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            # Skip already matched or mismatched classes
            if gt_idx in matched or pred_cls != gt_cls:
                continue
                
            current_iou = cls.compute_overlap(pred_box, gt_box)
            if current_iou > best['iou']:
                best['iou'] = current_iou
                best['index'] = gt_idx
                
        return best

    @staticmethod
    def _calculate_metrics(tp: int, fp: int, fn: int, ious: List) -> Dict[str, float]:
        """
        Calculate evaluation metrics from counts
        
        Args:
            tp: True positive count
            fp: False positive count
            fn: False negative count
            ious: List of IoU values for true positives
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'detection_accuracy': tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0,
            'classification_accuracy': precision,
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'mean_iou': np.mean(ious) if ious else 0,
            'total_detections': len(ious)
        }

class ResultVisualizer:
    """Handles visualization of training results and metrics"""
    
    @staticmethod
    def display_training_metrics(run_identifier: str = 'yolo_voc_run') -> None:
        """
        Plot training metrics from results CSV file
        
        Args:
            run_identifier: Name of the training run directory
        """
        results_file = Path(f"runs/detect/{run_identifier}/results.csv")
        if not results_file.exists():
            print("Training metrics data not available!")
            return

        # Load and clean data
        data = pd.read_csv(results_file)
        data.columns = data.columns.str.strip()

        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot training losses
        axs[0,0].plot(data.get('train/box_loss', []), 'b-', label='Bounding Box Loss')
        axs[0,0].plot(data.get('train/cls_loss', []), 'r-', label='Classification Loss')
        axs[0,0].set_title('Training Loss Components')
        axs[0,0].legend()
        axs[0,0].grid(True, alpha=0.3)

        # Plot mAP metrics
        axs[0,1].plot(data.get('metrics/mAP50(B)', []), 'g-', label='mAP@0.5')
        axs[0,1].plot(data.get('metrics/mAP50-95(B)', []), 'y-', label='mAP@0.5-0.95')
        axs[0,1].set_title('Detection Accuracy Metrics')
        axs[0,1].legend()
        axs[0,1].grid(True, alpha=0.3)

        # Plot precision-recall
        axs[1,0].plot(data.get('metrics/precision(B)', []), 'm-', label='Precision')
        axs[1,0].plot(data.get('metrics/recall(B)', []), 'c-', label='Recall')
        axs[1,0].set_title('Precision & Recall Evolution')
        axs[1,0].legend()
        axs[1,0].grid(True, alpha=0.3)

        # Plot F1 score
        if 'metrics/precision(B)' in data.columns and 'metrics/recall(B)' in data.columns:
            prec = data['metrics/precision(B)']
            rec = data['metrics/recall(B)']
            f1 = 2 * (prec * rec) / (prec + rec)
            f1 = f1.fillna(0)
            axs[1,1].plot(f1, 'r-', label='F1 Score')
        axs[1,1].set_title('F1 Score Trend')
        axs[1,1].legend()
        axs[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def execute_pipeline():
    """Main execution pipeline for training and evaluation"""
    print("YOLOv8 Object Detection Pipeline")
    print("=" * 45)

    # Step 1: Prepare dataset
    data_handler = VOCDataHandler()
    validation_set = data_handler.prepare_dataset()

    # Step 2: Train model
    print("\nInitiating model training...")
    detector = YOLO('yolov8n.pt')

    training_results = detector.train(
        data='voc_setup.yaml',
        name='yolo_voc_run',
        epochs=50,
        batch=16,
        imgsz=640,
        lr0=0.01,
        device=DEVICE,
        patience=10,
        save_period=10
    )

    # Step 3: Validate model
    print("\nRunning model validation...")
    validation_results = detector.val(data='voc_setup.yaml')

    # Step 4: Detailed evaluation
    print("\nComputing detailed metrics...")
    metrics = ModelEvaluator.assess_model(detector, validation_set, 300)

    # Step 5: Display results
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    print(f"mAP@0.5:          {validation_results.box.map50:.4f}")
    print(f"mAP@0.5-0.95:     {validation_results.box.map:.4f}")
    print(f"Detection Accuracy:       {metrics['detection_accuracy']:.4f}")
    print(f"Classification Accuracy:  {metrics['classification_accuracy']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"Mean IoU:         {metrics['mean_iou']:.4f}")
    print("=" * 55)

    # Step 6: Visualize results
    ResultVisualizer.display_training_metrics('yolo_voc_run')

    # Step 7: Save model
    detector.save('final_yolo_voc_model.pt')
    print("\nModel saved successfully as 'final_yolo_voc_model.pt'")

if __name__ == "__main__":
    execute_pipeline()
