"""
utils.py - CSV loading, saving, and general helpers.
"""

import csv
import os
from pathlib import Path


def load_ground_truth(csv_path):
    """
    Load ground truth from train.csv.
    
    Returns:
        dict: {image_id: [(marker_id, x, y), ...]}
    """
    gt = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            img_id = row[0].strip()
            pred_str = row[1].strip()
            dets = []
            if pred_str:
                parts = pred_str.split()
                for i in range(0, len(parts), 3):
                    dets.append((int(parts[i]), float(parts[i+1]), float(parts[i+2])))
            gt[img_id] = dets
    return gt


def save_submission(predictions, output_path):
    """
    Save predictions as Kaggle submission CSV.
    
    Args:
        predictions: dict {image_id: [(marker_id, x, y), ...]}
        output_path: path to save CSV
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'prediction_string'])
        
        for img_id in sorted(predictions.keys()):
            dets = predictions[img_id]
            if dets:
                pred_str = " ".join(f"{m} {x:.3f} {y:.3f}" for m, x, y in dets)
            else:
                # Kaggle rejects null/empty values.
                # Use a dummy prediction: marker ID 0 at (0, 0).
                # If the image truly has no GT markers, any prediction = score 0.
                # If it has GT markers, this is wrong but better than Kaggle rejecting.
                pred_str = "0 0.000 0.000"
            writer.writerow([img_id, pred_str])


def get_image_list(directory, extension=".jpg"):
    """Get sorted list of image paths in a directory."""
    return sorted(Path(directory).glob(f"*{extension}"))