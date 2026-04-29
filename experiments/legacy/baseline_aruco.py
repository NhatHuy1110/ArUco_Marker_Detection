"""
=============================================================================
Stage 1: Baseline ArUco Marker Detection Pipeline
=============================================================================
CO3057 - Computer Vision and Digital Image Processing, HK252
ArUco Marker Detection Challenge

Pipeline:
  1. Load images from train/test folders
  2. Detect ArUco markers using OpenCV's built-in detector (DICT_ARUCO_MIP_36h12)
  3. Evaluate on training set (local score)
  4. Generate submission CSV for test set

Usage:
  python baseline_aruco.py --data_dir ./aruco_data/aruco_data \
                           --train_csv ./train.csv \
                           --output_csv ./submission_baseline.csv

Author: [Your Name]
=============================================================================
"""

import os
import csv
import math
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# 1. ARUCO DETECTOR SETUP
# =============================================================================

def create_aruco_detector():
    """
    Create an ArUco detector with DICT_ARUCO_MIP_36h12 dictionary
    and default parameters.
    
    The ARUCO_MIP_36h12 dictionary contains 250 markers (ID 0-249),
    each encoded as a 6x6 binary grid with a 1-cell black border,
    making it 8x8 total. "36h12" means 36-bit code with min Hamming
    distance of 12 between any two codewords -> very robust to errors.
    
    Returns:
        detector: cv2.aruco.ArucoDetector instance
    """
    # Load the predefined dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    
    # Create detector parameters (using defaults for baseline)
    params = cv2.aruco.DetectorParameters()
    
    # --- Key parameters explained (defaults shown) ---
    # Adaptive thresholding: converts grayscale to binary
    # params.adaptiveThreshWinSizeMin = 3      # smallest window
    # params.adaptiveThreshWinSizeMax = 23     # largest window
    # params.adaptiveThreshWinSizeStep = 10    # step between windows
    # params.adaptiveThreshConstant = 7        # constant subtracted from mean
    
    # Contour filtering: reject contours that are too small/large
    # params.minMarkerPerimeterRate = 0.03     # min perimeter relative to image
    # params.maxMarkerPerimeterRate = 4.0      # max perimeter relative to image
    
    # Corner refinement: improve corner localization accuracy
    # Options: CORNER_REFINE_NONE, CORNER_REFINE_SUBPIX, 
    #          CORNER_REFINE_CONTOUR, CORNER_REFINE_APRILTAG
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Error correction: how many bits can be corrected
    # Higher = more tolerant but more false positives
    # params.errorCorrectionRate = 0.6  # default
    
    # Create the detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    
    return detector


# =============================================================================
# 2. DETECTION FUNCTION
# =============================================================================

def detect_markers_in_image(detector, image_path):
    """
    Detect ArUco markers in a single image.
    
    OpenCV's detectMarkers returns corners in this order for each marker:
        corners[i] = [[top-left, top-right, bottom-right, bottom-left]]
    
    These are in the marker's CANONICAL orientation (determined by the
    bit pattern), not the geometric position in the image. So corners[0]
    is always the "top-left" of the marker as defined by its encoding.
    
    Args:
        detector: cv2.aruco.ArucoDetector
        image_path: path to the image file
    
    Returns:
        list of (marker_id, x_topleft, y_topleft) tuples
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARNING] Cannot read image: {image_path}")
        return []
    
    # Convert to grayscale (ArUco detection works on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    # corners: list of Nx4x2 arrays (N markers, 4 corners each, x-y coords)
    # ids: Nx1 array of marker IDs
    # rejected: list of rejected candidate quadrilaterals
    corners, ids, rejected = detector.detectMarkers(gray)
    
    detections = []
    
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            # corners[i] shape: (1, 4, 2) -> squeeze to (4, 2)
            # Index 0 = top-left corner of the marker
            top_left_x = float(corners[i][0][0][0])
            top_left_y = float(corners[i][0][0][1])
            
            detections.append((marker_id, top_left_x, top_left_y))
    
    return detections


# =============================================================================
# 3. EVALUATION METRIC (matches Kaggle's scoring)
# =============================================================================

def compute_score_single_image(gt_detections, pred_detections, img_height, img_width, sigma=0.02, lam=1.0):
    """
    Compute the per-image score following the exact Kaggle metric.
    
    Matching rule:
        - Match predictions to ground truth by marker ID
        - For each ID, if more predictions than GT, keep closest ones
        - Extra predictions = spam
    
    Distance score:
        phi(d_norm) = exp(-d_norm^2 / (2 * sigma^2))
        where d_norm = euclidean_distance / image_diagonal
    
    Per-image score:
        Score = sum(phi) / (N_gt + lambda * N_spam)
    
    Special case: if N_gt = 0, score is 1.0 if no predictions, else 0.0
    """
    N_gt = len(gt_detections)
    
    # Special case: no ground truth markers
    if N_gt == 0:
        return 1.0 if len(pred_detections) == 0 else 0.0
    
    # Image diagonal for normalization
    diagonal = math.sqrt(img_height ** 2 + img_width ** 2)
    
    # Group GT and predictions by marker ID
    gt_by_id = {}
    for (mid, x, y) in gt_detections:
        gt_by_id.setdefault(mid, []).append((x, y))
    
    pred_by_id = {}
    for (mid, x, y) in pred_detections:
        pred_by_id.setdefault(mid, []).append((x, y))
    
    total_phi = 0.0
    total_spam = 0
    
    # Process each predicted ID
    all_pred_ids = set(pred_by_id.keys())
    all_gt_ids = set(gt_by_id.keys())
    
    for pred_id in all_pred_ids:
        preds = pred_by_id[pred_id]
        
        if pred_id not in gt_by_id:
            # All predictions for this ID are spam (wrong ID)
            total_spam += len(preds)
            continue
        
        gts = gt_by_id[pred_id]
        
        # Compute distances from each prediction to each GT
        distances = []
        for j, (px, py) in enumerate(preds):
            min_dist = float('inf')
            for (gx, gy) in gts:
                d = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                min_dist = min(min_dist, d)
            distances.append((min_dist, j))
        
        # Sort by distance (closest first)
        distances.sort()
        
        # Keep only |Gk| best matches, rest are spam
        n_valid = min(len(preds), len(gts))
        n_spam_this_id = max(0, len(preds) - len(gts))
        total_spam += n_spam_this_id
        
        # Compute phi for valid matches
        for k in range(n_valid):
            d_min = distances[k][0]
            d_norm = d_min / diagonal
            phi = math.exp(-(d_norm ** 2) / (2 * sigma ** 2))
            total_phi += phi
    
    # Final per-image score
    score = total_phi / (N_gt + lam * total_spam)
    return score


def evaluate_on_trainset(detector, train_dir, train_csv_path):
    """
    Evaluate the detector on the training set and compute:
    - Per-image scores
    - Overall mean score (= Kaggle final score)
    - Statistics on detections
    
    Returns:
        mean_score, per_image_results (list of dicts)
    """
    # Load ground truth
    gt_data = {}
    with open(train_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            image_id = row[0].strip()
            pred_str = row[1].strip()
            
            detections = []
            if pred_str:
                parts = pred_str.split()
                for i in range(0, len(parts), 3):
                    mid = int(parts[i])
                    x = float(parts[i + 1])
                    y = float(parts[i + 2])
                    detections.append((mid, x, y))
            
            gt_data[image_id] = detections
    
    print(f"Loaded ground truth for {len(gt_data)} images")
    print(f"Evaluating on training set...\n")
    
    scores = []
    results = []
    total_gt = 0
    total_detected = 0
    total_correct_id = 0
    
    sorted_ids = sorted(gt_data.keys())
    
    for idx, image_id in enumerate(sorted_ids):
        # Build image path
        image_path = os.path.join(train_dir, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"  [SKIP] Image not found: {image_path}")
            continue
        
        # Get image dimensions
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Detect
        pred_detections = detect_markers_in_image(detector, image_path)
        gt_detections = gt_data[image_id]
        
        # Score
        score = compute_score_single_image(gt_detections, pred_detections, h, w)
        scores.append(score)
        
        # Statistics
        n_gt = len(gt_detections)
        n_pred = len(pred_detections)
        gt_ids = set(d[0] for d in gt_detections)
        pred_ids = set(d[0] for d in pred_detections)
        correct_ids = gt_ids & pred_ids
        
        total_gt += n_gt
        total_detected += n_pred
        total_correct_id += len(correct_ids)
        
        results.append({
            'image_id': image_id,
            'score': score,
            'n_gt': n_gt,
            'n_pred': n_pred,
            'n_correct_ids': len(correct_ids),
            'missed_ids': gt_ids - pred_ids,
            'false_ids': pred_ids - gt_ids,
        })
        
        # Print progress every 200 images
        if (idx + 1) % 200 == 0:
            running_mean = np.mean(scores)
            print(f"  [{idx+1}/{len(sorted_ids)}] Running mean score: {running_mean:.4f}")
    
    mean_score = np.mean(scores) if scores else 0.0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Images evaluated:     {len(scores)}")
    print(f"Mean score (Kaggle):  {mean_score:.4f}")
    print(f"Total GT markers:     {total_gt}")
    print(f"Total detected:       {total_detected}")
    print(f"Detection rate:       {total_detected/total_gt*100:.1f}%")
    print(f"{'='*60}")
    
    # Score distribution
    score_arr = np.array(scores)
    print(f"\nScore distribution:")
    print(f"  Min:    {score_arr.min():.4f}")
    print(f"  Q1:     {np.percentile(score_arr, 25):.4f}")
    print(f"  Median: {np.percentile(score_arr, 50):.4f}")
    print(f"  Q3:     {np.percentile(score_arr, 75):.4f}")
    print(f"  Max:    {score_arr.max():.4f}")
    
    # Worst images
    results_sorted = sorted(results, key=lambda x: x['score'])
    print(f"\n10 worst images:")
    for r in results_sorted[:10]:
        print(f"  {r['image_id']}: score={r['score']:.4f}, "
              f"gt={r['n_gt']}, pred={r['n_pred']}, "
              f"missed={r['missed_ids']}")
    
    # Score -> marks estimate
    if mean_score >= 0.78:
        marks = (1 - (0.97 - mean_score) / (0.97 - 0.78)) * 8
        marks = max(0, min(8, marks))
        print(f"\nEstimated Kaggle marks (score part): {marks:.1f}/8")
    else:
        print(f"\nWARNING: Score {mean_score:.4f} is below baseline (0.78)!")
        print(f"You need to surpass 0.78 to get any marks.")
    
    return mean_score, results


# =============================================================================
# 4. GENERATE SUBMISSION CSV
# =============================================================================

def generate_submission(detector, test_dir, output_csv_path):
    """
    Run detection on all test images and generate Kaggle submission CSV.
    
    Format:
        image_id,prediction_string
        000000000089,29 481.785 261.833 102 273.434 321.559 ...
    """
    # Get all test image files
    test_images = sorted(Path(test_dir).glob("*.jpg"))
    print(f"Found {len(test_images)} test images")
    
    rows = []
    total_markers = 0
    
    for idx, img_path in enumerate(test_images):
        image_id = img_path.stem  # filename without extension
        
        # Detect markers
        detections = detect_markers_in_image(detector, img_path)
        total_markers += len(detections)
        
        # Build prediction string: "id1 x1 y1 id2 x2 y2 ..."
        if len(detections) > 0:
            parts = []
            for (mid, x, y) in detections:
                parts.append(f"{mid} {x:.3f} {y:.3f}")
            pred_string = " ".join(parts)
        else:
            pred_string = ""
        
        rows.append([image_id, pred_string])
        
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(test_images)}] processed, "
                  f"{total_markers} markers detected so far")
    
    # Write CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'prediction_string'])
        writer.writerows(rows)
    
    print(f"\nSubmission saved to: {output_csv_path}")
    print(f"Total test images: {len(rows)}")
    print(f"Total markers detected: {total_markers}")
    print(f"Average markers per image: {total_markers/len(rows):.1f}")


# =============================================================================
# 5. VISUALIZATION (for debugging and report)
# =============================================================================

def visualize_detections(detector, image_path, save_path=None):
    """
    Draw detected markers on the image with IDs and corner points.
    Useful for debugging and for including in the report.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Draw detected markers (green)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        
        # Draw top-left corner explicitly (red dot)
        for i in range(len(ids)):
            tl = corners[i][0][0]  # top-left corner
            cv2.circle(img, (int(tl[0]), int(tl[1])), 5, (0, 0, 255), -1)
            cv2.putText(img, f"ID:{ids[i][0]}", 
                       (int(tl[0]) - 10, int(tl[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw rejected candidates (red outline) - helps understand failures
    if rejected is not None and len(rejected) > 0:
        for rej in rejected:
            pts = rej[0].astype(int)
            cv2.polylines(img, [pts], True, (0, 0, 200), 1)
    
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Visualization saved to: {save_path}")
    
    return img


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Baseline ArUco Marker Detection"
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to aruco_data/aruco_data folder (contains train/ and test/)'
    )
    parser.add_argument(
        '--train_csv', type=str, required=True,
        help='Path to train.csv (ground truth)'
    )
    parser.add_argument(
        '--output_csv', type=str, default='submission_baseline.csv',
        help='Output CSV path for Kaggle submission'
    )
    parser.add_argument(
        '--mode', type=str, default='all',
        choices=['eval', 'submit', 'all', 'visualize'],
        help='Mode: eval (evaluate on train), submit (generate test CSV), '
             'all (both), visualize (save detection images)'
    )
    parser.add_argument(
        '--vis_dir', type=str, default='visualizations',
        help='Directory to save visualization images'
    )
    parser.add_argument(
        '--vis_count', type=int, default=20,
        help='Number of images to visualize'
    )
    
    args = parser.parse_args()
    
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    # Verify paths
    assert os.path.isdir(train_dir), f"Train dir not found: {train_dir}"
    assert os.path.isdir(test_dir), f"Test dir not found: {test_dir}"
    assert os.path.isfile(args.train_csv), f"Train CSV not found: {args.train_csv}"
    
    print("=" * 60)
    print("ArUco Marker Detection - Stage 1 Baseline")
    print("=" * 60)
    print(f"Data directory:  {args.data_dir}")
    print(f"Train CSV:       {args.train_csv}")
    print(f"Output CSV:      {args.output_csv}")
    print(f"Mode:            {args.mode}")
    print()
    
    # Create detector
    detector = create_aruco_detector()
    print("Detector created: DICT_ARUCO_MIP_36h12")
    print(f"Corner refinement: SUBPIX")
    print()
    
    # === EVALUATE ON TRAINING SET ===
    if args.mode in ['eval', 'all']:
        print("-" * 60)
        print("PHASE 1: Evaluating on training set")
        print("-" * 60)
        start = time.time()
        mean_score, results = evaluate_on_trainset(
            detector, train_dir, args.train_csv
        )
        elapsed = time.time() - start
        print(f"Evaluation time: {elapsed:.1f}s "
              f"({elapsed/len(results)*1000:.0f}ms per image)")
    
    # === GENERATE SUBMISSION ===
    if args.mode in ['submit', 'all']:
        print()
        print("-" * 60)
        print("PHASE 2: Generating test submission")
        print("-" * 60)
        start = time.time()
        generate_submission(detector, test_dir, args.output_csv)
        elapsed = time.time() - start
        print(f"Submission time: {elapsed:.1f}s")
    
    # === VISUALIZATION ===
    if args.mode in ['visualize', 'all']:
        print()
        print("-" * 60)
        print("PHASE 3: Saving visualizations")
        print("-" * 60)
        os.makedirs(args.vis_dir, exist_ok=True)
        
        train_images = sorted(Path(train_dir).glob("*.jpg"))[:args.vis_count]
        for img_path in train_images:
            save_path = os.path.join(args.vis_dir, f"det_{img_path.name}")
            visualize_detections(detector, img_path, save_path)
        
        print(f"Saved {len(train_images)} visualizations to {args.vis_dir}/")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()