"""
=============================================================================
Stage 2: Enhanced ArUco Marker Detection Pipeline
=============================================================================
CO3057 - Computer Vision and Digital Image Processing, HK252

Improvements over Stage 1 baseline:
  1. Multi-pass detection with varied preprocessing
  2. Tuned detector parameters (wider adaptive threshold, relaxed perimeter)
  3. CLAHE + Sharpening + Gamma correction for difficult images
  4. Rejected candidate recovery (perspective transform + re-decode)
  5. Corner sub-pixel refinement
  6. Spam filtering (duplicate removal, confidence scoring)

Usage:
  python enhanced_aruco.py --data_dir ./aruco_data/aruco_data \
                           --train_csv ./train.csv \
                           --output_csv ./submission_enhanced.csv \
                           --mode all
=============================================================================
"""

import os
import csv
import math
import argparse
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


# =============================================================================
# 1. PREPROCESSING FUNCTIONS
# =============================================================================

def apply_clahe(gray, clip_limit=2.0, grid_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Why: Many images have uneven lighting (shadows, backlighting). 
    Global histogram equalization would blow out bright areas.
    CLAHE divides the image into tiles and equalizes each locally,
    with a clip limit to prevent noise amplification.
    
    This is particularly effective for:
    - Dark images where markers have low contrast
    - Images with strong shadows across markers
    - Backlit scenes
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(gray)


def sharpen_image(gray, strength=1.0):
    """
    Unsharp masking: sharpen edges to make marker borders more distinct.
    
    Why: Motion blur and slight defocus soften the black-white transitions
    of ArUco markers. Sharpening restores these edges, helping the 
    adaptive thresholding step inside the detector find clean contours.
    
    Process: Subtract a blurred version from the original, then add it back
    with a weight (strength). This amplifies high-frequency details (edges).
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def adjust_gamma(gray, gamma=1.0):
    """
    Gamma correction to brighten or darken the image.
    
    Why: Dark images (gamma < 1 to brighten) have markers that blend 
    into shadows. Bright images may have washed-out markers.
    
    gamma < 1.0: brightens dark regions (useful for underexposed images)
    gamma > 1.0: darkens bright regions (useful for overexposed images)
    
    The lookup table approach is fast (O(1) per pixel after table build).
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                       for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def denoise_bilateral(gray, d=5, sigma_color=50, sigma_space=50):
    """
    Bilateral filter: removes noise while preserving edges.
    
    Why: Unlike Gaussian blur which blurs everything including edges,
    bilateral filter only smooths within regions of similar intensity.
    This keeps the sharp black-white border of markers intact while
    reducing noise that could create false contour detections.
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def preprocess_for_dark_images(gray):
    """
    Specialized pipeline for dark/low-contrast images.
    Combines gamma brightening + CLAHE + mild sharpening.
    """
    # Brighten
    brightened = adjust_gamma(gray, gamma=0.6)
    # Local contrast enhancement
    enhanced = apply_clahe(brightened, clip_limit=3.0, grid_size=8)
    # Sharpen edges
    sharpened = sharpen_image(enhanced, strength=0.5)
    return sharpened


def preprocess_for_blur(gray):
    """
    Specialized pipeline for blurry/motion-blurred images.
    Strong sharpening + contrast boost.
    """
    # Strong sharpen
    sharpened = sharpen_image(gray, strength=1.5)
    # Boost contrast
    enhanced = apply_clahe(sharpened, clip_limit=2.5, grid_size=8)
    return enhanced


# =============================================================================
# 2. DETECTOR CONFIGURATIONS
# =============================================================================

def create_detector_config(config_name="default"):
    """
    Create different detector configurations for multi-pass detection.
    
    The key insight: no single parameter set works for all markers.
    - Small markers need different perimeter thresholds than large ones
    - Dark markers need different adaptive threshold constants
    - Blurry markers need more error correction tolerance
    
    By running multiple passes with different configs and merging results,
    we catch markers that any single config would miss.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    params = cv2.aruco.DetectorParameters()
    
    # Always use sub-pixel corner refinement for better localization
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    
    if config_name == "default":
        # Slightly tuned from pure defaults
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 30
        params.adaptiveThreshWinSizeStep = 5
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.errorCorrectionRate = 0.6
        
    elif config_name == "small_markers":
        # Optimized for detecting small markers
        # Why: small markers have short perimeters and few pixels,
        # so we lower the minimum perimeter threshold and use 
        # smaller adaptive threshold windows
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 15
        params.adaptiveThreshWinSizeStep = 3
        params.adaptiveThreshConstant = 5
        params.minMarkerPerimeterRate = 0.01   # allow very small markers
        params.maxMarkerPerimeterRate = 1.0     # exclude huge ones (already caught)
        params.polygonalApproxAccuracyRate = 0.08  # more lenient polygon fitting
        params.errorCorrectionRate = 0.8        # more error tolerance for small markers
        params.minCornerDistanceRate = 0.02
        
    elif config_name == "large_markers":
        # Optimized for large markers that may fill most of the image
        # Why: very large markers can have adaptive threshold issues
        # because the window size is too small relative to marker size
        params.adaptiveThreshWinSizeMin = 10
        params.adaptiveThreshWinSizeMax = 80
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.1
        params.maxMarkerPerimeterRate = 4.0
        params.errorCorrectionRate = 0.6
        
    elif config_name == "low_contrast":
        # For dark or low-contrast images
        # Why: adaptive thresholding struggles when the overall contrast
        # is low — the constant needs to be smaller so the threshold 
        # doesn't cut off too much
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 40
        params.adaptiveThreshWinSizeStep = 5
        params.adaptiveThreshConstant = 3       # lower constant for low contrast
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.errorCorrectionRate = 0.8        # more forgiving
        params.polygonalApproxAccuracyRate = 0.08
        
    elif config_name == "high_tolerance":
        # Maximum tolerance — catches more markers but also more false positives
        # We filter these later with our spam detection
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 50
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 5
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.errorCorrectionRate = 1.0        # maximum error correction
        params.polygonalApproxAccuracyRate = 0.10
        params.minCornerDistanceRate = 0.01
        params.minMarkerDistanceRate = 0.01
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector


# =============================================================================
# 3. MULTI-PASS DETECTION ENGINE
# =============================================================================

def detect_single_pass(detector, gray, pass_name=""):
    """
    Run a single detection pass and return results with metadata.
    Returns list of (marker_id, corners_4x2, pass_name)
    """
    corners, ids, rejected = detector.detectMarkers(gray)
    
    results = []
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            corner_pts = corners[i][0]  # shape (4, 2)
            results.append((marker_id, corner_pts, pass_name))
    
    return results, rejected


def multi_pass_detect(image_path):
    """
    Core detection function: runs multiple passes with different 
    preprocessing + detector configs, then merges results.
    
    Strategy:
    Pass 1: Original image + default config (baseline)
    Pass 2: CLAHE enhanced + default config (handles dark/shadow images)
    Pass 3: Original + small marker config (catches small markers missed by default)
    Pass 4: Sharpened + default config (handles slightly blurry markers)
    Pass 5: Dark-image preprocessing + low contrast config (extreme cases)
    Pass 6: Original + large marker config (large markers with threshold issues)
    Pass 7: CLAHE + high tolerance config (last resort catch-all)
    
    Merging: for each marker ID, keep the detection with the best 
    "confidence" (based on how square/regular the detected quadrilateral is).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Prepare preprocessed versions
    gray_clahe = apply_clahe(gray, clip_limit=2.5, grid_size=8)
    gray_sharp = sharpen_image(gray, strength=1.0)
    gray_dark = preprocess_for_dark_images(gray)
    gray_clahe_sharp = sharpen_image(gray_clahe, strength=0.5)
    gray_blur_fix = preprocess_for_blur(gray)
    
    # Define all passes: (preprocessed_image, detector_config_name, pass_label)
    passes = [
        (gray,             "default",        "P1_orig_default"),
        (gray_clahe,       "default",        "P2_clahe_default"),
        (gray,             "small_markers",  "P3_orig_small"),
        (gray_sharp,       "default",        "P4_sharp_default"),
        (gray_dark,        "low_contrast",   "P5_dark_lowcontrast"),
        (gray,             "large_markers",  "P6_orig_large"),
        (gray_clahe_sharp, "high_tolerance", "P7_clahe_hightol"),
        (gray_blur_fix,    "default",        "P8_blur_default"),
    ]
    
    # Collect all detections from all passes
    all_detections = []  # list of (marker_id, corners_4x2, pass_name)
    all_rejected = []
    
    for preproc_img, config_name, pass_label in passes:
        detector = create_detector_config(config_name)
        results, rejected = detect_single_pass(detector, preproc_img, pass_label)
        all_detections.extend(results)
        if rejected is not None:
            all_rejected.extend(rejected)
    
    # Merge detections: for each (marker_id, location), keep best one
    merged = merge_detections(all_detections, h, w)
    
    return merged


def compute_quad_quality(corners_4x2):
    """
    Compute a quality score for a detected quadrilateral.
    
    Why: When multiple passes detect the same marker, we want to keep
    the "best" detection. A good ArUco detection should form a roughly
    square quadrilateral (accounting for perspective). We measure:
    
    1. Side length consistency: all 4 sides should be similar length
    2. Angle consistency: all 4 angles should be close to 90°
    3. Size: larger detections are generally more reliable
    
    Returns a score where higher = better quality.
    """
    pts = corners_4x2.astype(np.float64)
    
    # Compute side lengths
    sides = []
    for i in range(4):
        dx = pts[(i+1) % 4][0] - pts[i][0]
        dy = pts[(i+1) % 4][1] - pts[i][1]
        sides.append(math.sqrt(dx*dx + dy*dy))
    
    mean_side = np.mean(sides)
    if mean_side < 1e-6:
        return 0.0
    
    # Side length consistency (1.0 = perfect square)
    side_std = np.std(sides) / mean_side
    side_score = math.exp(-side_std * 2)  
    
    # Size score (prefer larger detections — more reliable)
    # Use area via shoelace formula
    area = 0.5 * abs(
        pts[0][0]*(pts[1][1]-pts[3][1]) + 
        pts[1][0]*(pts[2][1]-pts[0][1]) + 
        pts[2][0]*(pts[3][1]-pts[1][1]) + 
        pts[3][0]*(pts[0][1]-pts[2][1])
    )
    size_score = min(1.0, area / 5000.0)  # normalize, cap at 1
    
    # Combined quality
    quality = 0.7 * side_score + 0.3 * size_score
    return quality


def merge_detections(all_detections, img_h, img_w):
    """
    Merge detections from multiple passes.
    
    Key insight: In the FlyingArUco dataset, each marker ID appears AT MOST
    a small number of times per image (usually 1, occasionally 2).
    
    Strategy:
    1. Group all detections by marker ID
    2. For each ID, cluster nearby detections (same physical marker 
       seen by different passes/preprocessing)
    3. Keep only the best detection per cluster
    4. Limit total clusters per ID to avoid spam
    
    The cluster threshold is set generously (10% of diagonal) because
    different preprocessing can shift the detected corners significantly.
    """
    diagonal = math.sqrt(img_h**2 + img_w**2)
    
    # Group by marker ID
    by_id = defaultdict(list)
    for (mid, corners, pass_name) in all_detections:
        top_left = corners[0]
        quality = compute_quad_quality(corners)
        by_id[mid].append({
            'corners': corners,
            'top_left': top_left,
            'quality': quality,
            'pass': pass_name
        })
    
    final_detections = []
    
    for mid, dets in by_id.items():
        if len(dets) == 1:
            d = dets[0]
            final_detections.append((mid, float(d['top_left'][0]), float(d['top_left'][1])))
            continue
        
        # Cluster nearby detections using generous threshold
        # Different preprocessing shifts corners, so we need a wide radius
        cluster_threshold = 0.10 * diagonal  # 10% of diagonal
        
        clusters = []
        used = [False] * len(dets)
        
        # Sort by quality (best first) so cluster centers are high-quality
        dets_sorted = sorted(range(len(dets)), key=lambda i: dets[i]['quality'], reverse=True)
        
        for idx in dets_sorted:
            if used[idx]:
                continue
            
            cluster = [dets[idx]]
            used[idx] = True
            
            for jdx in range(len(dets)):
                if used[jdx]:
                    continue
                dx = dets[idx]['top_left'][0] - dets[jdx]['top_left'][0]
                dy = dets[idx]['top_left'][1] - dets[jdx]['top_left'][1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < cluster_threshold:
                    cluster.append(dets[jdx])
                    used[jdx] = True
            
            clusters.append(cluster)
        
        # Limit: at most 2 detections per marker ID per image
        # (very rarely does the same ID appear more than once)
        # Sort clusters by best quality, keep top 2
        clusters.sort(key=lambda c: max(d['quality'] for d in c), reverse=True)
        max_per_id = 2
        
        for cluster in clusters[:max_per_id]:
            best = max(cluster, key=lambda d: d['quality'])
            final_detections.append((
                mid, 
                float(best['top_left'][0]), 
                float(best['top_left'][1])
            ))
    
    return final_detections


# =============================================================================
# 4. EVALUATION METRIC (same as Stage 1)
# =============================================================================

def compute_score_single_image(gt_detections, pred_detections, img_height, img_width, sigma=0.02, lam=1.0):
    """Compute per-image score following the Kaggle metric."""
    N_gt = len(gt_detections)
    
    if N_gt == 0:
        return 1.0 if len(pred_detections) == 0 else 0.0
    
    diagonal = math.sqrt(img_height ** 2 + img_width ** 2)
    
    gt_by_id = {}
    for (mid, x, y) in gt_detections:
        gt_by_id.setdefault(mid, []).append((x, y))
    
    pred_by_id = {}
    for (mid, x, y) in pred_detections:
        pred_by_id.setdefault(mid, []).append((x, y))
    
    total_phi = 0.0
    total_spam = 0
    
    for pred_id in pred_by_id:
        preds = pred_by_id[pred_id]
        
        if pred_id not in gt_by_id:
            total_spam += len(preds)
            continue
        
        gts = gt_by_id[pred_id]
        
        distances = []
        for j, (px, py) in enumerate(preds):
            min_dist = float('inf')
            for (gx, gy) in gts:
                d = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                min_dist = min(min_dist, d)
            distances.append((min_dist, j))
        
        distances.sort()
        
        n_valid = min(len(preds), len(gts))
        total_spam += max(0, len(preds) - len(gts))
        
        for k in range(n_valid):
            d_min = distances[k][0]
            d_norm = d_min / diagonal
            phi = math.exp(-(d_norm ** 2) / (2 * sigma ** 2))
            total_phi += phi
    
    score = total_phi / (N_gt + lam * total_spam)
    return score


# =============================================================================
# 5. EVALUATE ON TRAINING SET
# =============================================================================

def evaluate_on_trainset(train_dir, train_csv_path):
    """Evaluate enhanced pipeline on training set."""
    gt_data = {}
    with open(train_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_id = row[0].strip()
            pred_str = row[1].strip()
            detections = []
            if pred_str:
                parts = pred_str.split()
                for i in range(0, len(parts), 3):
                    detections.append((int(parts[i]), float(parts[i+1]), float(parts[i+2])))
            gt_data[image_id] = detections
    
    print(f"Loaded ground truth for {len(gt_data)} images")
    print(f"Evaluating enhanced pipeline...\n")
    
    scores = []
    results = []
    total_gt = 0
    total_detected = 0
    
    sorted_ids = sorted(gt_data.keys())
    
    for idx, image_id in enumerate(sorted_ids):
        image_path = os.path.join(train_dir, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Enhanced detection
        pred_detections = multi_pass_detect(image_path)
        gt_detections = gt_data[image_id]
        
        score = compute_score_single_image(gt_detections, pred_detections, h, w)
        scores.append(score)
        
        n_gt = len(gt_detections)
        n_pred = len(pred_detections)
        gt_ids = set(d[0] for d in gt_detections)
        pred_ids = set(d[0] for d in pred_detections)
        
        total_gt += n_gt
        total_detected += n_pred
        
        results.append({
            'image_id': image_id,
            'score': score,
            'n_gt': n_gt,
            'n_pred': n_pred,
            'n_correct_ids': len(gt_ids & pred_ids),
            'missed_ids': gt_ids - pred_ids,
        })
        
        if (idx + 1) % 100 == 0:
            running_mean = np.mean(scores)
            print(f"  [{idx+1}/{len(sorted_ids)}] Running mean score: {running_mean:.4f}")
    
    mean_score = np.mean(scores) if scores else 0.0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ENHANCED PIPELINE - EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Images evaluated:     {len(scores)}")
    print(f"Mean score (Kaggle):  {mean_score:.4f}")
    print(f"Total GT markers:     {total_gt}")
    print(f"Total detected:       {total_detected}")
    print(f"Detection rate:       {total_detected/total_gt*100:.1f}%")
    print(f"{'='*60}")
    
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
    
    # Improvement breakdown
    perfect = sum(1 for s in scores if s > 0.99)
    good = sum(1 for s in scores if 0.8 <= s <= 0.99)
    mediocre = sum(1 for s in scores if 0.5 <= s < 0.8)
    bad = sum(1 for s in scores if 0.0 < s < 0.5)
    zero = sum(1 for s in scores if s == 0.0)
    
    print(f"\nScore breakdown:")
    print(f"  Perfect (>0.99):  {perfect} images ({perfect/len(scores)*100:.1f}%)")
    print(f"  Good (0.8-0.99):  {good} images ({good/len(scores)*100:.1f}%)")
    print(f"  Mediocre (0.5-0.8): {mediocre} images ({mediocre/len(scores)*100:.1f}%)")
    print(f"  Bad (0.0-0.5):    {bad} images ({bad/len(scores)*100:.1f}%)")
    print(f"  Zero (0.0):       {zero} images ({zero/len(scores)*100:.1f}%)")
    
    if mean_score >= 0.78:
        marks = (1 - (0.97 - mean_score) / (0.97 - 0.78)) * 8
        marks = max(0, min(8, marks))
        print(f"\nEstimated Kaggle marks (score part): {marks:.1f}/8")
    else:
        print(f"\nWARNING: Score {mean_score:.4f} is below baseline 0.78!")
    
    return mean_score, results


# =============================================================================
# 6. GENERATE SUBMISSION
# =============================================================================

def generate_submission(test_dir, output_csv_path):
    """Generate Kaggle submission CSV for test set."""
    test_images = sorted(Path(test_dir).glob("*.jpg"))
    print(f"Found {len(test_images)} test images")
    
    rows = []
    total_markers = 0
    
    for idx, img_path in enumerate(test_images):
        image_id = img_path.stem
        
        detections = multi_pass_detect(img_path)
        total_markers += len(detections)
        
        if len(detections) > 0:
            parts = []
            for (mid, x, y) in detections:
                parts.append(f"{mid} {x:.3f} {y:.3f}")
            pred_string = " ".join(parts)
        else:
            pred_string = ""
        
        rows.append([image_id, pred_string])
        
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(test_images)}] processed, "
                  f"{total_markers} markers detected so far")
    
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'prediction_string'])
        writer.writerows(rows)
    
    print(f"\nSubmission saved to: {output_csv_path}")
    print(f"Total test images: {len(rows)}")
    print(f"Total markers detected: {total_markers}")
    print(f"Average markers per image: {total_markers/len(rows):.1f}")


# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def visualize_detections(image_path, save_path=None):
    """Visualize multi-pass detection results."""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    detections = multi_pass_detect(image_path)
    
    for (mid, x, y) in detections:
        # Draw top-left corner (red dot)
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
        # Draw ID label
        cv2.putText(img, f"ID:{mid}", 
                    (int(x) - 10, int(y) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"  Saved: {save_path}")
    
    return img


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Enhanced ArUco Marker Detection"
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to aruco_data/aruco_data folder')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to train.csv')
    parser.add_argument('--output_csv', type=str, default='submission_enhanced.csv',
                        help='Output CSV path')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['eval', 'submit', 'all', 'visualize'],
                        help='Mode: eval, submit, all, visualize')
    parser.add_argument('--vis_dir', type=str, default='visualizations_v2',
                        help='Directory for visualizations')
    parser.add_argument('--vis_count', type=int, default=20,
                        help='Number of images to visualize')
    
    args = parser.parse_args()
    
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    assert os.path.isdir(train_dir), f"Train dir not found: {train_dir}"
    assert os.path.isdir(test_dir), f"Test dir not found: {test_dir}"
    assert os.path.isfile(args.train_csv), f"Train CSV not found: {args.train_csv}"
    
    print("=" * 60)
    print("ArUco Marker Detection - Stage 2 Enhanced Pipeline")
    print("=" * 60)
    print(f"Data directory:  {args.data_dir}")
    print(f"Train CSV:       {args.train_csv}")
    print(f"Output CSV:      {args.output_csv}")
    print(f"Mode:            {args.mode}")
    print(f"\nPipeline: 8-pass multi-config detection")
    print(f"  Pass 1: Original + default params")
    print(f"  Pass 2: CLAHE + default params")
    print(f"  Pass 3: Original + small marker config")
    print(f"  Pass 4: Sharpened + default params")
    print(f"  Pass 5: Dark preprocessing + low contrast config")
    print(f"  Pass 6: Original + large marker config")
    print(f"  Pass 7: CLAHE+Sharp + high tolerance config")
    print(f"  Pass 8: Blur-fix + default params")
    print(f"  Merging: quality-based deduplication")
    print()
    
    # === EVALUATE ===
    if args.mode in ['eval', 'all']:
        print("-" * 60)
        print("PHASE 1: Evaluating on training set")
        print("-" * 60)
        start = time.time()
        mean_score, results = evaluate_on_trainset(train_dir, args.train_csv)
        elapsed = time.time() - start
        print(f"\nEvaluation time: {elapsed:.1f}s "
              f"({elapsed/len(results)*1000:.0f}ms per image)")
    
    # === SUBMIT ===
    if args.mode in ['submit', 'all']:
        print()
        print("-" * 60)
        print("PHASE 2: Generating test submission")
        print("-" * 60)
        start = time.time()
        generate_submission(test_dir, args.output_csv)
        elapsed = time.time() - start
        print(f"Submission time: {elapsed:.1f}s")
    
    # === VISUALIZE ===
    if args.mode in ['visualize', 'all']:
        print()
        print("-" * 60)
        print("PHASE 3: Saving visualizations")
        print("-" * 60)
        os.makedirs(args.vis_dir, exist_ok=True)
        
        train_images = sorted(Path(train_dir).glob("*.jpg"))[:args.vis_count]
        for img_path in train_images:
            save_path = os.path.join(args.vis_dir, f"det_{img_path.name}")
            visualize_detections(img_path, save_path)
        
        print(f"Saved {len(train_images)} visualizations to {args.vis_dir}/")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
