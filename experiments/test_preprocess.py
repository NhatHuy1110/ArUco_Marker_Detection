"""
test_preprocess.py - Test each preprocessing method independently.

This is the MOST IMPORTANT experiment script. It answers:
  "Does this preprocessing help or hurt detection?"

For each method, it runs the detector on ALL training images
with that preprocessing applied, computes the Kaggle score,
and compares to baseline (no preprocessing).

Usage:
  python test_preprocess.py

Output: A table showing score, #predictions, #spam for each method.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aruco_marker_detection.config import TRAIN_CSV, TRAIN_DIR
from aruco_marker_detection.detector import create_detector, detect_on_gray
from aruco_marker_detection.metrics import compute_score_image
from aruco_marker_detection.preprocessing import ALL_PREPROCESS
from aruco_marker_detection.utils import load_ground_truth


def test_single_preprocess(detector, gt_data, train_dir, preprocess_fn, name):
    """
    Evaluate one preprocessing method on the full training set.
    
    Returns:
        dict with score, n_pred, n_spam, n_missed, etc.
    """
    scores = []
    total_pred = 0
    total_spam = 0
    total_gt = 0
    total_missed_ids = 0
    
    sorted_ids = sorted(gt_data.keys())
    
    for img_id in sorted_ids:
        img_path = os.path.join(train_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Apply preprocessing
        processed = preprocess_fn(gray)
        
        # Detect
        raw = detect_on_gray(detector, processed)
        
        # Convert to (mid, x, y) format
        pred_dets = []
        for (mid, corners) in raw:
            pred_dets.append((mid, float(corners[0][0]), float(corners[0][1])))
        
        gt_dets = gt_data[img_id]
        
        # Score
        score = compute_score_image(gt_dets, pred_dets, h, w)
        scores.append(score)
        
        # Stats
        total_pred += len(pred_dets)
        total_gt += len(gt_dets)
        
        gt_ids = set(d[0] for d in gt_dets)
        pred_ids = set(d[0] for d in pred_dets)
        total_spam += sum(1 for d in pred_dets if d[0] not in gt_ids)
        total_missed_ids += len(gt_ids - pred_ids)
    
    mean_score = np.mean(scores)
    return {
        'name': name,
        'score': mean_score,
        'n_pred': total_pred,
        'n_gt': total_gt,
        'n_spam': total_spam,
        'n_missed_ids': total_missed_ids,
        'detection_rate': total_pred / total_gt * 100,
    }


def main():
    print("="*70)
    print("PREPROCESSING EXPERIMENT: Testing each method independently")
    print("="*70)
    print()
    
    # Load ground truth
    gt_data = load_ground_truth(TRAIN_CSV)
    print(f"Loaded GT: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} markers")
    
    # Create detector (same for all tests)
    detector = create_detector("subpix")
    print(f"Detector: DICT_ARUCO_MIP_36h12, CORNER_REFINE_SUBPIX")
    print()
    
    # Test each preprocessing
    results = []
    
    for name, fn in ALL_PREPROCESS.items():
        print(f"Testing: {name:20s} ...", end=" ", flush=True)
        t = time.time()
        result = test_single_preprocess(detector, gt_data, TRAIN_DIR, fn, name)
        elapsed = time.time() - t
        
        print(f"score={result['score']:.4f}  "
              f"pred={result['n_pred']:5d}  "
              f"spam={result['n_spam']:4d}  "
              f"missed={result['n_missed_ids']:4d}  "
              f"({elapsed:.1f}s)")
        
        results.append(result)
    
    # Sort by score (best first)
    results.sort(key=lambda r: r['score'], reverse=True)
    
    # Print summary table
    baseline = next(r for r in results if r['name'] == 'none')
    
    print()
    print("="*70)
    print(f"{'Method':20s} {'Score':>8s} {'Delta':>8s} {'Pred':>6s} "
          f"{'Spam':>6s} {'Missed':>7s} {'Rate%':>6s}")
    print("-"*70)
    
    for r in results:
        delta = r['score'] - baseline['score']
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        marker = " <-- BASELINE" if r['name'] == 'none' else ""
        marker = " *** BEST" if r == results[0] and r['name'] != 'none' else marker
        
        print(f"{r['name']:20s} {r['score']:8.4f} {delta_str:>8s} "
              f"{r['n_pred']:6d} {r['n_spam']:6d} {r['n_missed_ids']:7d} "
              f"{r['detection_rate']:6.1f}{marker}")
    
    print("="*70)
    print()
    print("INTERPRETATION:")
    print(f"  Baseline score: {baseline['score']:.4f}")
    print(f"  Best method:    {results[0]['name']} ({results[0]['score']:.4f})")
    
    # Find methods that improve over baseline
    improvements = [r for r in results if r['score'] > baseline['score'] and r['name'] != 'none']
    if improvements:
        print(f"\n  Methods that IMPROVE over baseline:")
        for r in improvements:
            d = r['score'] - baseline['score']
            print(f"    {r['name']:20s}: +{d:.4f} (spam={r['n_spam']}, missed={r['n_missed_ids']})")
    else:
        print(f"\n  NO method improves over baseline!")
        print(f"  --> Focus on post-filtering and corner refinement instead.")
    
    hurting = [r for r in results if r['score'] < baseline['score']]
    if hurting:
        print(f"\n  Methods that HURT (avoid these):")
        for r in hurting:
            d = r['score'] - baseline['score']
            print(f"    {r['name']:20s}: {d:.4f}")


if __name__ == "__main__":
    main()
