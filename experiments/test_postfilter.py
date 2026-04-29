"""
test_postfilter.py - Tune post-filter thresholds on gamma_2.0 preprocessing.

Tests different combinations of:
  - min_perim_ratio: how small a marker can be (fraction of diagonal)
  - min_side_ratio: how skewed a marker can be (min/max side length)

Usage:
  python test_postfilter.py
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
from aruco_marker_detection.postprocessing import filter_detections
from aruco_marker_detection.preprocessing import ALL_PREPROCESS
from aruco_marker_detection.utils import load_ground_truth


def evaluate_with_filter(detector, gt_data, preprocess_fn,
                         min_perim_ratio, min_side_ratio, require_convex):
    """Evaluate with specific filter settings."""
    scores = []
    total_pred = total_spam = total_gt = total_filtered = 0
    
    for img_id in sorted(gt_data.keys()):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        processed = preprocess_fn(gray)
        raw = detect_on_gray(detector, processed)
        n_raw = len(raw)
        
        filtered = filter_detections(raw, h, w,
                                      min_perim_ratio=min_perim_ratio,
                                      min_side_ratio=min_side_ratio,
                                      require_convex=require_convex)
        
        pred = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in filtered]
        gt = gt_data[img_id]
        
        score = compute_score_image(gt, pred, h, w)
        scores.append(score)
        total_pred += len(pred)
        total_gt += len(gt)
        total_filtered += (n_raw - len(filtered))
        gt_ids = set(d[0] for d in gt)
        total_spam += sum(1 for d in pred if d[0] not in gt_ids)
    
    return {
        'score': np.mean(scores),
        'n_pred': total_pred,
        'n_spam': total_spam,
        'n_filtered': total_filtered,
    }


def main():
    print("=" * 70)
    print("POST-FILTER TUNING on gamma_2.0")
    print("=" * 70)
    
    gt_data = load_ground_truth(TRAIN_CSV)
    detector = create_detector("subpix")
    preprocess_fn = ALL_PREPROCESS["gamma_2.0"]
    
    print(f"GT: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} markers")
    
    # Baseline: no filter
    print(f"\nBaseline (no filter)...", end=" ", flush=True)
    t = time.time()
    base = evaluate_with_filter(detector, gt_data, preprocess_fn,
                                 min_perim_ratio=0.0, min_side_ratio=0.0,
                                 require_convex=False)
    print(f"score={base['score']:.4f}  spam={base['n_spam']}  ({time.time()-t:.1f}s)")
    
    # Test different min_perim_ratio values
    print(f"\n--- Varying min_perim_ratio (side_ratio=0.15, convex=True) ---")
    perim_values = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]
    
    best_perim = 0.015
    best_perim_score = 0
    
    for p in perim_values:
        r = evaluate_with_filter(detector, gt_data, preprocess_fn,
                                  min_perim_ratio=p, min_side_ratio=0.15,
                                  require_convex=True)
        delta = r['score'] - base['score']
        ds = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        marker = " ***" if r['score'] > best_perim_score else ""
        if r['score'] > best_perim_score:
            best_perim_score = r['score']
            best_perim = p
        print(f"  perim={p:.3f}: score={r['score']:.4f} ({ds})  "
              f"pred={r['n_pred']:5d}  spam={r['n_spam']:4d}  "
              f"filtered={r['n_filtered']:3d}{marker}")
    
    print(f"  Best min_perim_ratio: {best_perim}")
    
    # Test different min_side_ratio values with best perim
    print(f"\n--- Varying min_side_ratio (perim={best_perim}, convex=True) ---")
    side_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    
    best_side = 0.15
    best_side_score = 0
    
    for s in side_values:
        r = evaluate_with_filter(detector, gt_data, preprocess_fn,
                                  min_perim_ratio=best_perim, min_side_ratio=s,
                                  require_convex=True)
        delta = r['score'] - base['score']
        ds = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        marker = " ***" if r['score'] > best_side_score else ""
        if r['score'] > best_side_score:
            best_side_score = r['score']
            best_side = s
        print(f"  side_ratio={s:.2f}: score={r['score']:.4f} ({ds})  "
              f"pred={r['n_pred']:5d}  spam={r['n_spam']:4d}  "
              f"filtered={r['n_filtered']:3d}{marker}")
    
    print(f"  Best min_side_ratio: {best_side}")
    
    # Test convex filter impact
    print(f"\n--- Convex filter impact (perim={best_perim}, side={best_side}) ---")
    
    r_with = evaluate_with_filter(detector, gt_data, preprocess_fn,
                                   min_perim_ratio=best_perim, min_side_ratio=best_side,
                                   require_convex=True)
    r_without = evaluate_with_filter(detector, gt_data, preprocess_fn,
                                      min_perim_ratio=best_perim, min_side_ratio=best_side,
                                      require_convex=False)
    
    print(f"  With convex:    score={r_with['score']:.4f}  spam={r_with['n_spam']:4d}  filtered={r_with['n_filtered']:3d}")
    print(f"  Without convex: score={r_without['score']:.4f}  spam={r_without['n_spam']:4d}  filtered={r_without['n_filtered']:3d}")
    
    best_convex = r_with['score'] >= r_without['score']
    
    # Final result
    print(f"\n{'='*70}")
    print(f"OPTIMAL FILTER SETTINGS:")
    print(f"  min_perim_ratio = {best_perim}")
    print(f"  min_side_ratio  = {best_side}")
    print(f"  require_convex  = {best_convex}")
    print(f"  Score: {max(r_with['score'], r_without['score']):.4f}")
    print(f"  (vs no filter: {base['score']:.4f})")
    print(f"{'='*70}")
    print(f"\nUpdate postprocess.py filter_detections() default params accordingly.")


if __name__ == "__main__":
    main()
