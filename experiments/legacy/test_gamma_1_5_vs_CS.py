"""
test_extended.py - Extended experiments based on initial results.

Finding from test_preprocess.py: gamma_1.5 (darken) is the ONLY method 
that improves over baseline (+0.0557).

This script tests:
  1. Fine-tune gamma value (1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0)
  2. A few promising combos with gamma
  3. Impact of post-filter on best preprocessing
  4. Complement merge: gamma as primary + baseline as supplement

Usage:
  python test_extended.py
"""

import os
import sys
import time
import math
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aruco_marker_detection.config import TRAIN_CSV, TRAIN_DIR
from aruco_marker_detection.detector import create_detector, detect_on_gray
from aruco_marker_detection.metrics import compute_score_image
from aruco_marker_detection.postprocessing import filter_detections
from aruco_marker_detection.utils import load_ground_truth


# =============================================================================
# GAMMA FUNCTIONS WITH DIFFERENT VALUES
# =============================================================================

def make_gamma_fn(gamma_val):
    """Create a gamma correction function for a specific gamma value."""
    table = np.array([((i / 255.0) ** (1.0 / gamma_val)) * 255
                       for i in range(256)]).astype("uint8")
    def fn(gray):
        return cv2.LUT(gray, table)
    return fn


def make_combo_fn(*fns):
    """Chain multiple preprocessing functions."""
    def fn(gray):
        result = gray
        for f in fns:
            result = f(result)
        return result
    return fn


def clahe_fn(clip=2.0):
    def fn(gray):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(gray)
    return fn


def sharpen_fn(strength=0.5):
    def fn(gray):
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return fn


def contrast_stretch_fn():
    def fn(gray):
        lo = np.percentile(gray, 2)
        hi = np.percentile(gray, 98)
        if hi - lo < 10:
            return gray
        return np.clip((gray.astype(float) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return fn


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_method(detector, gt_data, preprocess_fn, name, use_filter=False):
    """Run detection on all train images with given preprocessing."""
    scores = []
    total_pred = total_spam = total_gt = total_missed = 0
    
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
        
        if use_filter:
            raw = filter_detections(raw, h, w)
        
        pred_dets = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in raw]
        gt_dets = gt_data[img_id]
        
        score = compute_score_image(gt_dets, pred_dets, h, w)
        scores.append(score)
        
        total_pred += len(pred_dets)
        total_gt += len(gt_dets)
        gt_ids = set(d[0] for d in gt_dets)
        pred_ids = set(d[0] for d in pred_dets)
        total_spam += sum(1 for d in pred_dets if d[0] not in gt_ids)
        total_missed += len(gt_ids - pred_ids)
    
    return {
        'name': name,
        'score': np.mean(scores),
        'n_pred': total_pred,
        'n_spam': total_spam,
        'n_missed': total_missed,
        'rate': total_pred / total_gt * 100,
    }


def evaluate_complement_merge(detector, gt_data, primary_fn, secondary_fn, name):
    """
    Complement merge: run primary, then add NEW IDs from secondary.
    Only useful if secondary catches markers that primary misses,
    without adding too much spam.
    """
    scores = []
    total_pred = total_spam = total_gt = total_new = 0
    
    for img_id in sorted(gt_data.keys()):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Primary pass
        p1_raw = detect_on_gray(detector, primary_fn(gray))
        p1_dets = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in p1_raw]
        p1_ids = set(d[0] for d in p1_dets)
        
        # Secondary pass
        p2_raw = detect_on_gray(detector, secondary_fn(gray))
        
        # Add NEW IDs only
        final = list(p1_dets)
        final_ids = set(p1_ids)
        new_count = 0
        for mid, corners in p2_raw:
            if mid not in final_ids:
                final.append((mid, float(corners[0][0]), float(corners[0][1])))
                final_ids.add(mid)
                new_count += 1
        
        total_new += new_count
        gt_dets = gt_data[img_id]
        score = compute_score_image(gt_dets, final, h, w)
        scores.append(score)
        
        total_pred += len(final)
        total_gt += len(gt_dets)
        gt_ids = set(d[0] for d in gt_dets)
        total_spam += sum(1 for d in final if d[0] not in gt_ids)
    
    return {
        'name': name,
        'score': np.mean(scores),
        'n_pred': total_pred,
        'n_spam': total_spam,
        'n_missed': 0,  # not calculated for brevity
        'rate': total_pred / total_gt * 100,
        'n_new': total_new,
    }


def print_table(results, baseline_score):
    """Print comparison table."""
    print(f"\n{'Method':35s} {'Score':>8s} {'Delta':>8s} {'Pred':>6s} "
          f"{'Spam':>6s} {'Missed':>7s} {'Rate%':>6s}")
    print("-" * 85)
    for r in results:
        delta = r['score'] - baseline_score
        ds = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        missed = r.get('n_missed', '-')
        missed_str = f"{missed:7d}" if isinstance(missed, int) else f"{'':>7s}"
        extra = ""
        if 'n_new' in r:
            extra = f"  (new={r['n_new']})"
        print(f"{r['name']:35s} {r['score']:8.4f} {ds:>8s} "
              f"{r['n_pred']:6d} {r['n_spam']:6d} {missed_str} "
              f"{r['rate']:6.1f}{extra}")


def main():
    print("=" * 70)
    print("EXTENDED EXPERIMENTS")
    print("=" * 70)
    
    gt_data = load_ground_truth(TRAIN_CSV)
    detector = create_detector("subpix")
    print(f"GT: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} markers\n")
    
    all_results = []
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: Fine-tune gamma value
    # -----------------------------------------------------------------
    print("EXPERIMENT 1: Fine-tuning gamma value")
    print("-" * 50)
    
    gamma_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    
    for g in gamma_values:
        name = f"gamma_{g}"
        fn = make_gamma_fn(g)
        print(f"  Testing {name:15s} ...", end=" ", flush=True)
        t = time.time()
        r = evaluate_method(detector, gt_data, fn, name)
        print(f"score={r['score']:.4f}  spam={r['n_spam']:4d}  "
              f"missed={r['n_missed']:4d}  ({time.time()-t:.1f}s)")
        all_results.append(r)
    
    # Also test baseline for reference
    print(f"  Testing {'none':15s} ...", end=" ", flush=True)
    t = time.time()
    baseline = evaluate_method(detector, gt_data, lambda g: g, "none (baseline)")
    print(f"score={baseline['score']:.4f}  ({time.time()-t:.1f}s)")
    
    print_table(sorted(all_results, key=lambda r: r['score'], reverse=True), 
                baseline['score'])
    
    # Find best gamma
    best_gamma_result = max(all_results, key=lambda r: r['score'])
    best_gamma_name = best_gamma_result['name']
    best_gamma_val = float(best_gamma_name.split('_')[1])
    print(f"\nBest gamma: {best_gamma_val} (score={best_gamma_result['score']:.4f})")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 2: Combos with best gamma
    # -----------------------------------------------------------------
    print(f"\n\nEXPERIMENT 2: Combos with {best_gamma_name}")
    print("-" * 50)
    
    best_gamma_fn = make_gamma_fn(best_gamma_val)
    
    combos = [
        (f"{best_gamma_name}+clahe_2.0",
         make_combo_fn(best_gamma_fn, clahe_fn(2.0))),
        (f"{best_gamma_name}+sharpen_mild",
         make_combo_fn(best_gamma_fn, sharpen_fn(0.5))),
        (f"{best_gamma_name}+contrast_stretch",
         make_combo_fn(best_gamma_fn, contrast_stretch_fn())),
        (f"contrast_stretch+{best_gamma_name}",
         make_combo_fn(contrast_stretch_fn(), best_gamma_fn)),
        (f"{best_gamma_name}+clahe_2.0+sharpen",
         make_combo_fn(best_gamma_fn, clahe_fn(2.0), sharpen_fn(0.5))),
    ]
    
    combo_results = []
    for name, fn in combos:
        print(f"  Testing {name:35s} ...", end=" ", flush=True)
        t = time.time()
        r = evaluate_method(detector, gt_data, fn, name)
        print(f"score={r['score']:.4f}  spam={r['n_spam']:4d}  "
              f"missed={r['n_missed']:4d}  ({time.time()-t:.1f}s)")
        combo_results.append(r)
    
    # Add best gamma alone for comparison
    combo_results.append(best_gamma_result)
    print_table(sorted(combo_results, key=lambda r: r['score'], reverse=True),
                baseline['score'])
    
    # -----------------------------------------------------------------
    # EXPERIMENT 3: Post-filter impact on best method
    # -----------------------------------------------------------------
    print(f"\n\nEXPERIMENT 3: Post-filter impact")
    print("-" * 50)
    
    # Best method WITHOUT filter (already have this)
    best_no_filter = best_gamma_result
    
    # Best method WITH filter
    print(f"  Testing {best_gamma_name}+filter ...", end=" ", flush=True)
    t = time.time()
    best_filtered = evaluate_method(detector, gt_data, best_gamma_fn,
                                     f"{best_gamma_name}+filter", use_filter=True)
    print(f"score={best_filtered['score']:.4f}  spam={best_filtered['n_spam']:4d}  "
          f"missed={best_filtered['n_missed']:4d}  ({time.time()-t:.1f}s)")
    
    # Baseline with filter
    print(f"  Testing none+filter         ...", end=" ", flush=True)
    t = time.time()
    base_filtered = evaluate_method(detector, gt_data, lambda g: g,
                                     "none+filter", use_filter=True)
    print(f"score={base_filtered['score']:.4f}  spam={base_filtered['n_spam']:4d}  "
          f"missed={base_filtered['n_missed']:4d}  ({time.time()-t:.1f}s)")
    
    filter_results = [best_no_filter, best_filtered, baseline, base_filtered]
    print_table(sorted(filter_results, key=lambda r: r['score'], reverse=True),
                baseline['score'])
    
    # -----------------------------------------------------------------
    # EXPERIMENT 4: Complement merge (gamma primary + baseline secondary)
    # -----------------------------------------------------------------
    print(f"\n\nEXPERIMENT 4: Complement merge")
    print("-" * 50)
    
    identity_fn = lambda g: g
    
    # gamma primary + baseline supplement
    print(f"  Testing gamma→baseline merge ...", end=" ", flush=True)
    t = time.time()
    merge1 = evaluate_complement_merge(detector, gt_data,
                                        best_gamma_fn, identity_fn,
                                        f"{best_gamma_name} → none (complement)")
    print(f"score={merge1['score']:.4f}  ({time.time()-t:.1f}s)")
    
    # baseline primary + gamma supplement  
    print(f"  Testing baseline→gamma merge ...", end=" ", flush=True)
    t = time.time()
    merge2 = evaluate_complement_merge(detector, gt_data,
                                        identity_fn, best_gamma_fn,
                                        f"none → {best_gamma_name} (complement)")
    print(f"score={merge2['score']:.4f}  ({time.time()-t:.1f}s)")
    
    merge_results = [best_gamma_result, merge1, merge2, baseline]
    print_table(sorted(merge_results, key=lambda r: r['score'], reverse=True),
                baseline['score'])
    
    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------
    everything = [baseline, base_filtered, best_gamma_result, best_filtered,
                  merge1, merge2] + combo_results
    everything_sorted = sorted(everything, key=lambda r: r['score'], reverse=True)
    
    print(f"\n\n{'='*70}")
    print(f"FINAL RANKING (all experiments)")
    print(f"{'='*70}")
    print_table(everything_sorted[:10], baseline['score'])
    
    best = everything_sorted[0]
    print(f"\n>>> RECOMMENDATION: Use '{best['name']}' (score={best['score']:.4f})")
    print(f">>> Set PREPROCESS_METHOD in run_pipeline.py accordingly")


if __name__ == "__main__":
    main()
