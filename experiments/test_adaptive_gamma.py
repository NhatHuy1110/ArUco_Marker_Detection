"""
test_adaptive_gamma.py - Fine-tune gamma + adaptive gamma per-image.

Experiments:
  1. Test gamma 2.0 to 4.0 to find peak
  2. Adaptive gamma: analyze each image's brightness, pick gamma accordingly
  3. Compare fixed vs adaptive

Usage:
  python test_adaptive_gamma.py
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aruco_marker_detection.config import TRAIN_CSV, TRAIN_DIR
from aruco_marker_detection.detector import create_detector, detect_on_gray
from aruco_marker_detection.metrics import compute_score_image
from aruco_marker_detection.postprocessing import filter_detections
from aruco_marker_detection.utils import load_ground_truth


def make_gamma_fn(gamma_val):
    """Create gamma correction LUT."""
    table = np.array([((i / 255.0) ** (1.0 / gamma_val)) * 255
                       for i in range(256)]).astype("uint8")
    def fn(gray):
        return cv2.LUT(gray, table)
    return fn


# Pre-build gamma LUTs for adaptive use (avoid rebuilding per image)
GAMMA_LUTS = {}
for g in [1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]:
    table = np.array([((i / 255.0) ** (1.0 / g)) * 255
                       for i in range(256)]).astype("uint8")
    GAMMA_LUTS[g] = table


def apply_gamma(gray, gamma_val):
    """Apply gamma using pre-built LUT."""
    return cv2.LUT(gray, GAMMA_LUTS[gamma_val])


def evaluate_fixed_gamma(detector, gt_data, gamma_val):
    """Evaluate with a fixed gamma value + post-filter."""
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
        
        processed = apply_gamma(gray, gamma_val)
        raw = detect_on_gray(detector, processed)
        raw = filter_detections(raw, h, w)
        
        pred = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in raw]
        gt = gt_data[img_id]
        
        score = compute_score_image(gt, pred, h, w)
        scores.append(score)
        total_pred += len(pred)
        total_gt += len(gt)
        gt_ids = set(d[0] for d in gt)
        total_spam += sum(1 for d in pred if d[0] not in gt_ids)
        total_missed += len(gt_ids - set(d[0] for d in pred))
    
    return np.mean(scores), total_pred, total_spam, total_missed


def adaptive_gamma_v1(gray):
    """
    Adaptive gamma based on mean brightness.
    
    Logic:
    - Dark images (mean < 80): use lower gamma to preserve contrast
    - Medium images (80-160): use moderate gamma
    - Bright images (mean > 160): use stronger mid-tone lifting
    
    Why: Fixed gamma 2.0 works great for bright images but may hurt 
    dark images by making them too dark. Adaptive gamma adjusts per-image.
    """
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 60:
        return apply_gamma(gray, 1.0)   # very dark: no change
    elif mean_brightness < 80:
        return apply_gamma(gray, 1.2)   # dark: mild lift
    elif mean_brightness < 100:
        return apply_gamma(gray, 1.5)   # medium-dark: moderate lift
    elif mean_brightness < 130:
        return apply_gamma(gray, 1.8)   # medium: moderate-strong lift
    elif mean_brightness < 160:
        return apply_gamma(gray, 2.0)   # medium-bright: strong lift
    else:
        return apply_gamma(gray, 2.5)   # bright: very strong lift


def adaptive_gamma_v2(gray):
    """
    Adaptive gamma v2: finer granularity based on brightness.
    Uses linear interpolation between gamma values.
    
    Dark (mean=0): gamma=1.0 (no change)
    Bright (mean=255): gamma=3.0 (strong mid-tone lift)
    """
    mean_b = np.mean(gray)
    # Linear interpolation: brightness 0 to gamma 1.0, brightness 200 to gamma 3.0
    gamma = 1.0 + (mean_b / 200.0) * 2.0
    gamma = max(1.0, min(3.5, gamma))
    
    # Snap to nearest pre-built LUT
    available = sorted(GAMMA_LUTS.keys())
    best_g = min(available, key=lambda g: abs(g - gamma))
    return apply_gamma(gray, best_g)


def adaptive_gamma_v3(gray):
    """
    Adaptive gamma v3: based on contrast (std dev) + brightness.
    
    Low contrast + bright: strong gamma (helps separate marker from background)
    Low contrast + dark: mild gamma (avoid washing out useful contrast)
    High contrast: moderate gamma (already good separation)
    """
    mean_b = np.mean(gray)
    std_b = np.std(gray)
    
    if std_b < 40:  # low contrast
        if mean_b > 140:
            return apply_gamma(gray, 2.5)  # bright + low contrast: strong
        elif mean_b > 100:
            return apply_gamma(gray, 2.0)
        else:
            return apply_gamma(gray, 1.4)  # dark + low contrast: mild
    elif std_b < 60:  # medium contrast
        if mean_b > 140:
            return apply_gamma(gray, 2.2)
        elif mean_b > 100:
            return apply_gamma(gray, 2.0)
        else:
            return apply_gamma(gray, 1.5)
    else:  # high contrast
        if mean_b > 140:
            return apply_gamma(gray, 2.0)
        elif mean_b > 100:
            return apply_gamma(gray, 1.8)
        else:
            return apply_gamma(gray, 1.4)


def evaluate_adaptive(detector, gt_data, adaptive_fn, name):
    """Evaluate adaptive preprocessing."""
    scores = []
    total_pred = total_spam = total_gt = total_missed = 0
    gamma_usage = defaultdict(int)
    
    for img_id in sorted(gt_data.keys()):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        processed = adaptive_fn(gray)
        raw = detect_on_gray(detector, processed)
        raw = filter_detections(raw, h, w)
        
        pred = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in raw]
        gt = gt_data[img_id]
        
        score = compute_score_image(gt, pred, h, w)
        scores.append(score)
        total_pred += len(pred)
        total_gt += len(gt)
        gt_ids = set(d[0] for d in gt)
        total_spam += sum(1 for d in pred if d[0] not in gt_ids)
        total_missed += len(gt_ids - set(d[0] for d in pred))
    
    return {
        'name': name,
        'score': np.mean(scores),
        'n_pred': total_pred,
        'n_spam': total_spam,
        'n_missed': total_missed,
    }


def main():
    print("=" * 70)
    print("GAMMA FINE-TUNING + ADAPTIVE GAMMA EXPERIMENTS")
    print("=" * 70)
    
    gt_data = load_ground_truth(TRAIN_CSV)
    detector = create_detector("subpix")
    n_gt = sum(len(v) for v in gt_data.values())
    print(f"GT: {len(gt_data)} images, {n_gt} markers\n")
    
    # =========================================================
    # EXPERIMENT 1: Fixed gamma from 1.0 to 4.0
    # =========================================================
    print("EXPERIMENT 1: Fixed gamma values (all with post-filter)")
    print("-" * 55)
    
    gamma_values = [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]
    fixed_results = []
    
    for g in gamma_values:
        print(f"  gamma={g:.1f} ...", end=" ", flush=True)
        t = time.time()
        score, pred, spam, missed = evaluate_fixed_gamma(detector, gt_data, g)
        print(f"score={score:.4f}  spam={spam:4d}  missed={missed:4d}  "
              f"pred={pred:5d}  ({time.time()-t:.1f}s)")
        fixed_results.append((g, score, pred, spam, missed))
    
    # Find best
    best_g, best_score = max(fixed_results, key=lambda x: x[1])[:2]
    print(f"\n  Best fixed gamma: {best_g} (score={best_score:.4f})")
    
    # =========================================================
    # EXPERIMENT 2: Adaptive gamma strategies
    # =========================================================
    print(f"\n\nEXPERIMENT 2: Adaptive gamma strategies (all with post-filter)")
    print("-" * 55)
    
    # First, analyze brightness distribution of training images
    print("  Analyzing image brightness distribution...", end=" ", flush=True)
    brightnesses = []
    contrasts = []
    for img_id in sorted(gt_data.keys()):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightnesses.append(np.mean(gray))
        contrasts.append(np.std(gray))
    
    ba = np.array(brightnesses)
    ca = np.array(contrasts)
    print(f"done")
    print(f"  Brightness: min={ba.min():.0f} Q1={np.percentile(ba,25):.0f} "
          f"med={np.median(ba):.0f} Q3={np.percentile(ba,75):.0f} max={ba.max():.0f}")
    print(f"  Contrast:   min={ca.min():.0f} Q1={np.percentile(ca,25):.0f} "
          f"med={np.median(ca):.0f} Q3={np.percentile(ca,75):.0f} max={ca.max():.0f}")
    print(f"  Dark images (mean<80):  {np.sum(ba<80)}")
    print(f"  Medium (80-160):        {np.sum((ba>=80)&(ba<160))}")
    print(f"  Bright (>160):          {np.sum(ba>=160)}")
    print()
    
    adaptive_methods = [
        ("adaptive_v1 (brightness bins)", adaptive_gamma_v1),
        ("adaptive_v2 (linear interp)", adaptive_gamma_v2),
        ("adaptive_v3 (contrast+bright)", adaptive_gamma_v3),
    ]
    
    adaptive_results = []
    for name, fn in adaptive_methods:
        print(f"  {name:40s} ...", end=" ", flush=True)
        t = time.time()
        r = evaluate_adaptive(detector, gt_data, fn, name)
        print(f"score={r['score']:.4f}  spam={r['n_spam']:4d}  "
              f"missed={r['n_missed']:4d}  ({time.time()-t:.1f}s)")
        adaptive_results.append(r)
    
    # =========================================================
    # FINAL COMPARISON
    # =========================================================
    print(f"\n\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':45s} {'Score':>8s} {'Spam':>6s} {'Missed':>7s}")
    print("-" * 70)
    
    # Add fixed gamma results
    all_results = []
    for g, score, pred, spam, missed in fixed_results:
        all_results.append((f"fixed gamma={g:.1f}", score, spam, missed))
    for r in adaptive_results:
        all_results.append((r['name'], r['score'], r['n_spam'], r['n_missed']))
    
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    for name, score, spam, missed in all_results:
        marker = " <<<" if score == all_results[0][1] else ""
        print(f"  {name:43s} {score:8.4f} {spam:6d} {missed:7d}{marker}")
    
    best_name, best_score = all_results[0][0], all_results[0][1]
    print(f"\n>>> BEST: {best_name} (score={best_score:.4f})")
    
    # Marks estimate
    if best_score >= 0.78:
        marks = max(0, min(8, (1-(0.97-best_score)/(0.97-0.78))*8))
        print(f">>> Estimated marks: {marks:.1f}/8")


if __name__ == "__main__":
    main()
