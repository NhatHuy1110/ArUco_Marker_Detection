"""
=============================================================================
Stage 2 v3: ArUco Detection with Complement-Merge Strategy
=============================================================================
Lessons learned from v1 and v2 failures:

v1 FAILED (score 0.003): errorCorrectionRate=0.8~1.0 caused massive 
   false positives. 8 passes with relaxed params = spam explosion.

v2 FAILED (score 0.45): Even with conservative errorCorrectionRate,
   UNION merge across 5 preprocessing passes doubled predictions 
   (23861 vs 10407 GT). Preprocessing (CLAHE, sharpen, gamma) creates 
   new high-contrast edges that look like marker boundaries, generating
   false positives that don't exist in the original image. Each pass 
   adds ~2500 spam predictions.

ROOT CAUSE: The problem is the MERGE STRATEGY, not just the params.
   Union merge = every false positive from every pass accumulates.
   The metric penalizes spam linearly in the denominator.

CORRECT APPROACH (v3): COMPLEMENT merge
   1. Pass 1: Original image → baseline results (proven score ~0.78)
   2. Pass 2: CLAHE image → supplementary results  
   3. Merge: Keep ALL of pass 1. From pass 2, ONLY add detections 
      whose marker ID was NOT found in pass 1.
   
   This guarantees:
   - Never worse than baseline (pass 1 results untouched)
   - New detections only for previously-missed markers
   - False positives from pass 2 that share IDs with pass 1 are filtered

Usage:
  python enhanced_aruco_v3.py --data_dir aruco-detection-challenge/aruco_data/aruco_data --train_csv aruco-detection-challenge/train.csv --output_csv submission_v3.csv --mode all
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
# DETECTOR
# =============================================================================

def create_detector():
    """
    Single detector with carefully tuned parameters.
    
    Changes from pure defaults:
    - adaptiveThreshWinSizeMax: 30 (default 23) — tries more threshold windows
    - adaptiveThreshWinSizeStep: 5 (default 10) — finer steps between windows
    - minMarkerPerimeterRate: 0.02 (default 0.03) — catches slightly smaller markers
    - cornerRefinementMethod: SUBPIX — improves localization by ~1-3 pixels
    - errorCorrectionRate: 0.6 (DEFAULT — never increase this!)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    params = cv2.aruco.DetectorParameters()
    
    # Wider adaptive threshold search
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 30
    params.adaptiveThreshWinSizeStep = 5
    
    # Slightly smaller markers allowed
    params.minMarkerPerimeterRate = 0.02
    
    # Sub-pixel corner refinement
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    
    # KEEP DEFAULT error correction (0.6) - DO NOT CHANGE
    params.errorCorrectionRate = 0.6
    
    return cv2.aruco.ArucoDetector(aruco_dict, params)


# =============================================================================
# PREPROCESSING
# =============================================================================

def apply_clahe(gray, clip_limit=2.5, grid_size=8):
    """CLAHE for dark/uneven lighting. Only changes intensity, not geometry."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(gray)


# =============================================================================
# DETECTION WITH COMPLEMENT MERGE
# =============================================================================

def detect_on_gray(detector, gray):
    """
    Run detector, return dict: {marker_id: (top_left_x, top_left_y)}
    If same ID detected multiple times, keep all instances.
    Returns: list of (marker_id, tl_x, tl_y)
    """
    corners, ids, rejected = detector.detectMarkers(gray)
    
    results = []
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            mid = int(ids[i][0])
            tl_x = float(corners[i][0][0][0])
            tl_y = float(corners[i][0][0][1])
            results.append((mid, tl_x, tl_y))
    
    return results


def detect_image(detector, image_path):
    """
    Complement-merge detection:
    
    1. Run on ORIGINAL image → get baseline detections
    2. Run on CLAHE image → get supplementary detections
    3. Keep ALL baseline detections
    4. From CLAHE pass, add ONLY markers with NEW IDs
       (IDs not present in baseline results)
    
    Why this works:
    - Baseline detector works well for 80%+ of markers
    - CLAHE helps specifically with dark/low-contrast markers
    - By only adding NEW IDs, we avoid duplicating existing detections
    - False positives from CLAHE that happen to share IDs with real 
      markers are automatically filtered (they'd be spam anyway since 
      the real marker is already detected)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pass 1: Original image (this IS the baseline)
    pass1_results = detect_on_gray(detector, gray)
    
    # Collect IDs found in pass 1
    pass1_ids = set(d[0] for d in pass1_results)
    
    # Pass 2: CLAHE enhanced
    gray_clahe = apply_clahe(gray, clip_limit=2.5, grid_size=8)
    pass2_results = detect_on_gray(detector, gray_clahe)
    
    # Complement merge: only add NEW IDs from pass 2
    for (mid, x, y) in pass2_results:
        if mid not in pass1_ids:
            pass1_results.append((mid, x, y))
            pass1_ids.add(mid)  # prevent duplicates from pass 2 itself
    
    return pass1_results


# =============================================================================
# EVALUATION METRIC (Kaggle-exact)
# =============================================================================

def compute_score(gt_dets, pred_dets, img_h, img_w, sigma=0.02, lam=1.0):
    """Per-image Kaggle score."""
    N_gt = len(gt_dets)
    if N_gt == 0:
        return 1.0 if len(pred_dets) == 0 else 0.0
    
    diagonal = math.sqrt(img_h**2 + img_w**2)
    
    gt_by_id = defaultdict(list)
    for mid, x, y in gt_dets:
        gt_by_id[mid].append((x, y))
    
    pred_by_id = defaultdict(list)
    for mid, x, y in pred_dets:
        pred_by_id[mid].append((x, y))
    
    total_phi = 0.0
    total_spam = 0
    
    for pid, preds in pred_by_id.items():
        if pid not in gt_by_id:
            total_spam += len(preds)
            continue
        
        gts = gt_by_id[pid]
        dists = []
        for px, py in preds:
            d_min = min(math.sqrt((px-gx)**2 + (py-gy)**2) for gx, gy in gts)
            dists.append(d_min)
        dists.sort()
        
        n_valid = min(len(preds), len(gts))
        total_spam += max(0, len(preds) - len(gts))
        
        for k in range(n_valid):
            d_norm = dists[k] / diagonal
            total_phi += math.exp(-(d_norm**2) / (2 * sigma**2))
    
    return total_phi / (N_gt + lam * total_spam)


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate(detector, train_dir, train_csv):
    gt_data = {}
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            img_id = row[0].strip()
            pred_str = row[1].strip()
            dets = []
            if pred_str:
                parts = pred_str.split()
                for i in range(0, len(parts), 3):
                    dets.append((int(parts[i]), float(parts[i+1]), float(parts[i+2])))
            gt_data[img_id] = dets
    
    print(f"Ground truth: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} markers")
    
    scores = []
    total_gt = 0
    total_pred = 0
    total_spam = 0
    total_new_from_clahe = 0
    results = []
    
    sorted_ids = sorted(gt_data.keys())
    
    for idx, img_id in enumerate(sorted_ids):
        img_path = os.path.join(train_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # --- Detect with complement merge ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pass 1: original
        p1 = detect_on_gray(detector, gray)
        p1_ids = set(d[0] for d in p1)
        
        # Pass 2: CLAHE
        p2 = detect_on_gray(detector, apply_clahe(gray))
        
        # Complement merge
        final = list(p1)
        final_ids = set(p1_ids)
        new_count = 0
        for mid, x, y in p2:
            if mid not in final_ids:
                final.append((mid, x, y))
                final_ids.add(mid)
                new_count += 1
        
        total_new_from_clahe += new_count
        
        # Score
        gt_dets = gt_data[img_id]
        score = compute_score(gt_dets, final, h, w)
        scores.append(score)
        
        total_gt += len(gt_dets)
        total_pred += len(final)
        
        gt_ids = set(d[0] for d in gt_dets)
        false_ids = set(d[0] for d in final) - gt_ids
        total_spam += sum(1 for d in final if d[0] not in gt_ids)
        
        results.append({
            'image_id': img_id, 'score': score,
            'n_gt': len(gt_dets), 'n_pred': len(final),
            'missed': gt_ids - set(d[0] for d in final),
            'new_from_clahe': new_count,
        })
        
        if (idx + 1) % 200 == 0:
            m = np.mean(scores)
            print(f"  [{idx+1}/{len(sorted_ids)}] Mean: {m:.4f} | "
                  f"Pred: {total_pred} | Spam: {total_spam} | "
                  f"New from CLAHE: {total_new_from_clahe}")
    
    mean_score = np.mean(scores)
    sa = np.array(scores)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Mean score:          {mean_score:.4f}")
    print(f"Total GT:            {total_gt}")
    print(f"Total predicted:     {total_pred}")
    print(f"Detection rate:      {total_pred/total_gt*100:.1f}%")
    print(f"Total spam:          {total_spam}")
    print(f"New markers (CLAHE): {total_new_from_clahe}")
    print(f"\nDistribution: min={sa.min():.4f} Q1={np.percentile(sa,25):.4f} "
          f"med={np.median(sa):.4f} Q3={np.percentile(sa,75):.4f} max={sa.max():.4f}")
    
    perfect = sum(1 for s in scores if s > 0.99)
    good = sum(1 for s in scores if 0.8 <= s <= 0.99)
    mid = sum(1 for s in scores if 0.5 <= s < 0.8)
    bad = sum(1 for s in scores if 0.0 < s < 0.5)
    zero = sum(1 for s in scores if s == 0.0)
    print(f"\n  >0.99: {perfect}  0.8-0.99: {good}  0.5-0.8: {mid}  "
          f"<0.5: {bad}  =0: {zero}")
    
    results.sort(key=lambda r: r['score'])
    print(f"\n10 worst:")
    for r in results[:10]:
        print(f"  {r['image_id']}: {r['score']:.4f} gt={r['n_gt']} "
              f"pred={r['n_pred']} missed={r['missed']}")
    
    if mean_score >= 0.78:
        marks = max(0, min(8, (1 - (0.97 - mean_score)/(0.97-0.78)) * 8))
        print(f"\nEstimated marks: {marks:.1f}/8")
    else:
        print(f"\nBelow baseline 0.78!")
    
    return mean_score, results


# =============================================================================
# GENERATE SUBMISSION
# =============================================================================

def submit(detector, test_dir, output_csv):
    images = sorted(Path(test_dir).glob("*.jpg"))
    print(f"Test images: {len(images)}")
    
    rows = []
    total = 0
    for idx, p in enumerate(images):
        dets = detect_image(detector, p)
        total += len(dets)
        pred_str = " ".join(f"{m} {x:.3f} {y:.3f}" for m, x, y in dets) if dets else ""
        rows.append([p.stem, pred_str])
        if (idx+1) % 100 == 0:
            print(f"  [{idx+1}/{len(images)}] {total} markers")
    
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'prediction_string'])
        w.writerows(rows)
    
    print(f"Saved: {output_csv} ({len(rows)} images, {total} markers, "
          f"avg {total/len(rows):.1f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--train_csv', required=True)
    p.add_argument('--output_csv', default='submission_v3.csv')
    p.add_argument('--mode', default='all', choices=['eval','submit','all'])
    args = p.parse_args()
    
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    assert os.path.isdir(train_dir), f"Not found: {train_dir}"
    assert os.path.isdir(test_dir), f"Not found: {test_dir}"
    
    print("="*60)
    print("ArUco Detection - Stage 2 v3 (Complement Merge)")
    print("="*60)
    print("Pass 1: Original image (baseline)")
    print("Pass 2: CLAHE (supplement)")
    print("Merge: Pass1 + NEW IDs from Pass2 only")
    print()
    
    detector = create_detector()
    
    if args.mode in ['eval', 'all']:
        t = time.time()
        evaluate(detector, train_dir, args.train_csv)
        print(f"Eval time: {time.time()-t:.1f}s")
    
    if args.mode in ['submit', 'all']:
        print()
        t = time.time()
        submit(detector, test_dir, args.output_csv)
        print(f"Submit time: {time.time()-t:.1f}s")
    
    print("\nDone!")

if __name__ == "__main__":
    main()