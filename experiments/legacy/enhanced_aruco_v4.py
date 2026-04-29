"""
=============================================================================
Stage 2 v4: Single-Pass with Spam Filtering & Corner Refinement
=============================================================================
LESSONS FROM ALL PREVIOUS FAILURES:
  v1 (0.003): High errorCorrectionRate → false positive explosion
  v2 (0.45):  Union merge across 5 preprocessed images → spam doubled
  v3 (0.63):  Complement merge still adds CLAHE false positives + 
              wider detector params increased pass 1 spam too

KEY INSIGHT: The baseline (0.78) already OVER-detects (11149 pred vs 
10407 GT). The problem is NOT missing markers — it's TOO MANY false 
positives. Every preprocessing variant and param relaxation makes this 
WORSE, not better.

STRATEGY v4: Improve by REMOVING bad detections, not adding more.
  1. Single pass on ORIGINAL image (no preprocessing)
  2. Default detector params (proven to work)  
  3. Post-filter: remove low-confidence detections
  4. Better corner refinement for localization accuracy
  5. Test both SUBPIX and APRILTAG corner refinement

Usage:
  python enhanced_aruco_v4.py --data_dir aruco-detection-challenge/aruco_data/aruco_data --train_csv aruco-detection-challenge/train.csv --output_csv submission_v4.csv --mode all
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
# POST-FILTER: Remove low-quality detections to reduce spam
# =============================================================================

def compute_marker_perimeter(corners):
    """Compute perimeter of the detected quadrilateral."""
    pts = corners  # shape (4, 2)
    perim = 0
    for i in range(4):
        dx = pts[(i+1)%4][0] - pts[i][0]
        dy = pts[(i+1)%4][1] - pts[i][1]
        perim += math.sqrt(dx*dx + dy*dy)
    return perim


def compute_marker_area(corners):
    """Compute area of quadrilateral using shoelace formula."""
    pts = corners
    area = 0.5 * abs(
        pts[0][0]*(pts[1][1]-pts[3][1]) +
        pts[1][0]*(pts[2][1]-pts[0][1]) +
        pts[2][0]*(pts[3][1]-pts[1][1]) +
        pts[3][0]*(pts[0][1]-pts[2][1])
    )
    return area


def compute_side_ratio(corners):
    """
    Ratio of shortest to longest side. 
    Perfect square = 1.0, very skewed = close to 0.
    ArUco markers are squares, so after perspective transform they should
    still have a reasonable side ratio (>0.3 even with strong perspective).
    Very low ratio = likely not a real marker.
    """
    pts = corners
    sides = []
    for i in range(4):
        dx = pts[(i+1)%4][0] - pts[i][0]
        dy = pts[(i+1)%4][1] - pts[i][1]
        sides.append(math.sqrt(dx*dx + dy*dy))
    
    if max(sides) < 1e-6:
        return 0.0
    return min(sides) / max(sides)


def is_convex(corners):
    """Check if quadrilateral is convex. Non-convex = likely false positive."""
    pts = corners
    cross_signs = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        p3 = pts[(i+2)%4]
        # Cross product of consecutive edges
        cross = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
        cross_signs.append(cross > 0)
    
    # All should be same sign for convex
    return all(cross_signs) or not any(cross_signs)


def post_filter(detections, img_h, img_w):
    """
    Remove low-quality detections that are likely false positives.
    
    Filters:
    1. Minimum perimeter: very tiny quads are usually noise
    2. Convexity: real markers are always convex quadrilaterals
    3. Side ratio: reject extremely elongated/skewed detections
    4. Minimum area: reject markers that are too small to decode reliably
    
    These filters are CONSERVATIVE — they only remove obvious junk,
    not borderline cases. Better to keep a few false positives than
    accidentally remove real markers.
    """
    filtered = []
    diagonal = math.sqrt(img_h**2 + img_w**2)
    
    for (marker_id, corners) in detections:
        perim = compute_marker_perimeter(corners)
        area = compute_marker_area(corners)
        side_ratio = compute_side_ratio(corners)
        convex = is_convex(corners)
        
        # Filter 1: minimum perimeter (must be at least 1.5% of diagonal)
        # A 6x6 ArUco marker needs enough pixels to decode 36 bits
        min_perim = 0.015 * diagonal
        if perim < min_perim:
            continue
        
        # Filter 2: must be convex
        if not convex:
            continue
        
        # Filter 3: side ratio must be reasonable
        # Even with strong perspective, ratio shouldn't drop below 0.15
        if side_ratio < 0.15:
            continue
        
        # Filter 4: minimum area
        min_area = (diagonal * 0.005) ** 2  # ~4x4 pixels minimum
        if area < min_area:
            continue
        
        filtered.append((marker_id, corners))
    
    return filtered


# =============================================================================
# DETECTOR
# =============================================================================

def create_detector(corner_method="subpix"):
    """
    Create detector with default params + specified corner refinement.
    
    NO changes to detection parameters from OpenCV defaults except:
    - Corner refinement method (SUBPIX or APRILTAG)
    
    This keeps false positive rate the same as the proven baseline.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    params = cv2.aruco.DetectorParameters()
    
    # Corner refinement - the ONLY parameter we change
    if corner_method == "apriltag":
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    else:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
    
    # Everything else = OpenCV defaults
    # adaptiveThreshWinSizeMin = 3 (default)
    # adaptiveThreshWinSizeMax = 23 (default)
    # adaptiveThreshWinSizeStep = 10 (default)
    # adaptiveThreshConstant = 7 (default)
    # minMarkerPerimeterRate = 0.03 (default)
    # maxMarkerPerimeterRate = 4.0 (default)
    # errorCorrectionRate = 0.6 (default)
    
    return cv2.aruco.ArucoDetector(aruco_dict, params)


# =============================================================================
# DETECTION: Single pass + post-filter
# =============================================================================

def detect_image(detector, image_path):
    """
    Single-pass detection with post-filtering.
    
    1. Read image → grayscale
    2. Run detector (default params)
    3. Post-filter to remove obvious false positives
    4. Return (marker_id, top_left_x, top_left_y) for each detection
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Detect
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or len(ids) == 0:
        return []
    
    # Collect raw detections
    raw = []
    for i in range(len(ids)):
        mid = int(ids[i][0])
        corner_pts = corners[i][0]  # (4, 2)
        raw.append((mid, corner_pts))
    
    # Post-filter
    filtered = post_filter(raw, h, w)
    
    # Extract top-left corner
    results = []
    for (mid, corner_pts) in filtered:
        tl_x = float(corner_pts[0][0])
        tl_y = float(corner_pts[0][1])
        results.append((mid, tl_x, tl_y))
    
    return results


# =============================================================================
# EVALUATION
# =============================================================================

def compute_score(gt_dets, pred_dets, img_h, img_w, sigma=0.02, lam=1.0):
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
        dists = sorted(
            min(math.sqrt((px-gx)**2+(py-gy)**2) for gx,gy in gts)
            for px,py in preds
        )
        n_valid = min(len(preds), len(gts))
        total_spam += max(0, len(preds) - len(gts))
        for k in range(n_valid):
            d_norm = dists[k] / diagonal
            total_phi += math.exp(-(d_norm**2)/(2*sigma**2))
    
    return total_phi / (N_gt + lam * total_spam)


def evaluate(detector, train_dir, train_csv, method_name=""):
    gt_data = {}
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            img_id = row[0].strip()
            ps = row[1].strip()
            dets = []
            if ps:
                parts = ps.split()
                for i in range(0, len(parts), 3):
                    dets.append((int(parts[i]), float(parts[i+1]), float(parts[i+2])))
            gt_data[img_id] = dets
    
    print(f"[{method_name}] GT: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} markers")
    
    scores = []
    total_gt = total_pred = total_spam = filtered_count = 0
    results = []
    
    for idx, img_id in enumerate(sorted(gt_data.keys())):
        img_path = os.path.join(train_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Detect with filter
        pred = detect_image(detector, img_path)
        gt = gt_data[img_id]
        
        # Also detect WITHOUT filter for comparison
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_corners, raw_ids, _ = detector.detectMarkers(gray)
        n_raw = len(raw_ids) if raw_ids is not None else 0
        n_filtered = n_raw - len(pred)
        filtered_count += n_filtered
        
        score = compute_score(gt, pred, h, w)
        scores.append(score)
        total_gt += len(gt)
        total_pred += len(pred)
        
        gt_ids = set(d[0] for d in gt)
        total_spam += sum(1 for d in pred if d[0] not in gt_ids)
        
        results.append({
            'image_id': img_id, 'score': score,
            'n_gt': len(gt), 'n_pred': len(pred),
            'n_raw': n_raw, 'n_filtered': n_filtered,
            'missed': gt_ids - set(d[0] for d in pred),
        })
        
        if (idx+1) % 200 == 0:
            print(f"  [{idx+1}/2000] Mean: {np.mean(scores):.4f} | "
                  f"Pred: {total_pred} | Spam: {total_spam} | "
                  f"Filtered out: {filtered_count}")
    
    ms = np.mean(scores)
    sa = np.array(scores)
    
    print(f"\n--- {method_name} ---")
    print(f"Mean score: {ms:.4f}")
    print(f"Pred: {total_pred} | GT: {total_gt} | Rate: {total_pred/total_gt*100:.1f}%")
    print(f"Spam: {total_spam} | Filtered out: {filtered_count}")
    print(f"Dist: min={sa.min():.4f} Q1={np.percentile(sa,25):.4f} "
          f"med={np.median(sa):.4f} Q3={np.percentile(sa,75):.4f}")
    
    if ms >= 0.78:
        marks = max(0, min(8, (1-(0.97-ms)/(0.97-0.78))*8))
        print(f"Estimated marks: {marks:.1f}/8")
    
    return ms, results


# =============================================================================
# SUBMISSION
# =============================================================================

def submit(detector, test_dir, output_csv):
    images = sorted(Path(test_dir).glob("*.jpg"))
    rows = []
    total = 0
    for idx, p in enumerate(images):
        dets = detect_image(detector, p)
        total += len(dets)
        ps = " ".join(f"{m} {x:.3f} {y:.3f}" for m,x,y in dets) if dets else ""
        rows.append([p.stem, ps])
        if (idx+1) % 100 == 0:
            print(f"  [{idx+1}/{len(images)}] {total} markers")
    
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id','prediction_string'])
        w.writerows(rows)
    print(f"Saved: {output_csv} ({total} markers, avg {total/len(rows):.1f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--train_csv', required=True)
    p.add_argument('--output_csv', default='submission_v4.csv')
    p.add_argument('--mode', default='all', choices=['eval','submit','all'])
    args = p.parse_args()
    
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    assert os.path.isdir(train_dir)
    assert os.path.isdir(test_dir)
    
    print("="*60)
    print("ArUco Detection v4 - Single Pass + Spam Filter")
    print("="*60)
    print("Strategy: Default detector params + post-filter junk")
    print("No preprocessing. No multi-pass. Just cleaner output.")
    print()
    
    # Test both corner refinement methods
    if args.mode in ['eval', 'all']:
        print("Testing SUBPIX corner refinement...")
        det_subpix = create_detector("subpix")
        t = time.time()
        score_sp, _ = evaluate(det_subpix, train_dir, args.train_csv, "SUBPIX")
        print(f"Time: {time.time()-t:.1f}s\n")
        
        print("Testing APRILTAG corner refinement...")
        det_april = create_detector("apriltag")
        t = time.time()
        score_ap, _ = evaluate(det_april, train_dir, args.train_csv, "APRILTAG")
        print(f"Time: {time.time()-t:.1f}s\n")
        
        # Pick the better one
        if score_ap >= score_sp:
            best_method = "apriltag"
            best_score = score_ap
        else:
            best_method = "subpix"
            best_score = score_sp
        
        print(f"{'='*60}")
        print(f"BEST: {best_method.upper()} with score {best_score:.4f}")
        print(f"{'='*60}")
    else:
        best_method = "subpix"  # default
    
    if args.mode in ['submit', 'all']:
        print(f"\nGenerating submission with {best_method}...")
        detector = create_detector(best_method)
        submit(detector, test_dir, args.output_csv)
    
    print("\nDone!")


if __name__ == "__main__":
    main()