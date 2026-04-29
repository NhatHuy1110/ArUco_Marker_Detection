"""
train_ml_filter.py - Train an ML classifier to filter false positive detections.

Workflow:
  1. Run gamma_2.5 detector on all training images
  2. Label each detection as REAL or FAKE by matching with ground truth
  3. Extract 26 features per detection
  4. Train Random Forest with cross-validation
  5. Analyze feature importance
  6. Save trained model

Usage:
  python train_ml_filter.py
"""

import sys
import math
import time
import pickle
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aruco_marker_detection.config import ML_MODEL_PATH, TRAIN_CSV, TRAIN_DIR
from aruco_marker_detection.detector import create_detector, detect_on_gray
from aruco_marker_detection.metrics import compute_score_image
from aruco_marker_detection.ml_features import (
    FEATURE_NAMES,
    extract_features,
    features_to_vector,
)
from aruco_marker_detection.preprocessing import ALL_PREPROCESS
from aruco_marker_detection.utils import load_ground_truth


def label_detections(detections, gt_dets, img_h, img_w):
    """
    Label each detection as REAL (1) or FAKE (0).
    
    A detection is REAL if:
      - Its marker ID exists in ground truth
      - Its position is within 5% of image diagonal from the GT position
    
    Everything else is FAKE (wrong ID or too far from any GT).
    """
    diagonal = math.sqrt(img_h**2 + img_w**2)
    threshold = 0.05 * diagonal
    
    gt_by_id = defaultdict(list)
    for mid, x, y in gt_dets:
        gt_by_id[mid].append((x, y))
    
    labels = []
    for mid, corners in detections:
        tl_x, tl_y = float(corners[0][0]), float(corners[0][1])
        
        if mid not in gt_by_id:
            labels.append(0)  # FAKE: wrong ID
            continue
        
        # Check if close to any GT with same ID
        min_dist = min(math.sqrt((tl_x-gx)**2 + (tl_y-gy)**2) 
                       for gx, gy in gt_by_id[mid])
        
        if min_dist < threshold:
            labels.append(1)  # REAL
        else:
            labels.append(0)  # FAKE: right ID but wrong position
    
    return labels


def build_dataset(detector, gt_data, preprocess_fn):
    """
    Build feature matrix X and label vector y from all training images.
    """
    print("Building ML dataset...")
    
    X_list = []
    y_list = []
    meta_list = []  # (image_id, marker_id) for analysis
    
    sorted_ids = sorted(gt_data.keys())
    
    for idx, img_id in enumerate(sorted_ids):
        img_path = TRAIN_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_proc = preprocess_fn(gray_orig)
        h, w = gray_orig.shape[:2]
        
        # Detect
        raw = detect_on_gray(detector, gray_proc)
        if len(raw) == 0:
            continue
        
        # Label
        gt_dets = gt_data[img_id]
        labels = label_detections(raw, gt_dets, h, w)
        
        # Extract features (use ORIGINAL gray for intensity features)
        for i, (mid, corners) in enumerate(raw):
            feat = extract_features(corners, gray_orig, h, w, raw)
            X_list.append(features_to_vector(feat))
            y_list.append(labels[i])
            meta_list.append((img_id, mid))
        
        if (idx + 1) % 200 == 0:
            n_real = sum(y_list)
            n_fake = len(y_list) - n_real
            print(f"  [{idx+1}/{len(sorted_ids)}] samples={len(y_list)} "
                  f"(real={n_real}, fake={n_fake})")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nDataset built:")
    print(f"  Total samples: {len(y)}")
    print(f"  Real (1): {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
    print(f"  Fake (0): {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
    
    return X, y, meta_list


def train_and_evaluate(X, y):
    """
    Train Random Forest with cross-validation.
    Returns best model.
    """
    print("\n" + "="*60)
    print("TRAINING ML CLASSIFIER")
    print("="*60)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # ---- Random Forest ----
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced',  # handle class imbalance
        random_state=42,
        n_jobs=1,
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validation
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1')
    print(f"  F1 scores: {cv_scores}")
    print(f"  Mean F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    cv_acc = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    print(f"  Mean accuracy: {cv_acc.mean():.4f}")
    
    # Train on full data
    print("\nTraining on full dataset...")
    rf.fit(X, y)
    
    # Feature importance
    print("\nTop 10 most important features:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices[:10]):
        print(f"  {rank+1}. {FEATURE_NAMES[idx]:25s}: {importances[idx]:.4f}")
    
    # Classification report on training data (for reference)
    y_pred = rf.predict(X)
    print(f"\nTraining set classification report:")
    print(classification_report(y, y_pred, target_names=['FAKE', 'REAL']))
    
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion matrix:")
    print(f"  True Fake correctly rejected: {cm[0][0]} / {cm[0].sum()}")
    print(f"  True Fake incorrectly kept:   {cm[0][1]} / {cm[0].sum()}")
    print(f"  True Real correctly kept:      {cm[1][1]} / {cm[1].sum()}")
    print(f"  True Real incorrectly removed: {cm[1][0]} / {cm[1].sum()}")
    
    # ---- Gradient Boosting (optional comparison) ----
    print("\n--- Gradient Boosting ---")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=42,
    )
    
    # Need sample weights for imbalance
    sample_weights = np.ones(len(y))
    n_fake = np.sum(y == 0)
    n_real = np.sum(y == 1)
    sample_weights[y == 0] = n_real / n_fake  # upweight minority class
    
    print("Running 5-fold cross-validation...")
    cv_scores_gb = cross_val_score(gb, X, y, cv=cv, scoring='f1')
    print(f"  Mean F1: {cv_scores_gb.mean():.4f} +/- {cv_scores_gb.std():.4f}")
    
    gb.fit(X, y, sample_weight=sample_weights)
    
    y_pred_gb = gb.predict(X)
    cm_gb = confusion_matrix(y, y_pred_gb)
    print(f"  True Fake correctly rejected: {cm_gb[0][0]} / {cm_gb[0].sum()}")
    print(f"  True Real correctly kept:      {cm_gb[1][1]} / {cm_gb[1].sum()}")
    
    # ---- Pick best ----
    if cv_scores_gb.mean() > cv_scores.mean():
        print(f"\n>>> Best model: GradientBoosting (F1={cv_scores_gb.mean():.4f})")
        return gb, "gradient_boosting"
    else:
        print(f"\n>>> Best model: RandomForest (F1={cv_scores.mean():.4f})")
        return rf, "random_forest"


def evaluate_with_ml_filter(detector, gt_data, preprocess_fn, model):
    """
    Evaluate the full pipeline with ML post-filter.
    Compare against heuristic filter and no filter.
    """
    print("\n" + "="*60)
    print("EVALUATING ML POST-FILTER ON TRAINING SET")
    print("="*60)
    
    from aruco_marker_detection.postprocessing import filter_detections
    
    scores_no_filter = []
    scores_heuristic = []
    scores_ml = []
    
    total_pred_nf = total_pred_h = total_pred_ml = 0
    total_spam_nf = total_spam_h = total_spam_ml = 0
    
    for idx, img_id in enumerate(sorted(gt_data.keys())):
        img_path = TRAIN_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_proc = preprocess_fn(gray_orig)
        h, w = gray_orig.shape[:2]
        gt_dets = gt_data[img_id]
        gt_ids = set(d[0] for d in gt_dets)
        
        # Detect
        raw = detect_on_gray(detector, gray_proc)
        
        # --- No filter ---
        pred_nf = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in raw]
        s_nf = compute_score_image(gt_dets, pred_nf, h, w)
        scores_no_filter.append(s_nf)
        total_pred_nf += len(pred_nf)
        total_spam_nf += sum(1 for d in pred_nf if d[0] not in gt_ids)
        
        # --- Heuristic filter ---
        filtered_h = filter_detections(raw, h, w)
        pred_h = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in filtered_h]
        s_h = compute_score_image(gt_dets, pred_h, h, w)
        scores_heuristic.append(s_h)
        total_pred_h += len(pred_h)
        total_spam_h += sum(1 for d in pred_h if d[0] not in gt_ids)
        
        # --- ML filter ---
        if len(raw) > 0:
            # Extract features
            X_test = []
            for mid, corners in raw:
                feat = extract_features(corners, gray_orig, h, w, raw)
                X_test.append(features_to_vector(feat))
            X_test = np.nan_to_num(np.array(X_test))
            
            # Predict
            preds = model.predict(X_test)
            
            # Keep only detections classified as REAL
            pred_ml = []
            for i, (mid, corners) in enumerate(raw):
                if preds[i] == 1:
                    pred_ml.append((mid, float(corners[0][0]), float(corners[0][1])))
        else:
            pred_ml = []
        
        s_ml = compute_score_image(gt_dets, pred_ml, h, w)
        scores_ml.append(s_ml)
        total_pred_ml += len(pred_ml)
        total_spam_ml += sum(1 for d in pred_ml if d[0] not in gt_ids)
        
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/2000] "
                  f"no_filter={np.mean(scores_no_filter):.4f} "
                  f"heuristic={np.mean(scores_heuristic):.4f} "
                  f"ML={np.mean(scores_ml):.4f}")
    
    print(f"\n{'Method':20s} {'Score':>8s} {'Pred':>6s} {'Spam':>6s}")
    print("-" * 45)
    print(f"{'No filter':20s} {np.mean(scores_no_filter):8.4f} "
          f"{total_pred_nf:6d} {total_spam_nf:6d}")
    print(f"{'Heuristic filter':20s} {np.mean(scores_heuristic):8.4f} "
          f"{total_pred_h:6d} {total_spam_h:6d}")
    print(f"{'ML filter':20s} {np.mean(scores_ml):8.4f} "
          f"{total_pred_ml:6d} {total_spam_ml:6d}")
    
    return np.mean(scores_ml)


def main():
    print("=" * 60)
    print("ML POST-FILTER TRAINING")
    print("=" * 60)
    
    gt_data = load_ground_truth(TRAIN_CSV)
    detector = create_detector("subpix")
    preprocess_fn = ALL_PREPROCESS["gamma_2.5"]
    
    # Step 1: Build dataset
    t = time.time()
    X, y, meta = build_dataset(detector, gt_data, preprocess_fn)
    print(f"Dataset build time: {time.time()-t:.1f}s")
    
    # Step 2: Train and evaluate classifier
    model, model_name = train_and_evaluate(X, y)
    
    # Step 3: Evaluate on training set (compare methods)
    ml_score = evaluate_with_ml_filter(detector, gt_data, preprocess_fn, model)
    
    # Step 4: Save model
    ML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ML_MODEL_PATH.open('wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {ML_MODEL_PATH}")
    
    # Marks estimate
    if ml_score >= 0.78:
        marks = max(0, min(8, (1-(0.97-ml_score)/(0.97-0.78))*8))
        print(f"Estimated marks: {marks:.1f}/8")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
