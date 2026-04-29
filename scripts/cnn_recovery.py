"""
cnn_recovery.py - CNN-based recovery of missed ArUco markers.

Purpose: The OpenCV ArUco detector misses ~500 markers. It finds many
quadrilateral "candidates" but fails to decode their bit patterns.
This CNN classifies each rejected candidate as REAL marker or NOT,
then attempts enhanced decoding on CNN-verified candidates.

Architecture: Simple 4-layer ConvNet (no pretrained model needed)
  Input:  32x32 grayscale (perspective-warped candidate)
  Output: binary (1=marker, 0=not-marker)

Workflow:
  1. python cnn_recovery.py --mode prepare   # Build training dataset
  2. python cnn_recovery.py --mode train      # Train CNN
  3. python cnn_recovery.py --mode evaluate   # Test impact on score

Requirements:
  pip install torch torchvision
"""

import os
import math
import time
import pickle
import argparse
import csv
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aruco_marker_detection.config import (
    CNN_DATA_DIR,
    CNN_MODEL_PATH,
    ML_MODEL_PATH,
    TRAIN_CSV,
    TRAIN_DIR,
)
from aruco_marker_detection.detector import create_detector, detect_on_gray
from aruco_marker_detection.metrics import compute_score_image
from aruco_marker_detection.preprocessing import ALL_PREPROCESS
from aruco_marker_detection.utils import load_ground_truth


# =============================================================================
# 1. DATASET PREPARATION
# =============================================================================

WARP_SIZE = 32  # Input size for CNN

def warp_candidate(corners_4x2, gray):
    """
    Perspective-transform a quadrilateral candidate to a 32x32 square.
    This normalizes the view regardless of rotation/perspective.
    """
    pts = corners_4x2.astype(np.float32)
    dst = np.array([[0, 0], [WARP_SIZE, 0], 
                     [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(gray, M, (WARP_SIZE, WARP_SIZE))
        return warped
    except:
        return None


def prepare_dataset(detector, gt_data, preprocess_fn, save_dir=CNN_DATA_DIR):
    """
    Build training dataset for CNN from detected + rejected candidates.
    
    POSITIVE samples: Warped images of successfully detected markers
      (we know these are real because they matched GT)
    
    NEGATIVE samples: Warped images of rejected candidates
      (quadrilaterals that OpenCV found but couldn't decode)
      + some that were decoded but are spam (wrong ID)
    
    Also extract HARD POSITIVES: markers present in GT but not detected
      (if we can find their approximate location from rejected candidates)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    positives = []  # warped images of real markers
    negatives = []  # warped images of non-markers
    
    sorted_ids = sorted(gt_data.keys())
    
    for idx, img_id in enumerate(sorted_ids):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_proc = preprocess_fn(gray)
        h, w = gray.shape[:2]
        diagonal = math.sqrt(h**2 + w**2)
        
        # Get GT info
        gt_dets = gt_data[img_id]
        gt_by_id = defaultdict(list)
        for mid, gx, gy in gt_dets:
            gt_by_id[mid].append((gx, gy))
        
        # Run detector to get both detections and rejected candidates
        corners, ids, rejected = detector.detectMarkers(gray_proc)
        
        # POSITIVES: successfully detected markers that match GT
        if ids is not None:
            for i in range(len(ids)):
                mid = int(ids[i][0])
                pts = corners[i][0]
                tl_x, tl_y = float(pts[0][0]), float(pts[0][1])
                
                # Check if this detection is real (matches GT)
                if mid in gt_by_id:
                    for gx, gy in gt_by_id[mid]:
                        dist = math.sqrt((tl_x-gx)**2 + (tl_y-gy)**2)
                        if dist < 0.05 * diagonal:
                            warped = warp_candidate(pts, gray)
                            if warped is not None:
                                positives.append(warped)
                            break
                else:
                    # False positive detection: negative sample
                    warped = warp_candidate(pts, gray)
                    if warped is not None:
                        negatives.append(warped)
        
        # NEGATIVES: rejected candidates (found quad but couldn't decode)
        if rejected is not None:
            for rej in rejected:
                pts = rej[0]
                if len(pts) == 4:
                    warped = warp_candidate(pts, gray)
                    if warped is not None:
                        # Check if this rejected candidate is actually near a GT marker
                        # (it might be a real marker that couldn't be decoded)
                        center = np.mean(pts, axis=0)
                        is_near_gt = False
                        for mid_gt, gx, gy in gt_dets:
                            dist = math.sqrt((center[0]-gx)**2 + (center[1]-gy)**2)
                            if dist < 0.08 * diagonal:
                                is_near_gt = True
                                break
                        
                        if is_near_gt:
                            # This rejected candidate is near a GT marker
                            # Treat as positive: a marker the detector could not decode.
                            positives.append(warped)
                        else:
                            # Not near any GT: negative sample.
                            negatives.append(warped)
        
        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{len(sorted_ids)}] "
                  f"pos={len(positives)} neg={len(negatives)}")
    
    # Balance dataset (undersample majority class)
    print(f"\nRaw: {len(positives)} positives, {len(negatives)} negatives")
    
    # Save
    pos_arr = np.array(positives)
    neg_arr = np.array(negatives)
    
    np.save(os.path.join(save_dir, "positives.npy"), pos_arr)
    np.save(os.path.join(save_dir, "negatives.npy"), neg_arr)
    
    print(f"Saved to {save_dir}/")
    print(f"  positives.npy: {pos_arr.shape}")
    print(f"  negatives.npy: {neg_arr.shape}")
    
    return pos_arr, neg_arr


# =============================================================================
# 2. CNN MODEL
# =============================================================================

class MarkerCNN(nn.Module):
    """
    Simple 4-layer ConvNet for binary classification.
    
    Architecture:
      Conv(1 to 16, 3x3), ReLU, MaxPool(2): 16x16
      Conv(16 to 32, 3x3), ReLU, MaxPool(2): 8x8
      Conv(32 to 64, 3x3), ReLU, MaxPool(2): 4x4
      Flatten, FC(1024 to 128), ReLU, Dropout, FC(128 to 1), Sigmoid
    
    Input: (batch, 1, 32, 32) grayscale
    Output: (batch, 1) probability of being a real marker
    
    Why this architecture:
    - Small input (32x32) doesn't need deep network
    - 3 conv layers capture: edges (layer1), patterns (layer2), structure (layer3)
    - Dropout prevents overfitting on small dataset
    - Binary output with sigmoid for marker/not-marker
    """
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 32 to 16
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 16 to 8
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 8 to 4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CandidateDataset(Dataset):
    """PyTorch dataset for warped candidate images."""
    
    def __init__(self, images, labels):
        self.images = images.astype(np.float32) / 255.0  # normalize to [0,1]
        self.labels = labels.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        # Add channel dimension: (32, 32) to (1, 32, 32)
        img = img[np.newaxis, :, :]
        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(label)


# =============================================================================
# 3. TRAINING
# =============================================================================

def train_cnn(data_dir=CNN_DATA_DIR, model_path=CNN_MODEL_PATH,
              epochs=30, batch_size=64, lr=0.001):
    """
    Train CNN on prepared dataset.
    """
    # Load data
    pos = np.load(os.path.join(data_dir, "positives.npy"))
    neg = np.load(os.path.join(data_dir, "negatives.npy"))
    
    print(f"Loaded: {len(pos)} positives, {len(neg)} negatives")
    
    # Create labels
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)
    
    # Shuffle
    perm = np.random.RandomState(42).permutation(len(y))
    X, y = X[perm], y[perm]
    
    # Split: 80% train, 20% validation
    split = int(0.8 * len(y))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Train: {len(y_train)} (pos={np.sum(y_train==1):.0f}, neg={np.sum(y_train==0):.0f})")
    print(f"Val:   {len(y_val)} (pos={np.sum(y_val==1):.0f}, neg={np.sum(y_val==0):.0f})")
    
    # Datasets
    train_ds = CandidateDataset(X_train, y_train)
    val_ds = CandidateDataset(X_val, y_val)
    
    # Class weights for imbalanced data
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([n_neg / n_pos])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = MarkerCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Override forward to remove sigmoid for BCEWithLogitsLoss
    # Actually, let's use BCELoss since model already has sigmoid
    criterion = nn.BCELoss()
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>5s} {'Train Loss':>12s} {'Val Loss':>12s} {'Val Acc':>10s} "
          f"{'Val Prec':>10s} {'Val Recall':>10s}")
    print("-" * 65)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
        
        train_loss /= len(train_ds)
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * len(labels)
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_ds)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = np.mean(all_preds == all_labels)
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"{epoch+1:5d} {train_loss:12.4f} {val_loss:12.4f} "
              f"{acc:10.4f} {precision:10.4f} {recall:10.4f}")
        
        # Save best model
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), model_path)
        
        scheduler.step()
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved: {model_path}")
    
    return model


# =============================================================================
# 4. INFERENCE: Recover missed markers using CNN
# =============================================================================

def recover_markers_cnn(detector, gray_proc, gray_orig, h, w,
                        cnn_model, device, confidence_threshold=0.7):
    """
    Try to recover markers from rejected candidates using CNN.
    
    Process:
    1. Get rejected candidates from ArUco detector
    2. Warp each to 32x32
    3. CNN predicts: is this a real marker?
    4. For high-confidence positives: try to decode with relaxed settings
    
    Returns: list of (marker_id, top_left_x, top_left_y) recovered markers
    """
    # Get rejected candidates
    corners, ids, rejected = detector.detectMarkers(gray_proc)
    
    if rejected is None or len(rejected) == 0:
        return []
    
    # Create a more tolerant detector for re-decoding
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    tolerant_params = cv2.aruco.DetectorParameters()
    tolerant_params.errorCorrectionRate = 0.8  # more tolerant for CNN-verified candidates
    tolerant_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    tolerant_detector = cv2.aruco.ArucoDetector(aruco_dict, tolerant_params)
    
    recovered = []
    
    # Already detected IDs (avoid duplicates)
    existing_ids = set()
    if ids is not None:
        existing_ids = set(int(ids[i][0]) for i in range(len(ids)))
    
    cnn_model.eval()
    
    for rej in rejected:
        pts = rej[0]
        if len(pts) != 4:
            continue
        
        # Warp to 32x32
        warped = warp_candidate(pts.astype(np.float32), gray_orig)
        if warped is None:
            continue
        
        # CNN prediction
        img_tensor = torch.tensor(warped.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            prob = cnn_model(img_tensor).item()
        
        if prob < confidence_threshold:
            continue
        
        # CNN says this is likely a marker!
        # Try to decode with multiple strategies:
        
        # Strategy 1: Crop region around the candidate, try detection
        x_min = max(0, int(pts[:, 0].min()) - 10)
        y_min = max(0, int(pts[:, 1].min()) - 10)
        x_max = min(w, int(pts[:, 0].max()) + 10)
        y_max = min(h, int(pts[:, 1].max()) + 10)
        
        crop = gray_proc[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        
        # Try with enhanced contrast on crop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        crop_enhanced = clahe.apply(crop)
        
        # Try decode on enhanced crop
        c2, i2, _ = tolerant_detector.detectMarkers(crop_enhanced)
        
        if i2 is not None and len(i2) > 0:
            for j in range(len(i2)):
                mid = int(i2[j][0])
                if mid not in existing_ids:
                    # Adjust coordinates back to full image
                    tl_x = float(c2[j][0][0][0]) + x_min
                    tl_y = float(c2[j][0][0][1]) + y_min
                    recovered.append((mid, tl_x, tl_y))
                    existing_ids.add(mid)
    
    return recovered


# =============================================================================
# 5. EVALUATE WITH CNN RECOVERY
# =============================================================================

def evaluate_with_cnn(detector, gt_data, preprocess_fn, cnn_model, ml_model, device):
    """Evaluate full pipeline: gamma, detection, ML filter, CNN recovery."""
    from aruco_marker_detection.ml_features import extract_features, features_to_vector
    
    scores = []
    total_pred = total_spam = total_recovered = 0
    
    for idx, img_id in enumerate(sorted(gt_data.keys())):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_proc = preprocess_fn(gray_orig)
        h, w = gray_orig.shape[:2]
        gt_dets = gt_data[img_id]
        
        # Step 1: Detect
        raw = detect_on_gray(detector, gray_proc)
        
        # Step 2: ML filter
        if len(raw) > 0 and ml_model is not None:
            X = []
            for mid, corners in raw:
                feat = extract_features(corners, gray_orig, h, w, raw)
                X.append(features_to_vector(feat))
            X = np.nan_to_num(np.array(X))
            preds = ml_model.predict(X)
            filtered = [raw[i] for i in range(len(raw)) if preds[i] == 1]
        else:
            filtered = raw
        
        # Convert to result format
        results = [(mid, float(c[0][0]), float(c[0][1])) for mid, c in filtered]
        
        # Step 3: CNN recovery of rejected candidates
        recovered = recover_markers_cnn(detector, gray_proc, gray_orig, h, w,
                                         cnn_model, device, confidence_threshold=0.7)
        
        # Add recovered markers (avoid duplicate IDs)
        existing_ids = set(d[0] for d in results)
        for mid, x, y in recovered:
            if mid not in existing_ids:
                results.append((mid, x, y))
                existing_ids.add(mid)
                total_recovered += 1
        
        # Score
        score = compute_score_image(gt_dets, results, h, w)
        scores.append(score)
        total_pred += len(results)
        gt_ids = set(d[0] for d in gt_dets)
        total_spam += sum(1 for d in results if d[0] not in gt_ids)
        
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/2000] Mean: {np.mean(scores):.4f} | "
                  f"Recovered: {total_recovered} | Spam: {total_spam}")
    
    ms = np.mean(scores)
    print(f"\nScore: {ms:.4f}")
    print(f"Pred: {total_pred} | Spam: {total_spam} | Recovered: {total_recovered}")
    
    if ms >= 0.78:
        marks = max(0, min(8, (1-(0.97-ms)/(0.97-0.78))*8))
        print(f"Estimated marks: {marks:.1f}/8")
    
    return ms


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, 
                        choices=['prepare', 'train', 'evaluate', 'all'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--data_dir', type=Path, default=CNN_DATA_DIR)
    parser.add_argument('--model_path', type=Path, default=CNN_MODEL_PATH)
    args = parser.parse_args()
    
    gt_data = load_ground_truth(TRAIN_CSV)
    detector = create_detector("subpix")
    preprocess_fn = ALL_PREPROCESS["gamma_2.5"]
    
    print("=" * 60)
    print("CNN MARKER RECOVERY")
    print("=" * 60)
    
    if args.mode in ['prepare', 'all']:
        print("\n--- PREPARING DATASET ---")
        prepare_dataset(detector, gt_data, preprocess_fn, args.data_dir)
    
    if args.mode in ['train', 'all']:
        print("\n--- TRAINING CNN ---")
        cnn_model = train_cnn(args.data_dir, args.model_path, epochs=args.epochs)
    
    if args.mode in ['evaluate', 'all']:
        print("\n--- EVALUATING ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load CNN model
        cnn_model = MarkerCNN().to(device)
        cnn_model.load_state_dict(torch.load(args.model_path, map_location=device))
        cnn_model.eval()
        print(f"CNN model loaded: {args.model_path}")
        
        # Load ML filter model
        ml_model = None
        if ML_MODEL_PATH.exists():
            with ML_MODEL_PATH.open('rb') as f:
                ml_model = pickle.load(f)
            print("ML filter loaded")
        
        evaluate_with_cnn(detector, gt_data, preprocess_fn, cnn_model, ml_model, device)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
