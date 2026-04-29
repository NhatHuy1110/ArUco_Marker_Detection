"""Competition metric for marker ID and top-left localization quality."""

import math
from collections import defaultdict

from aruco_marker_detection.config import LAMBDA_SPAM, SIGMA


def compute_score_image(gt_dets, pred_dets, img_h, img_w, sigma=SIGMA, lam=LAMBDA_SPAM):
    """Compute the per-image Kaggle score.

    Args:
        gt_dets: list of (marker_id, x, y) ground-truth detections.
        pred_dets: list of (marker_id, x, y) predicted detections.
        img_h: image height.
        img_w: image width.
        sigma: Gaussian tolerance in normalized image-diagonal units.
        lam: false-positive penalty weight.

    Returns:
        Score in [0, 1].
    """
    n_gt = len(gt_dets)
    if n_gt == 0:
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

    for marker_id, preds in pred_by_id.items():
        if marker_id not in gt_by_id:
            total_spam += len(preds)
            continue

        gts = gt_by_id[marker_id]
        dists = sorted(
            min(math.sqrt((px - gx) ** 2 + (py - gy) ** 2) for gx, gy in gts)
            for px, py in preds
        )

        n_valid = min(len(preds), len(gts))
        total_spam += max(0, len(preds) - len(gts))

        for dist in dists[:n_valid]:
            d_norm = dist / diagonal
            total_phi += math.exp(-(d_norm**2) / (2 * sigma**2))

    return total_phi / (n_gt + lam * total_spam)
