"""Run the final ArUco marker detection pipeline.

Examples:
  python scripts/run_pipeline.py --mode eval
  python scripts/run_pipeline.py --mode submit
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aruco_marker_detection.config import (  # noqa: E402
    DEFAULT_PREPROCESS,
    DEFAULT_SUBMISSION_PATH,
    ML_MODEL_PATH,
    TEST_DIR,
    TRAIN_CSV,
    TRAIN_DIR,
)
from aruco_marker_detection.detector import create_detector, detect_on_gray  # noqa: E402
from aruco_marker_detection.metrics import compute_score_image  # noqa: E402
from aruco_marker_detection.ml_features import extract_features, features_to_vector  # noqa: E402
from aruco_marker_detection.postprocessing import filter_detections  # noqa: E402
from aruco_marker_detection.preprocessing import ALL_PREPROCESS  # noqa: E402
from aruco_marker_detection.utils import get_image_list, load_ground_truth, save_submission  # noqa: E402


def load_ml_model(model_path):
    if model_path is None or not model_path.exists():
        return None
    try:
        with model_path.open("rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError, pickle.UnpicklingError) as exc:
        print(f"Could not load ML model from {model_path}: {exc}")
        print("Falling back to the heuristic post-filter. Re-run scripts/train_ml_filter.py to rebuild it.")
        return None


def detect_image_full(detector, image_path, preprocess_fn, ml_model=None, use_heuristic=True):
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray_orig.shape[:2]
    processed = preprocess_fn(gray_orig)

    raw = detect_on_gray(detector, processed)
    if not raw:
        return []

    if ml_model is not None:
        features = [
            features_to_vector(extract_features(corners, gray_orig, img_h, img_w, raw))
            for _, corners in raw
        ]
        preds = ml_model.predict(np.nan_to_num(np.array(features)))
        raw = [det for det, keep in zip(raw, preds) if keep == 1]
    elif use_heuristic:
        raw = filter_detections(raw, img_h, img_w)

    return [(mid, float(corners[0][0]), float(corners[0][1])) for mid, corners in raw]


def evaluate(detector, preprocess_fn, ml_model, use_heuristic):
    gt_data = load_ground_truth(TRAIN_CSV)
    print(f"GT: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} markers")

    scores = []
    total_pred = 0
    total_spam = 0
    per_image = []

    for idx, img_id in enumerate(sorted(gt_data.keys()), start=1):
        img_path = TRAIN_DIR / f"{img_id}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        pred = detect_image_full(detector, img_path, preprocess_fn, ml_model, use_heuristic)
        gt = gt_data[img_id]

        score = compute_score_image(gt, pred, img_h, img_w)
        scores.append(score)
        total_pred += len(pred)
        gt_ids = {det[0] for det in gt}
        total_spam += sum(1 for det in pred if det[0] not in gt_ids)
        per_image.append({"image_id": img_id, "score": score, "n_gt": len(gt), "n_pred": len(pred)})

        if idx % 200 == 0:
            print(
                f"  [{idx}/{len(gt_data)}] mean={np.mean(scores):.4f} "
                f"pred={total_pred} spam={total_spam}"
            )

    score_arr = np.array(scores)
    mean_score = float(np.mean(score_arr))
    print(f"\nMean score: {mean_score:.4f}")
    print(f"Predictions: {total_pred} | Spam: {total_spam}")
    print(
        f"Q1={np.percentile(score_arr, 25):.4f} "
        f"Median={np.median(score_arr):.4f} "
        f"Q3={np.percentile(score_arr, 75):.4f}"
    )

    print("\n10 worst images:")
    for row in sorted(per_image, key=lambda r: r["score"])[:10]:
        print(
            f"  {row['image_id']}: score={row['score']:.4f} "
            f"gt={row['n_gt']} pred={row['n_pred']}"
        )

    return mean_score


def generate_submission(detector, preprocess_fn, ml_model, use_heuristic, output_path):
    images = get_image_list(TEST_DIR)
    print(f"Test images: {len(images)}")

    predictions = {}
    total = 0
    for idx, image_path in enumerate(images, start=1):
        dets = detect_image_full(detector, image_path, preprocess_fn, ml_model, use_heuristic)
        predictions[image_path.stem] = dets
        total += len(dets)
        if idx % 100 == 0:
            print(f"  [{idx}/{len(images)}] markers={total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_submission(predictions, output_path)
    print(f"Saved: {output_path} ({total} markers, avg {total / len(images):.1f})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["eval", "submit", "all"])
    parser.add_argument("--preprocess", default=DEFAULT_PREPROCESS, choices=sorted(ALL_PREPROCESS))
    parser.add_argument("--model", type=Path, default=ML_MODEL_PATH)
    parser.add_argument("--no-ml-filter", action="store_true")
    parser.add_argument("--no-heuristic-filter", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBMISSION_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess_fn = ALL_PREPROCESS[args.preprocess]
    use_heuristic = not args.no_heuristic_filter
    ml_model = None if args.no_ml_filter else load_ml_model(args.model)

    print("=" * 60)
    print("ArUco Detection Pipeline")
    print("=" * 60)
    print(f"Preprocess: {args.preprocess}")
    print(f"ML filter: {'enabled' if ml_model is not None else 'disabled'}")
    print(f"Heuristic fallback: {use_heuristic}")

    detector = create_detector("subpix")

    if args.mode in {"eval", "all"}:
        start = time.time()
        evaluate(detector, preprocess_fn, ml_model, use_heuristic)
        print(f"Eval time: {time.time() - start:.1f}s")

    if args.mode in {"submit", "all"}:
        start = time.time()
        generate_submission(detector, preprocess_fn, ml_model, use_heuristic, args.output)
        print(f"Submit time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
