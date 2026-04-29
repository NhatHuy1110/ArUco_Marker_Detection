# Experiment Report

## Problem

Detect ArUco markers from the FlyingArUco challenge dataset and submit marker IDs with top-left corner coordinates. The scoring function rewards correct localization with a Gaussian distance term and penalizes false positives in the denominator.

Dataset summary:

- Train: 2000 images, 10,407 markers
- Test: 500 images
- Dictionary: `DICT_ARUCO_MIP_36h12`
- Metric parameters: `sigma=0.02`, `lambda_spam=1.0`

## Final Pipeline

```text
image
  -> grayscale
  -> inverse-gamma preprocessing (gamma=2.5)
  -> OpenCV ArUco detector with SUBPIX corner refinement
  -> Gradient Boosting false-positive filter
  -> CSV prediction string
```

The gamma LUT is implemented as:

```text
I' = 255 * (I / 255) ** (1 / gamma)
```

With this convention, `gamma > 1` lifts mid-tone intensity. Older notes called this "darkening", but the implementation is inverse-gamma brightening.

## Main Results

| Pipeline | Train score | Public test score | Spam | Notes |
| --- | ---: | ---: | ---: | --- |
| OpenCV baseline | 0.780 | 0.783 | 1700 | SUBPIX, default detector params |
| Gamma + heuristic filter | 0.890 | - | 697 | Perimeter and side-ratio thresholds |
| Gamma + ML filter | 0.952 | 0.940 | 0 | Final submitted pipeline |

The final `python scripts/run_pipeline.py --mode eval` run produced:

- Mean train score: `0.9516`
- Predictions: `9925`
- Spam: `0`

The final `python scripts/run_pipeline.py --mode submit` run produced:

- Output: `outputs/submission_final.csv`
- Test predictions: `2444`

## Preprocessing Study

Most image enhancement methods hurt the score because they add marker-like texture and increase false positives. Gamma correction was the only preprocessing family that improved the baseline.

| Method | Score | Gain | Spam | Missed IDs |
| --- | ---: | ---: | ---: | ---: |
| Gamma 1.5 | 0.836 | +0.056 | 1500 | 475 |
| None | 0.780 | - | 1700 | 966 |
| Contrast stretch | 0.772 | -0.008 | 1826 | 955 |
| Sharpen mild | 0.743 | -0.037 | 2069 | 1186 |
| CLAHE | 0.741 | -0.039 | 2177 | 1123 |
| Histogram equalization | 0.730 | -0.050 | 1763 | 1560 |
| Bilateral filter | 0.722 | -0.058 | 1243 | 1913 |
| Gamma 0.6 | 0.602 | -0.178 | 1567 | 3172 |

Fine-tuning with the heuristic filter found the best score at `gamma=2.5`.

## ML Filter

Training data is built by running the detector on all training images. A detection is labeled real if it has the correct marker ID and the detected top-left corner is within 5% of the image diagonal from a matching ground truth point.

Training samples:

- Real: 9925
- Fake: 1191
- Total: 11116

Feature groups:

- Geometry: area, perimeter, side ratio, corner angles, convexity
- Intensity: statistics from the perspective-warped marker patch
- Binary pattern: border darkness and 8x8 cell-grid statistics
- Context: number of detections, nearest neighbor distance, relative area

The selected model is `GradientBoostingClassifier`, which achieved cross-validated F1 around `0.994` and removed all training-set spam in the final evaluation.

## Negative Results

| Approach | Score | Main issue |
| --- | ---: | --- |
| Relaxed multi-pass detection | 0.003 | High error correction decoded many background quads as markers |
| Union merge across preprocessing passes | 0.450 | Every pass added false positives |
| Complement merge | 0.630 | New false marker IDs bypassed the merge rule |
| CNN rejected-candidate recovery | 0.887 | Recovered detections had too many false positives |

The core lesson is that this metric rewards precision. Methods that add detections must be extremely accurate; otherwise the spam penalty dominates.
