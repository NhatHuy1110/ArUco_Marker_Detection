"""Feature extraction for the machine-learning false-positive filter."""

import math

import cv2
import numpy as np


FEATURE_NAMES = [
    "perimeter",
    "perimeter_norm",
    "area",
    "area_norm",
    "side_ratio",
    "side_std_norm",
    "angle_mean",
    "angle_std",
    "angle_dev_from_90",
    "is_convex",
    "warp_mean",
    "warp_std",
    "warp_min",
    "warp_max",
    "dark_ratio",
    "contrast",
    "border_mean",
    "border_std",
    "border_dark_ratio",
    "inner_mean",
    "inner_std",
    "clear_cell_ratio",
    "cell_uniformity",
    "n_detections",
    "dist_to_nearest",
    "area_vs_median",
]


def extract_features(corners_4x2, gray, img_h, img_w, all_detections_in_image):
    """Extract the 26-dimensional feature dictionary for one detection."""
    pts = corners_4x2.astype(np.float32)
    diagonal = math.sqrt(img_h**2 + img_w**2)

    features = {}
    sides = _side_lengths(pts)
    mean_side = np.mean(sides)

    perimeter = sum(sides)
    area = _quad_area(pts)
    features["perimeter"] = perimeter
    features["perimeter_norm"] = perimeter / diagonal
    features["area"] = area
    features["area_norm"] = area / (img_h * img_w)
    features["side_ratio"] = min(sides) / max(sides) if max(sides) > 1e-6 else 0.0
    features["side_std_norm"] = np.std(sides) / mean_side if mean_side > 1e-6 else 1.0

    angles = _corner_angles(pts)
    features["angle_mean"] = np.mean(angles)
    features["angle_std"] = np.std(angles)
    features["angle_dev_from_90"] = np.mean([abs(angle - 90) for angle in angles])
    features["is_convex"] = int(_is_convex(pts))

    warped = _warp_marker(gray, pts)
    features["warp_mean"] = np.mean(warped)
    features["warp_std"] = np.std(warped)
    features["warp_min"] = float(np.min(warped))
    features["warp_max"] = float(np.max(warped))
    features["dark_ratio"] = np.mean(warped < 128)
    features["contrast"] = (
        float(np.max(warped)) - float(np.min(warped))
    ) / (np.mean(warped) + 1e-8)

    _add_binary_pattern_features(features, warped)
    _add_context_features(features, pts, area, diagonal, all_detections_in_image)
    return features


def features_to_vector(feature_dict):
    """Convert a feature dict to a numpy vector in training-time order."""
    return np.array([feature_dict[name] for name in FEATURE_NAMES])


def _side_lengths(pts):
    sides = []
    for i in range(4):
        dx = pts[(i + 1) % 4][0] - pts[i][0]
        dy = pts[(i + 1) % 4][1] - pts[i][1]
        sides.append(math.sqrt(dx * dx + dy * dy))
    return sides


def _quad_area(pts):
    return 0.5 * abs(
        pts[0][0] * (pts[1][1] - pts[3][1])
        + pts[1][0] * (pts[2][1] - pts[0][1])
        + pts[2][0] * (pts[3][1] - pts[1][1])
        + pts[3][0] * (pts[0][1] - pts[2][1])
    )


def _corner_angles(pts):
    angles = []
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles.append(math.degrees(math.acos(np.clip(cos_angle, -1, 1))))
    return angles


def _is_convex(pts):
    signs = []
    for i in range(4):
        p1, p2, p3 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (
            p3[0] - p2[0]
        )
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def _warp_marker(gray, pts, dst_size=48):
    dst_pts = np.array(
        [[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]],
        dtype=np.float32,
    )
    try:
        matrix = cv2.getPerspectiveTransform(pts, dst_pts)
        return cv2.warpPerspective(gray, matrix, (dst_size, dst_size))
    except cv2.error:
        return np.zeros((dst_size, dst_size), dtype=np.uint8)


def _add_binary_pattern_features(features, warped):
    cell_size = warped.shape[0] // 8
    cell_means = np.zeros((8, 8))
    cell_stds = np.zeros((8, 8))

    for row in range(8):
        for col in range(8):
            cell = warped[
                row * cell_size : (row + 1) * cell_size,
                col * cell_size : (col + 1) * cell_size,
            ]
            cell_means[row, col] = np.mean(cell)
            cell_stds[row, col] = np.std(cell)

    border_cells = [
        cell_means[row, col]
        for row in range(8)
        for col in range(8)
        if row in (0, 7) or col in (0, 7)
    ]
    inner_means = cell_means[1:7, 1:7].flatten()
    inner_stds = cell_stds[1:7, 1:7].flatten()

    features["border_mean"] = np.mean(border_cells)
    features["border_std"] = np.std(border_cells)
    features["border_dark_ratio"] = np.mean(np.array(border_cells) < 128)
    features["inner_mean"] = np.mean(inner_means)
    features["inner_std"] = np.std(inner_means)
    features["clear_cell_ratio"] = (
        np.sum(inner_means < 80) + np.sum(inner_means > 180)
    ) / 36.0
    features["cell_uniformity"] = np.mean(inner_stds)


def _add_context_features(features, pts, area, diagonal, all_detections_in_image):
    features["n_detections"] = len(all_detections_in_image)

    if len(all_detections_in_image) <= 1:
        features["dist_to_nearest"] = 1.0
        features["area_vs_median"] = 1.0
        return

    my_center = np.mean(pts, axis=0)
    min_dist = float("inf")
    other_areas = []

    for _, other_corners in all_detections_in_image:
        other_pts = other_corners.astype(np.float32)
        other_center = np.mean(other_pts, axis=0)
        dist = np.linalg.norm(my_center - other_center)
        if dist <= 1:
            continue
        min_dist = min(min_dist, dist)
        other_areas.append(_quad_area(other_pts))

    features["dist_to_nearest"] = min_dist / diagonal if min_dist < float("inf") else 1.0
    features["area_vs_median"] = area / (np.median(other_areas) + 1e-8) if other_areas else 1.0
