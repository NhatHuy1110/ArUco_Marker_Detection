"""
Post-filtering utilities for removing low-quality ArUco detections.

These filters run AFTER detection to remove low-quality results.
They are conservative: only remove obvious junk.
"""

import math
import numpy as np


def compute_perimeter(corners):
    """Perimeter of detected quadrilateral."""
    pts = corners  # (4, 2)
    perim = 0
    for i in range(4):
        dx = pts[(i+1)%4][0] - pts[i][0]
        dy = pts[(i+1)%4][1] - pts[i][1]
        perim += math.sqrt(dx*dx + dy*dy)
    return perim


def compute_area(corners):
    """Area via shoelace formula."""
    p = corners
    return 0.5 * abs(
        p[0][0]*(p[1][1]-p[3][1]) + p[1][0]*(p[2][1]-p[0][1]) +
        p[2][0]*(p[3][1]-p[1][1]) + p[3][0]*(p[0][1]-p[2][1])
    )


def compute_side_ratio(corners):
    """Ratio of shortest/longest side. 1.0 = perfect square."""
    sides = []
    for i in range(4):
        dx = corners[(i+1)%4][0] - corners[i][0]
        dy = corners[(i+1)%4][1] - corners[i][1]
        sides.append(math.sqrt(dx*dx + dy*dy))
    return min(sides) / max(sides) if max(sides) > 1e-6 else 0.0


def is_convex(corners):
    """Check if quad is convex."""
    signs = []
    for i in range(4):
        p1, p2, p3 = corners[i], corners[(i+1)%4], corners[(i+2)%4]
        cross = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def filter_detections(detections_with_corners, img_h, img_w,
                      min_perim_ratio=0.05, min_side_ratio=0.35,
                      require_convex=True):
    """
    Remove low-quality detections.
    
    Args:
        detections_with_corners: list of (marker_id, corners_4x2)
        img_h, img_w: image dimensions
        min_perim_ratio: minimum perimeter as fraction of image diagonal
        min_side_ratio: minimum shortest/longest side ratio
        require_convex: reject non-convex quadrilaterals
    
    Returns:
        filtered list of (marker_id, corners_4x2)
    """
    diagonal = math.sqrt(img_h**2 + img_w**2)
    min_perim = min_perim_ratio * diagonal
    
    filtered = []
    for (mid, corners) in detections_with_corners:
        if compute_perimeter(corners) < min_perim:
            continue
        if require_convex and not is_convex(corners):
            continue
        if compute_side_ratio(corners) < min_side_ratio:
            continue
        filtered.append((mid, corners))
    
    return filtered
