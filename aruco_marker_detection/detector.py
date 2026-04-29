"""
detector.py - ArUco detector creation and raw detection.

The detector uses OpenCV's built-in ArUco detection with
DICT_ARUCO_MIP_36h12 dictionary. Parameters are kept at 
OpenCV defaults unless explicitly overridden.
"""

import cv2
import numpy as np


def create_detector(corner_method="subpix"):
    """
    Create an ArUco detector.
    
    Args:
        corner_method: "subpix" or "none"
            - "subpix": sub-pixel corner refinement (recommended)
            - "none": no corner refinement
    
    Returns:
        cv2.aruco.ArucoDetector
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    params = cv2.aruco.DetectorParameters()
    
    if corner_method == "subpix":
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
    else:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    
    # All other params = OpenCV defaults
    # adaptiveThreshWinSizeMin = 3
    # adaptiveThreshWinSizeMax = 23
    # adaptiveThreshWinSizeStep = 10
    # adaptiveThreshConstant = 7
    # minMarkerPerimeterRate = 0.03
    # maxMarkerPerimeterRate = 4.0
    # errorCorrectionRate = 0.6
    
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_on_gray(detector, gray):
    """
    Run ArUco detection on a grayscale image.
    
    Args:
        detector: cv2.aruco.ArucoDetector
        gray: grayscale image (numpy array, uint8)
    
    Returns:
        list of (marker_id, corners_4x2_array)
        Each corners_4x2 has shape (4, 2): [top-left, top-right, bottom-right, bottom-left]
        These are in the marker's canonical orientation (defined by bit encoding).
    """
    corners, ids, rejected = detector.detectMarkers(gray)
    
    results = []
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            mid = int(ids[i][0])
            pts = corners[i][0]  # shape (4, 2)
            results.append((mid, pts))
    
    return results


def detect_image(detector, image_path, preprocess_fn=None):
    """
    Full detection pipeline for a single image.
    
    Args:
        detector: cv2.aruco.ArucoDetector
        image_path: path to image file
        preprocess_fn: optional function(gray) -> gray
            If provided, apply to grayscale before detection.
    
    Returns:
        list of (marker_id, top_left_x, top_left_y)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if preprocess_fn is not None:
        gray = preprocess_fn(gray)
    
    raw = detect_on_gray(detector, gray)
    
    results = []
    for (mid, corners) in raw:
        tl_x = float(corners[0][0])
        tl_y = float(corners[0][1])
        results.append((mid, tl_x, tl_y))
    
    return results