"""Project configuration and default paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "aruco-detection-challenge"
DATA_DIR = DATASET_ROOT / "aruco_data" / "aruco_data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = DATASET_ROOT / "train.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

ML_MODEL_PATH = MODELS_DIR / "ml_filter_model.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_marker_model.pth"
CNN_DATA_DIR = MODELS_DIR / "cnn_data"
DEFAULT_SUBMISSION_PATH = OUTPUTS_DIR / "submission_final.csv"

# ARUCO_MIP_36h12 has 250 marker IDs and a 6x6 payload with minimum
# Hamming distance 12.
ARUCO_DICT_ID = "DICT_ARUCO_MIP_36h12"

# Competition metric parameters.
SIGMA = 0.02
LAMBDA_SPAM = 1.0

# Final pipeline defaults selected from the ablation study.
DEFAULT_PREPROCESS = "gamma_2.5"
USE_HEURISTIC_FILTER = True
