"""
Central configuration for the whole project.

Every script imports its settings from here, so a value is changed in ONE place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"          # put the extracted Kaggle folder here (or pass --data)
CACHE_DIR = ROOT / "cache"           # preprocessed image arrays are stored here
RESULTS_DIR = ROOT / "results"       # all metrics, tables and figures
MODELS_DIR = ROOT / "models"         # the final trained CNN used by the demo

# ---------------------------------------------------------------- data
CLASS_NAMES = {0: "genuine", 1: "counterfeit"}   # counterfeit is the POSITIVE class
# Folder names that are recognised as class labels (lower-case).
LABEL_ALIASES = {
    "fake": 1, "counterfeit": 1, "falsified": 1,
    "real": 0, "genuine": 0, "authentic": 0, "original": 0,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

# ---------------------------------------------------------------- preprocessing
IMG_SIZE = 128            # final input size: 128 x 128 x 3 (see report, section "Image size")
JPEG_QUALITY = 75         # every image is re-encoded through the SAME JPEG bottleneck
TARGET_L_MEAN = 55.0      # target brightness (LAB lightness, 0-100 scale)
TARGET_L_STD = 20.0       # target contrast (std of LAB lightness)

# Near-duplicate grouping (perceptual hash). Two images whose 64-bit pHash differs
# in <= PHASH_THRESHOLD bits (under any 90-degree rotation) are treated as the
# same underlying package/photo and are always kept in the same split.
PHASH_THRESHOLD = 10

# ---------------------------------------------------------------- evaluation
SEED = 42
N_FOLDS = 5               # 5-fold cross-validation: every image is tested exactly once
VAL_FRACTION_FOLDS = 6    # inside each training part, 1/6 of groups -> validation set

# ---------------------------------------------------------------- CNN training
BATCH_SIZE = 32
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10
