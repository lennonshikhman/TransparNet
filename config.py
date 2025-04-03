# config.py

import os
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global settings
random_seed = 6904

# Image & dataset config
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_WORKERS = 0

# Training config
NUM_EPOCHS = 20
WARMUP_EPOCHS = 5
INITIAL_LR = 0.01
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# PCA config
PCA_VARIANCE_PERCENT = 0.8

# Dataset URLs and paths
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATASET_DIR = Path("./imagenette2-320")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# Output paths
OUTPUT_DIR = Path("outputs")
TREE_VISUALIZATION_PATH = OUTPUT_DIR / "decision_tree.svg"
HEATMAP_DIR = OUTPUT_DIR / "feature_heatmap_grids"
HIGHLIGHTED_DIR = OUTPUT_DIR / "highlighted_images"
COMPOSITE_DIR = OUTPUT_DIR / "composite_images"

# Class names for Imagenette
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]

# Ensure output directories exist
for dir_path in [OUTPUT_DIR, HEATMAP_DIR, HIGHLIGHTED_DIR, COMPOSITE_DIR]:
    os.makedirs(dir_path, exist_ok=True)
