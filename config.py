from pathlib import Path
import torch

# Paths
ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"

FOOD101_PATH = DATA_DIR / "food-101"

IMAGES_DIR = FOOD101_PATH / "images"

META_DIR = FOOD101_PATH / "meta"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

RESULTS_DIR = ROOT_DIR / "results"


# Dataset
SELECTED_CLASSES = [
    "pizza",
    "sushi",
    "steak",
]

NUM_CLASSES = len(SELECTED_CLASSES)

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 4


# Training
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
