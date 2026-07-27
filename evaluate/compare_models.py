import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
import matplotlib.pyplot as plt

from data.data_setup import class_names

from models.model_factory import create_models

HISTORY_DIR = ROOT / "results" / "history"
FIGURES_DIR = ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Create models
models = create_models()

# Load history
histories = {
    "AlexNet": torch.load(HISTORY_DIR / "alexnet_history.pth"),
    "GoogLeNet": torch.load(HISTORY_DIR / "googlenet_history.pth"),
    "ResNet50": torch.load(HISTORY_DIR / "resnet50_history.pth"),
    "FoodNet": torch.load(HISTORY_DIR / "foodnet_history.pth"),
    "ConvNeXt": torch.load(HISTORY_DIR / "convnext_history.pth")
}


# Collect information

results = []

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for model_name, info in models.items():
    model = info["model"]
    history = histories[model_name]

    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)

    best_acc = history["best_acc"]

    best_epoch = history["best_epoch"]

    avg_epoch_time = sum(history["epoch_time"]) / len(history["epoch_time"])

    total_time = sum(history["epoch_time"])

    results.append({
        "Model": model_name,
        "Total Param": total_params,
        "Trainable Param": trainable_params,
        "Best Accuracy": best_acc,
        "Best Epoch": best_epoch,
        "Avg Epoch Time (s)": avg_epoch_time,
        "Total Training Time (s)": total_time
    })


df = pd.DataFrame(results)

print(df)

# Save table

df.to_csv(
    ROOT / "results" / "model_comparison.csv",
    index=False
)


# Bar chart - Accuracy

plt.figure(figsize=(8,5))

plt.bar(
    df["Model"],
    df["Best Accuracy"]
)

plt.ylabel("Accuracy (%)")

plt.title("Best Test Accuracy")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "compare_accuracy.png",
    dpi=300
)

plt.show()


# Bar chart - Parameters


plt.figure(figsize=(8,5))

plt.bar(
    df["Model"],
    df["Total Param"]
)

plt.ylabel("Parameters")

plt.title("Number of Total Parameters")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "compare_total_parameters.png",
    dpi=300
)

plt.show()

# Trainable Parameters

plt.figure(figsize=(8,5))

plt.bar(
    df["Model"],
    df["Trainable Param"]
)

plt.ylabel("Trainable Parameters")

plt.title("Number of Trainable Parameters")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "compare_trainable_parameters.png",
    dpi=300
)

plt.show()

print("\nComparison Finished Successfully.")