import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt

HISTORY_DIR = ROOT / "results" / "history"
FIGURES_DIR = ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

histories = {
    "AlexNet": torch.load(HISTORY_DIR / "alexnet_history.pth"),
    "GoogLeNet": torch.load(HISTORY_DIR / "googlenet_history.pth"),
    "ResNet50": torch.load(HISTORY_DIR / "resnet50_history.pth"),
    "FoodNet": torch.load(HISTORY_DIR / "foodnet_history.pth"),
    "ConvNeXt": torch.load(HISTORY_DIR / "convnext_history.pth")
}

def draw(metric_name,
         ylabel,
         save_name):
    plt.figure(figsize=(8,6))

    for model_name, history in histories.items():
        epochs = range(
            1,
            len(history[metric_name]) + 1
        )

        plt.plot(
            epochs,
            history[metric_name],
            marker="o",
            linewidth=2,
            label=model_name
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.title(f"Comparison of {ylabel}")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / save_name,
        dpi=300
    )

    plt.show()


draw(
    "train_acc",
    "Train Accuracy (%)",
    "compare_train_accuracy.png"
)

draw(
    "test_acc",
    "Test Accuracy (%)",
    "compare_test_accuracy.png"
)

draw(
    "train_loss",
    "Train Loss",
    "compare_train_loss.png"
)

draw(
    "test_loss",
    "Test Loss",
    "compare_test_loss.png"
)

draw(
    "epoch_time",
    "Epoch Time (s)",
    "compare_epoch_time.png"
)

print("\nAll figure saved successfully!")