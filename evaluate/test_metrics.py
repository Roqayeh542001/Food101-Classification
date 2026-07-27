import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from config import DEVICE
from data.data_setup import test_dataloader


import matplotlib.pyplot as plt

import pandas as pd

from models.model_factory import create_models


def evaluate_model(
        model,
        checkpoint_path,
        model_name,
        dataloader,
        device
):
    model = model.to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loss_fn = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)

            loss = loss_fn(logits, y)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()

            total += y.size(0)

    test_loss = total_loss / len(dataloader)

    test_acc = correct / total * 100

    print("="*30)
    print(model_name)
    print("="*30)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    return test_loss, test_acc

def plot_metrics(results):
    figures_dir = ROOT / "results" / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())

    losses = [results[m]["Test Loss"] for m in names]

    accs = [results[m]["Test Accuracy"] for m in names]

    plt.figure(figsize=(8,5))

    bars = plt.bar(names, accs)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom"
        )

    plt.ylabel("Accuracy (%)")

    plt.title("Test Accuracy Comparison")

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "compare_test_accuracy.png",
        dpi=300
    )

    plt.show()


    plt.figure(figsize=(8,5))

    bars = plt.bar(names, losses)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom"
        )

    plt.ylabel("Loss")

    plt.title("Test Loss Comparison")

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "compare_test_loss.png",
        dpi=300
    )

    plt.show()


    df = pd.DataFrame(results).T
    df.to_csv(
        ROOT / "results" / "model_test_metrics.csv"
    )


def main():
    models = create_models()

    results = {}

    for model_name, info in models.items():
        loss, acc = evaluate_model(
            model=info["model"],
            checkpoint_path=info["checkpoint"],
            model_name=model_name,
            dataloader=test_dataloader,
            device=DEVICE
        )

        results[model_name] = {
            "Test Loss": loss,
            "Test Accuracy": acc
        }

    plot_metrics(results)

    

if __name__ == "__main__":
    main()