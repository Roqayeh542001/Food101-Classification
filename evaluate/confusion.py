import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from data.data_setup import (
    test_dataloader,
    class_names
)

from config import DEVICE

import torch

from models.model_factory import create_models

def evaluate_model(model,
                   model_path,
                   model_name,
                   dataloader,
                   device):
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    sample_images, sample_labels = next(iter(dataloader))

    sample_images = sample_images.to(device)

    model.eval()

    
    print("="*30)
    print(model_name)
    print("="*30)

    print("\n[1] Sample Predictions")
    print("-"*40)

    with torch.inference_mode():
        preds = model(sample_images).argmax(dim=1)

    for i in range(len(sample_images)):
        print(
            f"Image {i:2d} | Pred: {class_names[preds[i].item()]} | True: {class_names[sample_labels[i].item()]}"
        )

    print("\n[2] Accuracy")
    print("-"*40)

    all_preds = []
    all_labels = []

    correct, total = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)

            preds = logits.argmax(dim=1).cpu()

            correct += (preds == y).sum().item()

            total += y.size(0)

            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    accuracy = correct / total * 100

    print(f"Accuracy = {accuracy:.2f}%")

    print("\n[3] Confusion Matrix")
    print("-"*40)

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nConfusion Matrix:\n", cm)
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    figures_dir = ROOT / "results" / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        figures_dir / f"{model_name.lower()}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # Classification Report

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )
    print("\n[4] Classification Report")
    print("-"*40)

    print(report)

    reports_dir = ROOT / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(
        reports_dir / f"{model_name}_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)


    # Prediction Confidence

    print("\n[5] Prediction Confidence")
    print("-"*40)

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)

            logits = model(images)

            probs = torch.softmax(logits, dim=1)

            for i in range(min(5, len(images))):
                pred = probs[i].argmax().item()
                conf = probs[i][pred].item()

                print(f"Image {i}: True: {class_names[labels[i].item()]} | Pred: {class_names[pred]} | Confidence: ({conf*100:.2f}%)")

            break
    SHOW_PARAMS = False

    if SHOW_PARAMS:
        for name, param in model.named_parameters():
            print(name, param.shape)

    
    history = checkpoint["history"]
    
    return history

def main():
    models = create_models()

    for model_name, info in models.items():
        model = info["model"]
        model_path = info["checkpoint"]
        evaluate_model(model,
                    model_path,
                    model_name,
                    test_dataloader,
                    DEVICE)

if __name__ == "__main__":
    main()