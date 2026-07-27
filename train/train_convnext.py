import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)

import torch
from torch import nn

from data.data_setup import (class_names,
                             train_dataloader,
                             test_dataloader)

from config import DEVICE, NUM_EPOCHS

from utils.engine import train


def main():
    model = convnext_tiny (
        weights = ConvNeXt_Tiny_Weights.DEFAULT
    )

    for param in model.parameters ():
        param.requires_grad = False

    # Fine tuning
    for param in model.features[7].parameters():
        param.requires_grad = True

    model.classifier[2] = nn.Linear(768, len(class_names))

    for param in model.classifier.parameters():
        param.requires_grad = True


    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=None,
        epochs=NUM_EPOCHS,
        device=DEVICE
    )

    checkpoint_dir = ROOT / "results" / "history"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        results,
        checkpoint_dir / "convnext_history.pth"
    )

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Trainable parameters:", trainable)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    return results


if __name__ == "__main__":
    main()