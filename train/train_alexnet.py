import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from torchvision.models import (
    alexnet,
    AlexNet_Weights
)
from config import DEVICE, NUM_EPOCHS
from data.data_setup import (
    train_dataloader,
    test_dataloader,
    class_names
)

from utils.engine import train


def main():
    weights = AlexNet_Weights.DEFAULT

    model = alexnet(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    print(model.classifier)

    model.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=len(class_names)
    )

    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-3
    )

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=DEVICE
    )

    checkpoint_dir = ROOT / "results" / "history"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        results,
        checkpoint_dir / "alexnet_history.pth"
    )

    print("AlexNet history saved successfully.")

    return results

if __name__ == "__main__":
    main()