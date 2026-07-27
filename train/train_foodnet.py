import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from config import(
    DEVICE,
    NUM_EPOCHS
)

from data.data_setup import (
    train_dataloader,
    test_dataloader,
    class_names
)

from utils.engine import train

from models.foodnet import FoodNet

from utils.plot_results import plot_results



def main():
    model = FoodNet(
        num_classes=len(class_names)
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-4
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("trainable_parameters:", trainable_params)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


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
        checkpoint_dir / "foodnet_history.pth"
    )

    print("FoodNet history saved successfully.")

    return results

if __name__ == "__main__":
    main()