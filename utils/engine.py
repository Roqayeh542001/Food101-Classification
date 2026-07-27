import time

import torch
from torch import nn
from tqdm.auto import tqdm

from pathlib import Path

import copy


def train_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    model.train()

    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        output = model(X)

        if hasattr(output, "logits"):
            y_pred = output.logits
        else:
            y_pred = output

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        y_pred_class = torch.argmax(
            torch.softmax(y_pred, dim=1),
            dim=1
        )

        train_acc += ((y_pred_class == y).sum().item() / len(y)) * 100

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            output = model(X)

            if hasattr(output, "logits"):
                y_pred = output.logits
            else:
                y_pred = output

            loss = loss_fn(y_pred, y)

            test_loss += loss.item()

            y_pred_class = torch.argmax(
                torch.softmax(y_pred, dim=1),
                dim=1
            )

            test_acc += ((y_pred_class == y).sum().item() / len(y)) * 100

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device,
        scheduler=None
):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time": []
    }

    best_acc = 0.0
    best_epoch = 0
    best_model_state = None

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    model_name = model.__class__.__name__.lower()

    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["epoch_time"].append(epoch_time)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            print(f"\nNew best model! Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")


    results["best_acc"] = best_acc
    results["best_epoch"] = best_epoch

    if best_model_state is not None:
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": best_model_state,
                "history": results,
                "best_epoch": best_epoch,
                "best_acc": best_acc,
                "training_time": sum(results["epoch_time"])
            },
            f"{checkpoint_dir}/{model_name}_best.pth"
        )

    return results