"""Regression metrics and the training-curve figure."""

from pathlib import Path

import torch


def compute_metrics(pred, target) -> dict[str, float]:
    """MAE, MSE, RMSE and R² between two 1-D (or [n, 1]) tensors."""
    pred = torch.as_tensor(pred, dtype=torch.float32).flatten()
    target = torch.as_tensor(target, dtype=torch.float32).flatten()

    mae = torch.mean(torch.abs(pred - target))
    mse = torch.mean((pred - target) ** 2)
    ss_total = torch.sum((target - target.mean()) ** 2)
    ss_residual = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_residual / ss_total  # 1 = perfect, 0 = as good as predicting the mean

    return {"mae": mae.item(), "mse": mse.item(), "rmse": mse.sqrt().item(), "r2": r2.item()}


def plot_history(history: dict[str, list[dict[str, float]]], path: str | Path, title: str = "") -> None:
    """Training and validation metrics side by side, like the notebooks' live plot."""
    import matplotlib

    matplotlib.use("Agg")  # write a file, never open a window
    import matplotlib.pyplot as plt

    colors = {"loss": "blue", "mae": "green", "mse": "red", "rmse": "purple", "r2": "brown"}
    names = {"loss": "Loss", "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, split, label in zip(axes, ("train", "val"), ("Training", "Validation")):
        for key, color in colors.items():
            ax.plot([row[key] for row in history[split]], label=f"{label} {names[key]}", color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(f"{label} Metrics")
        ax.legend()
        ax.grid(True)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
