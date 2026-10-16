"""Train one model on ESOL, keep the best checkpoint, report test metrics.

    mgnn-train --model gcn
    mgnn-train --model mpnn --epochs 500
    mgnn-train --model dcnn --hops 3 --output-dir checkpoints/dcnn-3hops

Writes to --output-dir:
    best.pt        weights + config of the epoch with the lowest validation loss
    history.json   per-epoch train / val metrics
    curves.png     the training-curve figure
    metrics.json   final metrics of best.pt on train / val / test
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

from mgnn.config import MODELS, Config
from mgnn.data import load_esol, make_loaders, split_dataset
from mgnn.metrics import compute_metrics, plot_history
from mgnn.models import build_model, count_parameters


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    """One forward pass on a PyG batch. Features arrive as integers, models want floats."""
    e_attr = batch.edge_attr.float().to(device) if batch.edge_attr is not None else None
    return model(
        x=batch.x.float().to(device),
        e_idx=batch.edge_index.to(device),
        b_idx=batch.batch.to(device),
        e_attr=e_attr,
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict[str, float]:
    """Train for one pass over `loader` if `optimizer` is given, else just evaluate."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            y = batch.y.float().to(device).view(-1)
            pred = forward(model, batch, device).view(-1)
            loss = loss_fn(pred, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds.append(pred.detach().cpu())
            targets.append(y.cpu())

    metrics = compute_metrics(torch.cat(preds), torch.cat(targets))
    return {"loss": total_loss / len(loader), **metrics}


def fit(cfg: Config, graphs: list[Data] | None = None, verbose: bool = True) -> dict:
    """Full run. `graphs` defaults to ESOL; tests pass a small synthetic set instead."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)

    graphs = load_esol(cfg.data_dir) if graphs is None else graphs
    train_set, val_set, test_set = split_dataset(graphs, seed=cfg.seed)
    train_loader, val_loader, test_loader = make_loaders(train_set, val_set, test_set, cfg.batch_size)

    model = build_model(cfg).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    log = partial(print, flush=True) if verbose else (lambda *a, **k: None)
    log(f"{cfg.model.upper()} | {count_parameters(model):,} parameters | device {device}")
    log(f"graphs: train {len(train_set)} | val {len(val_set)} | test {len(test_set)}")

    history: dict[str, list] = {"train": [], "val": []}
    best_val, best_epoch = float("inf"), -1
    for epoch in range(cfg.epochs):
        train_m = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_m = run_epoch(model, val_loader, loss_fn, device)
        history["train"].append(train_m)
        history["val"].append(val_m)

        if val_m["loss"] < best_val:
            best_val, best_epoch = val_m["loss"], epoch
            torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "epoch": epoch}, out / "best.pt")

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            log(
                f"epoch {epoch:4d} | train loss {train_m['loss']:.4f} | "
                f"val loss {val_m['loss']:.4f}  rmse {val_m['rmse']:.3f}  r2 {val_m['r2']:.3f}"
            )
        if cfg.patience and epoch - best_epoch >= cfg.patience:
            log(f"early stop: no val improvement for {cfg.patience} epochs")
            break

    # final numbers come from the best checkpoint, not the last epoch
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    final = {
        "best_epoch": best_epoch,
        **{
            name: run_epoch(model, loader, loss_fn, device)
            for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader))
        },
    }
    (out / "history.json").write_text(json.dumps(history))
    (out / "metrics.json").write_text(json.dumps(final, indent=2))
    plot_history(history, out / "curves.png", title=f"{cfg.model.upper()} on ESOL")

    t = final["test"]
    log(f"best epoch {best_epoch} | test rmse {t['rmse']:.3f}  mae {t['mae']:.3f}  r2 {t['r2']:.3f}")
    log(f"saved {out / 'best.pt'}, metrics.json, history.json, curves.png")
    return final


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Train a GNN on ESOL molecular solubility.")
    p.add_argument("--model", choices=MODELS, default="gcn")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--hidden", type=int, help="hidden size (GCN/MPNN) or MLP head size (DCNN)")
    p.add_argument("--layers", type=int, help="number of GCN / MPNN / DCNN layers")
    p.add_argument("--hops", type=int, help="DCNN only: diffusion hops")
    p.add_argument("--dropout", type=float)
    p.add_argument("--patience", type=int, help="early-stopping patience in epochs, 0 = off")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    args = vars(p.parse_args(argv))
    return Config.for_model(args.pop("model"), **args)


def main(argv=None) -> None:
    fit(parse_args(argv))


if __name__ == "__main__":
    main()
