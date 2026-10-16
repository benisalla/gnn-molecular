"""Predict aqueous solubility (log mol/L) of molecules with a trained model.

mgnn-predict --checkpoint checkpoints/gcn/best.pt "CCO" "c1ccccc1"
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch

from mgnn.config import Config
from mgnn.data import smiles_to_graph
from mgnn.models import build_model
from mgnn.train import forward, pick_device


def load_model(checkpoint: str | Path, device: str = "auto") -> tuple[nn.Module, Config]:
    """Rebuild the model from the config stored in the checkpoint, then load its weights."""
    dev = pick_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


@torch.no_grad()
def predict(model: nn.Module, smiles: list[str]) -> list[float]:
    device = next(model.parameters()).device
    batch = Batch.from_data_list([smiles_to_graph(s) for s in smiles])
    return forward(model.eval(), batch, device).view(-1).tolist()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Predict ESOL solubility from SMILES.")
    p.add_argument("smiles", nargs="+", help="one or more SMILES strings")
    p.add_argument("--checkpoint", required=True, help="best.pt written by mgnn-train")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    model, _ = load_model(args.checkpoint, args.device)
    for s, value in zip(args.smiles, predict(model, args.smiles)):
        print(f"{value:8.3f}  log(mol/L)  {s}")


if __name__ == "__main__":
    main()
