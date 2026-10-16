# Training guide

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

This installs PyTorch, PyTorch Geometric, RDKit and matplotlib. ESOL is small
(1,128 molecules), so a CPU is enough. For a GPU, first install the matching
PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/), then
run the command above.

## Train

```bash
mgnn-train --model gcn
mgnn-train --model mpnn
mgnn-train --model dcnn
mgnn-train --model gin          # also: gat, sage, attentivefp
```

The first run downloads ESOL into `data/`. Each run:

1. cleans the dataset (1,127 molecules) and splits it 80 / 10 / 10 with a fixed seed,
2. trains with Adam and an MSE loss, evaluating on validation after every epoch,
3. keeps the epoch with the **lowest validation loss** in `best.pt`,
4. stops early if validation has not improved for `--patience` epochs,
5. reloads `best.pt` and reports train / val / test metrics.

Everything is written to `checkpoints/<model>/` (or `--output-dir`):

| File | Content |
|------|---------|
| `best.pt` | weights + the full config (enough to rebuild the model) |
| `metrics.json` | MAE, MSE, RMSE, R² of `best.pt` on train / val / test |
| `history.json` | every epoch's train and val metrics |
| `curves.png` | training and validation curves |

## Options

Every option has a sensible default per model; pass only what you want to change.

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `gcn` | `gcn`, `mpnn`, `dcnn`, `gin`, `gat`, `sage` or `attentivefp` |
| `--epochs` | 300 | maximum epochs |
| `--patience` | 100 | early-stopping patience in epochs, `0` turns it off |
| `--lr` | 5e-4 | Adam learning rate |
| `--hidden`, `--layers`, `--dropout`, `--batch-size` | per model, below | model size and regularisation |
| `--hops` | 2 | DCNN only: diffusion hops |
| `--heads` | 4 | GAT only: attention heads (`hidden` must be divisible by it) |
| `--seed` | 42 | split shuffle and weight init |
| `--device` | auto | `cpu`, `cuda`, `mps` or `auto` |
| `--data-dir` | `data` | where ESOL is stored |
| `--output-dir` | `checkpoints/<model>` | where results go |

Per-model defaults (GCN, MPNN, DCNN from the original notebooks):

| Model | hidden | layers | dropout | batch size | bond features |
|-------|-------:|-------:|--------:|-----------:|:-------------:|
| gcn | 16 | 4 | 0.0 | 64 | no |
| mpnn | 32 | 6 | – | 64 | yes |
| dcnn | 128 (MLP head) | 4 | 0.1 | 32 | no |
| gin | 64 | 4 | 0.1 | 64 | yes |
| gat | 64 | 4 | 0.1 | 64 | yes |
| sage | 64 | 4 | 0.0 | 64 | no |
| attentivefp | 200 | 2 | 0.2 | 64 | yes |

Examples:

```bash
mgnn-train --model dcnn --hops 4 --output-dir checkpoints/dcnn-4hops
mgnn-train --model mpnn --hidden 64 --layers 4 --epochs 500
mgnn-train --model gcn --seed 1 --output-dir checkpoints/gcn-seed1   # a second seed
```

## From Python

```python
from mgnn.config import Config
from mgnn.train import fit

final = fit(Config.for_model("mpnn", epochs=100))
print(final["test"])     # {'loss': ..., 'mae': ..., 'mse': ..., 'rmse': ..., 'r2': ...}
```

`fit(cfg, graphs=my_graphs)` trains on your own list of PyG `Data` objects
instead of ESOL. Each needs `x` (9 features per node), `edge_index`, `y` and,
for models that use bonds, `edge_attr` (3 features per edge); change `node_dim` / `edge_dim`
in the config for other sizes.

## Reading the metrics

Solubility is in `log(mol/L)`, so an RMSE of 1.0 means a typical error of
about a factor of 10 in solubility. R² is 1 for a perfect model and 0 for a
model that always predicts the mean. ESOL is small: with 112 test molecules
the test RMSE moves by several hundredths between seeds, so compare models over
a few `--seed` values before drawing conclusions.
