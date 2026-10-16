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

Every option has a default that comes from the original notebooks; you only
pass what you want to change.

| Flag | GCN | MPNN | DCNN | Meaning |
|------|----:|-----:|-----:|---------|
| `--hidden` | 16 | 32 | 128 | hidden size (DCNN: size of the MLP head) |
| `--layers` | 4 | 6 | 4 | number of graph layers |
| `--hops` | – | – | 2 | diffusion hops (DCNN only) |
| `--dropout` | 0.0 | – | 0.1 | dropout (the MPNN has none) |
| `--batch-size` | 64 | 64 | 32 | molecules per batch |
| `--lr` | 5e-4 | 5e-4 | 5e-4 | Adam learning rate |
| `--epochs` | 300 | 300 | 300 | maximum epochs |
| `--patience` | 100 | 100 | 100 | early-stopping patience, `0` turns it off |
| `--seed` | 42 | 42 | 42 | split shuffle and weight init |
| `--device` | auto | auto | auto | `cpu`, `cuda`, `mps` or `auto` |
| `--data-dir` | data | data | data | where ESOL is stored |
| `--output-dir` | checkpoints/gcn | checkpoints/mpnn | checkpoints/dcnn | where results go |

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
for the MPNN, `edge_attr` (3 features per edge); change `node_dim` / `edge_dim`
in the config for other sizes.

## Reading the metrics

Solubility is in `log(mol/L)`, so an RMSE of 1.0 means a typical error of
about a factor of 10 in solubility. R² is 1 for a perfect model and 0 for a
model that always predicts the mean. ESOL is small: with 112 test molecules
the test RMSE moves by several hundredths between seeds, so compare models over
a few `--seed` values before drawing conclusions.
