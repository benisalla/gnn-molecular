# Testing guide

```bash
pytest                          # 75 tests, CPU only, no internet (~2-4 min on a laptop)
pytest tests/test_models.py     # one file
pytest -k mpnn -v               # tests whose name contains "mpnn"
ruff check src tests            # lint
```

The tests use small fake molecules (random connected graphs with ESOL-shaped
features, see `tests/conftest.py`), so they need no download. The only test that
touches RDKit parses a few SMILES strings locally.

| File | What it proves |
|------|----------------|
| `test_models.py` | every test runs on all seven models: output shapes; **permutation invariance**; **equivariance** of one MPNN layer; models that use bonds (MPNN, GIN, GAT, AttentiveFP) really use them and still run without them; the diffusion kernel equals `[I, P, P², …]` with `P = D⁻¹A` and stays finite on padding; a molecule's prediction does not depend on the other molecules in its batch; every parameter gets a gradient; each model can memorise one batch; same seed gives the same weights |
| `test_data.py` | isolated-atom detection and filtering; the 80/10/10 split is complete, disjoint and reproducible; loaders batch graphs correctly; SMILES featurization matches ESOL (9 atom and 3 bond features) and bad SMILES raise `ValueError` |
| `test_metrics_config.py` | MAE / MSE / RMSE / R² on hand-computed cases; per-model defaults; a config survives the checkpoint round trip; invalid configs are rejected |
| `test_train_predict.py` | a short training run lowers the loss and writes `best.pt`, `history.json`, `metrics.json`, `curves.png` for each of the seven models; final metrics come from the best epoch; early stopping; a checkpoint reloads into exactly the same weights; `predict()` works on SMILES and does not depend on batching; the CLI picks up model defaults |

## End-to-end check on real data

This is the check that the whole pipeline works on the real dataset:

```bash
mgnn-train --model gcn --epochs 3 --output-dir checkpoints/smoke
mgnn-predict --checkpoint checkpoints/smoke/best.pt "CCO"
```

It downloads ESOL (~100 KB) on the first run, prints
`graphs: train 901 | val 114 | test 112`, trains 3 epochs and prints one
prediction. After 3 epochs the numbers are meaningless; this only checks the
wiring. Real training results are in the README.
