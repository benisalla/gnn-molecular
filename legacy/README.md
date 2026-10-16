# Legacy notebooks

The original Colab notebooks this project grew from, kept as they were for
reference. `src/mgnn/` contains the same models, cleaned up and tested.

| Notebook | Became |
|----------|--------|
| `GCN.ipynb` | `mgnn.models.GCNModel` |
| `MPNN.ipynb` | `mgnn.models.MPNN`, `mgnn.models.MPNNModel`, the permutation tests in `tests/test_models.py` |
| `DCNN.ipynb` | `mgnn.models.DCNN`, `mgnn.models.GDCNNModel` |
| `live_plot`, `compute_metrics`, `trainer`, `validator` (all three) | `mgnn.metrics`, `mgnn.train.run_epoch` |
| `has_isolated_nodes`, `filter_dataset` | `mgnn.data` |

## What changed on the way

- **One interface.** All models take `(x, e_idx, b_idx, e_attr=None)`, so one
  training loop trains all three.
- **MPNN: train on the training set.** The MPNN notebook's training loop called
  `trainer(model, val_loader)`, so it trained on the validation molecules. That
  is why its "validation" R² reaches ~1.0 in the curves; those curves are not a
  held-out score. `mgnn-train` trains on `train` and selects on `val`.
- **MPNN: no debug print.** `aggregate()` printed every call; PyG's built-in
  aggregation does the same sum / mean, so the override was removed.
- **DCNN: padding does not count.** The notebook averaged over the padded dense
  tensor, so a small molecule's mean was diluted by padding zeros and its
  prediction depended on the biggest molecule in the batch. The mean now uses
  the padding mask (`tests/test_models.py` checks this).
- **DCNN: no matrix inverse.** `D` is diagonal, so `D⁻¹A` is computed as each
  row of `A` divided by `(degree + 1)`. Same result, no `torch.inverse`.
- **DCNN: hops.** The notebook trained with `n_hop=1`; the default is now 2.
- **Shuffled split.** The notebooks sliced the dataset in file order, which is
  not random; the split is now shuffled once with a fixed seed.
- **No `torch_scatter`.** Everything uses PyTorch Geometric's built-in ops, so
  installing is one `pip install`.
