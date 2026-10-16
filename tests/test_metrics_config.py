import math

import pytest
import torch

from mgnn.config import Config
from mgnn.metrics import compute_metrics


def test_perfect_prediction():
    y = torch.tensor([1.0, 2.0, 3.0])
    m = compute_metrics(y, y)
    assert m == {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 1.0}


def test_known_values():
    m = compute_metrics(torch.tensor([2.0, 2.0, 2.0, 2.0]), torch.tensor([1.0, 2.0, 3.0, 2.0]))
    assert m["mae"] == pytest.approx(0.5)
    assert m["mse"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(math.sqrt(0.5))
    assert m["r2"] == pytest.approx(0.0)  # predicting the mean scores exactly 0


def test_accepts_column_vectors():
    m = compute_metrics(torch.zeros(5, 1), torch.arange(5.0).view(5, 1))
    assert m["mae"] == pytest.approx(2.0)


def test_config_defaults_per_model():
    assert Config.for_model("mpnn").layers == 6
    assert Config.for_model("dcnn").batch_size == 32
    assert Config.for_model("gcn").output_dir == "checkpoints/gcn"
    assert Config.for_model("gcn", hidden=64, lr=None).hidden == 64  # None = keep default


def test_config_round_trip():
    cfg = Config.for_model("dcnn", hops=3)
    assert Config.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "bad", [{"model": "gnn"}, {"hidden": 0}, {"hops": -1}, {"dropout": 1.0}, {"model": "gat", "hidden": 30, "heads": 4}]
)
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        Config(**bad)
