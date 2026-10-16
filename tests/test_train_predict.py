import json

import pytest
import torch

from mgnn.config import MODELS, Config
from mgnn.predict import load_model, predict
from mgnn.train import fit, parse_args


@pytest.mark.parametrize("name", MODELS)
def test_fit_lowers_loss_and_writes_outputs(name, graphs, tmp_path):
    cfg = Config.for_model(name, epochs=40, batch_size=8, lr=5e-3, output_dir=str(tmp_path), device="cpu")
    final = fit(cfg, graphs=graphs, verbose=False)

    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history["train"]) == 40
    assert history["train"][-1]["loss"] < history["train"][0]["loss"]
    for f in ("best.pt", "metrics.json", "curves.png"):
        assert (tmp_path / f).exists()
    assert set(final) == {"best_epoch", "train", "val", "test"}
    assert final["val"]["loss"] == pytest.approx(min(r["loss"] for r in history["val"]), rel=1e-4)


def test_early_stopping(graphs, tmp_path):
    cfg = Config.for_model("gcn", epochs=500, lr=0.0, patience=5, output_dir=str(tmp_path), device="cpu")
    fit(cfg, graphs=graphs, verbose=False)  # lr = 0: nothing improves after epoch 0
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history["train"]) == 6


def test_checkpoint_reloads_the_same_model(graphs, tmp_path):
    cfg = Config.for_model("mpnn", epochs=3, output_dir=str(tmp_path), device="cpu")
    fit(cfg, graphs=graphs, verbose=False)
    model, loaded_cfg = load_model(tmp_path / "best.pt", device="cpu")
    assert loaded_cfg == cfg
    saved = torch.load(tmp_path / "best.pt", weights_only=True)["model"]
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved[k])


def test_predict_from_smiles(graphs, tmp_path):
    fit(Config.for_model("gcn", epochs=2, output_dir=str(tmp_path), device="cpu"), graphs=graphs, verbose=False)
    model, _ = load_model(tmp_path / "best.pt", device="cpu")
    values = predict(model, ["CCO", "c1ccccc1", "CC(=O)O"])
    assert len(values) == 3 and all(isinstance(v, float) for v in values)
    assert predict(model, ["CCO"])[0] == pytest.approx(values[0], abs=1e-5)  # batching does not matter


def test_cli_parsing_uses_model_defaults():
    cfg = parse_args(["--model", "dcnn", "--epochs", "7"])
    assert cfg.model == "dcnn" and cfg.epochs == 7 and cfg.batch_size == 32
    assert cfg.output_dir == "checkpoints/dcnn"
