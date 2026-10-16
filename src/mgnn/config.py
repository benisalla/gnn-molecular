"""All hyperparameters live here.

One frozen dataclass per run. It is saved inside every checkpoint, so a model
can always be rebuilt (and a run reproduced) from the checkpoint alone.
"""

from dataclasses import asdict, dataclass
from typing import Any

MODELS = ("gcn", "mpnn", "dcnn")

# Defaults taken from the original notebooks (see legacy/).
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "gcn": {"hidden": 16, "layers": 4, "batch_size": 64, "dropout": 0.0},
    "mpnn": {"hidden": 32, "layers": 6, "batch_size": 64, "dropout": 0.0},
    "dcnn": {"hidden": 128, "layers": 4, "hops": 2, "batch_size": 32, "dropout": 0.1},
}


@dataclass(frozen=True, slots=True)
class Config:
    model: str = "gcn"  # "gcn" | "mpnn" | "dcnn"

    # data
    data_dir: str = "data"  # ESOL is downloaded here on first run
    seed: int = 42  # controls the train/val/test shuffle and the weight init

    # model
    node_dim: int = 9  # atom features produced by torch_geometric.utils.from_smiles
    edge_dim: int = 3  # bond features (only the MPNN uses them)
    hidden: int = 16
    layers: int = 4
    hops: int = 2  # DCNN only: diffusion steps P^0 .. P^hops
    dropout: float = 0.0

    # training
    epochs: int = 300
    batch_size: int = 64
    lr: float = 5e-4
    patience: int = 100  # stop when val loss has not improved for this many epochs (0 = off)
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    output_dir: str = "checkpoints/gcn"

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if self.hidden <= 0 or self.layers <= 0:
            raise ValueError("hidden and layers must be positive")
        if self.hops < 0:
            raise ValueError("hops must be >= 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def for_model(cls, model: str, **overrides: Any) -> "Config":
        """Config with the notebook defaults for `model`, then any overrides."""
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}")
        values = {"model": model, "output_dir": f"checkpoints/{model}", **MODEL_DEFAULTS[model]}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Config":
        return cls(**values)
