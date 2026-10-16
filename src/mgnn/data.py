"""ESOL molecules as graphs: load, clean, split, batch.

Every molecule is a graph: atoms are nodes (9 integer features each), bonds are
edges (3 integer features each), and the label is log solubility in mol/L.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, from_smiles


def load_esol(root: str = "data") -> list[Data]:
    """Download (first run only) and return ESOL as a list of graphs, cleaned."""
    return filter_dataset(MoleculeNet(root=root, name="ESOL"))


def has_isolated_nodes(data: Data) -> bool:
    """True if some atom has no bond at all."""
    deg = degree(data.edge_index[0], data.num_nodes)
    return bool((deg == 0).any())


def filter_dataset(dataset) -> list[Data]:
    """Keep molecules with more than one atom and no isolated atoms.

    Single atoms and disconnected ions have nothing to pass messages along.
    On ESOL this removes 1 molecule (1128 -> 1127).
    """
    return [d for d in dataset if d.num_nodes > 1 and not has_isolated_nodes(d)]


def split_dataset(
    graphs: list[Data] | Dataset, seed: int = 42, train: float = 0.8, test: float = 0.1
) -> tuple[list[Data], list[Data], list[Data]]:
    """Shuffle once with `seed`, then cut 80 / 10 / 10 (validation gets the remainder).

    Same proportions as the notebooks. The shuffle matters: the ESOL file is not
    in random order, so a plain slice gives the three splits different molecules.
    """
    n = len(graphs)
    n_train, n_test = int(n * train), int(n * test)
    n_val = n - n_train - n_test
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [graphs[i] for i in order]
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def make_loaders(
    train: list[Data], val: list[Data], test: list[Data], batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """PyG merges each batch of small graphs into one big disconnected graph."""
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


def smiles_to_graph(smiles: str) -> Data:
    """Featurize one SMILES string exactly like MoleculeNet does (via RDKit)."""
    try:
        graph = from_smiles(smiles)
    except Exception as err:  # RDKit raises different errors for different bad inputs
        raise ValueError(f"could not parse SMILES {smiles!r}") from err
    if graph.num_nodes == 0:
        raise ValueError(f"could not parse SMILES {smiles!r}")
    return graph
