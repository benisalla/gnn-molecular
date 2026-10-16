"""Small fake molecules so the tests run offline, on CPU, in seconds."""

import pytest
import torch
from torch_geometric.data import Batch, Data


def random_molecule(n_nodes: int, gen: torch.Generator) -> Data:
    """A connected graph (a chain plus a few random bonds), ESOL-shaped features."""
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    for _ in range(n_nodes // 3):
        a, b = torch.randint(0, n_nodes, (2,), generator=gen).tolist()
        if a != b:
            src.append(a)
            dst.append(b)
    # bonds are undirected: store both directions, like MoleculeNet does
    e_idx = torch.tensor([src + dst, dst + src])
    x = torch.randint(0, 10, (n_nodes, 9), generator=gen)
    e_attr = torch.randint(0, 4, (e_idx.size(1) // 2, 3), generator=gen).repeat(2, 1)
    y = x.float().mean().view(1, 1)  # learnable target: depends on the atoms only
    return Data(x=x, edge_index=e_idx, edge_attr=e_attr, y=y)


@pytest.fixture
def graphs() -> list[Data]:
    gen = torch.Generator().manual_seed(0)
    return [random_molecule(int(torch.randint(3, 12, (1,), generator=gen)), gen) for _ in range(40)]


@pytest.fixture
def batch(graphs) -> Batch:
    return Batch.from_data_list(graphs[:6])


def within_graph_perm(batch: Batch, seed: int = 0) -> torch.Tensor:
    """Shuffle atoms inside each molecule; molecules stay in order (PyG expects a sorted batch vector)."""
    gen = torch.Generator().manual_seed(seed)
    start, parts = 0, []
    for n in torch.bincount(batch.batch).tolist():
        parts.append(start + torch.randperm(n, generator=gen))
        start += n
    return torch.cat(parts)


def permute_batch(batch: Batch, perm: torch.Tensor) -> Batch:
    """Renumber the atoms with `perm` (new node i is old node perm[i]); bonds follow."""
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    out = batch.clone()
    out.x = batch.x[perm]
    out.batch = batch.batch[perm]
    out.edge_index = inverse[batch.edge_index]
    return out
