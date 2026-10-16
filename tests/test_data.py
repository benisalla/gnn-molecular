import pytest
import torch
from torch_geometric.data import Data

from mgnn.data import filter_dataset, has_isolated_nodes, make_loaders, smiles_to_graph, split_dataset


def chain(n: int) -> Data:
    src = list(range(n - 1))
    e_idx = torch.tensor([src + [i + 1 for i in src], [i + 1 for i in src] + src], dtype=torch.long)
    return Data(x=torch.zeros(n, 9, dtype=torch.long), edge_index=e_idx.view(2, -1), y=torch.zeros(1, 1))


def test_has_isolated_nodes():
    assert not has_isolated_nodes(chain(4))
    g = chain(4)
    g.x = torch.zeros(5, 9, dtype=torch.long)  # node 4 has no bond
    assert has_isolated_nodes(g)


def test_filter_dataset_drops_single_atoms_and_isolated_atoms():
    lonely = chain(3)
    lonely.x = torch.zeros(4, 9, dtype=torch.long)
    single = Data(x=torch.zeros(1, 9, dtype=torch.long), edge_index=torch.zeros(2, 0, dtype=torch.long))
    kept = filter_dataset([chain(3), single, lonely, chain(5)])
    assert [g.num_nodes for g in kept] == [3, 5]


def test_split_sizes_disjoint_and_deterministic(graphs):
    train, val, test = split_dataset(graphs, seed=0)
    assert (len(train), len(val), len(test)) == (32, 4, 4)  # 80 / 10 / 10 of 40
    ids = [id(g) for g in train + val + test]
    assert len(set(ids)) == len(graphs)  # every graph exactly once
    again = split_dataset(graphs, seed=0)
    assert [id(g) for g in again[0]] == [id(g) for g in train]
    other = split_dataset(graphs, seed=1)
    assert [id(g) for g in other[0]] != [id(g) for g in train]


def test_loaders_batch_graphs(graphs):
    train_loader, val_loader, _ = make_loaders(*split_dataset(graphs), batch_size=8)
    batch = next(iter(train_loader))
    assert batch.num_graphs == 8
    assert batch.batch.max().item() == 7  # one id per molecule
    assert sum(b.num_graphs for b in val_loader) == 4


def test_smiles_to_graph_matches_esol_features():
    g = smiles_to_graph("CCO")  # ethanol: 3 heavy atoms, 2 bonds
    assert g.x.shape == (3, 9)
    assert g.edge_index.shape == (2, 4)  # each bond stored in both directions
    assert g.edge_attr.shape == (4, 3)


def test_smiles_to_graph_rejects_garbage():
    with pytest.raises(ValueError):
        smiles_to_graph("this is not a molecule")
