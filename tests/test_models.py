import pytest
import torch
from conftest import permute_batch, within_graph_perm

from mgnn.config import MODELS, Config
from mgnn.models import DCNN, MPNN, GDCNNModel, build_model


def run(model, batch):
    return model(x=batch.x.float(), e_idx=batch.edge_index, b_idx=batch.batch, e_attr=batch.edge_attr.float())


@pytest.mark.parametrize("name", MODELS)
def test_output_shape(name, batch):
    model = build_model(Config.for_model(name)).eval()
    assert run(model, batch).shape == (batch.num_graphs, 1)


@pytest.mark.parametrize("name", MODELS)
def test_graph_output_is_permutation_invariant(name, batch):
    """Renumbering the atoms of a molecule must not change its prediction."""
    model = build_model(Config.for_model(name)).eval()
    perm = within_graph_perm(batch, seed=1)
    torch.testing.assert_close(run(model, batch), run(model, permute_batch(batch, perm)), atol=1e-4, rtol=1e-4)


def test_mpnn_layer_is_permutation_equivariant(batch):
    """A layer's node outputs must move with the nodes: f(P·x) = P·f(x)."""
    layer = MPNN(hh_emb=9, he_emb=3).eval()
    perm = within_graph_perm(batch, seed=2)
    out = layer(batch.x.float(), batch.edge_index, batch.edge_attr.float())
    p = permute_batch(batch, perm)
    out_p = layer(p.x.float(), p.edge_index, batch.edge_attr.float())  # edges keep their order
    torch.testing.assert_close(out_p, out[perm], atol=1e-5, rtol=1e-5)


BOND_MODELS = ["mpnn", "gin", "gat", "attentivefp"]


@pytest.mark.parametrize("name", BOND_MODELS)
def test_bond_models_use_bond_features(name, batch):
    model = build_model(Config.for_model(name)).eval()
    changed = batch.clone()
    changed.edge_attr = batch.edge_attr + 1
    assert not torch.allclose(run(model, batch), run(model, changed))


@pytest.mark.parametrize("name", BOND_MODELS)
def test_bond_models_run_without_bond_features(name, batch):
    model = build_model(Config.for_model(name)).eval()
    out = model(x=batch.x.float(), e_idx=batch.edge_index, b_idx=batch.batch)
    assert out.shape == (batch.num_graphs, 1)


def test_diffusion_kernel():
    # path graph 0 - 1 - 2
    A = torch.tensor([[[0.0, 1, 0], [1, 0, 1], [0, 1, 0]]])
    Pi = GDCNNModel.create_diffusion_kernel(A, k=3)
    assert Pi.shape == (1, 4, 3, 3)
    torch.testing.assert_close(Pi[0, 0], torch.eye(3))  # P^0 = I
    P = A[0] / (A[0].sum(-1, keepdim=True) + 1)  # D^-1 A with D = deg + I
    torch.testing.assert_close(Pi[0, 1], P)
    torch.testing.assert_close(Pi[0, 3], P @ P @ P)


def test_diffusion_kernel_padding_rows_are_finite():
    A = torch.zeros(2, 4, 4)  # an all-padding graph has degree 0 everywhere
    assert torch.isfinite(GDCNNModel.create_diffusion_kernel(A, k=2)).all()


def test_dcnn_layer_shape():
    layer = DCNN(n_feature=9, n_hop=2)
    P = torch.rand(3, 3, 5, 5)
    assert layer(P, torch.rand(3, 5, 9)).shape == (3, 5, 9)


@pytest.mark.parametrize("name", MODELS)
def test_prediction_does_not_depend_on_other_graphs_in_batch(name, graphs):
    """Padding (DCNN) or batching must not leak between molecules."""
    from torch_geometric.data import Batch

    model = build_model(Config.for_model(name)).eval()
    small = graphs[0]
    alone = run(model, Batch.from_data_list([small]))
    with_big = run(model, Batch.from_data_list([small, graphs[1], graphs[2]]))
    torch.testing.assert_close(alone[0], with_big[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", MODELS)
def test_every_parameter_gets_a_gradient(name, batch):
    model = build_model(Config.for_model(name)).train()
    run(model, batch).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("name", MODELS)
def test_can_overfit_one_batch(name, batch):
    """If a model cannot memorise 6 molecules, something is wired wrong."""
    model = build_model(Config.for_model(name, dropout=0.0)).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    y = batch.y.view(-1)
    first = None
    for _ in range(200):
        loss = torch.nn.functional.mse_loss(run(model, batch).view(-1), y)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_same_seed_same_weights():
    a = build_model(Config.for_model("gcn", seed=3))
    b = build_model(Config.for_model("gcn", seed=3))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
