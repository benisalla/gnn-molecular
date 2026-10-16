"""Three graph neural networks for graph-level regression.

All models share one call signature so the training loop does not care which
one it trains:

    model(x, e_idx, b_idx, e_attr=None) -> [n_graphs, 1]

    x      : [n_nodes, node_dim]   atom features of every graph in the batch
    e_idx  : [2, n_edges]          bonds as (source, target) pairs, COO format
    b_idx  : [n_nodes]             which graph each atom belongs to
    e_attr : [n_edges, edge_dim]   bond features (only the MPNN uses them)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from mgnn.config import Config

# --------------------------------------------------------------------------- GCN


class GCNModel(nn.Module):
    """Stacked graph convolutions, then [max pool ; mean pool] -> linear."""

    def __init__(self, in_feature: int, out_feature: int, n_layers: int = 4, drop_rate: float = 0.0):
        super().__init__()
        # first GCN layer maps atom features to the hidden size, the rest keep it
        dims = [in_feature] + [out_feature] * n_layers
        self.gcns = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(n_layers))
        self.dropout = nn.Dropout(p=drop_rate)
        # regression head: [gmp_size + gap_size] -> 1
        self.ln = nn.Linear(out_feature * 2, 1)

    def forward(self, x, e_idx, b_idx, e_attr=None):
        for gcn in self.gcns:
            x = self.dropout(torch.tanh(gcn(x, e_idx)))

        # Global Pooling: [global max pooling ; global average pooling]
        x = torch.cat([global_max_pool(x, b_idx), global_mean_pool(x, b_idx)], dim=1)
        return self.ln(x)


# -------------------------------------------------------------------------- MPNN


class MPNN(MessagePassing):
    """One message-passing layer.

    message:   m_ij = M(h_i, h_j, e_ij)
    aggregate: a_i  = sum / mean / max over neighbours j of m_ij
    update:    h_i' = U(h_i, a_i)
    """

    def __init__(self, hh_emb: int, he_emb: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.hh_emb = hh_emb
        self.he_emb = he_emb

        # MESSAGE NETWORK: [2*hh_emb + he_emb] -> [hh_emb]
        self.msg_net = nn.Sequential(
            nn.Linear(2 * hh_emb + he_emb, hh_emb),
            nn.LayerNorm(hh_emb),
            nn.ReLU(),
            nn.Linear(hh_emb, hh_emb),
            nn.LayerNorm(hh_emb),
            nn.ReLU(),
        )

        # UPDATE NETWORK: [2*hh_emb] -> [hh_emb]
        self.upd_net = nn.Sequential(
            nn.Linear(2 * hh_emb, hh_emb),
            nn.LayerNorm(hh_emb),
            nn.ReLU(),
            nn.Linear(hh_emb, hh_emb),
            nn.LayerNorm(hh_emb),
            nn.ReLU(),
        )

    def forward(self, x, e_idx, e_attr=None):
        # propagate() calls message() -> aggregate() -> update()
        return self.propagate(edge_index=e_idx, h=x, edge_attr=e_attr)

    def message(self, h_i, h_j, edge_attr=None):
        if edge_attr is None:  # no bond features: use zeros so the input size stays fixed
            edge_attr = h_i.new_zeros(h_i.size(0), self.he_emb)
        return self.msg_net(torch.cat([h_i, h_j, edge_attr], dim=-1))

    def update(self, h_aggr, h):
        return self.upd_net(torch.cat([h, h_aggr], dim=-1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hh_emb={self.hh_emb}, aggr={self.aggr})"


class MPNNModel(nn.Module):
    """Linear embed -> n_mpnn x (MPNN + residual) -> mean pool -> linear."""

    def __init__(self, n_mpnn: int, i_dim: int, hh_emb: int, he_emb: int, o_dim: int = 1, aggr: str = "sum"):
        super().__init__()
        self.ln1 = nn.Linear(i_dim, hh_emb)
        self.mpnns = nn.ModuleList(MPNN(hh_emb=hh_emb, he_emb=he_emb, aggr=aggr) for _ in range(n_mpnn))
        self.ln2 = nn.Linear(hh_emb, o_dim)

    def forward(self, x, e_idx, b_idx, e_attr=None):
        x = self.ln1(x)
        for mpnn in self.mpnns:
            x = x + mpnn(x=x, e_idx=e_idx, e_attr=e_attr)  # residual connection
        x = global_mean_pool(x, b_idx)
        return self.ln2(x)


# -------------------------------------------------------------------------- DCNN


class DCNN(nn.Module):
    """One diffusion-convolution layer: H = f( sum_k  W_k ⊙ (P^k X) ).

    P^k X averages features over k-step random walks, so each hop k gets its
    own per-feature weight W_k.
    """

    def __init__(self, n_feature: int, n_hop: int, non_ln: str = "relu"):
        super().__init__()
        self.n_feature = n_feature
        self.n_hop = n_hop
        # one weight per (hop, feature), hops 0..n_hop (hop 0 = the node itself)
        self.W = nn.Parameter(torch.empty(1, n_hop + 1, 1, n_feature))
        nn.init.xavier_uniform_(self.W)
        self.non_ln = getattr(F, non_ln)

    def forward(self, P, X):
        # P: (b_size, n_hop+1, n_node, n_node), X: (b_size, n_node, n_feature)
        PX = torch.matmul(P, X.unsqueeze(1))  # (b_size, n_hop+1, n_node, n_feature)
        H = self.non_ln(PX * self.W)
        return H.sum(dim=1)  # (b_size, n_node, n_feature)


class GDCNNModel(nn.Module):
    """BatchNorm -> n_dcnn x (DCNN + residual) -> mean pool -> MLP."""

    def __init__(self, n_hop: int, n_hide: int, n_feature: int, n_dcnn: int, non_ln: str = "relu", d_rate: float = 0.0):
        super().__init__()
        self.n_hop = n_hop
        self.feature_norm = nn.BatchNorm1d(n_feature)
        self.dcnns = nn.ModuleList(DCNN(n_feature=n_feature, n_hop=n_hop, non_ln=non_ln) for _ in range(n_dcnn))
        self.fc = nn.Sequential(
            nn.Linear(n_feature, n_hide),
            nn.ReLU(),
            nn.BatchNorm1d(n_hide),
            nn.Dropout(p=d_rate),
            nn.Linear(n_hide, 1),
        )

    def forward(self, x, e_idx, b_idx, e_attr=None):
        A = to_dense_adj(edge_index=e_idx, batch=b_idx)  # (b_size, n_node, n_node)
        P = self.create_diffusion_kernel(A, self.n_hop)  # (b_size, n_hop+1, n_node, n_node)

        x = self.feature_norm(x)
        x, mask = to_dense_batch(x, batch=b_idx)  # (b_size, n_node, n_feature), padded

        for dcnn in self.dcnns:
            x = x + dcnn(P, x)  # residual connection

        # mean over the real atoms only: padding rows must not count
        return self.fc(global_mean_pool(x[mask], b_idx))

    @staticmethod
    def create_diffusion_kernel(A: torch.Tensor, k: int) -> torch.Tensor:
        """Pi = [P^0, P^1, ..., P^k] with P = D^-1 A and D = degree(A) + I.

        D is diagonal, so D^-1 A is just each row of A divided by (degree + 1).
        The +1 keeps padding rows (degree 0) finite.
        """
        b_size, n_node, _ = A.shape
        P = A / (A.sum(dim=-1, keepdim=True) + 1.0)
        Pi = [torch.eye(n_node, device=A.device).expand(b_size, n_node, n_node)]  # P^0 = I
        for _ in range(k):
            Pi.append(torch.matmul(Pi[-1], P))
        return torch.stack(Pi, dim=1)


# ---------------------------------------------------------------------- factory


def build_model(cfg: Config) -> nn.Module:
    """The model named in `cfg`, built from its hyperparameters."""
    torch.manual_seed(cfg.seed)
    if cfg.model == "gcn":
        return GCNModel(cfg.node_dim, cfg.hidden, n_layers=cfg.layers, drop_rate=cfg.dropout)
    if cfg.model == "mpnn":
        return MPNNModel(n_mpnn=cfg.layers, i_dim=cfg.node_dim, hh_emb=cfg.hidden, he_emb=cfg.edge_dim)
    if cfg.model == "dcnn":
        return GDCNNModel(
            n_hop=cfg.hops, n_hide=cfg.hidden, n_feature=cfg.node_dim, n_dcnn=cfg.layers, d_rate=cfg.dropout
        )
    raise ValueError(f"unknown model {cfg.model!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
