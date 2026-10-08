"""First iteration of the hybrid neural network architecture.

The design follows the conceptual model of the thesis: separate processing
of heterogeneous information with subsequent fusion and explicit modelling
of temporal dynamics.

Two-branch encoder + discrete-time survival head:

1. **Static branch** - an MLP over time-invariant structured features
   (demographics, criminal history, sentence attributes).
2. **Dynamics branch** - an MLP over supervision-period features
   (violations, drug tests, employment, programme attendance).
3. **Gated fusion** - a learnable gate weighs the contribution of each branch.
4. **Temporal head (GRU)** - the fused representation initialises a GRU
   unrolled over 3 annual steps; each step outputs a *hazard*
   h_t = P(arrest in year t | no arrest before t).
   P(recidivism within 3 years) = 1 - prod_t (1 - h_t).

The survival formulation uses the per-year labels of the NIJ dataset
(Recidivism_Arrest_Year1/2/3) instead of a single binary outcome,
which lets the network learn *when* risk materialises, not only *whether*.

``MLPNet`` is included as a plain neural baseline for ablation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_YEARS = 3


def _mlp(in_dim: int, hidden: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers)


class MLPNet(nn.Module):
    """Plain feed-forward baseline over the full feature vector."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.3):
        super().__init__()
        self.body = _mlp(in_dim, hidden, dropout)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)  # logits


class HybridSurvivalNet(nn.Module):
    """Two-branch hybrid network with a discrete-time survival GRU head.

    Ablation switches: ``dynamic_dim=0`` collapses the model to a single
    branch; ``use_gate=False`` replaces gated fusion with plain concatenation.
    """

    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int = 0,
        static_hidden: tuple[int, ...] = (192, 96),
        dynamic_hidden: tuple[int, ...] = (64, 32),
        fused_dim: int = 96,
        dropout: float = 0.3,
        use_gate: bool = True,
    ):
        super().__init__()
        self.static_branch = _mlp(static_dim, static_hidden, dropout)
        self.dynamic_branch = _mlp(dynamic_dim, dynamic_hidden, dropout) if dynamic_dim > 0 else None

        concat_dim = static_hidden[-1] + (dynamic_hidden[-1] if dynamic_dim > 0 else 0)
        self.gate = nn.Sequential(nn.Linear(concat_dim, concat_dim), nn.Sigmoid()) if use_gate else None
        self.fuse = nn.Sequential(nn.Linear(concat_dim, fused_dim), nn.ReLU(), nn.Dropout(dropout))

        self.year_emb = nn.Embedding(N_YEARS, fused_dim)
        self.gru = nn.GRU(fused_dim, fused_dim, batch_first=True)
        self.hazard_head = nn.Linear(fused_dim, 1)

    def forward(self, x_static: torch.Tensor, x_dynamic: torch.Tensor | None = None) -> torch.Tensor:
        """Returns hazard logits of shape ``(batch, 3)``."""
        z = self.static_branch(x_static)
        if self.dynamic_branch is not None:
            z = torch.cat([z, self.dynamic_branch(x_dynamic)], dim=1)
        if self.gate is not None:
            z = self.gate(z) * z
        z = self.fuse(z)

        batch = z.size(0)
        steps = self.year_emb.weight.unsqueeze(0).expand(batch, -1, -1)  # (b, 3, d)
        h0 = z.unsqueeze(0)  # (1, b, d)
        out, _ = self.gru(steps, h0)  # (b, 3, d)
        return self.hazard_head(out).squeeze(-1)  # (b, 3)


def survival_loss(hazard_logits: torch.Tensor, year_events: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of the discrete-time survival model.

    ``year_events[i, t] = 1`` iff the first arrest of person *i* happened
    in year ``t+1``. A person at risk in year t is one not arrested earlier.
    """
    log_h = nn.functional.logsigmoid(hazard_logits)
    log_1mh = nn.functional.logsigmoid(-hazard_logits)

    # at_risk[i, t] = 1 while no event has occurred in years < t
    cum_events = torch.cumsum(year_events, dim=1)
    at_risk = (cum_events - year_events) == 0

    ll = year_events * log_h + (1 - year_events) * log_1mh
    return -(ll * at_risk).sum(dim=1).mean()


def prob_within_horizon(hazard_logits: torch.Tensor) -> torch.Tensor:
    """P(event within 3 years) = 1 - prod_t (1 - h_t)."""
    h = torch.sigmoid(hazard_logits)
    return 1.0 - torch.prod(1.0 - h, dim=1)


def to_tensor(a: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
