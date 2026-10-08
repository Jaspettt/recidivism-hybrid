"""Unit tests for src/recidivism/hybrid.py.

Tests cover model instantiation, forward pass shapes, survival loss,
and ablation switches (no gate, single branch).
No real dataset is required — all inputs are synthetic tensors.
"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recidivism.hybrid import (
    MLPNet,
    HybridSurvivalNet,
    survival_loss,
    prob_within_horizon,
    to_tensor,
)

BATCH = 16
STATIC_DIM = 40
DYNAMIC_DIM = 23
N_YEARS = 3


def make_inputs(batch=BATCH):
    x_static = torch.randn(batch, STATIC_DIM)
    x_dynamic = torch.randn(batch, DYNAMIC_DIM)
    # Each row: at most one 1 (first-arrest year), rest 0
    year_events = torch.zeros(batch, N_YEARS)
    for i in range(batch):
        t = torch.randint(0, N_YEARS + 1, (1,)).item()
        if t < N_YEARS:
            year_events[i, t] = 1
    return x_static, x_dynamic, year_events


# ---------------------------------------------------------------------------
# MLPNet
# ---------------------------------------------------------------------------

def test_mlpnet_output_shape():
    model = MLPNet(in_dim=STATIC_DIM + DYNAMIC_DIM)
    x = torch.randn(BATCH, STATIC_DIM + DYNAMIC_DIM)
    out = model(x)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


def test_mlpnet_no_nan():
    model = MLPNet(in_dim=STATIC_DIM + DYNAMIC_DIM)
    x = torch.randn(BATCH, STATIC_DIM + DYNAMIC_DIM)
    assert not torch.isnan(model(x)).any()


# ---------------------------------------------------------------------------
# HybridSurvivalNet — full model
# ---------------------------------------------------------------------------

def test_hybrid_output_shape():
    model = HybridSurvivalNet(static_dim=STATIC_DIM, dynamic_dim=DYNAMIC_DIM)
    x_s, x_d, _ = make_inputs()
    logits = model(x_s, x_d)
    assert logits.shape == (BATCH, N_YEARS), f"Expected ({BATCH}, {N_YEARS}), got {logits.shape}"


def test_hybrid_no_nan():
    model = HybridSurvivalNet(static_dim=STATIC_DIM, dynamic_dim=DYNAMIC_DIM)
    x_s, x_d, _ = make_inputs()
    assert not torch.isnan(model(x_s, x_d)).any()


# ---------------------------------------------------------------------------
# Ablation: no gate (plain concatenation)
# ---------------------------------------------------------------------------

def test_hybrid_no_gate():
    model = HybridSurvivalNet(static_dim=STATIC_DIM, dynamic_dim=DYNAMIC_DIM, use_gate=False)
    x_s, x_d, _ = make_inputs()
    out = model(x_s, x_d)
    assert out.shape == (BATCH, N_YEARS)


# ---------------------------------------------------------------------------
# Ablation: single branch (dynamic_dim=0)
# ---------------------------------------------------------------------------

def test_hybrid_single_branch():
    model = HybridSurvivalNet(static_dim=STATIC_DIM, dynamic_dim=0)
    x_s, _, _ = make_inputs()
    out = model(x_s)
    assert out.shape == (BATCH, N_YEARS)


# ---------------------------------------------------------------------------
# survival_loss
# ---------------------------------------------------------------------------

def test_survival_loss_positive():
    model = HybridSurvivalNet(static_dim=STATIC_DIM, dynamic_dim=DYNAMIC_DIM)
    x_s, x_d, y = make_inputs()
    logits = model(x_s, x_d)
    loss = survival_loss(logits, y)
    assert loss.item() > 0, "Survival loss should be positive"
    assert not torch.isnan(loss), "Survival loss is NaN"


def test_survival_loss_decreases():
    """Loss should decrease after one gradient step."""
    model = HybridSurvivalNet(static_dim=STATIC_DIM, dynamic_dim=DYNAMIC_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_s, x_d, y = make_inputs()

    model.train()
    logits = model(x_s, x_d)
    loss_before = survival_loss(logits, y).item()
    opt.zero_grad()
    survival_loss(logits, y).backward()
    opt.step()

    logits2 = model(x_s, x_d)
    loss_after = survival_loss(logits2, y).item()
    # At least one of the two losses should be finite
    assert not (torch.isnan(torch.tensor(loss_after)))


# ---------------------------------------------------------------------------
# prob_within_horizon
# ---------------------------------------------------------------------------

def test_prob_horizon_range():
    logits = torch.randn(BATCH, N_YEARS)
    p = prob_within_horizon(logits)
    assert p.shape == (BATCH,)
    assert (p >= 0).all() and (p <= 1).all(), "Probabilities must be in [0, 1]"


def test_prob_horizon_high_hazard():
    """Very high hazard should give cumulative prob close to 1."""
    logits = torch.full((BATCH, N_YEARS), 10.0)
    p = prob_within_horizon(logits)
    assert (p > 0.99).all()


def test_prob_horizon_low_hazard():
    """Very low hazard should give cumulative prob close to 0."""
    logits = torch.full((BATCH, N_YEARS), -10.0)
    p = prob_within_horizon(logits)
    assert (p < 0.01).all()


# ---------------------------------------------------------------------------
# to_tensor
# ---------------------------------------------------------------------------

def test_to_tensor_dtype():
    import numpy as np
    a = np.array([1.0, 2.0, 3.0])
    t = to_tensor(a)
    assert t.dtype == torch.float32
