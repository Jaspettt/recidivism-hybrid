"""Training loops for the neural models (MLP baseline and hybrid network)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .hybrid import HybridSurvivalNet, MLPNet, prob_within_horizon, survival_loss, to_tensor


def _iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def _fit(
    model: nn.Module,
    forward_prob,          # callable(tensors_dict, batch_idx) -> prob tensor
    loss_fn,               # callable(tensors_dict, batch_idx) -> scalar loss
    n_train: int,
    val_prob_fn,           # callable() -> (y_val, p_val)
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    verbose: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_state, bad_epochs = -np.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx in _iterate_minibatches(n_train, batch_size, rng):
            optimizer.zero_grad()
            loss = loss_fn(batch_idx)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * len(batch_idx)

        model.eval()
        with torch.no_grad():
            y_val, p_val = val_prob_fn()
        val_auc = roc_auc_score(y_val, p_val)
        history.append({"epoch": epoch, "loss": epoch_loss / n_train, "val_auc": val_auc})
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch:3d} | loss {epoch_loss / n_train:.4f} | val AUC {val_auc:.4f}")

        if val_auc > best_auc + 1e-4:
            best_auc, best_state, bad_epochs = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_auc": best_auc, "history": history}


class TorchModelWrapper:
    """Unified predict_proba interface over the fitted torch models."""

    def __init__(self, model, scalers, static_cols=None, dynamic_cols=None):
        self.model = model
        self.scalers = scalers
        self.static_cols = static_cols
        self.dynamic_cols = dynamic_cols

    def _hybrid_inputs(self, X):
        xs = to_tensor(self.scalers["static"].transform(X[self.static_cols]))
        xd = None
        if self.dynamic_cols:
            xd = to_tensor(self.scalers["dynamic"].transform(X[self.dynamic_cols]))
        return xs, xd

    def predict_proba_1d(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, HybridSurvivalNet):
                xs, xd = self._hybrid_inputs(X)
                return prob_within_horizon(self.model(xs, xd)).numpy()
            x = to_tensor(self.scalers["all"].transform(X))
            return torch.sigmoid(self.model(x)).numpy()

    def predict_hazards(self, X) -> np.ndarray:
        """Per-year hazards (hybrid model only)."""
        self.model.eval()
        with torch.no_grad():
            xs, xd = self._hybrid_inputs(X)
            return torch.sigmoid(self.model(xs, xd)).numpy()


def train_mlp(
    X_train, y_train,
    epochs: int = 100, batch_size: int = 512, lr: float = 1e-3,
    weight_decay: float = 1e-4, dropout: float = 0.3,
    val_size: float = 0.15, patience: int = 10, seed: int = 42, verbose: bool = True,
) -> tuple[TorchModelWrapper, dict]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    xt, xv = to_tensor(scaler.transform(X_tr)), to_tensor(scaler.transform(X_val))
    yt = to_tensor(y_tr)

    torch.manual_seed(seed)
    model = MLPNet(xt.shape[1], dropout=dropout)
    bce = nn.BCEWithLogitsLoss()

    info = _fit(
        model,
        forward_prob=None,
        loss_fn=lambda idx: bce(model(xt[idx]), yt[idx]),
        n_train=len(xt),
        val_prob_fn=lambda: (y_val, torch.sigmoid(model(xv)).numpy()),
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        patience=patience, seed=seed, verbose=verbose,
    )
    return TorchModelWrapper(model, {"all": scaler}), info


def train_hybrid(
    X_train, y_years_train, static_cols, dynamic_cols=None,
    epochs: int = 100, batch_size: int = 512, lr: float = 1e-3,
    weight_decay: float = 1e-4, dropout: float = 0.3,
    static_hidden: tuple[int, ...] = (192, 96),
    dynamic_hidden: tuple[int, ...] = (64, 32),
    fused_dim: int = 96, use_gate: bool = True,
    val_size: float = 0.15, patience: int = 10, seed: int = 42, verbose: bool = True,
) -> tuple[TorchModelWrapper, dict]:
    """Train the hybrid survival network.

    ``dynamic_cols=None``/``[]`` trains a single-branch (static-only) ablation;
    ``use_gate=False`` disables gated fusion.
    """
    dynamic_cols = list(dynamic_cols or [])
    y_any = (y_years_train.sum(axis=1) > 0).astype(int)
    X_tr, X_val, yy_tr, yy_val = train_test_split(
        X_train, y_years_train, test_size=val_size, stratify=y_any, random_state=seed)

    sc_s = StandardScaler().fit(X_tr[static_cols])
    xs = to_tensor(sc_s.transform(X_tr[static_cols]))
    xvs = to_tensor(sc_s.transform(X_val[static_cols]))
    scalers = {"static": sc_s}
    xd = xvd = None
    if dynamic_cols:
        sc_d = StandardScaler().fit(X_tr[dynamic_cols])
        xd = to_tensor(sc_d.transform(X_tr[dynamic_cols]))
        xvd = to_tensor(sc_d.transform(X_val[dynamic_cols]))
        scalers["dynamic"] = sc_d
    yy = to_tensor(yy_tr)
    y_val_any = (yy_val.sum(axis=1) > 0).astype(int)

    torch.manual_seed(seed)
    model = HybridSurvivalNet(
        len(static_cols), len(dynamic_cols), dropout=dropout,
        static_hidden=static_hidden, dynamic_hidden=dynamic_hidden,
        fused_dim=fused_dim, use_gate=use_gate)

    def batch_loss(idx):
        xb_d = xd[idx] if xd is not None else None
        return survival_loss(model(xs[idx], xb_d), yy[idx])

    info = _fit(
        model,
        forward_prob=None,
        loss_fn=batch_loss,
        n_train=len(xs),
        val_prob_fn=lambda: (y_val_any, prob_within_horizon(model(xvs, xvd)).numpy()),
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        patience=patience, seed=seed, verbose=verbose,
    )
    wrapper = TorchModelWrapper(model, scalers, static_cols, dynamic_cols)
    return wrapper, info
