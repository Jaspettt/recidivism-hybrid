"""Main experiment: baselines vs neural models on the NIJ dataset.

Trains logistic regression, random forest, XGBoost, an MLP and the hybrid
survival network on the official training split; evaluates everything on the
official test split (quality + fairness by race and gender); writes results
to ``results/``.

Usage:  python scripts/run_experiments.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.baselines import make_baselines
from recidivism.data import prepare_nij
from recidivism.metrics import classification_metrics, fairness_gaps, group_fairness_table
from recidivism.train import train_hybrid, train_mlp

RESULTS = ROOT / "results"
SEED = 42


def main() -> None:
    print("Loading and preprocessing NIJ dataset ...")
    data = prepare_nij(ROOT / "data" / "raw" / "nij_recidivism_full.csv")
    print(f"  train: {data.X_train.shape}, test: {data.X_test.shape}")
    print(f"  static features: {len(data.static_cols)}, dynamic features: {len(data.dynamic_cols)}")
    print(f"  base rate (3y recidivism) train/test: "
          f"{data.y_train.mean():.3f} / {data.y_test.mean():.3f}")

    predictions: dict[str, np.ndarray] = {}

    # --- classical baselines -------------------------------------------------
    for name, model in make_baselines(SEED).items():
        t0 = time.time()
        model.fit(data.X_train, data.y_train)
        predictions[name] = model.predict_proba(data.X_test)[:, 1]
        print(f"[{name}] fitted in {time.time() - t0:.1f}s")

    # --- MLP baseline ---------------------------------------------------------
    t0 = time.time()
    mlp, mlp_info = train_mlp(data.X_train, data.y_train, seed=SEED, verbose=False)
    predictions["mlp"] = mlp.predict_proba_1d(data.X_test)
    print(f"[mlp] fitted in {time.time() - t0:.1f}s "
          f"(best val AUC {mlp_info['best_val_auc']:.4f})")

    # --- hybrid survival network ----------------------------------------------
    t0 = time.time()
    hybrid, hyb_info = train_hybrid(
        data.X_train, data.y_years_train, data.static_cols, data.dynamic_cols,
        seed=SEED, verbose=False)
    predictions["hybrid_survival"] = hybrid.predict_proba_1d(data.X_test)
    print(f"[hybrid_survival] fitted in {time.time() - t0:.1f}s "
          f"(best val AUC {hyb_info['best_val_auc']:.4f})")

    # --- evaluation -------------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    metric_rows, fairness_rows = [], []
    for name, prob in predictions.items():
        metric_rows.append({"model": name, **classification_metrics(data.y_test, prob)})
        for attr in data.sensitive_test.columns:
            table = group_fairness_table(data.y_test, prob, data.sensitive_test[attr])
            table.to_csv(RESULTS / f"fairness_{attr.lower()}_{name}.csv")
            fairness_rows.append({"model": name, "attribute": attr,
                                  **fairness_gaps(table)})

    metrics_df = pd.DataFrame(metric_rows).set_index("model").sort_values("roc_auc", ascending=False)
    fairness_df = pd.DataFrame(fairness_rows).set_index(["model", "attribute"])
    metrics_df.to_csv(RESULTS / "metrics_nij.csv")
    fairness_df.to_csv(RESULTS / "fairness_gaps_nij.csv")

    np.save(RESULTS / "test_predictions.npy",
            np.column_stack([predictions[m] for m in metrics_df.index]))
    (RESULTS / "test_predictions_models.txt").write_text("\n".join(metrics_df.index))

    print("\n=== Quality on the official NIJ test split (3-year recidivism) ===")
    print(metrics_df.round(4).to_string())
    print("\n=== Fairness gaps (max-min across groups) ===")
    print(fairness_df.round(4).to_string())
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
