"""Full experimental study for the article (plan items 4-5, due June 2026).

Steps:
  1. Baselines: logistic regression, random forest, XGBoost, MLP.
  2. Small hyperparameter search for the hybrid survival network (val AUC).
  3. Ablations: single-branch survival, no-gate hybrid, full hybrid.
  4. Ensemble: average of hybrid and XGBoost probabilities.
  5. Per-year hazard discrimination (discrete-time survival evaluation).
  6. Fairness on NIJ (race, gender) + equalized-odds post-processing.
  7. Explainability: SHAP (XGBoost) and permutation importance (hybrid).
  8. Publication figures (results/figures) and summary tables (results/).

Usage:  python scripts/run_full_study.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recidivism.baselines import make_baselines
from recidivism.data import prepare_nij
from recidivism.explain import permutation_importance, shap_tree_values
from recidivism.fairness import mitigation_report
from recidivism.metrics import classification_metrics, fairness_gaps, group_fairness_table, per_year_auc
from recidivism.train import train_hybrid, train_mlp

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEED = 42

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.bbox": "tight", "font.size": 9})


def save_fig(name: str) -> None:
    plt.savefig(FIGURES / name)
    plt.close()
    print(f"  figure: {name}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    print("=" * 70)
    print("STEP 0. Data")
    data = prepare_nij(ROOT / "data" / "raw" / "nij_recidivism_full.csv")
    all_cols = data.X_train.columns.tolist()
    print(f"  train {data.X_train.shape}, test {data.X_test.shape}; "
          f"static {len(data.static_cols)}, dynamic {len(data.dynamic_cols)}")

    probs: dict[str, np.ndarray] = {}
    fitted = {}

    print("STEP 1. Classical baselines + MLP")
    for name, model in make_baselines(SEED).items():
        t0 = time.time()
        model.fit(data.X_train, data.y_train)
        probs[name] = model.predict_proba(data.X_test)[:, 1]
        fitted[name] = model
        print(f"  [{name}] {time.time() - t0:.1f}s")
    mlp, _ = train_mlp(data.X_train, data.y_train, seed=SEED, verbose=False)
    probs["mlp"] = mlp.predict_proba_1d(data.X_test)
    print("  [mlp] done")

    print("STEP 2. Hybrid hyperparameter search (selection by internal val AUC)")
    grid = list(itertools.product(
        [((192, 96), (64, 32), 96), ((256, 128), (96, 48), 128)],  # (static_hidden, dynamic_hidden, fused_dim)
        [0.2, 0.3],                                                # dropout
        [1e-3, 5e-4],                                              # lr
    ))
    search_rows, best = [], None
    for (sh, dh, fd), dropout, lr in grid:
        t0 = time.time()
        wrapper, info = train_hybrid(
            data.X_train, data.y_years_train, data.static_cols, data.dynamic_cols,
            static_hidden=sh, dynamic_hidden=dh, fused_dim=fd,
            dropout=dropout, lr=lr, seed=SEED, verbose=False)
        row = {"static_hidden": str(sh), "dynamic_hidden": str(dh), "fused_dim": fd,
               "dropout": dropout, "lr": lr, "val_auc": info["best_val_auc"],
               "sec": round(time.time() - t0, 1)}
        search_rows.append(row)
        print(f"  {row}")
        if best is None or info["best_val_auc"] > best[1]:
            best = (wrapper, info["best_val_auc"], {"static_hidden": sh, "dynamic_hidden": dh,
                                                    "fused_dim": fd, "dropout": dropout, "lr": lr})
    pd.DataFrame(search_rows).to_csv(RESULTS / "hybrid_search.csv", index=False)
    hybrid, best_val, best_cfg = best
    print(f"  best config: {best_cfg} (val AUC {best_val:.4f})")
    probs["hybrid_survival"] = hybrid.predict_proba_1d(data.X_test)

    print("STEP 3. Ablations")
    ablations = {}
    surv_single, _ = train_hybrid(  # survival head, single branch over ALL features
        data.X_train, data.y_years_train, all_cols, None,
        static_hidden=best_cfg["static_hidden"], fused_dim=best_cfg["fused_dim"],
        dropout=best_cfg["dropout"], lr=best_cfg["lr"], use_gate=False, seed=SEED, verbose=False)
    ablations["survival_single_branch"] = surv_single.predict_proba_1d(data.X_test)
    print("  [survival_single_branch] done")

    no_gate, _ = train_hybrid(
        data.X_train, data.y_years_train, data.static_cols, data.dynamic_cols,
        static_hidden=best_cfg["static_hidden"], dynamic_hidden=best_cfg["dynamic_hidden"],
        fused_dim=best_cfg["fused_dim"], dropout=best_cfg["dropout"], lr=best_cfg["lr"],
        use_gate=False, seed=SEED, verbose=False)
    ablations["hybrid_no_gate"] = no_gate.predict_proba_1d(data.X_test)
    print("  [hybrid_no_gate] done")

    print("STEP 4. Ensemble (hybrid + XGBoost, equal weights)")
    probs["ensemble_hybrid_xgb"] = 0.5 * probs["hybrid_survival"] + 0.5 * probs["xgboost"]

    # ----- main metrics table ------------------------------------------------
    rows = [{"model": m, **classification_metrics(data.y_test, p)} for m, p in probs.items()]
    metrics_df = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(RESULTS / "metrics_nij.csv")
    print(metrics_df.round(4).to_string())

    ab_rows = [{"model": "mlp (BCE, single branch)", **classification_metrics(data.y_test, probs["mlp"])},
               {"model": "+ survival head", **classification_metrics(data.y_test, ablations["survival_single_branch"])},
               {"model": "+ two branches (no gate)", **classification_metrics(data.y_test, ablations["hybrid_no_gate"])},
               {"model": "+ gated fusion (full hybrid)", **classification_metrics(data.y_test, probs["hybrid_survival"])}]
    ablation_df = pd.DataFrame(ab_rows).set_index("model")
    ablation_df.to_csv(RESULTS / "ablation_nij.csv")
    print("\nAblation:")
    print(ablation_df.round(4).to_string())

    print("STEP 5. Per-year hazard discrimination")
    hazards = hybrid.predict_hazards(data.X_test)
    year_auc = per_year_auc(hazards, data.y_years_test)
    pd.DataFrame([year_auc]).to_csv(RESULTS / "per_year_auc.csv", index=False)
    print(f"  {year_auc}")

    print("STEP 6. Fairness on NIJ + equalized-odds mitigation")
    fair_rows = []
    fair_reports = {}
    for model_name, pred_fn in [
        ("xgboost", lambda X: fitted["xgboost"].predict_proba(X)[:, 1]),
        ("hybrid_survival", hybrid.predict_proba_1d),
    ]:
        for attr in ["Race", "Gender"]:
            rep = mitigation_report(
                pred_fn,
                data.X_train, data.y_train, data.sensitive_train[attr],
                data.X_test, data.y_test, data.sensitive_test[attr], seed=SEED)
            fair_reports[(model_name, attr)] = rep
            fair_rows.append({
                "model": model_name, "attribute": attr,
                **{f"{k}_before": v for k, v in rep["gaps_before"].items()},
                **{f"{k}_after": v for k, v in rep["gaps_after"].items()},
                "accuracy_before": rep["accuracy_before"],
                "accuracy_after": rep["accuracy_after"],
            })
            print(f"  [{model_name} / {attr}] EOdds {rep['gaps_before']['equalized_odds_diff']:.3f}"
                  f" -> {rep['gaps_after']['equalized_odds_diff']:.3f}, "
                  f"acc {rep['accuracy_before']:.3f} -> {rep['accuracy_after']:.3f}")
    fair_df = pd.DataFrame(fair_rows)
    fair_df.to_csv(RESULTS / "fairness_mitigation_nij.csv", index=False)

    print("STEP 7. Explainability")
    _, shap_values, _ = shap_tree_values(fitted["xgboost"], data.X_test, max_samples=2000, seed=SEED)
    imp = permutation_importance(hybrid.predict_proba_1d, data.X_test, data.y_test,
                                 n_repeats=3, max_samples=3000, seed=SEED)
    imp.to_csv(RESULTS / "hybrid_permutation_importance.csv", index=False)

    # ======================================================================
    print("STEP 8. Figures")

    # ROC + calibration
    show = ["logreg", "random_forest", "xgboost", "mlp", "hybrid_survival", "ensemble_hybrid_xgb"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for m in show:
        RocCurveDisplay.from_predictions(data.y_test, probs[m], name=m, ax=axes[0])
        CalibrationDisplay.from_predictions(data.y_test, probs[m], name=m, n_bins=10, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set_title("(a) ROC curves, NIJ test set")
    axes[1].set_title("(b) Calibration curves")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    save_fig("fig_roc_calibration.png")

    # hazard trajectories
    first_year = np.where(data.y_years_test.sum(axis=1) > 0,
                          data.y_years_test.argmax(axis=1) + 1, 0)
    labels = {0: "No recidivism", 1: "Arrest in year 1", 2: "Arrest in year 2", 3: "Arrest in year 3"}
    colors = {0: "#27ae60", 1: "#c0392b", 2: "#e67e22", 3: "#f1c40f"}
    plt.figure(figsize=(5.5, 4))
    for k, lbl in labels.items():
        mask = first_year == k
        plt.plot([1, 2, 3], hazards[mask].mean(axis=0), marker="o",
                 label=f"{lbl} (n={mask.sum()})", color=colors[k])
    plt.xticks([1, 2, 3])
    plt.xlabel("Year after release")
    plt.ylabel("Mean predicted hazard $h_t$")
    plt.title("Predicted hazard trajectories by observed outcome")
    plt.legend(fontsize=8)
    save_fig("fig_hazard_trajectories.png")

    # permutation importance
    plt.figure(figsize=(6, 5))
    top = imp.head(15).iloc[::-1]
    plt.barh(top["feature"], top["auc_drop_mean"], xerr=top["auc_drop_std"], color="#2980b9")
    plt.xlabel("AUC drop when feature is permuted")
    plt.title("Permutation importance, hybrid model (top 15)")
    save_fig("fig_hybrid_importance.png")

    # SHAP beeswarm
    import shap
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP summary, XGBoost")
    save_fig("fig_shap_xgb.png")

    # fairness before/after for the hybrid
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, attr in zip(axes, ["Race", "Gender"]):
        rep = fair_reports[("hybrid_survival", attr)]
        groups = rep["table_before"].index.tolist()
        x = np.arange(len(groups))
        w = 0.2
        ax.bar(x - 1.5 * w, rep["table_before"]["fpr"], w, label="FPR before", color="#c0392b")
        ax.bar(x - 0.5 * w, rep["table_after"]["fpr"], w, label="FPR after", color="#e6a09a")
        ax.bar(x + 0.5 * w, rep["table_before"]["fnr"], w, label="FNR before", color="#2980b9")
        ax.bar(x + 1.5 * w, rep["table_after"]["fnr"], w, label="FNR after", color="#a3c6e8")
        ax.set_xticks(x, groups)
        ax.set_title(f"Hybrid model: error rates by {attr}")
        ax.set_ylabel("Error rate")
    axes[0].legend(fontsize=7)
    save_fig("fig_fairness_mitigation.png")

    # architecture diagram
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2):
        from matplotlib.patches import FancyArrowPatch
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.0, color="black"))

    box(0.02, 0.62, 0.20, 0.24, "Static features (40)\ndemographics,\ncriminal history,\nsentence", "#d6eaf8")
    box(0.02, 0.12, 0.20, 0.24, "Supervision dynamics (23)\nviolations, drug tests,\nemployment, programs", "#fdebd0")
    box(0.30, 0.62, 0.16, 0.24, "Static branch\nMLP 256-128", "#aed6f1")
    box(0.30, 0.12, 0.16, 0.24, "Dynamics branch\nMLP 96-48", "#f8c471")
    box(0.54, 0.37, 0.14, 0.24, "Gated\nfusion", "#d2b4de")
    box(0.76, 0.37, 0.10, 0.24, "GRU\n3 steps", "#a9dfbf")
    box(0.90, 0.37, 0.08, 0.24, "$h_1$\n$h_2$\n$h_3$", "#f9e79f")
    arrow(0.22, 0.74, 0.30, 0.74); arrow(0.22, 0.24, 0.30, 0.24)
    arrow(0.46, 0.74, 0.54, 0.52); arrow(0.46, 0.24, 0.54, 0.46)
    arrow(0.68, 0.49, 0.76, 0.49); arrow(0.86, 0.49, 0.90, 0.49)
    ax.text(0.5, 0.02, r"$P(\mathrm{recidivism\ within\ 3\ years}) = 1 - \prod_t (1 - h_t)$",
            ha="center", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_fig("fig_architecture.png")

    # ----- summary for the article -------------------------------------------
    summary = {
        "best_hybrid_config": {k: str(v) for k, v in best_cfg.items()},
        "best_hybrid_val_auc": best_val,
        "metrics": metrics_df.round(4).to_dict(orient="index"),
        "ablation": ablation_df.round(4).to_dict(orient="index"),
        "per_year_auc": year_auc,
        "fairness_mitigation": fair_df.round(4).to_dict(orient="records"),
        "top10_hybrid_features": imp.head(10).round(4).to_dict(orient="records"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll outputs written to {RESULTS}")


if __name__ == "__main__":
    main()
