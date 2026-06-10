"""
Fairness & Bias Mitigation Analysis
====================================
Performs a three-part fairness audit on the trained Census Income classifier:

  1. Aequitas  — formal group fairness (demographic parity, equal opportunity,
                  equalized odds) across sex and race
  2. AIF360    — bias mitigation:
                   • Pre-processing:  Reweighting (sample-weight adjustment)
                   • Post-processing: Reject Option Classification
                     (per-group threshold optimization)
  3. Counterfactual fairness — flips a single protected attribute per record
                   (sex, race, both) and measures prediction change rate

Outputs (written to project root):
  fairness/aequitas_results.csv
  fairness/fairness_summary.json
  fairness/fig_aequitas_metrics.png
  fairness/fig_disparity_ratios.png
  fairness/fig_bias_mitigation.png
  fairness/fig_confusion_matrices.png
  fairness/fig_counterfactual.png

Usage:
    python -m starter.fairness_analysis          # from project root
    python starter/fairness_analysis.py          # direct run from project root
"""
import json
import os
import pickle
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

from starter.ml.data import process_data  # noqa: E402
from starter.train_model import clean_data  # noqa: E402

# ---------------------------------------------------------------------------
# Paths — all relative to the project root so the script works anywhere
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(_ROOT, "starter", "data", "census_clean.csv")
MODEL_PATH = os.path.join(_ROOT, "model", "model.pkl")
ENCODER_PATH = os.path.join(_ROOT, "model", "encoder.pkl")
LB_PATH = os.path.join(_ROOT, "model", "lb.pkl")
OUTPUT_DIR = os.path.join(_ROOT, "fairness")

CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BLUE = "#2E75B6"
RED = "#E95D4E"
GREEN = "#2E9B5E"
PURPLE = "#7B4EA8"
AMBER = "#F4A824"
TEAL = "#5EC4C4"
COLORS = [BLUE, RED, GREEN, PURPLE, AMBER, TEAL]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_fig(filename: str) -> None:
    """Save current matplotlib figure to OUTPUT_DIR."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fairness/{filename}")


def _get_preds(df: pd.DataFrame) -> np.ndarray:
    """Encode df (no label column) and return model predictions."""
    clean = df.drop(
        columns=[
            "salary", "pred_label", "true_label", "score",
            "label_value", "sex_bin", "race_bin", "sex_bin_f",
        ],
        errors="ignore",
    )
    X, _, _, _ = process_data(
        clean,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return model.predict(X)


def _make_counterfactual(df: pd.DataFrame,
                         col: str, original: str, flipped: str) -> pd.DataFrame:
    cf = df.copy()
    cf[col] = cf[col].replace({original: flipped})
    return cf


# ===========================================================================
# MAIN
# ===========================================================================
def run_fairness_analysis() -> dict:
    """
    Run the full fairness pipeline. Returns a summary dict.
    All artefacts are written to the ``fairness/`` directory.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Ensure census_clean.csv exists (idempotent — safe to call every time) ─
    clean_data()

    # ── Load data & artifacts ────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

    global model, encoder, lb
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(LB_PATH, "rb") as f:
        lb = pickle.load(f)

    X_test, y_test, _, _ = process_data(
        test_df, categorical_features=CAT_FEATURES,
        label="salary", training=False, encoder=encoder, lb=lb,
    )
    preds = model.predict(X_test)

    test = test_df.reset_index(drop=True).copy()
    test["pred_label"] = lb.inverse_transform(preds)
    test["true_label"] = lb.inverse_transform(y_test)
    test["score"] = preds
    test["label_value"] = y_test
    test["sex_bin"] = (test["sex"] == "Male").astype(int)
    test["race_bin"] = (test["race"] == "White").astype(int)

    # =========================================================================
    # SECTION 1 — AEQUITAS
    # =========================================================================
    print("\n=== SECTION 1: AEQUITAS FAIRNESS AUDIT ===")

    from aequitas.group import Group
    from aequitas.bias import Bias
    from aequitas.fairness import Fairness

    aq_df = test[["sex", "race", "label_value", "score"]].copy()
    aq_df["label_value"] = aq_df["label_value"].astype(int)
    aq_df["score"] = aq_df["score"].astype(int)

    xt, _ = Group().get_crosstabs(aq_df, attr_cols=["sex", "race"])
    bdf = Bias().get_disparity_predefined_groups(
        xt, original_df=aq_df,
        ref_groups_dict={"sex": "Male", "race": "White"},
        alpha=0.05,
    )
    fdf = Fairness().get_group_value_fairness(bdf)

    key_cols = [
        "attribute_name", "attribute_value",
        "tpr", "fpr", "tnr", "fnr", "precision", "ppr",
        "tpr_disparity", "fpr_disparity", "ppr_disparity", "precision_disparity",
        "TPR Parity", "FPR Parity", "Precision Parity",
    ]
    available = [c for c in key_cols if c in fdf.columns]
    fdf[available].to_csv(os.path.join(OUTPUT_DIR, "aequitas_results.csv"), index=False)
    print(fdf[available].to_string(index=False))

    # Fig 1 — TPR / FPR / Precision / PPR by group
    metrics_to_plot = ["tpr", "fpr", "precision", "ppr"]
    metric_labels = [
        "True Positive Rate", "False Positive Rate",
        "Precision", "Predicted Positive Rate",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric, label in zip(axes.flatten(), metrics_to_plot, metric_labels):
        if metric not in fdf.columns:
            continue
        combined = pd.concat([
            fdf[fdf["attribute_name"] == attr][["attribute_value", metric]].assign(attribute=attr)
            for attr in ["sex", "race"]
        ])
        unique_vals = combined["attribute_value"].unique()
        cmap = {v: COLORS[i % len(COLORS)] for i, v in enumerate(unique_vals)}
        bars = ax.bar(
            range(len(combined)), combined[metric].values,
            color=[cmap[v] for v in combined["attribute_value"]],
            edgecolor="white", linewidth=1.2,
        )
        ax.set_xticks(range(len(combined)))
        ax.set_xticklabels(
            [f"{r['attribute_value']}\n({r['attribute']})"
             for _, r in combined.iterrows()],
            fontsize=8.5, rotation=20, ha="right",
        )
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Rate", fontsize=10)
        ax.set_ylim(0, min(combined[metric].max() * 1.3, 1.05))
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, combined[metric].values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Aequitas Fairness Metrics by Sex and Race",
                 fontsize=14, fontweight="bold", y=1.01)
    _save_fig("fig_aequitas_metrics.png")

    # Fig 2 — Disparity ratios
    disp_cols = [
        c for c in fdf.columns
        if "disparity" in c.lower() and fdf[c].notna().any()
    ][:4]
    if disp_cols:
        fig, axes = plt.subplots(
            1,
            len(disp_cols),
            figsize=(14, 5),
            squeeze=False,
        )
        axes = axes[0]
        for ax, col in zip(axes, disp_cols):
            sub = fdf[fdf["attribute_name"].isin(["sex", "race"])][
                ["attribute_name", "attribute_value", col]].dropna()
            colors = [BLUE if r == "sex" else RED for r in sub["attribute_name"]]
            bar = ax.barh(sub["attribute_value"], sub[col],
                          color=colors, edgecolor="white", linewidth=1.1)
            ax.axvline(1.0, color="#333333", linestyle="--",
                       linewidth=1.5, label="Parity (1.0)")
            ax.set_title(col.replace("_", " ").title(),
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Disparity Ratio", fontsize=10)
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            for b2, v in zip(bar, sub[col].values):
                ax.text(v + 0.02, b2.get_y() + b2.get_height() / 2,
                        f"{v:.2f}", va="center", fontsize=8.5)
        fig.legend(
            handles=[
                mpatches.Patch(color=BLUE, label="Sex (ref: Male)"),
                mpatches.Patch(color=RED, label="Race (ref: White)"),
            ],
            loc="lower center", ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, -0.05),
        )
        fig.suptitle("Disparity Ratios vs. Reference Group (Male / White)",
                     fontsize=13, fontweight="bold")
        _save_fig("fig_disparity_ratios.png")

    # =========================================================================
    # SECTION 2 — AIF360
    # =========================================================================
    print("\n=== SECTION 2: AIF360 BIAS MITIGATION ===")

    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.algorithms.postprocessing import RejectOptionClassification
    from aif360.metrics import ClassificationMetric

    PRIV = [{"sex_bin": 1.0}]
    UNPRIV = [{"sex_bin": 0.0}]

    def _aif_dataset(df_in, label_col, prot_col="sex_bin"):
        d = df_in[[prot_col, label_col]].rename(
            columns={label_col: "label"}).copy()
        d["label"] = d["label"].astype(float)
        d[prot_col] = d[prot_col].astype(float)
        return BinaryLabelDataset(
            df=d, label_names=["label"],
            protected_attribute_names=[prot_col],
            favorable_label=1.0, unfavorable_label=0.0,
        )

    aif_true = _aif_dataset(test, "label_value")
    aif_pred_orig = _aif_dataset(test, "score")

    cm_orig = ClassificationMetric(
        aif_true, aif_pred_orig,
        unprivileged_groups=UNPRIV, privileged_groups=PRIV,
    )
    orig_metrics = {
        "Equal Opportunity Diff": cm_orig.equal_opportunity_difference(),
        "Average Odds Diff": cm_orig.average_odds_difference(),
        "Disparate Impact": cm_orig.disparate_impact(),
        "Statistical Parity Diff": cm_orig.statistical_parity_difference(),
        "Theil Index": cm_orig.theil_index(),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=1)),
        "recall": float(recall_score(y_test, preds, zero_division=1)),
    }
    print("Original model:", {k: round(v, 4) for k, v in orig_metrics.items()})

    # ── Reweighting ───────────────────────────────────────────────────────────
    train_r = train_df.reset_index(drop=True).copy()
    train_r["sex_bin"] = (train_r["sex"] == "Male").astype(float)
    train_r["label_val"] = (train_r["salary"] == ">50K").astype(float)

    aif_train = BinaryLabelDataset(
        df=train_r[["sex_bin", "label_val"]].rename(columns={"label_val": "label"}),
        label_names=["label"], protected_attribute_names=["sex_bin"],
        favorable_label=1.0, unfavorable_label=0.0,
    )
    rw = Reweighing(unprivileged_groups=UNPRIV, privileged_groups=PRIV)
    rw.fit(aif_train)
    weights = rw.transform(aif_train).instance_weights

    X_train_rw, y_train_rw, _, _ = process_data(
        train_df, categorical_features=CAT_FEATURES,
        label="salary", training=False, encoder=encoder, lb=lb,
    )
    rw_model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    rw_model.fit(X_train_rw, y_train_rw, sample_weight=weights)
    rw_preds = rw_model.predict(X_test)

    rw_df = test[["sex_bin"]].copy().astype(float)
    rw_df["label"] = rw_preds.astype(float)
    aif_pred_rw = BinaryLabelDataset(
        df=rw_df, label_names=["label"],
        protected_attribute_names=["sex_bin"],
        favorable_label=1.0, unfavorable_label=0.0,
    )
    cm_rw = ClassificationMetric(
        aif_true, aif_pred_rw,
        unprivileged_groups=UNPRIV, privileged_groups=PRIV,
    )
    rw_metrics = {
        "Equal Opportunity Diff": cm_rw.equal_opportunity_difference(),
        "Average Odds Diff": cm_rw.average_odds_difference(),
        "Disparate Impact": cm_rw.disparate_impact(),
        "Statistical Parity Diff": cm_rw.statistical_parity_difference(),
        "Theil Index": cm_rw.theil_index(),
        "f1": float(f1_score(y_test, rw_preds)),
        "precision": float(precision_score(y_test, rw_preds, zero_division=1)),
        "recall": float(recall_score(y_test, rw_preds, zero_division=1)),
    }
    print("After reweighting:", {k: round(v, 4) for k, v in rw_metrics.items()})

    # ── Threshold optimization ────────────────────────────────────────────────
    roc_available = False
    roc_metrics = orig_metrics.copy()
    try:
        val_df = train_df.sample(frac=0.2, random_state=7).reset_index(drop=True)
        X_val, y_val, _, _ = process_data(
            val_df, categorical_features=CAT_FEATURES,
            label="salary", training=False, encoder=encoder, lb=lb,
        )
        val_df["sex_bin"] = (val_df["sex"] == "Male").astype(float)
        val_scores = model.predict_proba(X_val)[:, 1]

        aif_val_true = BinaryLabelDataset(
            df=val_df[["sex_bin"]].assign(label=y_val.astype(float)),
            label_names=["label"], protected_attribute_names=["sex_bin"],
            favorable_label=1.0, unfavorable_label=0.0,
        )
        aif_val_pred = BinaryLabelDataset(
            df=val_df[["sex_bin"]].assign(
                label=(val_scores >= 0.5).astype(float)),
            label_names=["label"], protected_attribute_names=["sex_bin"],
            favorable_label=1.0, unfavorable_label=0.0,
        )
        roc = RejectOptionClassification(
            unprivileged_groups=UNPRIV, privileged_groups=PRIV,
            low_class_thresh=0.01, high_class_thresh=0.99,
            num_class_thresh=100, num_ROC_margin=50,
            metric_name="Statistical parity difference",
            metric_ub=0.05, metric_lb=-0.05,
        )
        roc.fit(aif_val_true, aif_val_pred)

        test_scores = model.predict_proba(X_test)[:, 1]
        test["sex_bin_f"] = test["sex_bin"].astype(float)
        aif_test_scored = BinaryLabelDataset(
            df=test[["sex_bin_f"]].rename(columns={"sex_bin_f": "sex_bin"}).assign(
                label=(test_scores >= 0.5).astype(float)),
            label_names=["label"], protected_attribute_names=["sex_bin"],
            favorable_label=1.0, unfavorable_label=0.0,
        )
        roc_preds = roc.predict(aif_test_scored).labels.ravel().astype(int)

        roc_df = test[["sex_bin"]].copy().astype(float)
        roc_df["label"] = roc_preds.astype(float)
        aif_pred_roc = BinaryLabelDataset(
            df=roc_df, label_names=["label"],
            protected_attribute_names=["sex_bin"],
            favorable_label=1.0, unfavorable_label=0.0,
        )
        cm_roc = ClassificationMetric(
            aif_true, aif_pred_roc,
            unprivileged_groups=UNPRIV, privileged_groups=PRIV,
        )
        roc_metrics = {
            "Equal Opportunity Diff": cm_roc.equal_opportunity_difference(),
            "Average Odds Diff": cm_roc.average_odds_difference(),
            "Disparate Impact": cm_roc.disparate_impact(),
            "Statistical Parity Diff": cm_roc.statistical_parity_difference(),
            "Theil Index": cm_roc.theil_index(),
            "f1": float(f1_score(y_test, roc_preds)),
            "precision": float(precision_score(y_test, roc_preds, zero_division=1)),
            "recall": float(recall_score(y_test, roc_preds, zero_division=1)),
        }
        roc_available = True
        print("After threshold opt.:",
              {k: round(v, 4) for k, v in roc_metrics.items()})
    except Exception as e:
        print(f"  Threshold optimization skipped: {e}")

    # Fig 3 — Before/After comparison
    intervention_names = ["Original", "Reweighting\n(pre-processing)"]
    sets = [orig_metrics, rw_metrics]
    if roc_available:
        intervention_names.append("Threshold Opt.\n(post-processing)")
        sets.append(roc_metrics)

    plot_metrics = [
        "Equal Opportunity Diff", "Average Odds Diff",
        "Statistical Parity Diff", "Disparate Impact",
    ]
    plot_labels = [
        "Equal Opp. Diff", "Avg Odds Diff",
        "Stat. Parity Diff", "Disparate Impact",
    ]
    bar_colors = [BLUE, GREEN, PURPLE]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric, label in zip(axes, plot_metrics, plot_labels):
        vals = [s[metric] for s in sets]
        bars = ax.bar(intervention_names, vals,
                      color=bar_colors[: len(vals)],
                      edgecolor="white", linewidth=1.2)
        ideal = 1.0 if metric == "Disparate Impact" else 0.0
        ax.axhline(ideal, color="#333", linewidth=1.5,
                   linestyle="--", label=f"Ideal ({ideal})")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Value", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            ypos = v + 0.005 if v >= 0 else v - 0.025
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{v:+.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Bias Mitigation — Before vs. After (Protected: Sex, Female vs. Male)",
        fontsize=13, fontweight="bold",
    )
    _save_fig("fig_bias_mitigation.png")

    # Fig 4 — Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, pred_arr, title in zip(
        axes, [preds, rw_preds], ["Original Model", "After Reweighting"]
    ):
        cm = confusion_matrix(y_test, pred_arr)
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["<=50K", ">50K"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        p = precision_score(y_test, pred_arr, zero_division=1)
        r = recall_score(y_test, pred_arr, zero_division=1)
        f = f1_score(y_test, pred_arr)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(
            f"Predicted\nPrec={p:.3f}  Rec={r:.3f}  F1={f:.3f}", fontsize=10
        )
    fig.suptitle(
        "Confusion Matrices — Before and After Bias Mitigation",
        fontsize=13, fontweight="bold",
    )
    _save_fig("fig_confusion_matrices.png")

    # =========================================================================
    # SECTION 3 — COUNTERFACTUAL FAIRNESS
    # =========================================================================
    print("\n=== SECTION 3: COUNTERFACTUAL FAIRNESS ===")

    n = len(preds)
    # Sex
    cf_sex = _make_counterfactual(test, "sex", "Male", "Female")
    preds_sex = _get_preds(cf_sex)
    flip_sex = preds != preds_sex
    male_mask = test["sex"] == "Male"
    flip_sex_pct = flip_sex.sum() / n * 100
    flip_up_sex = int(((preds == 0) & (preds_sex == 1) & male_mask).sum())
    flip_down_sex = int(((preds == 1) & (preds_sex == 0) & male_mask).sum())

    # Race
    cf_race = _make_counterfactual(test, "race", "White", "Black")
    preds_race = _get_preds(cf_race)
    flip_race = preds != preds_race
    white_mask = test["race"] == "White"
    flip_race_pct = flip_race.sum() / n * 100
    flip_up_race = int(((preds == 0) & (preds_race == 1) & white_mask).sum())
    flip_down_race = int(((preds == 1) & (preds_race == 0) & white_mask).sum())

    # Intersectional
    cf_both = _make_counterfactual(test, "sex", "Male", "Female")
    cf_both = _make_counterfactual(cf_both, "race", "White", "Black")
    flip_both_pct = (preds != _get_preds(cf_both)).sum() / n * 100

    print(f"  Sex flip rate         : {flip_sex_pct:.1f}%  "
          f"(↑{flip_up_sex} gained >50K, ↓{flip_down_sex} lost >50K)")
    print(f"  Race flip rate        : {flip_race_pct:.1f}%  "
          f"(↑{flip_up_race} gained >50K, ↓{flip_down_race} lost >50K)")
    print(f"  Intersectional rate   : {flip_both_pct:.1f}%")

    # Fig 5 — Counterfactual
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    bars0 = axes[0].bar(
        ["Sex\n(Male↔Female)", "Race\n(White→Black)", "Intersectional\n(Both)"],
        [flip_sex_pct, flip_race_pct, flip_both_pct],
        color=[BLUE, RED, PURPLE], edgecolor="white", linewidth=1.2,
    )
    axes[0].set_ylabel("% Records with Flipped Prediction", fontsize=10)
    axes[0].set_title("Counterfactual Flip Rate", fontsize=11, fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars0, [flip_sex_pct, flip_race_pct, flip_both_pct]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.1, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    axes[1].bar(
        ["Male→Female\n(gains >50K)", "Male→Female\n(loses >50K)"],
        [flip_up_sex, flip_down_sex],
        color=[GREEN, RED], edgecolor="white", linewidth=1.2,
    )
    axes[1].set_title("Sex CF — Direction\n(from Male records)",
                      fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].bar(
        ["White→Black\n(gains >50K)", "White→Black\n(loses >50K)"],
        [flip_up_race, flip_down_race],
        color=[GREEN, RED], edgecolor="white", linewidth=1.2,
    )
    axes[2].set_title("Race CF — Direction\n(from White records)",
                      fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Count", fontsize=10)
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Counterfactual Fairness Analysis",
                 fontsize=14, fontweight="bold")
    _save_fig("fig_counterfactual.png")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "original": orig_metrics,
        "reweighting": rw_metrics,
        "counterfactual_sex": {
            "pct_flipped": float(flip_sex_pct),
            "flip_up": flip_up_sex,
            "flip_down": flip_down_sex,
        },
        "counterfactual_race": {
            "pct_flipped": float(flip_race_pct),
            "flip_up": flip_up_race,
            "flip_down": flip_down_race,
        },
        "counterfactual_both": {"pct_flipped": float(flip_both_pct)},
    }
    if roc_available:
        summary["threshold_opt"] = roc_metrics

    json_path = os.path.join(OUTPUT_DIR, "fairness_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll fairness outputs written to: {OUTPUT_DIR}/")
    return summary


if __name__ == "__main__":
    run_fairness_analysis()
