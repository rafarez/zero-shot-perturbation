"""Evaluation script for EVA-RNA zero-shot perturbation benchmark.

Reads the perturbation_results.csv produced by perturb.py and computes:
  - Global AUROC across all drug-disease pairs
  - Per-disease AUROC breakdown
  - ROC curve plot (.png)
  - JSON summary of all metrics

Usage
-----
    python evaluate.py --results path/to/perturbation_results.csv
    python evaluate.py --results path/to/perturbation_results.csv --out-dir results/eval
    python evaluate.py --results path/to/perturbation_results.csv --nan-as-zero

Arguments
---------
    --results       Path to perturbation_results.csv (required).
    --out-dir       Directory for output files (default: same dir as results CSV).
    --nan-as-zero   If set, NaN/missing scores are imputed as 0 rather than dropped.
                    Use this to evaluate coverage penalisation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_COL    = "median_score"
LABEL_COL    = "expected_efficacy"
DISEASE_COL  = "disease_abbrev"
DRUG_COL     = "drug_name"


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_results(path: Path, nan_as_zero: bool) -> pd.DataFrame:
    """Load and validate the results CSV, handling NaN scores.

    Parameters
    ----------
    path : Path
        Path to perturbation_results.csv.
    nan_as_zero : bool
        If True, NaN/empty scores are imputed to 0.
        If False, rows with NaN scores are dropped and a warning is emitted.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for AUROC computation.
    """
    df = pd.read_csv(path)

    required = {SCORE_COL, LABEL_COL, DISEASE_COL, DRUG_COL}
    missing_cols = required - set(df.columns)
    if missing_cols:
        log.error("Results CSV is missing required columns: %s", missing_cols)
        sys.exit(1)

    # Coerce expected_efficacy to bool (CSV may store it as string "True"/"False")
    df[LABEL_COL] = df[LABEL_COL].map(
        lambda v: True if str(v).strip().lower() == "true" else False
    )

    n_total = len(df)
    nan_mask = df[SCORE_COL].isna() | (df[SCORE_COL] == "")

    n_nan = nan_mask.sum()
    if n_nan > 0:
        frac = n_nan / n_total
        if nan_as_zero:
            log.warning(
                "--nan-as-zero: imputing %d / %d rows (%.1f%%) with NaN score to 0.  "
                "These correspond to drug-disease pairs where no target gene survived "
                "HVG filtering.  AUROC will penalise missing perturbations.",
                n_nan, n_total, 100 * frac,
            )
            df.loc[nan_mask, SCORE_COL] = 0.0
        else:
            nan_labels = df.loc[nan_mask, LABEL_COL]
            n_nan_pos  = nan_labels.sum()
            n_nan_neg  = (~nan_labels).sum()
            log.warning(
                "Dropping %d / %d rows (%.1f%%) with NaN score.  "
                "Of those: %d effective, %d ineffective.  "
                "If the split is skewed, the AUROC is computed on a biased subset.",
                n_nan, n_total, 100 * frac, n_nan_pos, n_nan_neg,
            )
            df = df.loc[~nan_mask].copy()

    df[SCORE_COL] = df[SCORE_COL].astype(float)

    if len(df) == 0:
        log.error("No valid rows remain after NaN handling.  Cannot compute AUROC.")
        sys.exit(1)

    n_pos = df[LABEL_COL].sum()
    n_neg = (~df[LABEL_COL]).sum()
    if n_pos == 0 or n_neg == 0:
        log.error(
            "Labels are all %s — AUROC is undefined.  "
            "Check that expected_efficacy contains both True and False values.",
            "positive" if n_pos == 0 else "negative",
        )
        sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------

def safe_auroc(y_true: np.ndarray, y_score: np.ndarray, label: str = "") -> float | None:
    """Compute AUROC, returning None if the label set is degenerate."""
    n_pos = y_true.sum()
    n_neg = (~y_true.astype(bool)).sum()
    if n_pos == 0 or n_neg == 0:
        log.warning(
            "Skipping AUROC for %s: only %d positive and %d negative labels.",
            label or "subset", n_pos, n_neg,
        )
        return None
    auc = roc_auc_score(y_true, y_score)
    if auc < 0.5:
        log.warning(
            "AUROC for %s is %.4f < 0.5.  "
            "The score direction may be inverted — check compute_shift_score.",
            label or "global", auc,
        )
    return auc


# ---------------------------------------------------------------------------
# ROC plot
# ---------------------------------------------------------------------------

def plot_roc(
    df: pd.DataFrame,
    global_auc: float,
    out_path: Path,
) -> None:
    """Plot a single global ROC curve with chance diagonal and save to PNG.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned results DataFrame.
    global_auc : float
        Pre-computed global AUROC to display in the legend.
    out_path : Path
        Destination file path (.png).
    """
    y_true  = df[LABEL_COL].values.astype(int)
    y_score = df[SCORE_COL].values.astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    # ---- Style ---------------------------------------------------------------
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")

    # ---- Chance diagonal -----------------------------------------------------
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1.0,
        color="#30363d",
        label="Chance (AUC = 0.50)",
    )

    # ---- ROC curve -----------------------------------------------------------
    ax.plot(
        fpr, tpr,
        linewidth=2.0,
        color="#58a6ff",
        label=f"EVA-RNA zero-shot  (AUC = {global_auc:.3f})",
    )

    # ---- Fill under curve ----------------------------------------------------
    ax.fill_between(fpr, tpr, alpha=0.08, color="#58a6ff")

    # ---- Labels & legend -----------------------------------------------------
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve — Drug Efficacy Prediction", fontsize=11, color="#e6edf3", pad=10)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, color="#21262d", linewidth=0.6)

    legend = ax.legend(
        loc="lower right",
        fontsize=9,
        framealpha=0.3,
        edgecolor="#30363d",
        labelcolor="#e6edf3",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("ROC curve saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EVA-RNA perturbation benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to perturbation_results.csv produced by perturb.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same directory as results CSV).",
    )
    parser.add_argument(
        "--nan-as-zero",
        action="store_true",
        default=False,
        help=(
            "Impute NaN scores as 0 instead of dropping rows.  "
            "Penalises pairs where no target gene survived HVG filtering."
        ),
    )
    args = parser.parse_args()

    if not args.results.exists():
        log.error("Results file not found: %s", args.results)
        sys.exit(1)

    out_dir = args.out_dir if args.out_dir is not None else args.results.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & clean --------------------------------------------------------
    df = load_results(args.results, nan_as_zero=args.nan_as_zero)
    n_pairs = len(df)
    n_pos   = int(df[LABEL_COL].sum())
    n_neg   = n_pairs - n_pos

    log.info("Loaded %d valid drug-disease pairs (%d effective, %d ineffective).",
             n_pairs, n_pos, n_neg)

    # ---- Global AUROC --------------------------------------------------------
    y_true  = df[LABEL_COL].values.astype(int)
    y_score = df[SCORE_COL].values.astype(float)

    global_auc = safe_auroc(y_true, y_score, label="global")
    if global_auc is None:
        sys.exit(1)

    log.info("Global AUROC: %.4f  (n=%d, pos=%d, neg=%d)",
             global_auc, n_pairs, n_pos, n_neg)

    # ---- Per-disease AUROC ---------------------------------------------------
    per_disease: dict[str, dict] = {}
    for disease, group in df.groupby(DISEASE_COL):
        yt = group[LABEL_COL].values.astype(int)
        ys = group[SCORE_COL].values.astype(float)
        auc = safe_auroc(yt, ys, label=disease)
        per_disease[disease] = {
            "auroc":  round(auc, 4) if auc is not None else None,
            "n_pairs": len(group),
            "n_positive": int(yt.sum()),
            "n_negative": int((~yt.astype(bool)).sum()),
        }
        status = f"{auc:.4f}" if auc is not None else "N/A (degenerate labels)"
        log.info("  %-6s  AUROC = %s  (n=%d)", disease, status, len(group))

    # ---- ROC plot ------------------------------------------------------------
    plot_path = out_dir / "roc_curve.png"
    plot_roc(df, global_auc, plot_path)

    # ---- JSON summary --------------------------------------------------------
    summary = {
        "results_file":  str(args.results),
        "nan_as_zero":   args.nan_as_zero,
        "n_pairs_total": n_pairs,
        "n_positive":    n_pos,
        "n_negative":    n_neg,
        "global_auroc":  round(global_auc, 4),
        "per_disease":   per_disease,
        "roc_plot":      str(plot_path),
    }

    json_path = out_dir / "eval_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("JSON summary saved → %s", json_path)

    # ---- Stdout summary ------------------------------------------------------
    print("\n" + "=" * 52)
    print(f"  EVA-RNA Perturbation Benchmark — Evaluation")
    print("=" * 52)
    print(f"  Pairs evaluated : {n_pairs}  ({n_pos} effective / {n_neg} ineffective)")
    print(f"  NaN handling    : {'imputed to 0' if args.nan_as_zero else 'dropped'}")
    print(f"  Global AUROC    : {global_auc:.4f}")
    print()
    print(f"  Per-disease breakdown:")
    for disease, metrics in sorted(per_disease.items()):
        auc_str = f"{metrics['auroc']:.4f}" if metrics["auroc"] is not None else "N/A"
        print(f"    {disease:<8}  AUROC = {auc_str}  (n={metrics['n_pairs']})")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()