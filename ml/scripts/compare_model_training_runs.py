"""
Compare two or more on-disk model training runs (metadata + training history only).

By default, comparisons are **anchored to the latest run**: only model keys present in the
last ``--runs`` directory are used for pairwise metrics, run-level means, and feature churn
(so aggregates are comparable across experiments). Pass ``--no-anchor-to-latest`` for the
legacy behavior (all overlapping keys per pair).

Install plotting deps: ``pip install -r ml/requirements.txt -r ml/requirements-analysis.txt``

Two-run example (backward-compatible)::

    python -m ml.scripts.compare_model_training_runs \\
        --baseline s3_backup/models/currency/xp_20251118_191604 \\
        --candidate s3_backup/models/currency/xp_20260405_092445 \\
        --output-dir reports/model_compare_xp

Three-run (multi-experiment) example::

    python -m ml.scripts.compare_model_training_runs \\
        --runs s3_backup/models/currency/xp_20251118_191604 \\
               s3_backup/models/currency/xp_20260405_092445 \\
               s3_backup/models/currency/xp_20260406_004633 \\
        --output-dir reports/model_compare_all
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.utils.common_utils import MLLogger
from ml.utils.model_run_comparison import (
    METRIC_KEYS,
    ModelMetadataRecord,
    aggregate_run_metrics,
    build_comparison_frame,
    build_feature_evolution_frame,
    filter_records_to_model_keys,
    load_training_history,
    mean_normalized_importance_by_feature,
    records_by_key,
    summarize_metrics,
    top_k_importance_bar_data,
)

# Metrics where a lower value is better (candidate win = delta < 0).
_LOWER_IS_BETTER = {"mae", "rmse", "mape"}
# Metrics where a higher value is better (candidate win = delta > 0).
_HIGHER_IS_BETTER = {"r2", "directional_accuracy"}


def _safe_filename_fragment(name: str, max_len: int = 100) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return cleaned[:max_len]


# ---------------------------------------------------------------------------
# Shared scatter / histogram / box plots (pairwise)
# ---------------------------------------------------------------------------


def _plot_scatter_metric(
    df: pd.DataFrame,
    metric: str,
    baseline_label: str,
    candidate_label: str,
    out_path: Path,
) -> None:
    bx = f"baseline_{metric}"
    cy = f"candidate_{metric}"
    sub = df[[bx, cy]].dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sub[bx], sub[cy], alpha=0.35, s=12)
    lim_max = float(max(sub[bx].max(), sub[cy].max()))
    lim_min = float(min(sub[bx].min(), sub[cy].min()))
    if lim_max > lim_min:
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="y = x")
    ax.set_xlabel(f"{metric} ({baseline_label})")
    ax.set_ylabel(f"{metric} ({candidate_label})")
    ax.set_title(f"{metric.upper()}: candidate vs baseline (matched models)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_delta_histograms(df: pd.DataFrame, out_path: Path) -> None:
    delta_cols = [f"delta_{m}" for m in METRIC_KEYS if f"delta_{m}" in df.columns]
    if not delta_cols:
        return
    n = len(delta_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, col in enumerate(delta_cols):
        ax = axes_flat[i]
        series = df[col].dropna()
        if len(series) == 0:
            ax.set_visible(False)
            continue
        ax.hist(series, bins=40, color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(col)
        ax.set_ylabel("count")
    for j in range(len(delta_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Delta = candidate − baseline (validation metrics)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_boxplot_delta_mae_by_horizon(df: pd.DataFrame, out_path: Path) -> None:
    if "delta_mae" not in df.columns or "horizon" not in df.columns:
        return
    sub = df.dropna(subset=["delta_mae", "horizon"])
    if sub.empty:
        return
    horizons = sorted(sub["horizon"].unique())
    data = [sub[sub["horizon"] == h]["delta_mae"].values for h in horizons]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=horizons, showfliers=False)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("horizon")
    ax.set_ylabel("delta MAE (candidate − baseline)")
    ax.set_title("Distribution of MAE deltas by horizon")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_learning_curve_model(
    model_dir: Path,
    title: str,
    out_path: Path,
    max_learners: int = 2,
) -> None:
    hist = load_training_history(model_dir)
    if not hist:
        return
    items = list(hist.items())[:max_learners]
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), squeeze=False)
    for i, (learner_key, payload) in enumerate(items):
        ax = axes[i, 0]
        train_loss = payload.get("train_loss") or []
        val_loss = payload.get("val_loss") or []
        if not train_loss and not val_loss:
            continue
        if train_loss:
            ax.plot(range(1, len(train_loss) + 1), train_loss, label="train_loss", linewidth=1.2)
        if val_loss:
            ax.plot(range(1, len(val_loss) + 1), val_loss, label="val_loss", linewidth=1.2)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title(f"{title} — {learner_key}")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_global_importance_bars(series: pd.Series, title: str, out_path: Path, top_n: int) -> None:
    top = series.head(top_n)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.28)))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.values, color="teal", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean normalized importance across matched models")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_topk_pair(
    baseline_rec: ModelMetadataRecord,
    candidate_rec: ModelMetadataRecord,
    model_key: str,
    out_path: Path,
    top_k: int,
) -> None:
    bdf = top_k_importance_bar_data(baseline_rec, top_k)
    cdf = top_k_importance_bar_data(candidate_rec, top_k)
    if bdf.empty and cdf.empty:
        return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, max(4, top_k * 0.25)))
    if not bdf.empty:
        y = np.arange(len(bdf))
        ax0.barh(y, bdf["importance_norm"].values, color="slategray", alpha=0.9)
        ax0.set_yticks(y)
        ax0.set_yticklabels(list(bdf["feature"]), fontsize=8)
        ax0.invert_yaxis()
        ax0.set_title("Baseline (normalized)")
    if not cdf.empty:
        y = np.arange(len(cdf))
        ax1.barh(y, cdf["importance_norm"].values, color="darkcyan", alpha=0.9)
        ax1.set_yticks(y)
        ax1.set_yticklabels(list(cdf["feature"]), fontsize=8)
        ax1.invert_yaxis()
        ax1.set_title("Candidate (normalized)")
    fig.suptitle(f"Top-{top_k} feature importance — {_safe_filename_fragment(model_key)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _select_learning_curve_models(
    df: pd.DataFrame,
    baseline_root: Path,
    candidate_root: Path,
    max_models: int,
    rng: np.random.Generator,
) -> list[str]:
    """Pick model keys with history on both sides: stratified by horizon + best/worst MAE delta."""
    if max_models <= 0:
        return []
    keys_with_history: list[str] = []
    for key in df["model_key"].values:
        b_dir = baseline_root / key
        c_dir = candidate_root / key
        if load_training_history(b_dir) and load_training_history(c_dir):
            keys_with_history.append(str(key))
    if not keys_with_history:
        return []

    sub = df[df["model_key"].isin(keys_with_history)].copy()
    if sub.empty:
        return []

    picked: list[str] = []
    if "delta_mae" in sub.columns:
        ordered = sub.sort_values("delta_mae", na_position="last")
        worst = ordered.iloc[-1]["model_key"] if len(ordered) else None
        best = ordered.iloc[0]["model_key"] if len(ordered) else None
        for k in (best, worst):
            if k is not None and str(k) not in picked:
                picked.append(str(k))

    horizons = ["1d", "3d", "7d"]
    per_h = max(1, max_models // max(len(horizons), 1))
    for h in horizons:
        pool = sub[sub["horizon"] == h]["model_key"].tolist()
        if not pool:
            continue
        rng.shuffle(pool)
        for k in pool[:per_h]:
            if k not in picked:
                picked.append(k)
            if len(picked) >= max_models:
                return picked[:max_models]

    for k in keys_with_history:
        if k not in picked:
            picked.append(k)
        if len(picked) >= max_models:
            break
    return picked[:max_models]


# ---------------------------------------------------------------------------
# Multi-run specific plots
# ---------------------------------------------------------------------------


def _plot_metric_progression_across_runs(
    run_labels: list[str],
    run_metrics: list[dict[str, float]],
    out_path: Path,
) -> None:
    """
    Line chart of mean validation metrics across N experiment runs.

    Plots MAE, MAPE, R² and directional accuracy side by side so the progression
    across experiments is immediately visible.
    """
    metrics_to_plot = ["mae", "mape", "r2", "directional_accuracy"]
    available = [m for m in metrics_to_plot if any(m in rm for rm in run_metrics)]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    x = np.arange(len(run_labels))
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    for i, metric in enumerate(available):
        ax = axes[0, i]
        vals = [rm.get(metric, np.nan) for rm in run_metrics]
        color = colors[i % len(colors)]
        ax.plot(x, vals, marker="o", linewidth=2, color=color)
        ax.fill_between(x, vals, alpha=0.12, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=7)
        ax.set_title(metric.upper().replace("_", " "), fontsize=10)
        ax.set_ylabel(f"mean {metric}")
        for xi, vi in zip(x, vals):
            if not np.isnan(vi):
                ax.annotate(
                    f"{vi:.3f}",
                    (xi, vi),
                    textcoords="offset points",
                    xytext=(0, 7),
                    fontsize=7,
                    ha="center",
                )
    fig.suptitle("Mean validation metrics per experiment (all models)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_evolution_stacked(
    evo_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Stacked bar chart showing shared, added, and removed features between consecutive runs.

    Helps visualise the feature engineering churn across experiments.
    """
    if evo_df.empty:
        return
    labels = [f"{row['run_a']} → {row['run_b']}" for _, row in evo_df.iterrows()]
    shared = evo_df["n_features_shared"].tolist()
    added = evo_df["n_features_added"].tolist()
    removed = evo_df["n_features_removed"].tolist()

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 3), 5))
    bars_shared = ax.bar(x, shared, color="steelblue", alpha=0.85, label="Shared")
    bars_added = ax.bar(x, added, bottom=shared, color="seagreen", alpha=0.85, label="Added")
    bottom_removed = [s + a for s, a in zip(shared, added)]
    bars_removed = ax.bar(x, removed, bottom=bottom_removed, color="tomato", alpha=0.85, label="Removed")

    for bar, val in zip(bars_shared, shared):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, str(val), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for bar, base, val in zip(bars_added, shared, added):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, base + val / 2, f"+{val}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for bar, base, val in zip(bars_removed, bottom_removed, removed):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, base + val / 2, f"-{val}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Number of features")
    ax.set_title("Feature engineering evolution between consecutive experiments")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_win_rate_comparison(
    pair_labels: list[str],
    pair_summaries: list[pd.DataFrame],
    out_path: Path,
) -> None:
    """
    Grouped bar chart of win rates (fraction of models improved) for each pairwise comparison.

    Win rate > 0.5 = majority of models improved. Plots MAE, R² and directional_accuracy.
    """
    win_metrics = ["mae", "r2", "directional_accuracy"]
    available = [m for m in win_metrics if any(f"win_rate_{m}" in s.columns for s in pair_summaries)]
    if not available or not pair_labels:
        return

    n_pairs = len(pair_labels)
    n_metrics = len(available)
    x = np.arange(n_pairs)
    width = 0.8 / n_metrics
    colors = ["steelblue", "seagreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 2.5), 5))
    for mi, metric in enumerate(available):
        offsets = (mi - n_metrics / 2 + 0.5) * width
        vals = []
        for summary in pair_summaries:
            col = f"win_rate_{metric}"
            if col in summary.columns and len(summary):
                vals.append(float(summary.iloc[0][col]))
            else:
                vals.append(np.nan)
        bars = ax.bar(x + offsets, vals, width, label=metric, color=colors[mi % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Win rate (fraction of models improved)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Win rates per pairwise comparison")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pairwise comparison outputs (two runs)
# ---------------------------------------------------------------------------


def _run_pairwise_comparison(
    baseline_root: Path,
    candidate_root: Path,
    out_dir: Path,
    top_k: int,
    max_lc_models: int,
    rng: np.random.Generator,
    logger: Any,
    anchor_model_keys: frozenset[str] | None = None,
    anchor_run_label: str | None = None,
) -> tuple[dict[str, ModelMetadataRecord], dict[str, ModelMetadataRecord], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute a full pairwise comparison between two run directories.

    Writes all outputs (CSVs, figures, REPORT.md) into ``out_dir``.

    When ``anchor_model_keys`` is set, both sides are restricted to those keys so metrics
    match the cohort produced by the latest training run (intersected with keys present in
    each run).

    Returns:
        Tuple of ``(base_recs, cand_recs, comparison_df, summary_overall, summary_horizon)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_label = baseline_root.name
    candidate_label = candidate_root.name

    with logger.log_operation("load_metadata"):
        base_recs = records_by_key(baseline_root)
        cand_recs = records_by_key(candidate_root)

    if anchor_model_keys is not None:
        base_recs = filter_records_to_model_keys(base_recs, anchor_model_keys)
        cand_recs = filter_records_to_model_keys(cand_recs, anchor_model_keys)

    keys_b = set(base_recs.keys())
    keys_c = set(cand_recs.keys())
    matched = sorted(keys_b & keys_c)
    only_b = sorted(keys_b - keys_c)
    only_c = sorted(keys_c - keys_b)

    (out_dir / "keys_baseline_only.txt").write_text("\n".join(only_b), encoding="utf-8")
    (out_dir / "keys_candidate_only.txt").write_text("\n".join(only_c), encoding="utf-8")

    comp = build_comparison_frame(base_recs, cand_recs)
    comp.to_csv(out_dir / "comparison_matched.csv", index=False)

    summary_overall = summarize_metrics(comp, group_col=None)
    summary_overall.to_csv(out_dir / "summary_overall.csv", index=False)

    summary_horizon = summarize_metrics(comp, group_col="horizon")
    summary_horizon.to_csv(out_dir / "summary_by_horizon.csv", index=False)

    global_b = mean_normalized_importance_by_feature(base_recs, matched)
    global_c = mean_normalized_importance_by_feature(cand_recs, matched)
    global_b.to_csv(out_dir / "feature_importance_global_baseline.csv", header=["mean_norm_importance"])
    global_c.to_csv(out_dir / "feature_importance_global_candidate.csv", header=["mean_norm_importance"])

    _plot_global_importance_bars(global_b, "Baseline — mean normalized importance", out_dir / "feature_importance_global_baseline.png", top_n=min(top_k, 25))
    _plot_global_importance_bars(global_c, "Candidate — mean normalized importance", out_dir / "feature_importance_global_candidate.png", top_n=min(top_k, 25))

    for m in ("mae", "rmse"):
        _plot_scatter_metric(comp, m, baseline_label, candidate_label, out_dir / f"scatter_{m}.png")

    _plot_delta_histograms(comp, out_dir / "delta_histograms.png")
    _plot_boxplot_delta_mae_by_horizon(comp, out_dir / "boxplot_delta_mae_by_horizon.png")

    lc_keys = _select_learning_curve_models(comp, baseline_root, candidate_root, max_lc_models, rng)
    lc_dir = out_dir / "learning_curves"
    lc_dir.mkdir(exist_ok=True)
    for key in lc_keys:
        fragment = _safe_filename_fragment(key)
        _plot_learning_curve_model(baseline_root / key, f"{baseline_label} | {key}", lc_dir / f"baseline_{fragment}.png")
        _plot_learning_curve_model(candidate_root / key, f"{candidate_label} | {key}", lc_dir / f"candidate_{fragment}.png")

    pair_dir = out_dir / "feature_importance_pairs"
    pair_dir.mkdir(exist_ok=True)
    if len(lc_keys) >= 3:
        example_keys = [lc_keys[0], lc_keys[len(lc_keys) // 2], lc_keys[-1]]
    elif lc_keys:
        example_keys = list(lc_keys)
    else:
        example_keys = matched[: min(3, len(matched))]
    for key in example_keys:
        if key not in base_recs or key not in cand_recs:
            continue
        _plot_topk_pair(base_recs[key], cand_recs[key], key, pair_dir / f"pair_{_safe_filename_fragment(key)}.png", top_k=top_k)

    _write_pairwise_report_markdown(
        out_dir,
        baseline_root,
        candidate_root,
        len(matched),
        len(only_b),
        len(only_c),
        summary_overall,
        summary_horizon,
        anchor_run_label=anchor_run_label,
        n_anchor_keys=len(anchor_model_keys) if anchor_model_keys is not None else None,
    )

    logger.info(
        "Pairwise comparison complete",
        extra={
            "output_dir": str(out_dir),
            "baseline": baseline_label,
            "candidate": candidate_label,
            "n_matched": len(matched),
            "n_baseline_only": len(only_b),
            "n_candidate_only": len(only_c),
            "anchored_to_latest": anchor_model_keys is not None,
            "n_anchor_keys": len(anchor_model_keys) if anchor_model_keys is not None else None,
        },
    )
    return base_recs, cand_recs, comp, summary_overall, summary_horizon


def _write_pairwise_report_markdown(
    out_dir: Path,
    baseline: Path,
    candidate: Path,
    n_match: int,
    n_base_only: int,
    n_cand_only: int,
    summary_overall: pd.DataFrame,
    summary_horizon: pd.DataFrame,
    *,
    anchor_run_label: str | None = None,
    n_anchor_keys: int | None = None,
) -> None:
    disclaimer = textwrap.dedent(
        """
        **Disclaimer:** Metrics are validation-set values stored at training time. The two runs may
        differ in training sample counts, feature sets, and data windows — this is not a controlled
        holdout replay. Interpret aggregate shifts together with per-model sample sizes in
        `comparison_matched.csv`.
        """
    ).strip()

    def _df_to_md_table(df: pd.DataFrame) -> str:
        return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# Model training run comparison",
        "",
        f"- **Baseline:** `{baseline}`",
        f"- **Candidate:** `{candidate}`",
        f"- **Matched models:** {n_match}",
        f"- **Baseline-only keys:** {n_base_only}",
        f"- **Candidate-only keys:** {n_cand_only}",
    ]
    if anchor_run_label is not None and n_anchor_keys is not None:
        lines += [
            "",
            f"- **Comparison cohort:** Model keys restricted to those present in the latest run "
            f"`{anchor_run_label}` (**{n_anchor_keys}** keys). Pairwise metrics include only keys "
            "that exist on **both** sides within this cohort.",
        ]
    lines += [
        "",
        "## Overall summary (deltas = candidate − baseline)",
        "",
        _df_to_md_table(summary_overall),
        "",
        "## By horizon",
        "",
        _df_to_md_table(summary_horizon),
        "",
        "## Interpretation",
        "",
        disclaimer,
    ]
    if anchor_run_label is not None:
        lines += [
            "",
            "When a comparison cohort is set, aggregate metrics refer to the intersection of that "
            "cohort with models available in each run — not the full universe of each run alone.",
        ]
    lines += [
        "",
        "## Outputs",
        "",
        "- `comparison_matched.csv` — per-model metrics and feature-overlap stats",
        "- `summary_overall.csv`, `summary_by_horizon.csv`",
        "- `keys_baseline_only.txt`, `keys_candidate_only.txt`",
        "- Figures: `scatter_*.png`, `delta_histograms.png`, `boxplot_delta_mae_by_horizon.png`, "
        "`feature_importance_global_*.png`, `learning_curves/*.png`, `feature_importance_pairs/*.png`",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Multi-run comparison report
# ---------------------------------------------------------------------------


def _format_win_rate(rate: float) -> str:
    """Return a coloured (emoji-free) text indicator for win-rate quality."""
    if np.isnan(rate):
        return "n/a"
    pct = f"{rate:.1%}"
    if rate >= 0.60:
        return f"{pct} (strong improvement)"
    if rate >= 0.50:
        return f"{pct} (marginal improvement)"
    if rate >= 0.40:
        return f"{pct} (marginal regression)"
    return f"{pct} (strong regression)"


def _build_run_summary_table(
    run_labels: list[str],
    run_records: list[dict[str, ModelMetadataRecord]],
) -> pd.DataFrame:
    """Return a DataFrame with one row per run showing key aggregate metrics."""
    rows = []
    for label, recs in zip(run_labels, run_records):
        metrics = aggregate_run_metrics(recs)
        rows.append(
            {
                "run": label,
                "n_models": len(recs),
                "mean_mae": metrics.get("mae", np.nan),
                "mean_mape": metrics.get("mape", np.nan),
                "mean_r2": metrics.get("r2", np.nan),
                "mean_directional_accuracy": metrics.get("directional_accuracy", np.nan),
                "mean_rmse": metrics.get("rmse", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _generate_recommendations(
    run_labels: list[str],
    pair_labels: list[str],
    pair_summaries_overall: list[pd.DataFrame],
    evo_df: pd.DataFrame,
) -> list[str]:
    """
    Derive data-driven improvement recommendations from pairwise comparison summaries.

    Returns a list of markdown bullet-point strings.
    """
    bullets: list[str] = []

    for pair_label, summary in zip(pair_labels, pair_summaries_overall):
        if summary.empty:
            continue
        row = summary.iloc[0]

        wr_mae = float(row.get("win_rate_mae", np.nan))
        wr_da = float(row.get("win_rate_directional_accuracy", np.nan))
        wr_r2 = float(row.get("win_rate_r2", np.nan))
        delta_mae_med = float(row.get("delta_mae_median", np.nan))

        assessment_parts = []
        if not np.isnan(wr_mae):
            if wr_mae < 0.40:
                assessment_parts.append(f"MAE regressed in {1 - wr_mae:.0%} of models")
            elif wr_mae > 0.60:
                assessment_parts.append(f"MAE improved in {wr_mae:.0%} of models")
        if not np.isnan(wr_da):
            if wr_da > 0.55:
                assessment_parts.append(f"directional accuracy improved in {wr_da:.0%} of models (most actionable signal)")
            elif wr_da < 0.45:
                assessment_parts.append(f"directional accuracy regressed in {1 - wr_da:.0%} of models")

        if assessment_parts:
            bullets.append(f"**{pair_label}**: {'; '.join(assessment_parts)}.")

    # Feature-churn guidance
    if not evo_df.empty:
        for _, row in evo_df.iterrows():
            n_added = int(row.get("n_features_added", 0))
            n_removed = int(row.get("n_features_removed", 0))
            added_names = row.get("features_added") or []
            removed_names = row.get("features_removed") or []
            label = f"{row['run_a']} → {row['run_b']}"

            if n_added:
                display_added = ", ".join(f"`{f}`" for f in added_names[:8])
                suffix = f" … (+{n_added - 8} more)" if n_added > 8 else ""
                bullets.append(f"**{label}** added {n_added} feature(s): {display_added}{suffix}.")
            if n_removed:
                display_removed = ", ".join(f"`{f}`" for f in removed_names[:8])
                suffix = f" … (+{n_removed - 8} more)" if n_removed > 8 else ""
                bullets.append(f"**{label}** removed {n_removed} feature(s): {display_removed}{suffix}.")

    # General guidance based on last pair
    if pair_summaries_overall:
        last_summary = pair_summaries_overall[-1]
        if not last_summary.empty:
            row = last_summary.iloc[0]
            wr_mae = float(row.get("win_rate_mae", np.nan))
            wr_da = float(row.get("win_rate_directional_accuracy", np.nan))

            if not np.isnan(wr_da) and wr_da > 0.55:
                bullets.append(
                    "**Directional accuracy is improving** — the model is better at forecasting price direction. "
                    "Continue building on the latest feature set; consider adding cross-currency correlation "
                    "or order-book proxies if available."
                )
            if not np.isnan(wr_mae) and wr_mae < 0.45:
                bullets.append(
                    "**MAE is degrading** — the latest feature changes may be introducing noise or the training "
                    "data window has changed materially. Audit `new_feature_importance_share` in "
                    "`comparison_matched.csv` to confirm new features are carrying weight. "
                    "Consider re-enabling robust scaling or reducing tree depth to combat overfitting."
                )
            if not np.isnan(wr_mae) and not np.isnan(wr_da) and wr_da > 0.55 and wr_mae < 0.45:
                bullets.append(
                    "**Conflicting signals (directional up, MAE down)** — the models are better at direction but "
                    "worse at magnitude. This is common when volatility features improve sign prediction but "
                    "price-level features regress. Consider a two-stage loss function (direction + magnitude)."
                )

    if not bullets:
        bullets.append("Insufficient data to generate automated recommendations. Inspect per-model CSVs manually.")

    return bullets


def _write_multi_run_report_markdown(
    out_dir: Path,
    run_paths: list[Path],
    run_labels: list[str],
    run_summary_df: pd.DataFrame,
    evo_df: pd.DataFrame,
    pair_labels: list[str],
    pair_dirs: list[Path],
    pair_summaries_overall: list[pd.DataFrame],
    pair_summaries_horizon: list[pd.DataFrame],
    recommendations: list[str],
    *,
    anchor_to_latest: bool = True,
    latest_run_label: str | None = None,
    n_anchor_keys: int | None = None,
) -> None:
    """Write the comprehensive multi-experiment REPORT.md."""

    def _df_to_md_table(df: pd.DataFrame) -> str:
        return "```\n" + df.to_string(index=False) + "\n```"

    disclaimer = textwrap.dedent(
        """
        **Disclaimer:** All metrics are validation-set values recorded at training time. Runs may differ
        in training sample counts, feature sets, data windows, and the set of currencies trained — this
        is **not** a controlled holdout replay. Interpret aggregate shifts alongside per-model sample
        sizes and feature-overlap statistics in each pairwise `comparison_matched.csv`.
        """
    ).strip()

    lines: list[str] = [
        "# Multi-experiment model comparison",
        "",
    ]
    if anchor_to_latest and latest_run_label is not None and n_anchor_keys is not None:
        lines += [
            "## Comparison cohort (anchored to latest run)",
            "",
            f"Pairwise metrics, run-level means in the table below, and feature evolution "
            f"are restricted to model keys produced by the latest training run "
            f"`{latest_run_label}` (**{n_anchor_keys}** keys). Older runs contribute fewer rows "
            "if they did not train every key in that cohort.",
            "",
        ]
    lines += [
        "## Experiments",
        "",
    ]
    for i, (path, label) in enumerate(zip(run_paths, run_labels), start=1):
        lines.append(f"{i}. `{path}` (label: **{label}**)")
    lines += [""]

    lines += [
        "## Run summary",
        "",
        "Mean validation metrics across models in each run"
        + (
            " (cohort = keys from latest run, intersected with models available in that run)."
            if anchor_to_latest
            else "."
        ),
        "",
        _df_to_md_table(run_summary_df),
        "",
    ]

    if not evo_df.empty:
        evo_display = evo_df.drop(columns=["features_added", "features_removed"], errors="ignore")
        lines += [
            "## Feature engineering evolution",
            "",
            "Feature counts and churn between consecutive runs (union across common models"
            + ("; cohort anchored to latest run" if anchor_to_latest else "")
            + ").",
            "",
            _df_to_md_table(evo_display),
            "",
        ]
        for _, row in evo_df.iterrows():
            pair_label = f"{row['run_a']} → {row['run_b']}"
            added = row.get("features_added") or []
            removed = row.get("features_removed") or []
            if added or removed:
                lines.append(f"### {pair_label}")
                if added:
                    lines += ["", f"**Added ({len(added)}):** " + ", ".join(f"`{f}`" for f in added), ""]
                if removed:
                    lines += ["", f"**Removed ({len(removed)}):** " + ", ".join(f"`{f}`" for f in removed), ""]

    lines += [
        "## Pairwise comparison summaries",
        "",
        "All deltas = candidate − baseline.",
        "",
    ]
    for pair_label, summary_overall, summary_horizon in zip(pair_labels, pair_summaries_overall, pair_summaries_horizon):
        lines += [
            f"### {pair_label}",
            "",
            "**Overall:**",
            "",
            _df_to_md_table(summary_overall),
            "",
            "**By horizon:**",
            "",
            _df_to_md_table(summary_horizon),
            "",
        ]

    lines += [
        "## Findings and recommendations",
        "",
    ]
    for bullet in recommendations:
        lines.append(f"- {bullet}")
    lines += [""]

    lines += [
        "## Interpretation",
        "",
        disclaimer,
        "",
        "## Output structure",
        "",
        "- `run_summary.csv` — per-run mean metrics"
        + (" (same cohort as latest run)" if anchor_to_latest else ""),
        "- `feature_evolution.csv` — feature churn between consecutive runs",
        "- `metric_progression.png` — mean metric trend across experiments",
        "- `feature_evolution.png` — stacked bar of feature additions/removals",
        "- `win_rate_comparison.png` — win rates per pairwise comparison",
    ]
    for pair_label, pair_dir in zip(pair_labels, pair_dirs):
        lines.append(f"- `{pair_dir.name}/` — full pairwise output for **{pair_label}**")
    lines += [""]

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Multi-run orchestration
# ---------------------------------------------------------------------------


def _run_multi_experiment_comparison(
    run_paths: list[Path],
    out_dir: Path,
    top_k: int,
    max_lc_models: int,
    seed: int,
    logger: Any,
    anchor_to_latest: bool = True,
) -> None:
    """
    Orchestrate a multi-run (N >= 2) experiment comparison.

    For N runs, produces:
    - Pairwise comparisons between every consecutive pair (0→1, 1→2, …).
    - Pairwise comparison between first and last run (baseline vs latest).
    - Aggregate progression charts across all runs.
    - Comprehensive REPORT.md with feature evolution and data-driven recommendations.

    When ``anchor_to_latest`` is True (default), all pairwise metrics, run-level aggregates,
    and feature-evolution stats use only model keys present in the **last** run in
    ``run_paths`` (ordered oldest-to-newest), so experiments are comparable on the same cohort.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_labels = [p.name for p in run_paths]

    logger.info("Loading metadata for all runs", extra={"n_runs": len(run_paths), "labels": run_labels})

    all_records: list[dict[str, ModelMetadataRecord]] = []
    for path in run_paths:
        recs = records_by_key(path)
        all_records.append(recs)
        logger.info("Loaded run", extra={"run": path.name, "n_models": len(recs)})

    anchor_keys: frozenset[str] | None = None
    latest_label = run_labels[-1]
    if anchor_to_latest:
        anchor_keys = frozenset(all_records[-1].keys())
        logger.info(
            "Anchoring comparison cohort to latest run",
            extra={"latest_run": latest_label, "n_anchor_keys": len(anchor_keys)},
        )

    all_records_cohort: list[dict[str, ModelMetadataRecord]] = (
        [filter_records_to_model_keys(r, anchor_keys) for r in all_records]
        if anchor_keys is not None
        else all_records
    )

    # ---- Pairwise comparisons (consecutive pairs + baseline vs latest) ----
    rng = np.random.default_rng(seed)
    pairs_to_compare: list[tuple[int, int]] = [(i, i + 1) for i in range(len(run_paths) - 1)]
    if len(run_paths) > 2:
        first_last = (0, len(run_paths) - 1)
        if first_last not in pairs_to_compare:
            pairs_to_compare.append(first_last)

    pair_labels: list[str] = []
    pair_dirs: list[Path] = []
    pair_summaries_overall: list[pd.DataFrame] = []
    pair_summaries_horizon: list[pd.DataFrame] = []
    loaded_records: dict[int, dict[str, ModelMetadataRecord]] = {i: r for i, r in enumerate(all_records)}

    for idx_a, idx_b in pairs_to_compare:
        label = f"{run_labels[idx_a]}_vs_{run_labels[idx_b]}"
        pair_dir = out_dir / label
        logger.info("Running pairwise comparison", extra={"pair": label})
        _, _, _, so, sh = _run_pairwise_comparison(
            run_paths[idx_a],
            run_paths[idx_b],
            pair_dir,
            top_k,
            max_lc_models,
            rng,
            logger,
            anchor_model_keys=anchor_keys,
            anchor_run_label=latest_label if anchor_keys is not None else None,
        )
        pair_labels.append(label.replace("_", " "))
        pair_dirs.append(pair_dir)
        pair_summaries_overall.append(so)
        pair_summaries_horizon.append(sh)

    # ---- Aggregate run summary ----
    run_summary_df = _build_run_summary_table(run_labels, all_records_cohort)
    run_summary_df.to_csv(out_dir / "run_summary.csv", index=False)

    # ---- Feature evolution ----
    evo_df = build_feature_evolution_frame(
        run_labels,
        all_records_cohort,
        anchor_model_keys=anchor_keys,
    )
    evo_df.to_csv(out_dir / "feature_evolution.csv", index=False)

    # ---- Progression and evolution plots ----
    run_metrics_list = [aggregate_run_metrics(r) for r in all_records_cohort]
    _plot_metric_progression_across_runs(run_labels, run_metrics_list, out_dir / "metric_progression.png")
    _plot_feature_evolution_stacked(evo_df, out_dir / "feature_evolution.png")

    # Win-rate plot covers only the consecutive pairs for clarity (not baseline→latest repeat).
    consecutive_pair_labels = [f"{run_labels[i]} → {run_labels[i + 1]}" for i in range(len(run_paths) - 1)]
    consecutive_summaries = pair_summaries_overall[: len(run_paths) - 1]
    _plot_win_rate_comparison(consecutive_pair_labels, consecutive_summaries, out_dir / "win_rate_comparison.png")

    # ---- Recommendations ----
    recommendations = _generate_recommendations(run_labels, pair_labels, pair_summaries_overall, evo_df)

    # ---- Comprehensive report ----
    _write_multi_run_report_markdown(
        out_dir,
        run_paths,
        run_labels,
        run_summary_df,
        evo_df,
        pair_labels,
        pair_dirs,
        pair_summaries_overall,
        pair_summaries_horizon,
        recommendations,
        anchor_to_latest=anchor_to_latest,
        latest_run_label=latest_label,
        n_anchor_keys=len(anchor_keys) if anchor_keys is not None else None,
    )

    logger.info(
        "Multi-experiment comparison complete",
        extra={
            "output_dir": str(out_dir),
            "n_runs": len(run_paths),
            "n_pairs_compared": len(pairs_to_compare),
            "report": str(out_dir / "REPORT.md"),
            "anchored_to_latest": anchor_keys is not None,
            "n_anchor_keys": len(anchor_keys) if anchor_keys is not None else None,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more model training run directories. "
            "Use --runs for multi-experiment mode (recommended); "
            "--baseline/--candidate are kept for backward compatibility."
        )
    )
    # Multi-run interface
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        help="Two or more run root directories, ordered oldest-to-newest.",
    )
    # Legacy two-run interface
    parser.add_argument("--baseline", type=Path, help="(Legacy) Baseline run root directory.")
    parser.add_argument("--candidate", type=Path, help="(Legacy) Candidate run root directory.")
    # Shared options
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for all outputs.")
    parser.add_argument("--top-k-features", type=int, default=15, help="Top features for bar charts.")
    parser.add_argument(
        "--max-learning-curve-models",
        type=int,
        default=9,
        help="Max number of models to plot learning curves for (both runs).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for stratified LC sampling.")
    parser.add_argument(
        "--no-anchor-to-latest",
        action="store_true",
        help=(
            "Disable anchoring: use every model key each run has instead of restricting to keys "
            "from the last run in --runs (or the candidate run for --baseline/--candidate)."
        ),
    )
    args = parser.parse_args()

    logger = MLLogger("compare_model_training_runs", level="INFO", console_output=True)
    out_dir = args.output_dir.resolve()
    top_k = max(5, args.top_k_features)

    # Resolve run list: --runs takes priority; fall back to --baseline/--candidate.
    if args.runs:
        run_paths = [p.resolve() for p in args.runs]
        if len(run_paths) < 2:
            parser.error("--runs requires at least two directories.")
    elif args.baseline and args.candidate:
        run_paths = [args.baseline.resolve(), args.candidate.resolve()]
    else:
        parser.error("Provide either --runs (2+) or both --baseline and --candidate.")

    _run_multi_experiment_comparison(
        run_paths=run_paths,
        out_dir=out_dir,
        top_k=top_k,
        max_lc_models=args.max_learning_curve_models,
        seed=args.seed,
        logger=logger,
        anchor_to_latest=not args.no_anchor_to_latest,
    )


if __name__ == "__main__":
    main()
