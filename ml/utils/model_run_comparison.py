"""
Utilities for comparing two on-disk model training runs from ``model_metadata.json``
and ``training_history.json`` artifacts (no loaded estimators required).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METRIC_KEYS = ("mae", "rmse", "mape", "r2", "directional_accuracy")
HORIZON_PATTERN = re.compile(r"_(1d|3d|7d)$")


@dataclass(frozen=True)
class ModelMetadataRecord:
    """Single-model fields extracted from ``model_metadata.json``."""

    model_key: str
    model_dir: Path
    metrics: dict[str, float]
    training_samples: int
    n_features: int
    feature_names: list[str]
    feature_importance: dict[str, float]


def parse_horizon(model_key: str) -> str | None:
    """Return horizon suffix ``1d``/``3d``/``7d`` or ``None`` if not matched."""
    match = HORIZON_PATTERN.search(model_key)
    if match is None:
        return None
    return match.group(1)


def discover_model_dirs(run_root: Path) -> list[Path]:
    """
    Return sorted immediate child directories that contain ``model_metadata.json``.

    Args:
        run_root: Root directory for one training run (e.g. ``.../xp_20260405_092445``).
    """
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root is not a directory: {run_root}")
    result: list[Path] = []
    for child in sorted(run_root.iterdir()):
        if child.is_dir() and (child / "model_metadata.json").is_file():
            result.append(child)
    return result


def load_model_metadata_json(model_dir: Path) -> dict[str, Any]:
    """Load raw ``model_metadata.json`` from a model directory."""
    path = model_dir / "model_metadata.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def record_from_metadata(model_dir: Path, raw: dict[str, Any]) -> ModelMetadataRecord:
    """Build a :class:`ModelMetadataRecord` from parsed JSON."""
    currency = str(raw.get("currency", model_dir.name))
    metrics_raw = raw.get("metrics") or {}
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        val = metrics_raw.get(key)
        if val is None:
            continue
        metrics[key] = float(val)

    feature_names = list(raw.get("feature_names") or [])
    fi = raw.get("feature_importance") or {}
    feature_importance = {str(k): float(v) for k, v in fi.items()}

    return ModelMetadataRecord(
        model_key=currency,
        model_dir=model_dir,
        metrics=metrics,
        training_samples=int(raw.get("training_samples", 0)),
        n_features=int(raw.get("n_features", len(feature_names))),
        feature_names=feature_names,
        feature_importance=feature_importance,
    )


def iter_model_records(run_root: Path) -> Iterator[ModelMetadataRecord]:
    """Yield metadata records for every model directory under ``run_root``."""
    for model_dir in discover_model_dirs(run_root):
        raw = load_model_metadata_json(model_dir)
        yield record_from_metadata(model_dir, raw)


def records_by_key(run_root: Path) -> dict[str, ModelMetadataRecord]:
    """Map ``model_key`` -> record. Duplicate keys raise ``ValueError``."""
    out: dict[str, ModelMetadataRecord] = {}
    for rec in iter_model_records(run_root):
        if rec.model_key in out:
            raise ValueError(f"Duplicate model_key {rec.model_key!r} under {run_root}")
        out[rec.model_key] = rec
    return out


def filter_records_to_model_keys(
    records: dict[str, ModelMetadataRecord],
    model_keys: set[str] | frozenset[str],
) -> dict[str, ModelMetadataRecord]:
    """
    Return a subset of ``records`` whose keys appear in ``model_keys``.

    Used to anchor multi-run comparisons to a fixed cohort (e.g. models trained in the latest run).
    """
    return {k: records[k] for k in records if k in model_keys}


def load_training_history(model_dir: Path) -> dict[str, dict[str, list[float]]] | None:
    """
    Load ``training_history.json`` if present.

    Returns:
        Mapping ``base_learner_key -> {"train_loss": [...], "val_loss": [...]}``,
        or ``None`` if the file is missing.
    """
    path = model_dir / "training_history.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    result: dict[str, dict[str, list[float]]] = {}
    for learner_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        train_loss = [float(x) for x in payload.get("train_loss") or []]
        val_loss = [float(x) for x in payload.get("val_loss") or []]
        result[str(learner_key)] = {"train_loss": train_loss, "val_loss": val_loss}
    return result if result else None


def normalized_importances(feature_importance: dict[str, float]) -> dict[str, float]:
    """Normalize importances so they sum to 1; empty dict returns empty dict."""
    total = sum(feature_importance.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in feature_importance.items()}


def new_feature_importance_share(
    candidate_importance: dict[str, float],
    candidate_feature_names: set[str],
    baseline_feature_names: set[str],
) -> float:
    """
    Fraction of total candidate importance mass on features not in the baseline set.

    Returns:
        Value in ``[0, 1]``, or ``0.0`` if total importance is zero.
    """
    only_new = candidate_feature_names - baseline_feature_names
    total = sum(candidate_importance.values())
    if total <= 0.0:
        return 0.0
    new_sum = sum(candidate_importance.get(f, 0.0) for f in only_new)
    return float(new_sum / total)


def spearman_importance_correlation(
    baseline_imp: dict[str, float],
    candidate_imp: dict[str, float],
    feature_names_intersection: set[str],
) -> float | None:
    """
    Spearman correlation between importance vectors on the intersection of features.

    Returns:
        Correlation in ``[-1, 1]``, or ``None`` if fewer than two shared features.
    """
    names = sorted(feature_names_intersection)
    if len(names) < 2:
        return None
    a = np.array([baseline_imp.get(n, 0.0) for n in names], dtype=float)
    b = np.array([candidate_imp.get(n, 0.0) for n in names], dtype=float)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    ra = rankdata(a)
    rb = rankdata(b)
    corr = float(np.corrcoef(ra, rb)[0, 1])
    if np.isnan(corr):
        return None
    return corr


def build_comparison_frame(
    baseline: dict[str, ModelMetadataRecord],
    candidate: dict[str, ModelMetadataRecord],
) -> pd.DataFrame:
    """
    Inner-join baseline and candidate records on ``model_key``.

    Adds delta columns ``delta_<metric>`` (candidate - baseline), horizon, feature-set
    counts, Spearman correlation on shared features, and new-feature importance share.
    """
    keys = sorted(set(baseline.keys()) & set(candidate.keys()))
    rows: list[dict[str, Any]] = []
    for key in keys:
        b = baseline[key]
        c = candidate[key]
        row: dict[str, Any] = {
            "model_key": key,
            "horizon": parse_horizon(key),
            "baseline_training_samples": b.training_samples,
            "candidate_training_samples": c.training_samples,
            "baseline_n_features": b.n_features,
            "candidate_n_features": c.n_features,
        }
        for m in METRIC_KEYS:
            bv = b.metrics.get(m)
            cv = c.metrics.get(m)
            row[f"baseline_{m}"] = bv
            row[f"candidate_{m}"] = cv
            if bv is not None and cv is not None:
                row[f"delta_{m}"] = cv - bv
            else:
                row[f"delta_{m}"] = np.nan

        b_names = set(b.feature_names)
        c_names = set(c.feature_names)
        inter = b_names & c_names
        row["n_features_intersection"] = len(inter)
        row["n_features_only_baseline"] = len(b_names - c_names)
        row["n_features_only_candidate"] = len(c_names - b_names)
        row["new_feature_importance_share"] = new_feature_importance_share(
            c.feature_importance, c_names, b_names
        )
        row["spearman_importance_intersection"] = spearman_importance_correlation(
            b.feature_importance, c.feature_importance, inter
        )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """
    Mean/median/count and win rates for metric deltas.

    Win rate for MAE/RMSE/MAPE: fraction of rows where ``delta < 0`` (candidate better).
    Win rate for R2/directional_accuracy: fraction where ``delta > 0``.
    """
    delta_cols = [f"delta_{m}" for m in METRIC_KEYS if f"delta_{m}" in df.columns]
    if group_col is None:
        parts = [_summarize_group(df, "all", delta_cols)]
        return pd.DataFrame(parts)

    parts = []
    for group_val in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == group_val]
        parts.append(_summarize_group(sub, str(group_val), delta_cols))
    return pd.DataFrame(parts)


def _summarize_group(df: pd.DataFrame, label: str, delta_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"group": label, "n_models": len(df)}
    for col in delta_cols:
        metric = col.replace("delta_", "")
        series = df[col].dropna()
        out[f"{col}_mean"] = float(series.mean()) if len(series) else np.nan
        out[f"{col}_median"] = float(series.median()) if len(series) else np.nan
        if metric in ("mae", "rmse", "mape"):
            wins = (series < 0).sum()
        else:
            wins = (series > 0).sum()
        out[f"win_rate_{metric}"] = float(wins / len(series)) if len(series) else np.nan
    return out


def mean_normalized_importance_by_feature(
    records: dict[str, ModelMetadataRecord],
    model_keys: list[str] | None = None,
) -> pd.Series:
    """
    Mean of per-model normalized importances for each feature name.

    Args:
        records: Keyed by ``model_key``.
        model_keys: Subset to average over; default is all keys in ``records``.
    """
    keys = model_keys if model_keys is not None else sorted(records.keys())
    accum: dict[str, list[float]] = {}
    for mk in keys:
        rec = records.get(mk)
        if rec is None:
            continue
        norm = normalized_importances(rec.feature_importance)
        for fname, val in norm.items():
            accum.setdefault(fname, []).append(val)
    means = {k: float(np.mean(v)) for k, v in accum.items()}
    return pd.Series(means).sort_values(ascending=False)


def top_k_importance_bar_data(
    record: ModelMetadataRecord,
    k: int,
) -> pd.DataFrame:
    """Return top-``k`` features by importance with normalized mass."""
    norm = normalized_importances(record.feature_importance)
    if not norm:
        return pd.DataFrame(columns=["feature", "importance_norm"])
    s = pd.Series(norm).sort_values(ascending=False).head(k)
    return pd.DataFrame({"feature": s.index, "importance_norm": s.values})


def aggregate_run_metrics(
    records: dict[str, ModelMetadataRecord],
    model_keys: list[str] | None = None,
) -> dict[str, float]:
    """
    Return mean validation metrics across all (or a subset of) models in a run.

    Args:
        records: Map of ``model_key`` -> :class:`ModelMetadataRecord`.
        model_keys: Subset of keys to average over; defaults to all keys.

    Returns:
        Dict mapping each metric name to its mean value across matched models.
        Metrics absent from all records are omitted.
    """
    keys = model_keys if model_keys is not None else list(records.keys())
    accum: dict[str, list[float]] = {m: [] for m in METRIC_KEYS}
    for mk in keys:
        rec = records.get(mk)
        if rec is None:
            continue
        for m in METRIC_KEYS:
            val = rec.metrics.get(m)
            if val is not None:
                accum[m].append(val)
    return {m: float(np.mean(v)) for m, v in accum.items() if v}


def build_feature_evolution_frame(
    run_labels: list[str],
    run_records: list[dict[str, ModelMetadataRecord]],
    anchor_model_keys: set[str] | frozenset[str] | None = None,
) -> pd.DataFrame:
    """
    Summarise feature-set changes between each pair of consecutive runs.

    For every adjacent pair ``(run_labels[i], run_labels[i+1])``, the union of
    feature names across all *common* models is computed for each run, then
    partitioned into shared, added, and removed feature sets.

    Args:
        run_labels: Human-readable label for each run (e.g. experiment directory name).
        run_records: Parallel list of ``model_key -> ModelMetadataRecord`` dicts.
        anchor_model_keys: When set, only models whose keys appear here are used when
            forming the common-model set for each consecutive pair (stabilizes churn stats
            to the same cohort as the latest run).

    Returns:
        DataFrame with one row per consecutive pair and columns:
        ``run_a``, ``run_b``, ``n_common_models``, ``n_features_run_a``,
        ``n_features_run_b``, ``n_features_shared``, ``n_features_added``,
        ``n_features_removed``, ``net_feature_change``,
        ``features_added`` (sorted list), ``features_removed`` (sorted list).
    """
    if len(run_labels) != len(run_records):
        raise ValueError("run_labels and run_records must have the same length")

    rows: list[dict[str, Any]] = []
    for i in range(len(run_labels) - 1):
        label_a, label_b = run_labels[i], run_labels[i + 1]
        recs_a, recs_b = run_records[i], run_records[i + 1]
        common = set(recs_a.keys()) & set(recs_b.keys())
        if anchor_model_keys is not None:
            common &= anchor_model_keys
        common_keys = sorted(common)
        if not common_keys:
            continue
        feats_a: set[str] = set()
        feats_b: set[str] = set()
        for k in common_keys:
            feats_a |= set(recs_a[k].feature_names)
            feats_b |= set(recs_b[k].feature_names)
        added = feats_b - feats_a
        removed = feats_a - feats_b
        shared = feats_a & feats_b
        rows.append(
            {
                "run_a": label_a,
                "run_b": label_b,
                "n_common_models": len(common_keys),
                "n_features_run_a": len(feats_a),
                "n_features_run_b": len(feats_b),
                "n_features_shared": len(shared),
                "n_features_added": len(added),
                "n_features_removed": len(removed),
                "net_feature_change": len(feats_b) - len(feats_a),
                "features_added": sorted(added),
                "features_removed": sorted(removed),
            }
        )
    return pd.DataFrame(rows)
