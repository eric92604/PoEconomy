"""Unit tests for model run comparison utilities."""

from __future__ import annotations

from pathlib import Path

from ml.utils.model_run_comparison import (
    ModelMetadataRecord,
    build_feature_evolution_frame,
    filter_records_to_model_keys,
)


def _record(model_key: str, features: list[str]) -> ModelMetadataRecord:
    return ModelMetadataRecord(
        model_key=model_key,
        model_dir=Path("."),
        metrics={"mae": 1.0},
        training_samples=10,
        n_features=len(features),
        feature_names=features,
        feature_importance={f: 1.0 for f in features},
    )


def test_filter_records_to_model_keys_intersection() -> None:
    records = {
        "a": _record("a", ["x"]),
        "b": _record("b", ["y"]),
        "c": _record("c", ["z"]),
    }
    out = filter_records_to_model_keys(records, frozenset({"a", "c"}))
    assert set(out.keys()) == {"a", "c"}
    assert out["a"] is records["a"]


def test_build_feature_evolution_frame_with_anchor() -> None:
    """Anchor excludes keys that are not in the latest cohort from common-model union."""
    run_labels = ["old", "new"]
    recs_old = {
        "m1": _record("m1", ["f1", "f2"]),
        "m2": _record("m2", ["f1"]),
    }
    recs_new = {
        "m1": _record("m1", ["f1", "f2", "f3"]),
    }
    run_records = [recs_old, recs_new]
    full = build_feature_evolution_frame(run_labels, run_records, anchor_model_keys=None)
    assert len(full) == 1
    assert full.iloc[0]["n_common_models"] == 1

    anchored = build_feature_evolution_frame(
        run_labels,
        run_records,
        anchor_model_keys=frozenset({"m1"}),
    )
    assert len(anchored) == 1
    assert anchored.iloc[0]["n_common_models"] == 1

    anchored_m2_only = build_feature_evolution_frame(
        run_labels,
        run_records,
        anchor_model_keys=frozenset({"m2"}),
    )
    assert anchored_m2_only.empty
