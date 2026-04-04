#!/usr/bin/env python3
"""
Configuration for ML model inference (prediction) pipeline.

Uses the shared DataConfig and ProcessingConfig from training with
inference-appropriate defaults, eliminating duplicate class definitions.

Environment Variables:
    MAX_LEAGUE_DAYS: Maximum days of league history to use (default: 200)
    MIN_LEAGUE_DAYS: Minimum days required (default: 0)
    MAX_NAN_RATIO: Maximum NaN ratio per row before dropping (default: 0.9)
    MODELS_DIR: Directory containing trained model artifacts (default: /var/task/models)
    PREDICTION_CACHE_TTL_HOURS: TTL for prediction cache in hours (default: 2)
    DEFAULT_CONFIDENCE_LEVEL: Confidence level for prediction intervals (default: 0.60)
    LOG_LEVEL: Logging level (default: INFO)
    DISABLE_FILE_LOGGING: Set to 'true' to disable file logging (default: false)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from ml.config.training_config import DataConfig, DynamoConfig, LoggingConfig, ProcessingConfig


def _build_inference_data_config() -> DataConfig:
    """Create a DataConfig with inference-appropriate defaults.

    Key differences from training:
    - max_league_days is larger (200 vs 60) to use all available price history.
    - min_league_days can be overridden via env var.
    """
    config = DataConfig()
    config.max_league_days = int(os.getenv("MAX_LEAGUE_DAYS", "200"))
    config.min_league_days = int(os.getenv("MIN_LEAGUE_DAYS", "0"))
    return config


def _build_inference_processing_config() -> ProcessingConfig:
    """Create a ProcessingConfig with inference-appropriate defaults.

    Key differences from training:
    - Outlier removal is disabled — inference must predict on any input value.
    - max_nan_ratio is handled by ProcessingConfig itself via MAX_NAN_RATIO env var,
      keeping the value consistent with the training path.
    """
    config = ProcessingConfig()
    config.outlier_removal = False
    return config


@dataclass
class InferenceConfig:
    """Complete configuration for ML model inference."""

    data: DataConfig = field(default_factory=_build_inference_data_config)
    processing: ProcessingConfig = field(default_factory=_build_inference_processing_config)
    dynamo: DynamoConfig = field(default_factory=DynamoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "/var/task/models")))

    prediction_cache_ttl_hours: int = field(
        default_factory=lambda: int(os.getenv("PREDICTION_CACHE_TTL_HOURS", "2"))
    )

    # Confidence level used when computing ensemble prediction intervals.
    # 0.60 is the production-proven default; override with DEFAULT_CONFIDENCE_LEVEL env var.
    default_confidence_level: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_CONFIDENCE_LEVEL", "0.60"))
    )

    def __post_init__(self) -> None:
        """Ensure models directory exists for local development."""
        if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            self.models_dir.mkdir(parents=True, exist_ok=True)


def get_inference_config() -> InferenceConfig:
    """Get the inference configuration.

    All env vars are read at instantiation time via field default_factory.
    """
    return InferenceConfig()


# Alias kept so existing callers compile without change.
# Both functions are equivalent: env vars are already applied at dataclass init.
get_inference_config_from_env = get_inference_config


__all__ = [
    "InferenceConfig",
    "get_inference_config",
    "get_inference_config_from_env",
]
