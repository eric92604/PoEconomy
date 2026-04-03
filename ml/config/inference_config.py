#!/usr/bin/env python3
"""
Configuration for ML model inference (prediction) pipeline.

This module provides configurations specifically optimized for inference,
separate from training configurations. Inference has different requirements:
- Use all available historical data (not limited by training constraints)
- Optimize for prediction accuracy over training speed
- Handle real-time prediction scenarios
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ml.config.training_config import DynamoConfig, LoggingConfig


@dataclass
class InferenceDataConfig:
    """Data configuration optimized for inference."""
    
    # League parameters - more permissive for inference
    max_league_days: int = int(os.getenv("MAX_LEAGUE_DAYS", "200"))
    min_league_days: int = int(os.getenv("MIN_LEAGUE_DAYS", "0"))
    
    # League inclusion settings - use all available leagues for inference
    included_leagues: Optional[List[str]] = None  # None means use all available
    excluded_leagues: List[str] = field(default_factory=list)
    
    # Currency selection strategy - no minimum records for inference
    min_records_threshold: int = int(os.getenv("MIN_RECORDS_THRESHOLD", "1"))  # Minimal threshold
    max_currencies_to_train: Optional[int] = None  # No limit for inference
    
    # Target variables - use all available horizons
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 7])
    
    # Feature engineering windows — must match training DataConfig
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 7])
    momentum_periods: List[int] = field(default_factory=lambda: [3, 5, 7])

    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])

    # EMA / MACD parameters
    ema_spans: List[int] = field(default_factory=lambda: [3, 7, 14])
    macd_fast_span: int = 5
    macd_slow_span: int = 14
    macd_signal_span: int = 3

    # Volatility feature engineering
    volatility_windows: List[int] = field(default_factory=lambda: [3, 5, 7])
    include_volatility_features: bool = True
    volatility_types: List[str] = field(default_factory=lambda: ['std', 'cv', 'range', 'garch'])


@dataclass
class InferenceProcessingConfig:
    """Processing configuration optimized for inference."""
    
    # Feature Engineering - comprehensive for best predictions
    max_lag_features: int = int(os.getenv("MAX_LAG_FEATURES", "15"))  # More lag features
    technical_indicators: bool = True
    statistical_features: bool = True
    
    # Data Quality - more lenient for inference
    outlier_threshold: float = float(os.getenv("OUTLIER_THRESHOLD", "4.0"))
    missing_value_threshold: float = float(os.getenv("MISSING_VALUE_THRESHOLD", "0.9"))

    # Fraction of feature values that may be NaN before a row is dropped.
    # Must match training ProcessingConfig.max_nan_ratio.
    max_nan_ratio: float = float(os.getenv("MAX_NAN_RATIO", "0.9"))
    
    # Scaling - keep robust scaling
    robust_scaling: bool = True
    
    # Price transformations
    log_transform: bool = True
    log_transform_ratio_threshold: float = 10.0
    
    # Data processing options - more conservative for inference
    outlier_removal: bool = bool(os.getenv("OUTLIER_REMOVAL", "false").lower() == "true")  # Keep outliers by default


@dataclass
class InferenceConfig:
    """Complete configuration for ML model inference."""
    
    # Core configurations
    data: InferenceDataConfig = field(default_factory=InferenceDataConfig)
    processing: InferenceProcessingConfig = field(default_factory=InferenceProcessingConfig)
    dynamo: DynamoConfig = field(default_factory=DynamoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Inference-specific settings
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "/var/task/models")))
    enable_model_caching: bool = True
    prediction_timeout_seconds: int = int(os.getenv("PREDICTION_TIMEOUT_SECONDS", "30"))
    
    # Performance settings
    max_concurrent_predictions: int = int(os.getenv("MAX_CONCURRENT_PREDICTIONS", "10"))
    enable_prediction_caching: bool = bool(os.getenv("ENABLE_PREDICTION_CACHING", "true").lower() == "true")
    prediction_cache_ttl_hours: int = int(os.getenv("PREDICTION_CACHE_TTL_HOURS", "2"))
    
    # Prediction interval settings
    default_confidence_level: float = float(os.getenv("DEFAULT_CONFIDENCE_LEVEL", "0.86"))  # 60% confidence for narrower prediction ranges
    
    def __post_init__(self) -> None:
        """Post-initialization setup."""
        # Ensure models directory exists (for local development)
        if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            self.models_dir.mkdir(parents=True, exist_ok=True)


def get_inference_config() -> InferenceConfig:
    """
    Get the default inference configuration.
    
    This configuration is optimized for making predictions with trained models,
    using all available historical data and comprehensive feature engineering.
    """
    return InferenceConfig()


def get_inference_config_from_env() -> InferenceConfig:
    """
    Get inference configuration with environment variable overrides.
    
    This allows runtime configuration of inference parameters through
    environment variables, which is useful for Lambda deployments.
    """
    config = InferenceConfig()
    
    # Override with environment variables if present
    if os.getenv("MAX_LEAGUE_DAYS"):
        config.data.max_league_days = int(os.getenv("MAX_LEAGUE_DAYS"))
    if os.getenv("MIN_LEAGUE_DAYS"):
        config.data.min_league_days = int(os.getenv("MIN_LEAGUE_DAYS", "10"))
    if os.getenv("MIN_RECORDS_THRESHOLD"):
        config.data.min_records_threshold = int(os.getenv("MIN_RECORDS_THRESHOLD", "1"))
    if os.getenv("MAX_LAG_FEATURES"):
        config.processing.max_lag_features = int(os.getenv("MAX_LAG_FEATURES", "20"))
    if os.getenv("OUTLIER_THRESHOLD"):
        config.processing.outlier_threshold = float(os.getenv("OUTLIER_THRESHOLD", "3.0"))
    if os.getenv("MISSING_VALUE_THRESHOLD"):
        config.processing.missing_value_threshold = float(os.getenv("MISSING_VALUE_THRESHOLD", "0.1"))
    if os.getenv("OUTLIER_REMOVAL"):
        outlier_removal = os.getenv("OUTLIER_REMOVAL")
        if outlier_removal:
            config.processing.outlier_removal = outlier_removal.lower() == "true"
    if os.getenv("PREDICTION_TIMEOUT_SECONDS"):
        config.prediction_timeout_seconds = int(os.getenv("PREDICTION_TIMEOUT_SECONDS", "30"))
    if os.getenv("MAX_CONCURRENT_PREDICTIONS"):
        config.max_concurrent_predictions = int(os.getenv("MAX_CONCURRENT_PREDICTIONS", "10"))
    if os.getenv("ENABLE_PREDICTION_CACHING"):
        caching = os.getenv("ENABLE_PREDICTION_CACHING")
        if caching:
            config.enable_prediction_caching = caching.lower() == "true"
    if os.getenv("PREDICTION_CACHE_TTL_HOURS"):
        config.prediction_cache_ttl_hours = int(os.getenv("PREDICTION_CACHE_TTL_HOURS", "2"))
    
    return config


__all__ = [
    "InferenceDataConfig",
    "InferenceProcessingConfig", 
    "InferenceConfig",
    "get_inference_config",
    "get_inference_config_from_env"
]
