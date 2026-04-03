"""
Centralized configuration management for ML training pipeline.

Environment Variables:
    # Threading and Parallelization
    MAX_CURRENCY_WORKERS: Number of parallel currency workers (default: 8)
    MAX_OPTUNA_WORKERS: Number of Optuna workers per currency (default: 1)
    MODEL_N_JOBS: Number of model training jobs per currency (default: 4)

    # Hyperparameter Optimization
    N_HYPERPARAMETER_TRIALS: Number of hyperparameter optimization trials (default: 200)
    N_MODEL_TRIALS: Number of model training iterations (default: 1000)
    CV_FOLDS: Number of cross-validation folds (default: 5)

    # Model Performance
    MAX_DEPTH: Maximum tree depth (default: 8)
    LEARNING_RATE: Learning rate for models (default: 0.1)

    # Data Selection
    MIN_RECORDS_THRESHOLD: Minimum records required for training (default: 50)
    MAX_CURRENCIES_TO_TRAIN: Maximum number of currencies to train (default: 0 = no limit)
    MIN_AVG_VALUE_THRESHOLD: Minimum average value threshold in Chaos Orbs (default: 0.25)

    # AWS Configuration
    AWS_REGION: AWS region (default: us-west-2)
    DATA_LAKE_BUCKET: S3 bucket for data lake
    DYNAMO_CURRENCY_METADATA_TABLE: DynamoDB currency metadata table name
    DYNAMO_CURRENCY_PRICES_TABLE: DynamoDB currency prices table name
    DYNAMO_LEAGUE_METADATA_TABLE: DynamoDB league metadata table name
    DYNAMO_DAILY_PRICES_TABLE: DynamoDB daily prices table name
    DYNAMO_PREDICTIONS_TABLE: DynamoDB predictions table name

    # Logging
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
    DISABLE_FILE_LOGGING: Set to 'true' to disable file logging (default: false)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os


@dataclass
class ModelConfig:
    """Configuration for ML model training."""

    # Model Selection
    use_lightgbm: bool = True
    use_random_forest: bool = True
    use_extra_trees: bool = True

    # Ensemble Weight Optimization
    optimize_ensemble_weights: bool = True
    ensemble_weight_optimization_trials: int = 50

    max_currency_workers: int = field(default_factory=lambda: int(os.getenv('MAX_CURRENCY_WORKERS', '8')))
    max_optuna_workers: int = field(default_factory=lambda: int(os.getenv('MAX_OPTUNA_WORKERS', '1')))
    model_n_jobs: int = field(default_factory=lambda: int(os.getenv('MODEL_N_JOBS', '4')))

    # Training Parameters
    random_state: int = 42
    cv_folds: int = field(default_factory=lambda: int(os.getenv('CV_FOLDS', '5')))
    early_stopping_rounds: int = 50

    # Hyperparameter Optimization
    n_hyperparameter_trials: int = field(default_factory=lambda: int(os.getenv('N_HYPERPARAMETER_TRIALS', '200')))

    # Model Performance
    max_depth: int = field(default_factory=lambda: int(os.getenv('MAX_DEPTH', '8')))
    learning_rate: float = field(default_factory=lambda: float(os.getenv('LEARNING_RATE', '0.1')))

    # Model Training Iterations
    n_model_trials: int = field(default_factory=lambda: int(os.getenv('N_MODEL_TRIALS', '1000')))


@dataclass
class DataConfig:
    """Configuration for data processing and feature engineering parameters."""

    # League parameters
    max_league_days: int = 60
    min_league_days: int = 0

    # League inclusion settings
    included_leagues: Optional[List[str]] = None
    excluded_leagues: List[str] = field(default_factory=list)

    # Currency selection
    min_records_threshold: int = field(default_factory=lambda: int(os.getenv('MIN_RECORDS_THRESHOLD', '50')))
    max_currencies_to_train: Optional[int] = field(
        default_factory=lambda: (
            int(os.getenv('MAX_CURRENCIES_TO_TRAIN', '0'))
            if os.getenv('MAX_CURRENCIES_TO_TRAIN') and int(os.getenv('MAX_CURRENCIES_TO_TRAIN', '0')) > 0
            else None
        )
    )
    min_avg_value_threshold: float = field(default_factory=lambda: float(os.getenv('MIN_AVG_VALUE_THRESHOLD', '0.25')))

    # Target variables
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 7])

    # Feature engineering windows — all must match between training and inference paths.
    # Rolling window sizes (window=1 is skipped internally — mean/min/max equal raw price)
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 7])
    # Multi-day momentum periods (period=1 is skipped internally)
    momentum_periods: List[int] = field(default_factory=lambda: [3, 5, 7])
    # Explicit price lag look-back periods — highest-impact feature class for tree-based models
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])
    # Exponential moving average spans
    ema_spans: List[int] = field(default_factory=lambda: [3, 7, 14])
    # MACD parameters (scaled for ≤60-day league windows; standard 12/26/9 is too wide)
    macd_fast_span: int = 5
    macd_slow_span: int = 14
    macd_signal_span: int = 3
    # Volatility feature windows
    volatility_windows: List[int] = field(default_factory=lambda: [3, 5, 7])

    # Feature engineering options
    include_league_features: bool = True
    # IQR multiplier for per-league outlier removal (3.0 ≈ 99.7% of a normal distribution)
    outlier_removal_iqr_multiplier: float = 3.0

    # Validation data configuration
    validation_max_days: int = 60


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""

    # Scaling
    robust_scaling: bool = True

    # Price transformations
    log_transform: bool = True
    log_transform_ratio_threshold: float = 10.0

    # Outlier removal (disabled in inference; controlled by profile in training)
    outlier_removal: bool = True

    # Fraction of feature values that may be NaN before a training row is dropped.
    # Must match the inference path which uses the same threshold.
    max_nan_ratio: float = 0.9


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""

    ml_root: Path = field(default_factory=lambda: _get_ml_root())
    data_dir: Path = field(default_factory=lambda: _get_data_dir())
    models_dir: Path = field(default_factory=lambda: _get_models_dir())
    logs_dir: Path = field(default_factory=lambda: _get_logs_dir())

    def __post_init__(self) -> None:
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            return
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


def _get_ml_root() -> Path:
    """Get ML root directory, adapting to Fargate structure."""
    if os.path.exists('/app/ml'):
        return Path('/app/ml')
    return Path(__file__).parent.parent.resolve()


def _get_data_dir() -> Path:
    """Get data directory, adapting to Fargate structure."""
    if os.path.exists('/app/data'):
        return Path('/app/data')
    return _get_ml_root() / "training_data"


def _get_models_dir() -> Path:
    """Get models directory, adapting to Fargate structure."""
    if os.path.exists('/app/ml/models'):
        return Path('/app/ml/models') / "currency"
    return _get_ml_root() / "models" / "currency"


def _get_logs_dir() -> Path:
    """Get logs directory, adapting to Fargate structure."""
    if os.path.exists('/app/ml/logs'):
        return Path('/app/ml/logs')
    return _get_ml_root() / "logs"


@dataclass
class LoggingConfig:
    """Configuration for logging setup."""

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    console_logging: bool = True
    # Disable file logging in Fargate (logs go to CloudWatch via awslogs driver).
    # Set DISABLE_FILE_LOGGING=true to suppress local log files.
    file_logging: bool = field(default_factory=lambda: os.getenv("DISABLE_FILE_LOGGING", "false").lower() != "true")
    upload_logs_to_s3: bool = True
    s3_logs_prefix: str = "training_logs"
    suppress_external: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and metadata."""

    experiment_id: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # When True, each processed currency dataset is written to disk individually
    # in addition to the combined dataset.  Useful for debugging feature engineering.
    save_individual_datasets: bool = False
    create_combined_dataset: bool = True

    def __post_init__(self) -> None:
        if self.experiment_id is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"xp_{timestamp}"


@dataclass
class TrainingPipelineConfig:
    """Runtime configuration for the training pipeline execution."""

    # When set, only these currencies will be trained; all others are skipped.
    currencies_to_train: Optional[List[str]] = None


@dataclass
class DynamoConfig:
    """Configuration for DynamoDB-backed data access."""

    region_name: str = os.getenv("AWS_REGION", "us-west-2")
    currency_metadata_table: str = os.getenv("DYNAMO_CURRENCY_METADATA_TABLE", "")
    currency_prices_table: str = os.getenv("DYNAMO_CURRENCY_PRICES_TABLE", "")
    league_metadata_table: Optional[str] = os.getenv("DYNAMO_LEAGUE_METADATA_TABLE", "")
    daily_prices_table: str = os.getenv("DYNAMO_DAILY_PRICES_TABLE", "poeconomy-production-daily-prices")
    predictions_table: str = os.getenv("DYNAMO_PREDICTIONS_TABLE", "")


@dataclass
class MLConfig:
    """Master configuration class combining all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    dynamo: DynamoConfig = field(default_factory=DynamoConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLConfig':
        paths_dict = config_dict.get('paths', {})
        for key, value in paths_dict.items():
            if isinstance(value, str) and ('dir' in key or 'root' in key):
                paths_dict[key] = Path(value)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            paths=PathConfig(**paths_dict),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            pipeline=TrainingPipelineConfig(**config_dict.get('pipeline', {})),
            dynamo=DynamoConfig(**config_dict.get('dynamo', {}))
        )

    @classmethod
    def from_file(cls, config_path: str) -> 'MLConfig':
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        config_dict = asdict(self)

        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        return convert_paths(config_dict)  # type: ignore[no-any-return]

    def save(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_config(config_path: Optional[str] = None) -> 'MLConfig':
    """Get configuration, optionally loading from a file."""
    if config_path and os.path.exists(config_path):
        return MLConfig.from_file(config_path)
    return get_production_config()


def get_config_by_mode(mode: str = "production") -> 'MLConfig':
    """Get configuration for the given run mode.

    Args:
        mode: One of 'production', 'development', or 'test'.
              Unknown values fall back to production.
    """
    if mode == "development":
        return get_development_config()
    elif mode == "test":
        return get_test_config()
    else:
        return get_production_config()


def get_production_config() -> 'MLConfig':
    """Full-scale production configuration.

    Dataclass field defaults (and any env-var overrides) are the production
    baseline — no additional knobs are forced here.
    """
    config = MLConfig()
    config.experiment.tags = ["production"]
    return config


def get_development_config() -> 'MLConfig':
    """Reduced-scale development configuration for faster local iteration.

    Overrides scale/performance knobs only when the corresponding environment
    variable is not already set.  Correctness behaviour is identical to
    production; only throughput/dataset-scope settings differ.
    """
    config = MLConfig()
    config.experiment.tags = ["development"]

    if not os.getenv('N_HYPERPARAMETER_TRIALS'):
        config.model.n_hyperparameter_trials = 50
    if not os.getenv('CV_FOLDS'):
        config.model.cv_folds = 3
    if not os.getenv('MAX_CURRENCIES_TO_TRAIN'):
        config.data.max_currencies_to_train = 20
    config.data.max_league_days = 30

    return config


def get_test_config() -> 'MLConfig':
    """Minimal test configuration for smoke-level validation.

    Designed for fast, deterministic runs that verify pipeline correctness
    without full optimization.  Env-variable overrides are still respected.
    """
    config = MLConfig()
    config.experiment.tags = ["test"]

    if not os.getenv('N_HYPERPARAMETER_TRIALS'):
        config.model.n_hyperparameter_trials = 5
    if not os.getenv('CV_FOLDS'):
        config.model.cv_folds = 2
    if not os.getenv('MAX_CURRENCIES_TO_TRAIN'):
        config.data.max_currencies_to_train = 3
    if not os.getenv('MIN_RECORDS_THRESHOLD'):
        config.data.min_records_threshold = 10
    config.data.max_league_days = 14
    config.processing.outlier_removal = False

    return config
