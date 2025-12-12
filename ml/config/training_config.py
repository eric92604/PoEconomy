"""
Centralized configuration management for ML training pipeline.

Environment Variables:
    # Threading and Parallelization
    MAX_CURRENCY_WORKERS: Number of parallel currency workers (default: 4)
    MAX_OPTUNA_WORKERS: Number of Optuna workers per currency (default: 2)
    MODEL_N_JOBS: Number of model training jobs per currency (default: 2)
    
    # Hyperparameter Optimization
    N_HYPERPARAMETER_TRIALS: Number of hyperparameter optimization trials (default: 200)
    N_MODEL_TRIALS: Number of model training iterations (default: 1000)
    CV_FOLDS: Number of cross-validation folds (default: 5)
    
    # Model Performance
    MAX_DEPTH: Maximum tree depth (default: 8)
    LEARNING_RATE: Learning rate for models (default: 0.1)
    
    # Data Selection
    MIN_RECORDS_THRESHOLD: Minimum records required for training (default: 50)
    MAX_CURRENCIES_TO_TRAIN: Maximum number of currencies to train (default: 0, 0 = no limit)
    MIN_AVG_VALUE_THRESHOLD: Minimum average value threshold (default: 0.25)
    
    # AWS Configuration
    AWS_REGION: AWS region (default: us-west-2)
    DATA_LAKE_BUCKET: S3 bucket for data lake
    DYNAMO_*: Various DynamoDB table and attribute configurations
    
    # Logging
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os




@dataclass
class ModelConfig:
    """Configuration for ML model training with configurable threading."""
    
    # Model Selection
    use_lightgbm: bool = True
    use_random_forest: bool = True
    use_extra_trees: bool = True 
    use_ensemble: bool = True
    
    # Ensemble Weight Optimization
    optimize_ensemble_weights: bool = True  # Enable Optuna-based weight optimization
    ensemble_weight_optimization_trials: int = 50  # Number of Optuna trials for weight optimization
    
    max_currency_workers: int = field(default_factory=lambda: int(os.getenv('MAX_CURRENCY_WORKERS', '4')))
    max_optuna_workers: int = field(default_factory=lambda: int(os.getenv('MAX_OPTUNA_WORKERS', '2')))
    model_n_jobs: int = field(default_factory=lambda: int(os.getenv('MODEL_N_JOBS', '2')))
    
    # Training Parameters
    test_size: float = 0.2
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
    """Configuration for data processing parameters."""
    
    # League parameters
    max_league_days: int = 60
    min_league_days: int = 0
    
    # League inclusion settings
    included_leagues: Optional[List[str]] = None
    excluded_leagues: List[str] = field(default_factory=list)
    
    # Currency selection strategy
    min_records_threshold: int = field(default_factory=lambda: int(os.getenv('MIN_RECORDS_THRESHOLD', '5')))
    max_currencies_to_train: Optional[int] = field(default_factory=lambda: int(os.getenv('MAX_CURRENCIES_TO_TRAIN', '0')) if os.getenv('MAX_CURRENCIES_TO_TRAIN') and int(os.getenv('MAX_CURRENCIES_TO_TRAIN', '0')) > 0 else None)
    min_avg_value_threshold: float = field(default_factory=lambda: float(os.getenv('MIN_AVG_VALUE_THRESHOLD', '0.25')))
    train_all_currencies: bool = False
    
    # Target variables
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 7])
    
    # Feature engineering windows - use consistent horizons: 1, 3, 5, 7
    rolling_windows: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    momentum_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    
    # Volatility feature engineering - use consistent horizons: 3, 5, 7 (removed 1-day as not meaningful)
    volatility_windows: List[int] = field(default_factory=lambda: [3, 5, 7])
    include_volatility_features: bool = True
    volatility_types: List[str] = field(default_factory=lambda: ['std', 'cv', 'range', 'garch'])
    
    
    # Feature engineering options
    include_league_features: bool = True
    outlier_removal_iqr_multiplier: float = 2.0
    
    # Data validation
    min_variance_threshold: float = 1e-8


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    
    # Feature Engineering
    max_lag_features: int = 10
    technical_indicators: bool = True
    statistical_features: bool = True
    
    # Data Quality
    outlier_threshold: float = 3.0
    missing_value_threshold: float = 0.5
    max_missing_ratio: float = 0.3
    
    # Scaling
    robust_scaling: bool = True
    
    # Price transformations
    log_transform: bool = True
    log_transform_ratio_threshold: float = 10.0
    
    # Data processing options
    outlier_removal: bool = True
    use_parallel_processing: bool = False
    
    # Target Engineering
    target_columns: List[str] = field(default_factory=lambda: [
        'price_1d',
        'price_3d', 
        'price_7d',
        'price_14d'
    ])


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories - adapt to Fargate structure
    ml_root: Path = field(default_factory=lambda: _get_ml_root())
    data_dir: Path = field(default_factory=lambda: _get_data_dir())
    models_dir: Path = field(default_factory=lambda: _get_models_dir())
    logs_dir: Path = field(default_factory=lambda: _get_logs_dir())
    
    # File patterns
    combined_data_pattern: str = "combined_currency_features_{experiment_id}.parquet"
    model_file_name: str = "ensemble_model.pkl"
    scaler_file_name: str = "scaler.pkl"
    preprocessing_file_name: str = "preprocessing_info.json"
    
    def __post_init__(self) -> None:
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            return
            
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


def _get_ml_root() -> Path:
    """Get ML root directory, adapting to Fargate structure."""
    if os.path.exists('/app/ml'):
        return Path('/app/ml')
    else:
        return Path(__file__).parent.parent.resolve()


def _get_data_dir() -> Path:
    """Get data directory, adapting to Fargate structure."""
    if os.path.exists('/app/data'):
        return Path('/app/data')
    else:
        return _get_ml_root() / "training_data"


def _get_models_dir() -> Path:
    """Get models directory, adapting to Fargate structure."""
    if os.path.exists('/app/ml/models'):
        return Path('/app/ml/models') / "currency"
    else:
        return _get_ml_root() / "models" / "currency"


def _get_logs_dir() -> Path:
    """Get logs directory, adapting to Fargate structure."""
    if os.path.exists('/app/ml/logs'):
        return Path('/app/ml/logs')
    else:
        return _get_ml_root() / "logs"


@dataclass
class LoggingConfig:
    """Configuration for logging setup."""
    
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    console_logging: bool = True
    # Disable file logging in Fargate (logs go to CloudWatch via awslogs driver)
    # Set DISABLE_FILE_LOGGING=true to disable file logging and reduce storage writes
    file_logging: bool = field(default_factory=lambda: os.getenv("DISABLE_FILE_LOGGING", "false").lower() != "true")
    
    # S3 log upload configuration
    upload_logs_to_s3: bool = True
    s3_logs_prefix: str = "training_logs"
    
    # Suppress external library logs
    suppress_external: bool = True
    suppress_lightgbm: bool = True
    suppress_optuna: bool = True
    suppress_sklearn: bool = True
    suppress_boto3: bool = True
    suppress_requests: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and metadata."""
    
    experiment_id: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Output options
    save_individual_datasets: bool = False
    create_combined_dataset: bool = True
    save_model_artifacts: bool = True
    save_feature_importance: bool = True
    
    def __post_init__(self) -> None:
        if self.experiment_id is None:
            # Generate timestamp-based experiment ID (xp_YYYYMMDD_HHMMSS)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"xp_{timestamp}"


@dataclass
class TrainingPipelineConfig:
    """Configuration for the complete training pipeline."""
    
    # Model and processing configs
    model_config: ModelConfig = field(default_factory=ModelConfig)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Pipeline settings
    currencies_to_train: Optional[List[str]] = None
    output_dir: str = field(default_factory=lambda: str(Path(__file__).parent.parent / "models"))
    
    # Experiment settings
    experiment_id: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Currency selection settings
    min_records_threshold: int = 5  # Reduced from 100 to allow minimal data
    
    # Logging
    log_level: str = "INFO"
    save_predictions: bool = True
    save_feature_importance: bool = True
    
    # Performance monitoring
    enable_monitoring: bool = True
    monitoring_interval: int = 30  # seconds
    
    def __post_init__(self) -> None:
        if self.experiment_id is None:
            # Use consistent ID for production training
            self.experiment_id = "production_training"


@dataclass
class DynamoConfig:
    """Configuration for DynamoDB-backed data access."""

    region_name: str = os.getenv("AWS_REGION", "us-west-2")
    currency_metadata_table: str = os.getenv("DYNAMO_CURRENCY_METADATA_TABLE", "")
    currency_prices_table: str = os.getenv("DYNAMO_CURRENCY_PRICES_TABLE", "")
    league_metadata_table: Optional[str] = os.getenv("DYNAMO_LEAGUE_METADATA_TABLE", "")
    predictions_table: str = os.getenv("DYNAMO_PREDICTIONS_TABLE", "")
    currency_timestamp_index: Optional[str] = os.getenv("DYNAMO_CURRENCY_TIMESTAMP_INDEX", "currency-timestamp-index")
    partition_key: str = os.getenv("DYNAMO_PARTITION_KEY", "currency_league")
    sort_key: str = os.getenv("DYNAMO_SORT_KEY", "timestamp")
    currency_attribute: str = os.getenv("DYNAMO_CURRENCY_ATTRIBUTE", "currency")
    pay_currency_attribute: str = os.getenv("DYNAMO_PAY_CURRENCY_ATTRIBUTE", "pay_currency")
    price_attribute: str = os.getenv("DYNAMO_PRICE_ATTRIBUTE", "price")
    league_attribute: str = os.getenv("DYNAMO_LEAGUE_ATTRIBUTE", "league")
    timestamp_attribute: str = os.getenv("DYNAMO_TIMESTAMP_ATTRIBUTE", "timestamp")
    metadata_json_attribute: str = os.getenv("DYNAMO_METADATA_JSON_ATTRIBUTE", "metadata_json")


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
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


# Default configuration instance
DEFAULT_CONFIG = MLConfig()


def get_config(config_path: Optional[str] = None) -> MLConfig:
    """
    Get configuration instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        MLConfig instance
    """
    if config_path and os.path.exists(config_path):
        return MLConfig.from_file(config_path)
    return DEFAULT_CONFIG


def get_config_by_mode(mode: str = "production") -> MLConfig:
    """Get configuration based on mode."""
    if mode == "all_currencies":
        return get_all_currencies_config()
    elif mode == "high_value":
        return get_high_value_config()
    else:
        # Default to production configuration for any mode
        return get_default_config()


# Default configuration
def get_default_config() -> MLConfig:
    """Get default configuration optimized for production."""
    config = MLConfig()
    
    # Set production-optimized defaults (environment variables are already handled by dataclass fields)
    # Only set non-environment-variable properties
    # Note: logging.level is set from LOG_LEVEL env var in LoggingConfig, defaulting to "INFO"
    config.logging.suppress_lightgbm = True
    config.logging.suppress_optuna = True
    config.experiment.tags = ["production"]
    
    # Enhanced data quality settings for production
    config.processing.outlier_removal = True
    config.processing.robust_scaling = True
    config.processing.max_missing_ratio = 0.3  # Allow some missing data for production
    
    return config


def get_all_currencies_config() -> MLConfig:
    """Get configuration optimized for training all available currencies."""
    config = get_default_config()
    
    # Lower threshold for more currencies (only if not set via env var)
    if not os.getenv('MIN_RECORDS_THRESHOLD'):
        config.data.min_records_threshold = 30  # Balanced threshold for more currencies
    
    # Adjust processing for larger scale
    
    # Experiment settings
    config.experiment.experiment_id = "production_training"
    config.experiment.description = "Training models for all currencies with sufficient historical data"
    config.experiment.tags = ["production", "all_currencies", "comprehensive"]
    
    return config


def get_high_value_config() -> MLConfig:
    """Get configuration optimized for training only high-value currencies (including Mirror items)."""
    config = get_default_config()
    
    # Lower threshold for rare items
    config.data.min_records_threshold = 20  # Balanced threshold for rare items
    
    # Experiment settings
    config.experiment.experiment_id = "production_training"
    config.experiment.description = "Training models for high-value currencies including Mirror items"
    config.experiment.tags = ["production", "high_value", "comprehensive"]
    
    return config
