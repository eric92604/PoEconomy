"""
Centralized configuration management for ML training pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os


@dataclass
class ModelConfig:
    """Configuration for model training parameters."""
    
    # Model selection
    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_ensemble: bool = True
    
    # Hyperparameter optimization
    n_trials: int = 200
    cv_folds: int = 5
    early_stopping_rounds: int = 20
    
    # Model parameters
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class DataConfig:
    """Configuration for data processing parameters."""
    
    # League parameters
    max_league_days: int = 60
    min_league_days: int = 0
    
    # League inclusion settings
    include_settlers_league: bool = True
    included_leagues: List[str] = field(default_factory=lambda: [
        'Settlers', 'Necro Settlers', 'Necropolis', 'Affliction', 
        'Ancestor', 'Crucible', 'Sanctum', 'Kalandra', 'Sentinel'
    ])
    excluded_leagues: List[str] = field(default_factory=list)
    
    # Currency selection strategy
    train_all_currencies: bool = True  # If True, train models for all currencies with sufficient data
    min_avg_value_threshold: float = 1.0  # Minimum average value (in Chaos Orbs) for currency selection
    min_records_threshold: int = 50  # Minimum number of records required for training
    
    # Currency availability filtering
    filter_by_availability: bool = True
    only_train_available_currencies: bool = True
    availability_check_days: int = 60  # How recent the availability check should be
    
    # Target variables
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 7])
    
    # Feature engineering windows
    rolling_windows: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    momentum_periods: List[int] = field(default_factory=lambda: [3, 5, 7])
    
    # Feature engineering options
    include_league_features: bool = True
    outlier_removal_iqr_multiplier: float = 2.0
    
    # Data validation
    max_missing_ratio: float = 0.3
    min_variance_threshold: float = 1e-8


@dataclass
class ProcessingConfig:
    """Configuration for data processing strategies."""
    
    log_transform: bool = True
    robust_scaling: bool = True
    feature_selection: bool = True
    outlier_removal: bool = True
    advanced_cv: bool = True
    
    # Feature selection parameters
    max_features: int = 75 
    feature_selection_k: int = 30
    
    # Transformation thresholds
    log_transform_ratio_threshold: float = 10.0
    log_transform_cv_improvement: float = 0.8
    
    # Categorical encoding
    max_categorical_cardinality: int = 50


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories
    ml_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve())
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve() / "training_data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve() / "models" / "currency")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve() / "logs")
    
    # File patterns
    combined_data_pattern: str = "combined_currency_features_{experiment_id}.parquet"
    model_file_name: str = "ensemble_model.pkl"
    scaler_file_name: str = "scaler.pkl"
    preprocessing_file_name: str = "preprocessing_info.json"
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging setup."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_logging: bool = True
    file_logging: bool = True
    
    # Suppress external library logs
    suppress_lightgbm: bool = True
    suppress_optuna: bool = True
    suppress_sklearn: bool = True


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
    
    def __post_init__(self):
        """Generate experiment ID if not provided."""
        if self.experiment_id is None:
            from datetime import datetime
            self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class MLConfig:
    """Master configuration class combining all sub-configurations."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLConfig':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            paths=PathConfig(**config_dict.get('paths', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'MLConfig':
        """Load configuration from JSON file."""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        config_dict = asdict(self)
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        return convert_paths(config_dict)
    
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
    if mode == "production":
        return get_production_config()
    elif mode == "development":
        return get_development_config()
    elif mode == "test":
        return get_test_config()
    elif mode == "all_currencies":
        return get_all_currencies_config()
    elif mode == "high_value":
        return get_high_value_config()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'production', 'development', 'test', 'all_currencies', or 'high_value'")


# Environment-specific configurations
def get_development_config() -> MLConfig:
    """Get configuration optimized for development."""
    config = MLConfig()
    config.model.n_trials = 50  # Faster for development
    config.model.cv_folds = 3
    config.logging.level = "INFO"
    config.experiment.tags = ["development"]
    return config


def get_production_config() -> MLConfig:
    """Get configuration optimized for production."""
    config = MLConfig()
    config.model.n_trials = 200
    config.model.cv_folds = 5
    config.logging.level = "WARNING"
    config.logging.suppress_lightgbm = True
    config.logging.suppress_optuna = True
    config.experiment.tags = ["production"]
    return config


def get_test_config() -> MLConfig:
    """Get configuration optimized for testing."""
    config = MLConfig()
    config.model.n_trials = 3  # Very fast for testing
    config.model.cv_folds = 1
    config.logging.level = "DEBUG"
    config.experiment.tags = ["test"]
    return config


def get_all_currencies_config() -> MLConfig:
    """Get configuration optimized for training all available currencies."""
    config = get_production_config()
    
    # Enable all currencies mode
    config.data.train_all_currencies = True
    config.data.min_avg_value_threshold = 1.0  # Include low-value currencies
    config.data.min_records_threshold = 50     # Lower threshold for more currencies
    config.data.filter_by_availability = False  # Disable availability filtering to include high-value items
    config.data.only_train_available_currencies = False
    
    # Adjust processing for larger scale
    config.processing.max_features = 50        # Slightly reduced for performance
    config.processing.feature_selection_k = 25 # Proportionally reduced
    
    # Experiment settings
    import pandas as pd
    config.experiment.experiment_id = f"all_currencies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    config.experiment.description = "Training models for all currencies with sufficient historical data (no availability filtering)"
    config.experiment.tags = ["production", "all_currencies", "comprehensive", "no_availability_filter"]
    
    return config


def get_high_value_config() -> MLConfig:
    """Get configuration optimized for training only high-value currencies (including Mirror items)."""
    config = get_production_config()
    
    # Enable all currencies mode with high value threshold
    config.data.train_all_currencies = True
    config.data.min_avg_value_threshold = 1000.0  # Only very valuable items
    config.data.min_records_threshold = 50        # Lower threshold for rare items
    config.data.filter_by_availability = False    # Disable availability filtering for high-value items
    config.data.only_train_available_currencies = False
    
    # Experiment settings
    import pandas as pd
    config.experiment.experiment_id = f"high_value_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    config.experiment.description = "Training models for high-value currencies including Mirror items"
    config.experiment.tags = ["production", "high_value", "comprehensive", "no_availability_filter"]
    
    return config 