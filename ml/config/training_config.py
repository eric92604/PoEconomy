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
    min_samples_required: int = 150
    min_samples_after_cleaning: int = 30
    
    # Feature engineering
    max_features: int = 50
    feature_selection_k: int = 30
    rolling_windows: List[int] = field(default_factory=lambda: [1, 3, 7, 14])
    momentum_periods: List[int] = field(default_factory=lambda: [3, 7])
    volatility_windows: List[int] = field(default_factory=lambda: [7, 14])


@dataclass
class DataConfig:
    """Configuration for data processing parameters."""
    
    # League parameters
    max_league_days: int = 60
    min_league_days: int = 0
    
    # Data filtering
    min_records_per_pair: int = 50
    min_records_after_cleaning: int = 30
    
    # Target variables
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 7])
    
    # Feature engineering
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
    
    # Transformation thresholds
    log_transform_ratio_threshold: float = 10.0
    log_transform_cv_improvement: float = 0.8
    
    # Categorical encoding
    max_categorical_cardinality: int = 50


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories
    ml_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "training_data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models" / "currency")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
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
        return asdict(self)
    
    def save(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        import json
        config_dict = self.to_dict()
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        config_dict = convert_paths(config_dict)
        
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


# Environment-specific configurations
def get_development_config() -> MLConfig:
    """Get configuration optimized for development."""
    config = MLConfig()
    config.model.n_trials = 50  # Faster for development
    config.model.cv_folds = 3
    config.logging.level = "DEBUG"
    config.experiment.tags = ["development"]
    return config


def get_production_config() -> MLConfig:
    """Get configuration optimized for production."""
    config = MLConfig()
    config.model.n_trials = 200
    config.model.cv_folds = 5
    config.logging.level = "INFO"
    config.logging.suppress_lightgbm = True
    config.logging.suppress_optuna = True
    config.experiment.tags = ["production"]
    return config


def get_testing_config() -> MLConfig:
    """Get configuration optimized for testing."""
    config = MLConfig()
    config.model.n_trials = 10  # Very fast for testing
    config.model.cv_folds = 2
    config.model.min_samples_required = 50
    config.data.min_records_per_pair = 20
    config.logging.level = "WARNING"
    config.experiment.tags = ["testing"]
    return config 