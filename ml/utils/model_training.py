"""
Comprehensive model training utilities for ML pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import joblib
import json
import time
from abc import ABC, abstractmethod

# ML imports
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config.training_config import ModelConfig, ProcessingConfig
from utils.logging_utils import MLLogger


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mae: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray) -> 'ModelMetrics':
        """Calculate metrics from predictions."""
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE with robust calculation
        epsilon = 1e-8
        mape_values = []
        for true, pred in zip(y_true, y_pred):
            if abs(true) > epsilon:
                denominator = (abs(true) + abs(pred)) / 2 + epsilon
                mape_val = abs(true - pred) / denominator * 100
                mape_values.append(mape_val)
        mape = np.mean(mape_values) if mape_values else float('inf')
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0.0
        
        return cls(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=max(r2, -1.0),  # Cap at -1 for very bad predictions
            directional_accuracy=directional_accuracy
        )


@dataclass
class TrainingResult:
    """Result of model training process."""
    model: Any
    scaler: Optional[Any]
    metrics: ModelMetrics
    training_time: float
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    cross_validation_scores: Optional[List[float]]
    model_type: str
    currency_pair: str


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, config: ModelConfig, logger: Optional[MLLogger] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or MLLogger("BaseModel")
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    @abstractmethod
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> Any:
        """Create model instance with optional hyperparameter optimization."""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Get model type identifier."""
        pass
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        # Scale features if configured
        if hasattr(self.config, 'use_scaling') and self.config.use_scaling:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)
        
        # Fit model
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None
        
        return dict(enumerate(self.model.feature_importances_))


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> lgb.LGBMRegressor:
        """Create LightGBM model with optional hyperparameter optimization."""
        if trial is not None:
            # Hyperparameter optimization
            params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
                'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0.0, 10.0),
            }
        else:
            # Default parameters
            params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'num_leaves': 100,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
        
        # Common parameters
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbose': -1,
            'early_stopping_rounds': self.config.early_stopping_rounds if trial is None else None
        })
        
        return lgb.LGBMRegressor(**params)
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "lightgbm"


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> xgb.XGBRegressor:
        """Create XGBoost model with optional hyperparameter optimization."""
        if trial is not None:
            # Hyperparameter optimization
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 10.0),
            }
        else:
            # Default parameters
            params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
        
        # Common parameters
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': self.config.early_stopping_rounds if trial is None else None
        })
        
        return xgb.XGBRegressor(**params)
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "xgboost"


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> RandomForestRegressor:
        """Create Random Forest model with optional hyperparameter optimization."""
        if trial is not None:
            # Hyperparameter optimization
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            }
        else:
            # Default parameters
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
            }
        
        # Common parameters
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': -1,
        })
        
        return RandomForestRegressor(**params)
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "random_forest"


class EnsembleModel:
    """Ensemble model combining multiple base models."""
    
    def __init__(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None,
        logger: Optional[MLLogger] = None
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of base models
            weights: Optional weights for ensemble (default: equal weights)
            logger: Optional logger instance
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.logger = logger or MLLogger("EnsembleModel")
        self.is_fitted = False
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        for i, model in enumerate(self.models):
            self.logger.info(f"Training ensemble model {i+1}/{len(self.models)}: {model.get_model_type()}")
            model.fit(X, y, X_val, y_val)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Weighted ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def get_model_type(self) -> str:
        """Get ensemble model type identifier."""
        model_types = [model.get_model_type() for model in self.models]
        return f"ensemble_{'_'.join(model_types)}"


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        config: ModelConfig,
        logger: Optional[MLLogger] = None
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: Model configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or MLLogger("HyperparameterOptimizer")
    
    def optimize(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        currency_pair: str,
        cv_folds: int = 5
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_class: Model class to optimize
            X: Training features
            y: Training targets
            currency_pair: Currency pair identifier
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        def objective(trial):
            # Create model with trial parameters
            model = model_class(self.config, self.logger)
            model.model = model._create_model(trial)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate RMSE (optimization target)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        # Create study
        study_name = f"optimize_{currency_pair}_{model_class.__name__}"
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=study_name
        )
        
        # Optimize
        with self.logger.log_operation(f"Hyperparameter optimization for {currency_pair}"):
            study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        # Create best model
        best_model = model_class(self.config, self.logger)
        best_model.model = best_model._create_model()
        
        # Update model with best parameters
        best_params = study.best_params
        for param_name, param_value in best_params.items():
            # Remove model prefix from parameter name
            clean_param_name = param_name.split('_', 1)[1] if '_' in param_name else param_name
            if hasattr(best_model.model, clean_param_name):
                setattr(best_model.model, clean_param_name, param_value)
        
        self.logger.info(
            f"Optimization completed for {currency_pair}",
            extra={
                "best_score": study.best_value,
                "n_trials": len(study.trials),
                "best_params": best_params
            }
        )
        
        return best_model, best_params


class ModelTrainer:
    """Main model training orchestrator."""
    
    def __init__(
        self,
        config: ModelConfig,
        processing_config: ProcessingConfig,
        logger: Optional[MLLogger] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            config: Model configuration
            processing_config: Processing configuration
            logger: Optional logger instance
        """
        self.config = config
        self.processing_config = processing_config
        self.logger = logger or MLLogger("ModelTrainer")
        self.optimizer = HyperparameterOptimizer(config, logger)
    
    def train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        currency_pair: str,
        model_type: str = "ensemble",
        optimize_hyperparameters: bool = True
    ) -> TrainingResult:
        """
        Train a single model for a currency pair.
        
        Args:
            X: Training features
            y: Training targets
            currency_pair: Currency pair identifier
            model_type: Type of model to train
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            TrainingResult with trained model and metrics
        """
        start_time = time.time()
        
        with self.logger.log_operation(f"Training {model_type} model for {currency_pair}"):
            # Split data
            split_idx = int(len(X) * (1 - self.config.test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Further split training data for validation
            val_split_idx = int(len(X_train) * 0.8)
            X_train_fit, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
            y_train_fit, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
            
            if model_type == "ensemble":
                model = self._train_ensemble_model(
                    X_train_fit, y_train_fit, X_val, y_val,
                    currency_pair, optimize_hyperparameters
                )
            else:
                model = self._train_single_base_model(
                    X_train_fit, y_train_fit, X_val, y_val,
                    currency_pair, model_type, optimize_hyperparameters
                )
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            metrics = ModelMetrics.from_predictions(y_test, y_pred)
            
            # Cross-validation scores
            cv_scores = self._calculate_cv_scores(model, X_train, y_train)
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
            elif hasattr(model, 'models'):  # Ensemble
                # Average feature importance across models
                importances = []
                for base_model in model.models:
                    imp = base_model.get_feature_importance()
                    if imp:
                        importances.append(imp)
                
                if importances:
                    # Average importances
                    all_features = set()
                    for imp in importances:
                        all_features.update(imp.keys())
                    
                    feature_importance = {}
                    for feature in all_features:
                        values = [imp.get(feature, 0) for imp in importances]
                        feature_importance[feature] = np.mean(values)
            
            training_time = time.time() - start_time
            
            # Get hyperparameters
            hyperparameters = {}
            if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
                hyperparameters = model.model.get_params()
            elif hasattr(model, 'models'):  # Ensemble
                hyperparameters = {
                    f"{m.get_model_type()}_params": m.model.get_params() if hasattr(m.model, 'get_params') else {}
                    for m in model.models
                }
            
            self.logger.log_model_training_end(
                currency_pair, model_type, metrics.to_dict(), training_time
            )
            
            return TrainingResult(
                model=model,
                scaler=getattr(model, 'scaler', None),
                metrics=metrics,
                training_time=training_time,
                hyperparameters=hyperparameters,
                feature_importance=feature_importance,
                cross_validation_scores=cv_scores,
                model_type=model_type,
                currency_pair=currency_pair
            )
    
    def _train_ensemble_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency_pair: str,
        optimize_hyperparameters: bool
    ) -> EnsembleModel:
        """Train ensemble model."""
        models = []
        
        # Create base models
        if self.config.use_lightgbm:
            if optimize_hyperparameters:
                lgb_model, _ = self.optimizer.optimize(
                    LightGBMModel, X_train, y_train, currency_pair
                )
            else:
                lgb_model = LightGBMModel(self.config, self.logger)
                lgb_model.model = lgb_model._create_model()
            models.append(lgb_model)
        
        if self.config.use_xgboost:
            if optimize_hyperparameters:
                xgb_model, _ = self.optimizer.optimize(
                    XGBoostModel, X_train, y_train, currency_pair
                )
            else:
                xgb_model = XGBoostModel(self.config, self.logger)
                xgb_model.model = xgb_model._create_model()
            models.append(xgb_model)
        
        # Create ensemble
        ensemble = EnsembleModel(models, logger=self.logger)
        ensemble.fit(X_train, y_train, X_val, y_val)
        
        return ensemble
    
    def _train_single_base_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency_pair: str,
        model_type: str,
        optimize_hyperparameters: bool
    ) -> BaseModel:
        """Train single base model."""
        model_classes = {
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel,
            'random_forest': RandomForestModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_classes[model_type]
        
        if optimize_hyperparameters:
            model, _ = self.optimizer.optimize(
                model_class, X_train, y_train, currency_pair
            )
        else:
            model = model_class(self.config, self.logger)
            model.model = model._create_model()
        
        model.fit(X_train, y_train, X_val, y_val)
        return model
    
    def _calculate_cv_scores(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Calculate cross-validation scores."""
        try:
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create a copy of the model for CV
                if hasattr(model, 'models'):  # Ensemble
                    # For ensemble, use a simplified approach
                    temp_model = model.models[0]  # Use first model as proxy
                else:
                    temp_model = model
                
                temp_model.fit(X_train, y_train)
                y_pred = temp_model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {str(e)}")
            return []


def save_model_artifacts(
    result: TrainingResult,
    output_dir: Path,
    currency_pair: str
) -> Dict[str, str]:
    """
    Save model artifacts to disk.
    
    Args:
        result: Training result
        output_dir: Output directory
        currency_pair: Currency pair identifier
        
    Returns:
        Dictionary of saved file paths
    """
    # Create currency-specific directory
    currency_dir = output_dir / currency_pair
    currency_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save model
    model_path = currency_dir / "ensemble_model.pkl"
    joblib.dump(result.model, model_path)
    saved_files['model'] = str(model_path)
    
    # Save scaler if exists
    if result.scaler is not None:
        scaler_path = currency_dir / "scaler.pkl"
        joblib.dump(result.scaler, scaler_path)
        saved_files['scaler'] = str(scaler_path)
    
    # Save metadata
    metadata = {
        'currency_pair': result.currency_pair,
        'model_type': result.model_type,
        'metrics': result.metrics.to_dict(),
        'training_time': result.training_time,
        'hyperparameters': result.hyperparameters,
        'feature_importance': result.feature_importance,
        'cross_validation_scores': result.cross_validation_scores,
        'training_timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = currency_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    saved_files['metadata'] = str(metadata_path)
    
    return saved_files


def load_model_artifacts(model_dir: Path) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """
    Load model artifacts from disk.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Tuple of (model, scaler, metadata)
    """
    # Load model
    model_path = model_dir / "ensemble_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load scaler if exists
    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    
    # Load metadata
    metadata_path = model_dir / "model_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, scaler, metadata 