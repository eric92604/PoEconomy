"""
Model training utilities optimized for multi-core systems.
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
from datetime import datetime

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


def configure_environment_for_parallel_training(config: ModelConfig) -> None:
    """Configure environment variables for optimal parallel training."""
    import os
    
    # Set threading environment variables
    os.environ['OMP_NUM_THREADS'] = str(config.model_n_jobs)
    os.environ['MKL_NUM_THREADS'] = str(config.model_n_jobs)
    os.environ['OPENBLAS_NUM_THREADS'] = str(config.model_n_jobs)
    os.environ['NUMEXPR_NUM_THREADS'] = str(config.model_n_jobs)
    
    # LightGBM specific
    os.environ['LGB_NUM_THREADS'] = str(config.model_n_jobs)
    
    # XGBoost specific  
    os.environ['XGB_NUM_THREADS'] = str(config.model_n_jobs)


def configure_optimal_threading(config: ModelConfig) -> None:
    """
    Configure optimal threading strategy for ML training.
    
    Strategy: Use process-level parallelism for multiple currencies,
    limit individual model threading to 2-4 cores for optimal performance.
    """
    import os
    import multiprocessing
    
    total_cores = multiprocessing.cpu_count()
    
    # Optimal threading strategy based on research:
    # - Individual models perform best with 2-4 cores
    # - Use remaining cores for process-level parallelism
    optimal_model_threads = min(4, max(2, total_cores // config.max_currency_workers))
    
    print(f"Threading Configuration:")
    print(f"  Total CPU cores: {total_cores}")
    print(f"  Currency workers: {config.max_currency_workers}")
    print(f"  Threads per model: {optimal_model_threads}")
    
    # Set environment variables for consistent threading
    env_vars = {
        'OMP_NUM_THREADS': str(optimal_model_threads),
        'MKL_NUM_THREADS': str(optimal_model_threads),
        'NUMEXPR_NUM_THREADS': str(optimal_model_threads),
        'OPENBLAS_NUM_THREADS': str(optimal_model_threads),
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
    
    # Update config to use optimal threading
    config.model_n_jobs = optimal_model_threads
    
    print(f"  Model n_jobs set to: {optimal_model_threads}")


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mae: Union[float, Dict[str, float]]
    rmse: Union[float, Dict[str, float]]
    mape: Union[float, Dict[str, float]]
    r2: Union[float, Dict[str, float]]
    directional_accuracy: Union[float, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray, 
                        target_names: Optional[List[str]] = None) -> 'ModelMetrics':
        """Calculate metrics from predictions."""
        # Handle multi-output case
        if y_true.ndim > 1 and y_true.shape[1] > 1 and target_names is not None:
            return cls._from_multi_output_predictions(y_true, y_pred, target_names)
        
        # Single output case - flatten if needed
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
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
    
    @classmethod
    def _from_multi_output_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray, 
                                     target_names: List[str]) -> 'ModelMetrics':
        """Calculate metrics for multi-output predictions."""
        n_outputs = y_true.shape[1]
        
        mae_dict = {}
        rmse_dict = {}
        mape_dict = {}
        r2_dict = {}
        directional_accuracy_dict = {}
        
        for i, target_name in enumerate(target_names):
            y_true_single = y_true[:, i]
            y_pred_single = y_pred[:, i]
            
            # Skip if all values are NaN
            valid_mask = ~(np.isnan(y_true_single) | np.isnan(y_pred_single))
            if not np.any(valid_mask):
                mae_dict[target_name] = float('inf')
                rmse_dict[target_name] = float('inf')
                mape_dict[target_name] = float('inf')
                r2_dict[target_name] = -1.0
                directional_accuracy_dict[target_name] = 0.0
                continue
            
            y_true_valid = y_true_single[valid_mask]
            y_pred_valid = y_pred_single[valid_mask]
            
            # Basic metrics
            mae_dict[target_name] = mean_absolute_error(y_true_valid, y_pred_valid)
            rmse_dict[target_name] = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            r2_dict[target_name] = max(r2_score(y_true_valid, y_pred_valid), -1.0)
            
            # MAPE with robust calculation
            epsilon = 1e-8
            mape_values = []
            for true, pred in zip(y_true_valid, y_pred_valid):
                if abs(true) > epsilon:
                    denominator = (abs(true) + abs(pred)) / 2 + epsilon
                    mape_val = abs(true - pred) / denominator * 100
                    mape_values.append(mape_val)
            mape_dict[target_name] = np.mean(mape_values) if mape_values else float('inf')
            
            # Directional accuracy
            if len(y_true_valid) > 1:
                true_direction = np.diff(y_true_valid) > 0
                pred_direction = np.diff(y_pred_valid) > 0
                directional_accuracy_dict[target_name] = np.mean(true_direction == pred_direction) * 100
            else:
                directional_accuracy_dict[target_name] = 0.0
        
        return cls(
            mae=mae_dict,
            rmse=rmse_dict,
            mape=mape_dict,
            r2=r2_dict,
            directional_accuracy=directional_accuracy_dict
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
    currency: str


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, config: ModelConfig, logger: Optional[MLLogger] = None):
        self.config = config
        self.logger = logger
        self.model = None
        
        if self.logger:
            self.logger.info(f"Initializing {self.get_model_type()} model")
    
    @abstractmethod
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> Any:
        """Create the underlying model with optional hyperparameter optimization."""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return the model type identifier."""
        pass
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Fit the model to training data."""
        start_time = time.time()
        
        if X_val is not None and y_val is not None:
            # Handle early stopping for different model types
            if isinstance(self.model, lgb.LGBMRegressor):
                # LightGBM early stopping
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(self.config.early_stopping_rounds)]
                )
            elif isinstance(self.model, xgb.XGBRegressor):
                # XGBoost early stopping (early_stopping_rounds set in constructor)
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                # Other models without early stopping
                self.model.fit(X, y)
        else:
            # No validation data - train without early stopping
            self.model.fit(X, y)
        
        training_time = time.time() - start_time
        
        if self.logger:
            self.logger.info(f"{self.get_model_type()} training completed in {training_time:.2f}s")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        predictions = self.model.predict(X)
        
        # Handle potential shape issues
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            return dict(zip(feature_names, importance))
        
        return None


class LightGBMModel(BaseModel):
    """LightGBM model implementation with optimal threading."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> lgb.LGBMRegressor:
        """Create LightGBM model with optimal parameters."""
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
        
        # Optimal threading configuration
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': self.config.model_n_jobs,
            'force_row_wise': True,  # Better for multi-threading
            'verbose': -1,
            'early_stopping_rounds': self.config.early_stopping_rounds if trial is None else None
        })
        
        return lgb.LGBMRegressor(**params)
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "lightgbm"


class XGBoostModel(BaseModel):
    """XGBoost model implementation with optimal threading."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> xgb.XGBRegressor:
        """Create XGBoost model with optimal parameters."""
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
        
        # Optimal threading configuration
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': self.config.model_n_jobs,
            'tree_method': 'hist',  # Optimal for multi-core
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
        """Create Random Forest model."""
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
        
        # Threading configuration
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': self.config.model_n_jobs,
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
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.logger = logger
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        if self.logger:
            self.logger.info(f"Initialized ensemble with {len(models)} models")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Fit all models in the ensemble."""
        for i, model in enumerate(self.models):
            if self.logger:
                self.logger.info(f"Training ensemble model {i+1}/{len(self.models)}: {model.get_model_type()}")
            
            model.model = model._create_model()
            model.fit(X, y, X_val, y_val)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.models or any(model.model is None for model in self.models):
            raise ValueError("All models must be trained before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average of predictions
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get aggregated feature importance from all models."""
        importance_dicts = []
        
        for model in self.models:
            importance = model.get_feature_importance()
            if importance is not None:
                importance_dicts.append(importance)
        
        if not importance_dicts:
            return None
        
        # Aggregate importance across models
        all_features = set()
        for imp_dict in importance_dicts:
            all_features.update(imp_dict.keys())
        
        aggregated_importance = {}
        for feature in all_features:
            importances = [imp_dict.get(feature, 0.0) for imp_dict in importance_dicts]
            aggregated_importance[feature] = np.mean(importances)
        
        return aggregated_importance
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "ensemble"


class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna."""
    
    def __init__(
        self,
        config: ModelConfig,
        logger: Optional[MLLogger] = None
    ):
        self.config = config
        self.logger = logger
        
        # Configure Optuna to be less verbose
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def optimize(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        currency: str,
        cv_folds: int = 5
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Optimize hyperparameters for a model."""
        if self.logger:
            self.logger.info(f"Starting hyperparameter optimization for {model_class.__name__}")
        
        def objective(trial):
            # Create model with trial parameters
            model = model_class(self.config, self.logger)
            model.model = model._create_model(trial)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Train and evaluate
                model.model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                # Calculate MAE as objective
                mae = mean_absolute_error(y_val_fold, y_pred)
                scores.append(mae)
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials)
        
        # Create best model
        best_model = model_class(self.config, self.logger)
        best_model.model = best_model._create_model(study.best_trial)
        
        if self.logger:
            self.logger.info(f"Optimization completed. Best MAE: {study.best_value:.4f}")
        
        return best_model, study.best_params


class ModelTrainer:
    """Main model trainer with optimized multi-core performance."""
    
    def __init__(
        self,
        config: ModelConfig,
        processing_config: ProcessingConfig,
        logger: Optional[MLLogger] = None
    ):
        self.config = config
        self.processing_config = processing_config
        self.logger = logger
        
        # Configure optimal threading
        configure_optimal_threading(config)
        
        if self.logger:
            self.logger.info("ModelTrainer initialized with optimal threading configuration")
    
    def train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        currency: str,
        model_type: str = "ensemble",
        target_names: Optional[List[str]] = None
    ) -> TrainingResult:
        """Train a single model with optimal performance."""
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Training {model_type} model for {currency}")
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split training data for validation
        val_split_idx = int(len(X_train) * 0.8)
        X_train_split, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_split, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        # Scale features if needed
        scaler = None
        if self.processing_config.robust_scaling:
            scaler = RobustScaler()
            X_train_split = scaler.fit_transform(X_train_split)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Train model
        if model_type == "ensemble":
            model = self._train_ensemble_model(X_train_split, y_train_split, X_val, y_val, currency)
        else:
            model = self._train_single_base_model(X_train_split, y_train_split, X_val, y_val, currency, model_type)
        
        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        metrics = ModelMetrics.from_predictions(y_test, y_pred, target_names)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Calculate cross-validation scores
        cv_scores = self._calculate_cv_scores(model, X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Get hyperparameters
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            hyperparameters = model.model.get_params()
        elif hasattr(model, 'models'):
            hyperparameters = {f"model_{i}": m.model.get_params() for i, m in enumerate(model.models)}
        else:
            hyperparameters = {}
        
        return TrainingResult(
            model=model,
            scaler=scaler,
            metrics=metrics,
            training_time=training_time,
            hyperparameters=hyperparameters,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            model_type=model_type,
            currency=currency
        )
    
    def _train_ensemble_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency: str
    ) -> EnsembleModel:
        """Train ensemble model with optimal threading."""
        models = []
        
        if self.config.use_lightgbm:
            lgb_model = LightGBMModel(self.config, self.logger)
            lgb_model.model = lgb_model._create_model()
            models.append(lgb_model)
        
        if self.config.use_xgboost:
            xgb_model = XGBoostModel(self.config, self.logger)
            xgb_model.model = xgb_model._create_model()
            models.append(xgb_model)
        
        ensemble = EnsembleModel(models, logger=self.logger)
        ensemble.fit(X_train, y_train, X_val, y_val)
        
        return ensemble
    
    def _train_single_base_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency: str,
        model_type: str
    ) -> BaseModel:
        """Train a single base model."""
        if model_type == "lightgbm":
            model = LightGBMModel(self.config, self.logger)
        elif model_type == "xgboost":
            model = XGBoostModel(self.config, self.logger)
        elif model_type == "random_forest":
            model = RandomForestModel(self.config, self.logger)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.model = model._create_model()
        model.fit(X_train, y_train, X_val, y_val)
        
        return model
    
    def _calculate_cv_scores(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Calculate cross-validation scores."""
        try:
            tscv = TimeSeriesSplit(n_splits=min(self.config.cv_folds, len(X) // 50))
            
            if hasattr(model, 'model'):
                # Single model
                scores = cross_val_score(
                    model.model, X, y,
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=1  # Use single job for CV to avoid conflicts
                )
            else:
                # Ensemble - manual CV
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                    
                    # Create temporary model
                    temp_ensemble = EnsembleModel([
                        LightGBMModel(self.config),
                        XGBoostModel(self.config)
                    ])
                    temp_ensemble.fit(X_train_cv, y_train_cv)
                    
                    y_pred_cv = temp_ensemble.predict(X_val_cv)
                    mae = mean_absolute_error(y_val_cv, y_pred_cv)
                    scores.append(-mae)  # Negative for consistency
            
            return scores.tolist()
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Cross-validation failed: {e}")
            return []


def save_model_artifacts(
    result: TrainingResult,
    output_dir: Path,
    currency: str
) -> Dict[str, str]:
    """Save model artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save model with standard name expected by model inference utility
    model_path = output_dir / "ensemble_model.pkl"
    joblib.dump(result.model, model_path)
    saved_files['model'] = str(model_path)
    
    # Save scaler if exists
    if result.scaler is not None:
        scaler_path = output_dir / "scaler.pkl"
        joblib.dump(result.scaler, scaler_path)
        saved_files['scaler'] = str(scaler_path)
    
    # Save model metadata with comprehensive information
    metadata = {
        'model_type': result.model_type,
        'currency': currency,
        'training_timestamp': datetime.now().isoformat(),
        'training_time': result.training_time,
        'metrics': result.metrics.to_dict(),
        'hyperparameters': result.hyperparameters,
        'feature_importance': result.feature_importance,
        'cross_validation_scores': result.cross_validation_scores
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    saved_files['metadata'] = str(metadata_path)
    
    # Save individual metrics file for backward compatibility
    metrics_path = output_dir / f"{currency}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(result.metrics.to_dict(), f, indent=2)
    saved_files['metrics_legacy'] = str(metrics_path)
    
    # Save feature importance if exists
    if result.feature_importance is not None:
        importance_path = output_dir / f"{currency}_feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(result.feature_importance, f, indent=2)
        saved_files['feature_importance'] = str(importance_path)
    
    # Save hyperparameters
    params_path = output_dir / f"{currency}_hyperparameters.json"
    with open(params_path, 'w') as f:
        json.dump(result.hyperparameters, f, indent=2, default=str)
    saved_files['hyperparameters'] = str(params_path)
    
    return saved_files 