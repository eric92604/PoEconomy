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

from ml.config.training_config import ModelConfig, ProcessingConfig
from ml.utils.common_utils import MLLogger


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
            r2=r2,
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
            r2_dict[target_name] = r2_score(y_true_valid, y_pred_valid)
            
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
    training_history: Optional[Dict[str, Any]] = None  # Training losses per epoch/trial
    training_samples: int = 0  # Number of training samples
    residual_standard_error: float = 0.0  # RSE from test set
    n_features: int = 0  # Number of features for degrees of freedom calculation
    confidence_level: float = 0.95  # Confidence level for prediction intervals
    feature_names: Optional[List[str]] = None  # Exact feature names used during training


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, config: ModelConfig, logger: Optional[MLLogger] = None):
        self.config = config
        self.logger = logger
        self.model: Any = None
        
        # Suppress model initialization logging to reduce noise
    
    @abstractmethod
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> Any:
        """Create the underlying model with optional hyperparameter optimization."""
        pass
    
    def _create_model_with_params(self, params: Dict[str, Any]) -> Any:
        """Create model instance with specific parameters."""
        # Create a mock trial with the given parameters
        class MockTrial:
            def __init__(self, params: Dict[str, Any]):
                self.params = params
            
            def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
                # Try exact match first, then try with prefix stripped
                if name in self.params:
                    return self.params[name]
                # For parameters like 'learning_rate', try 'xgb_learning_rate' or 'lgb_learning_rate'
                for key, value in self.params.items():
                    if key.endswith(f"_{name}"):
                        return value
                raise ValueError(f"Parameter '{name}' not found in optimized parameters. This indicates hyperparameter optimization failure.")
            
            def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
                # Try exact match first, then try with prefix stripped
                if name in self.params:
                    return self.params[name]
                # For parameters like 'n_estimators', try 'xgb_n_estimators' or 'lgb_n_estimators'
                for key, value in self.params.items():
                    if key.endswith(f"_{name}"):
                        return value
                raise ValueError(f"Parameter '{name}' not found in optimized parameters. This indicates hyperparameter optimization failure.")
            
            def suggest_categorical(self, name: str, choices: List[Any], **kwargs) -> Any:
                # Try exact match first, then try with prefix stripped
                if name in self.params:
                    return self.params[name]
                # For parameters like 'boosting_type', try 'xgb_boosting_type' or 'lgb_boosting_type'
                for key, value in self.params.items():
                    if key.endswith(f"_{name}"):
                        return value
                raise ValueError(f"Parameter '{name}' not found in optimized parameters. This indicates hyperparameter optimization failure.")
        
        return self._create_model(MockTrial(params))
    
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
    ) -> Dict[str, Any]:
        """
        Fit the model to training data.
        
        Returns:
            Dictionary containing training history (losses per epoch) if available
        """
        if self.model is None:
            raise ValueError("Model has not been initialized. Call _create_model() first.")
        
        start_time = time.time()
        training_history = {}
        
        if X_val is not None and y_val is not None:
            # Always use early stopping when validation data exists
            if isinstance(self.model, lgb.LGBMRegressor):
                # LightGBM early stopping with history tracking
                evals_result = {}
                self.model.fit(
                    X, y,
                    eval_set=[(X, y), (X_val, y_val)],  # Include training set for training loss
                    eval_names=['training', 'validation'],
                    callbacks=[
                        lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                        lgb.record_evaluation(evals_result)
                    ]
                )
                # Extract training history from evals_result
                if evals_result:
                    training_history = {
                        'train_loss': evals_result.get('training', {}).get('l2', []),
                        'val_loss': evals_result.get('validation', {}).get('l2', []),
                        'epochs': len(evals_result.get('validation', {}).get('l2', []))
                    }
            elif isinstance(self.model, xgb.XGBRegressor):
                # XGBoost early stopping - use parameter-based approach when available
                try:
                    # Check if early_stopping_rounds is set in model params
                    model_params = self.model.get_params()
                    evals_result = {}
                    if 'early_stopping_rounds' in model_params and model_params['early_stopping_rounds'] is not None:
                        # Use parameter-based early stopping (for non-Optuna training)
                        self.model.fit(
                            X, y,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:
                        # When Optuna is active, train without XGBoost's built-in early stopping
                        # Optuna will handle pruning based on validation scores
                        self.model.fit(
                            X, y,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    # Extract training history from evals_result (XGBoost stores it in model)
                    if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
                        evals_result = self.model.evals_result_
                        # XGBoost uses different key structure
                        if 'validation' in evals_result or 0 in evals_result:
                            val_key = 'validation' if 'validation' in evals_result else 0
                            val_losses = evals_result[val_key].get('rmse', evals_result[val_key].get('l2', []))
                            train_key = 'train' if 'train' in evals_result else 0
                            train_losses = evals_result.get(train_key, {}).get('rmse', evals_result.get(train_key, {}).get('l2', []))
                            training_history = {
                                'train_loss': train_losses if train_losses else [],
                                'val_loss': val_losses if val_losses else [],
                                'epochs': len(val_losses) if val_losses else 0
                            }
                except (TypeError, AttributeError, ImportError) as e:
                    # If early stopping fails, train without it
                    if self.logger:
                        self.logger.warning(f"XGBoost early stopping failed: {e}. Training without early stopping.")
                    self.model.fit(X, y, verbose=False)
            else:
                # Other models without early stopping
                self.model.fit(X, y)
        else:
            # No validation data - train without early stopping
            # This is expected during cross-validation where validation folds are used separately for evaluation
            self.model.fit(X, y)
        
        training_time = time.time() - start_time
        
        # Store training history in model for later retrieval
        if training_history:
            if hasattr(self, 'training_history'):
                self.training_history = training_history
            else:
                setattr(self, 'training_history', training_history)
        
        return training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        predictions = self.model.predict(X)
        
        # Handle potential shape issues
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions  # type: ignore[no-any-return]
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            # Use provided feature names, then model's names, then fallback to generic names
            if feature_names is not None:
                names = feature_names
            elif hasattr(self.model, 'feature_names_in_'):
                names = self.model.feature_names_in_
            else:
                names = [f'feature_{i}' for i in range(len(importance))]
            
            return dict(zip(names, importance))
        
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
                'n_estimators': self.config.n_model_trials,
                'max_depth': self.config.max_depth,  # Use environment variable
                'learning_rate': self.config.learning_rate,  # Use environment variable
                'num_leaves': 100,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
        
        # Calculate optimal n_jobs based on system resources and parallel context
        import os
        max_currency_workers = int(os.getenv('MAX_CURRENCY_WORKERS', '4'))
        
        if max_currency_workers > 1:
            optimal_n_jobs = max(1, min(self.config.model_n_jobs, 2))
        else:
            # Single currency worker: can use more resources
            optimal_n_jobs = self.config.model_n_jobs
            
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': optimal_n_jobs,
            'force_row_wise': True,  # Better for multi-threading
            'verbose': -1
        })
        
        return lgb.LGBMRegressor(**params)  # type: ignore[arg-type]
    
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
                'n_estimators': self.config.n_model_trials,
                'max_depth': self.config.max_depth,  # Use environment variable
                'learning_rate': self.config.learning_rate,  # Use environment variable
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
        
        # Calculate optimal n_jobs based on system resources and parallel context
        import os
        max_currency_workers = int(os.getenv('MAX_CURRENCY_WORKERS', '4'))
        
        if max_currency_workers > 1:
            # For 16 vCPU system: 4 currency workers × 1-2 model jobs = 4-8 processes (optimal)
            optimal_n_jobs = max(1, min(self.config.model_n_jobs, 2))
        else:
            # Single currency worker: can use more resources
            optimal_n_jobs = self.config.model_n_jobs
            
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': optimal_n_jobs,
            'tree_method': 'hist',  # type: ignore[dict-item]  # Optimal for multi-core
            'verbosity': 0,
        })
        
        # Only set early_stopping_rounds when NOT using Optuna (trial is None)
        # Optuna handles pruning itself and conflicts with XGBoost's built-in early stopping
        if trial is None:
            params['early_stopping_rounds'] = self.config.early_stopping_rounds
        
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
        
        # Calculate optimal n_jobs based on system resources and parallel context
        import os
        max_currency_workers = int(os.getenv('MAX_CURRENCY_WORKERS', '4'))
        
        if max_currency_workers > 1:
            # For 16 vCPU system: 4 currency workers × 1-2 model jobs = 4-8 processes (optimal)
            optimal_n_jobs = max(1, min(self.config.model_n_jobs, 2))
        else:
            # Single currency worker: can use more resources
            optimal_n_jobs = self.config.model_n_jobs
            
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': optimal_n_jobs,
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

    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Fit all models in the ensemble.
        
        Returns:
            Dictionary containing training history from all models
        """
        ensemble_history = {}
        for i, model in enumerate(self.models):
            # Only create model if it doesn't already exist (may have been created with optimized params)
            if model.model is None:
                model.model = model._create_model()
            model_history = model.fit(X, y, X_val, y_val)
            if model_history:
                model_type = model.get_model_type()
                ensemble_history[f"{model_type}_{i}"] = model_history
        
        return ensemble_history
    
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
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
        """Get aggregated feature importance from all models."""
        importance_dicts = []
        
        for model in self.models:
            importance = model.get_feature_importance(feature_names=feature_names)
            if importance is not None:
                importance_dicts.append(importance)
        
        if not importance_dicts:
            return None
        
        # Aggregate importance across models
        all_features: set[str] = set()
        for imp_dict in importance_dicts:
            all_features.update(imp_dict.keys())
        
        aggregated_importance = {}
        for feature in all_features:
            importances = [imp_dict.get(feature, 0.0) for imp_dict in importance_dicts]
            aggregated_importance[feature] = np.mean(importances)
        
        return aggregated_importance  # type: ignore[return-value]
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "ensemble"


class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna with per-currency optimization."""
    
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
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency: str,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a model."""
        
        def objective(trial: Any) -> float:
            # Create model with trial parameters
            model = model_class(self.config, self.logger)
            model._is_optuna_trial = True  # Mark as Optuna trial to suppress logging
            model.model = model._create_model(trial)
            
            # Use provided validation data if available, otherwise use cross-validation
            if X_val is not None and y_val is not None and len(X_val) >= 5:
                # Use provided validation data with early stopping
                model.fit(X, y, X_val, y_val)
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                return float(mae)
            else:
                # Fall back to time series cross-validation with proper validation
                min_samples_per_fold = 10
                max_folds = max(2, len(X) // min_samples_per_fold)  # At least 2 folds
                n_splits = min(cv_folds, max_folds)
                
                if n_splits < 2:
                    if self.logger:
                        self.logger.warning(f"Insufficient data for cross-validation: {len(X)} samples, need at least {min_samples_per_fold * 2}. Using simple train/validation split.")
                    # Use simple train/validation split instead
                    split_idx = int(len(X) * 0.8)
                    X_train_fold, X_val_fold = X[:split_idx], X[split_idx:]
                    y_train_fold, y_val_fold = y[:split_idx], y[split_idx:]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    mae = mean_absolute_error(y_val_fold, y_pred)
                    return float(mae)
                
                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    # Train and evaluate
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    
                    # Calculate MAE as objective
                    mae = mean_absolute_error(y_val_fold, y_pred)
                    scores.append(mae)
                
                return float(np.mean(scores))
        
        # Create study with proper storage for parallelization
        # Make study name unique per currency and model type to avoid conflicts
        study_name = f"{model_class.__name__}_{currency}_{self.config.random_state}"
        
        # Initialize storage_file variable for cleanup
        storage_file = None
        
        if self.config.max_optuna_workers > 1:
            # Use SQLite storage for multi-threaded parallelization with proper configuration
            import tempfile
            import os
            import uuid
            
            # Create unique database file per currency and process to avoid conflicts
            temp_dir = tempfile.gettempdir()
            unique_id = str(uuid.uuid4())[:8]  # Short unique ID
            storage_file = os.path.join(temp_dir, f"optuna_{study_name}_{unique_id}.db")
            
            # Configure SQLite for better concurrent access
            sqlite_url = f"sqlite:///{storage_file}?timeout=30&check_same_thread=False"
            
            try:
                # Create storage with proper connection management
                storage = optuna.storages.RDBStorage(
                    sqlite_url,
                    engine_kwargs={'pool_pre_ping': True, 'pool_recycle': 300}
                )
                study = optuna.create_study(
                    direction='minimize',
                    sampler=TPESampler(seed=self.config.random_state),
                    pruner=MedianPruner(),
                    storage=storage,
                    study_name=study_name,
                    load_if_exists=True
                )
                if self.logger:
                    self.logger.debug(f"Using SQLite storage for {currency}: {storage_file}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"SQLite storage failed for {currency}: {e}. Falling back to in-memory storage.")
                # Fall back to in-memory storage if SQLite fails
                study = optuna.create_study(
                    direction='minimize',
                    sampler=TPESampler(seed=self.config.random_state),
                    pruner=MedianPruner(),
                    study_name=study_name,
                    load_if_exists=True
                )
                storage_file = None  # No file to clean up
        else:
            # Use in-memory storage for single-threaded optimization
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=self.config.random_state),
                pruner=MedianPruner(),
                study_name=study_name,
                load_if_exists=True
            )
        
        # Check if we're likely running in a parallel context (multiple currency workers)
        import os
        max_currency_workers = int(os.getenv('MAX_CURRENCY_WORKERS', '6'))
        if max_currency_workers > 1:
            # Reduce Optuna workers when multiple currency workers are running to avoid resource conflicts
            optuna_workers = max(1, min(self.config.max_optuna_workers, 2))
            if self.logger:
                self.logger.debug(f"Reduced Optuna workers to {optuna_workers} due to parallel currency processing")
        else:
            optuna_workers = self.config.max_optuna_workers
            
        study.optimize(objective, n_trials=self.config.n_hyperparameter_trials, n_jobs=optuna_workers)
        
        if self.logger:
            self.logger.info(f"Optimization completed. Best MAE: {study.best_value:.4f}")
        
        # Clean up temporary database file if it exists
        if storage_file is not None:
            try:
                import os
                import time
                
                # Give a moment for any pending writes to complete
                time.sleep(0.1)
                
                if os.path.exists(storage_file):
                    # Force close any open connections
                    try:
                        if 'storage' in locals():
                            storage._backend._engine.dispose()
                    except:
                        pass
                    
                    os.remove(storage_file)
                    if self.logger:
                        self.logger.debug(f"Cleaned up temporary database: {storage_file}")
                else:
                    if self.logger:
                        self.logger.debug(f"Temporary database file already cleaned up: {storage_file}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to clean up database file {storage_file}: {e}")
        
        return study.best_params
    


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
        
        # Initialize hyperparameter optimizer
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config, self.logger)
    
    
    def train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        currency: str,
        model_type: str = "ensemble",
        target_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> TrainingResult:
        """Train a single model with optimal performance."""
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Training {model_type} model for {currency}")
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split training data for validation (always create validation split)
        val_split_idx = int(len(X_train) * 0.8)
        X_train_split, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_split, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        # Scale features if needed
        scaler = None
        if self.processing_config.robust_scaling:
            scaler = RobustScaler()
            X_train_split = scaler.fit_transform(X_train_split)
            if X_val is not None:
                X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Train model (validation data is always available now)
        training_history = None
        if model_type == "ensemble":
            model: Any = self._train_ensemble_model(X_train_split, y_train_split, X_val, y_val, currency)
            # Get training history from ensemble
            if hasattr(model, 'models'):
                training_history = {}
                for i, base_model in enumerate(model.models):
                    if hasattr(base_model, 'training_history'):
                        model_type_name = base_model.get_model_type()
                        training_history[f"{model_type_name}_{i}"] = base_model.training_history
        else:
            model = self._train_single_base_model(X_train_split, y_train_split, X_val, y_val, currency, model_type)
            # Get training history from single model
            if hasattr(model, 'training_history'):
                training_history = model.training_history
        
        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        metrics = ModelMetrics.from_predictions(y_test, y_pred, target_names)
        
        # Calculate residual standard error (RSE) from test set
        residuals = y_test - y_pred
        n_test = len(residuals)
        n_features = X_test.shape[1]
        degrees_of_freedom = max(1, n_test - n_features - 1)
        rse = np.sqrt(np.sum(residuals**2) / degrees_of_freedom)
        
        # Get feature importance with feature names
        feature_importance = model.get_feature_importance(feature_names=feature_names)
        
        # Calculate cross-validation scores
        cv_scores = self._calculate_cv_scores(model, X_train, y_train)
        training_time = time.time() - start_time
        
        # Get hyperparameters
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            hyperparameters = model.model.get_params()
        elif hasattr(model, 'models'):
            hyperparameters = {f"model_{i}": m.model.get_params() if m.model else {} for i, m in enumerate(model.models)}
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
            currency=currency,
            training_history=training_history,
            training_samples=n_test,
            residual_standard_error=float(rse),
            n_features=n_features,
            confidence_level=0.95,
            feature_names=feature_names,  # Store exact feature names used during training
        )
    
    def _train_ensemble_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency: str
    ) -> EnsembleModel:
        """Train ensemble model using per-currency hyperparameter optimization."""
        models = []
        
        # Perform per-currency hyperparameter optimization if enabled
        if self.config.n_hyperparameter_trials > 1:
            if self.config.use_lightgbm:
                lgb_model = LightGBMModel(self.config, self.logger)
                
                # Perform individual optimization for this currency
                if self.logger:
                    self.logger.info(f"Optimizing LightGBM hyperparameters for {currency}")
                
                try:
                    optimized_params = self.hyperparameter_optimizer.optimize(
                        LightGBMModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                    )
                    lgb_model.model = lgb_model._create_model_with_params(optimized_params)
                    if self.logger:
                        self.logger.info(f"Using optimized LightGBM params for {currency}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"LightGBM optimization failed for {currency}: {e}. Using default parameters.")
                    lgb_model.model = lgb_model._create_model()
                
                models.append(lgb_model)
            
            if self.config.use_xgboost:
                xgb_model = XGBoostModel(self.config, self.logger)
                
                # Perform individual optimization for this currency
                if self.logger:
                    self.logger.info(f"Optimizing XGBoost hyperparameters for {currency}")
                
                try:
                    optimized_params = self.hyperparameter_optimizer.optimize(
                        XGBoostModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                    )
                    xgb_model.model = xgb_model._create_model_with_params(optimized_params)
                    if self.logger:
                        self.logger.info(f"Using optimized XGBoost params for {currency}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"XGBoost optimization failed for {currency}: {e}. Using default parameters.")
                    xgb_model.model = xgb_model._create_model()
                
                models.append(xgb_model)  # type: ignore[arg-type]
        else:
            # Use default parameters if optimization is disabled
            if self.logger:
                self.logger.warning(f"Hyperparameter optimization disabled for {currency}. Using default parameters.")
            
            if self.config.use_lightgbm:
                lgb_model = LightGBMModel(self.config, self.logger)
                lgb_model.model = lgb_model._create_model()
                models.append(lgb_model)
            
            if self.config.use_xgboost:
                xgb_model = XGBoostModel(self.config, self.logger)
                xgb_model.model = xgb_model._create_model()
                models.append(xgb_model)  # type: ignore[arg-type]
        
        ensemble = EnsembleModel(models, logger=self.logger)  # type: ignore[arg-type]
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
        """Train a single base model with optional hyperparameter optimization."""
        # Use hyperparameter optimization if enabled
        if self.config.n_hyperparameter_trials > 1:
            optimizer = HyperparameterOptimizer(self.config, self.logger)
            
            if model_type == "lightgbm":
                params = optimizer.optimize(
                    LightGBMModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                )
                model = LightGBMModel(self.config, self.logger)
                model.model = model._create_model_with_params(params)
            elif model_type == "xgboost":
                params = optimizer.optimize(
                    XGBoostModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                )
                model = XGBoostModel(self.config, self.logger)
                model.model = model._create_model_with_params(params)
            elif model_type == "random_forest":
                params = optimizer.optimize(
                    RandomForestModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                )
                model = RandomForestModel(self.config, self.logger)
                model.model = model._create_model_with_params(params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if self.logger:
                self.logger.info(f"{model_type} optimized params: {params}")
        else:
            # Use default parameters for fast training
            if model_type == "lightgbm":
                model: BaseModel = LightGBMModel(self.config, self.logger)
            elif model_type == "xgboost":
                model = XGBoostModel(self.config, self.logger)
            elif model_type == "random_forest":
                model = RandomForestModel(self.config, self.logger)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.model = model._create_model()
        
        model.fit(X_train, y_train, X_val, y_val)
        return model
    
    def _calculate_cv_scores(self, model: Any, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Calculate cross-validation scores."""
        try:
            # Ensure we have enough data for cross-validation
            min_samples_per_fold = 10
            max_folds = max(2, len(X) // min_samples_per_fold)  # At least 2 folds
            n_splits = min(self.config.cv_folds, max_folds)
            
            if n_splits < 2:
                if self.logger:
                    self.logger.warning(f"Insufficient data for cross-validation: {len(X)} samples, need at least {min_samples_per_fold * 2}")
                return []
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
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
                    
                    # Create temporary model with logger for proper error handling
                    temp_ensemble = EnsembleModel([
                        LightGBMModel(self.config, self.logger),
                        XGBoostModel(self.config, self.logger)
                    ], logger=self.logger)
                    temp_ensemble.fit(X_train_cv, y_train_cv)
                    
                    y_pred_cv = temp_ensemble.predict(X_val_cv)
                    mae = mean_absolute_error(y_val_cv, y_pred_cv)
                    scores.append(-mae)  # Negative for consistency
            
            return scores  # type: ignore[no-any-return]
            
        except Exception as e:
            # Use try/except to avoid logger errors masking CV errors
            try:
                if self.logger and hasattr(self.logger, 'logger') and self.logger.logger is not None:
                    self.logger.warning(f"Cross-validation failed: {str(e)}")
                else:
                    # Fallback to print if logger is not available
                    import sys
                    sys.stderr.write(f"Cross-validation failed: {str(e)}\n")
            except Exception as logger_error:
                # If logger itself fails, use stderr as last resort
                import sys
                sys.stderr.write(f"Cross-validation failed: {str(e)} (logger error: {str(logger_error)})\n")
            return []


def save_model_artifacts(
    result: TrainingResult,
    output_dir: Path,
    currency: str
) -> Dict[str, str]:
    """Save model artifacts to disk"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Check if model exists
    if result.model is None:
        raise ValueError(f"Model is None for currency {currency}")
    
    # Save model without compression
    model_path = output_dir / "ensemble_model.pkl"
    try:
        joblib.dump(result.model, model_path, compress=0)
        saved_files['model'] = str(model_path)
        
        # Calculate model size
        model_size_bytes = model_path.stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)
        saved_files['model_size_mb'] = model_size_mb
        saved_files['model_dir'] = str(output_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to save model for {currency}: {e}")
    
    # Save scaler without compression if exists
    if result.scaler is not None:
        scaler_path = output_dir / "scaler.pkl"
        joblib.dump(result.scaler, scaler_path, compress=0)
        saved_files['scaler'] = str(scaler_path)
    
    # Save minimal metadata (only what's needed for inference)
    metadata = {
        'model_type': result.model_type,
        'currency': currency,
        'training_timestamp': datetime.now().isoformat(),
        'training_time': result.training_time,
        'metrics': result.metrics.to_dict(),
        'training_samples': result.training_samples,
        'residual_standard_error': result.residual_standard_error,
        'n_features': result.n_features,
        'confidence_level': result.confidence_level,
    }
    
    # Store feature names if available (for exact feature matching during inference)
    if result.feature_names:
        metadata['feature_names'] = result.feature_names
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    saved_files['metadata'] = str(metadata_path)
    
    # Save training history if available
    if result.training_history:
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(result.training_history, f, indent=2, default=str)
        saved_files['training_history'] = str(history_path)

    return saved_files 