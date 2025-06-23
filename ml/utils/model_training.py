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
            y: Training targets (can be multi-output)
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        # Scale features if configured
        if hasattr(self.config, 'use_scaling') and self.config.use_scaling:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)
        
        # Handle multi-output case - only wrap if not already wrapped
        from sklearn.multioutput import MultiOutputRegressor
        self.is_multi_output = y.ndim > 1 and y.shape[1] > 1
        
        if self.is_multi_output and not isinstance(self.model, MultiOutputRegressor):
            # For multi-output regression, we need to recreate the model without early stopping
            from sklearn.multioutput import MultiOutputRegressor
            # Recreate the base model without early stopping for multi-output
            if hasattr(self, '_create_model_without_early_stopping'):
                base_model = self._create_model_without_early_stopping()
            else:
                base_model = self.model
                # Fallback: try to disable early stopping parameters
                if hasattr(base_model, 'callbacks'):
                    base_model.callbacks = None
                if hasattr(base_model, 'early_stopping_rounds'):
                    base_model.early_stopping_rounds = None
            
            self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        
        # Fit model - always use validation data if available
        model_type = type(self.model).__name__
        base_model_type = type(self.model.estimators_[0] if hasattr(self.model, 'estimators_') else self.model).__name__
        
        if (X_val is not None and y_val is not None and 
            base_model_type in ['LGBMRegressor', 'XGBRegressor'] and 
            not self.is_multi_output):
            # Use validation data for models that support eval_set (single-output only)
            self.model.fit(X, y, eval_set=[(X_val, y_val)])
        else:
            # For other models, multi-output, or when no validation data available
            self.model.fit(X, y)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions (can be multi-output)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        
        # Ensure predictions have correct shape for multi-output
        if hasattr(self, 'is_multi_output') and self.is_multi_output:
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_fitted:
            return None
            
        # Handle multi-output case
        if hasattr(self, 'is_multi_output') and self.is_multi_output:
            if hasattr(self.model, 'estimators_'):
                # Average importance across all outputs
                importances = []
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    return dict(enumerate(avg_importance))
        elif hasattr(self.model, 'feature_importances_'):
            return dict(enumerate(self.model.feature_importances_))
        
        return None


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
    
    def _create_model_without_early_stopping(self, trial: Optional[optuna.Trial] = None) -> lgb.LGBMRegressor:
        """Create LightGBM model without early stopping for multi-output."""
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
        
        # Common parameters - NO early stopping
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbose': -1
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
    
    def _create_model_without_early_stopping(self, trial: Optional[optuna.Trial] = None) -> xgb.XGBRegressor:
        """Create XGBoost model without early stopping for multi-output."""
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
        
        # Common parameters - NO early stopping
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbosity': 0
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
            y: Training targets (can be multi-output)
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        # Store multi-output information
        self.is_multi_output = y.ndim > 1 and y.shape[1] > 1
        
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
            Weighted ensemble predictions (can be multi-output)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Convert to numpy array for easier manipulation
        predictions = np.array(predictions)
        
        # Handle multi-output case
        if hasattr(self, 'is_multi_output') and self.is_multi_output:
            # For multi-output: predictions shape is (n_models, n_samples, n_outputs)
            # We want to average across models for each output
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            # For single output: predictions shape is (n_models, n_samples)
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
        currency: str,
        cv_folds: int = 5
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_class: Model class to optimize
            X: Training features
            y: Training targets
            currency: Currency identifier
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        def objective(trial):
            # Create model with trial parameters
            model = model_class(self.config, self.logger)
            model.model = model._create_model(trial)
            
            # Check if we need multi-output regression
            is_multi_output = y.ndim > 1 and y.shape[1] > 1
            if is_multi_output:
                from sklearn.multioutput import MultiOutputRegressor
                # Disable early stopping for MultiOutputRegressor
                if hasattr(model.model, 'callbacks') and model.model.callbacks:
                    model.model.callbacks = None
                if hasattr(model.model, 'early_stopping_rounds') and model.model.early_stopping_rounds:
                    model.model.early_stopping_rounds = None
                model.model = MultiOutputRegressor(model.model, n_jobs=-1)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Handle NaN values in targets for cross-validation
                if is_multi_output:
                    # For multi-output, remove rows where ALL targets are NaN
                    valid_train_mask = ~np.isnan(y_train).all(axis=1)
                    valid_val_mask = ~np.isnan(y_val).all(axis=1)
                else:
                    # For single output, remove rows where target is NaN
                    valid_train_mask = ~np.isnan(y_train)
                    valid_val_mask = ~np.isnan(y_val)
                
                X_train = X_train[valid_train_mask]
                y_train = y_train[valid_train_mask]
                X_val = X_val[valid_val_mask]
                y_val = y_val[valid_val_mask]
                
                # Skip if no valid samples
                if len(X_train) == 0 or len(X_val) == 0:
                    continue
                
                # Apply median imputation to remaining NaN values in targets
                if is_multi_output and np.isnan(y_train).any():
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    y_train = imputer.fit_transform(y_train)
                    if np.isnan(y_val).any():
                        y_val = imputer.transform(y_val)
                
                # Apply scaling if configured
                if hasattr(self.config, 'use_scaling') and self.config.use_scaling:
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                
                # Fit and predict
                model.model.fit(X_train, y_train)
                y_pred = model.model.predict(X_val)
                
                # Calculate RMSE (optimization target)
                if y_val.ndim > 1 and y_val.shape[1] > 1:
                    # Multi-output case: calculate average RMSE across all outputs
                    rmse_per_output = []
                    for i in range(y_val.shape[1]):
                        valid_mask = ~(np.isnan(y_val[:, i]) | np.isnan(y_pred[:, i]))
                        if np.any(valid_mask):
                            rmse_single = np.sqrt(mean_squared_error(
                                y_val[valid_mask, i], 
                                y_pred[valid_mask, i]
                            ))
                            rmse_per_output.append(rmse_single)
                    
                    rmse = np.mean(rmse_per_output) if rmse_per_output else float('inf')
                else:
                    # Single output case
                    if y_val.ndim > 1:
                        y_val = y_val.flatten()
                    if y_pred.ndim > 1:
                        y_pred = y_pred.flatten()
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                scores.append(rmse)
            
            return np.mean(scores)
        
        # Create study
        study_name = f"optimize_{currency}_{model_class.__name__}"
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=study_name
        )
        
        # Optimize
        with self.logger.log_operation(f"Hyperparameter optimization for {currency}"):
            study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        # Create best model
        best_model = model_class(self.config, self.logger)
        
        # Check if we need multi-output regression for the best model too
        is_multi_output = y.ndim > 1 and y.shape[1] > 1
        if is_multi_output:
            from sklearn.multioutput import MultiOutputRegressor
            # Create model without early stopping for multi-output
            if hasattr(best_model, '_create_model_without_early_stopping'):
                base_model = best_model._create_model_without_early_stopping()
            else:
                base_model = best_model._create_model()
                # Fallback: try to disable early stopping parameters
                if hasattr(base_model, 'callbacks'):
                    base_model.callbacks = None
                if hasattr(base_model, 'early_stopping_rounds'):
                    base_model.early_stopping_rounds = None
            
            best_model.model = MultiOutputRegressor(base_model, n_jobs=-1)
            # Store reference to base model for parameter updates
            actual_model = base_model
        else:
            best_model.model = best_model._create_model()
            actual_model = best_model.model
        
        # Update model with best parameters
        best_params = study.best_params
        for param_name, param_value in best_params.items():
            # Remove model prefix from parameter name
            clean_param_name = param_name.split('_', 1)[1] if '_' in param_name else param_name
            if hasattr(actual_model, clean_param_name):
                setattr(actual_model, clean_param_name, param_value)
        
        self.logger.info(
            f"Optimization completed for {currency}",
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
        currency: str,
        model_type: str = "ensemble",
        target_names: Optional[List[str]] = None
    ) -> TrainingResult:
        """
        Train a single model for a currency.
        
        Args:
            X: Feature matrix
            y: Target values (can be multi-output)
            currency: Currency identifier
            model_type: Type of model to train
            target_names: Names of target columns for multi-output
            
        Returns:
            Training result
        """
        start_time = time.time()
        
        # Determine if multi-output
        is_multi_output = y.ndim > 1 and y.shape[1] > 1
        
        # Split data into train/val
        n_samples = len(X)
        val_size = int(n_samples * self.config.test_size)
        train_idx = np.arange(n_samples - val_size)
        val_idx = np.arange(n_samples - val_size, n_samples)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Apply imputation to train/validation splits if needed
        if is_multi_output and (np.isnan(y_train).any() or np.isnan(y_val).any()):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            y_train = imputer.fit_transform(y_train)
            if np.isnan(y_val).any():
                y_val = imputer.transform(y_val)
        
        # Train model
        if model_type == "ensemble" and self.config.use_ensemble:
            model = self._train_ensemble_model(
                X_train, y_train,
                X_val, y_val,
                currency
            )
        else:
            model = self._train_single_base_model(
                X_train, y_train,
                X_val, y_val,
                currency,
                model_type
            )
        
        # Calculate metrics
        y_pred = model.predict(X)
        
        # Handle metrics calculation for multi-output
        if is_multi_output and target_names:
            metrics = ModelMetrics.from_predictions(y, y_pred, target_names)
        else:
            metrics = ModelMetrics.from_predictions(y, y_pred)
        
        # Calculate cross-validation scores if enough data
        cv_scores = self._calculate_cv_scores(model, X, y)
        
        # Get feature importance
        feature_importance = model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None
        
        # Get hyperparameters
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            hyperparameters = model.model.get_params()
        elif hasattr(model, 'get_params'):
            hyperparameters = model.get_params()
        else:
            hyperparameters = {}
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model=model,
            scaler=getattr(model, 'scaler', None),
            metrics=metrics,
            training_time=training_time,
            hyperparameters=hyperparameters,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            model_type=model.get_model_type(),
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
        """Train ensemble model."""
        models = []
        
        # Determine if we need multi-output regression
        is_multi_output = y_train.ndim > 1 and y_train.shape[1] > 1
        
        # Create base models
        if self.config.use_lightgbm:
            lgb_model, _ = self.optimizer.optimize(
                LightGBMModel, X_train, y_train, currency
            )
            # Ensure the optimized model is properly configured for multi-output
            if is_multi_output and not hasattr(lgb_model.model, 'estimators_'):
                # The optimizer should have wrapped it, but double-check
                from sklearn.multioutput import MultiOutputRegressor
                if not isinstance(lgb_model.model, MultiOutputRegressor):
                    base_model = lgb_model._create_model_without_early_stopping()
                    lgb_model.model = MultiOutputRegressor(base_model, n_jobs=-1)
                    lgb_model.is_multi_output = True
            
            # Fit the optimized model on the training data
            lgb_model.fit(X_train, y_train, X_val, y_val)
            models.append(lgb_model)
        
        if self.config.use_xgboost:
            xgb_model, _ = self.optimizer.optimize(
                XGBoostModel, X_train, y_train, currency
            )
            # Ensure the optimized model is properly configured for multi-output
            if is_multi_output and not hasattr(xgb_model.model, 'estimators_'):
                # The optimizer should have wrapped it, but double-check
                from sklearn.multioutput import MultiOutputRegressor
                if not isinstance(xgb_model.model, MultiOutputRegressor):
                    base_model = xgb_model._create_model_without_early_stopping()
                    xgb_model.model = MultiOutputRegressor(base_model, n_jobs=-1)
                    xgb_model.is_multi_output = True
            
            # Fit the optimized model on the training data
            xgb_model.fit(X_train, y_train, X_val, y_val)
            models.append(xgb_model)
        
        # Create ensemble - don't call fit here as individual models are already fitted
        ensemble = EnsembleModel(models, logger=self.logger)
        
        # Set ensemble properties
        ensemble.is_multi_output = is_multi_output
        ensemble.is_fitted = True
        
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
        """Train single base model."""
        model_classes = {
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel,
            'random_forest': RandomForestModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_classes[model_type]
        
        model, _ = self.optimizer.optimize(
            model_class, X_train, y_train, currency
        )
        
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
                
                # Handle multi-output case
                if y_val.ndim > 1 and y_val.shape[1] > 1:
                    # For multi-output, calculate average RMSE across all outputs
                    rmse_per_output = []
                    for i in range(y_val.shape[1]):
                        valid_mask = ~(np.isnan(y_val[:, i]) | np.isnan(y_pred[:, i]))
                        if np.any(valid_mask):
                            rmse_single = np.sqrt(mean_squared_error(
                                y_val[valid_mask, i], 
                                y_pred[valid_mask, i]
                            ))
                            rmse_per_output.append(rmse_single)
                    
                    rmse = np.mean(rmse_per_output) if rmse_per_output else float('inf')
                else:
                    # Single output case
                    if y_val.ndim > 1:
                        y_val = y_val.flatten()
                    if y_pred.ndim > 1:
                        y_pred = y_pred.flatten()
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                scores.append(rmse)
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {str(e)}")
            return []


def save_model_artifacts(
    result: TrainingResult,
    output_dir: Path,
    currency: str
) -> Dict[str, str]:
    """
    Save model artifacts to disk.
    
    Args:
        result: Training result
        output_dir: Output directory
        currency: Currency identifier
        
    Returns:
        Dictionary of saved file paths
    """
    # Sanitize currency name for file system
    safe_currency = currency.replace(" -> ", "_to_").replace("'", "").replace(":", "").replace("/", "_").replace("\\", "_").replace("?", "").replace("*", "").replace("|", "").replace("<", "").replace(">", "").replace('"', "")
    
    # Create currency-specific directory
    currency_dir = output_dir / safe_currency
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
        'currency': result.currency,
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