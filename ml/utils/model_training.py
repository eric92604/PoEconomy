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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)
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
    imputer: Optional[Any] = None  # Fitted imputer for models that require imputation (RandomForest/ExtraTrees)
    metrics: ModelMetrics = None  # type: ignore[assignment]
    training_time: float = 0.0
    hyperparameters: Dict[str, Any] = None  # type: ignore[assignment]
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    model_type: str = ""
    currency: str = ""
    training_history: Optional[Dict[str, Any]] = None  # Training losses per epoch/trial
    training_samples: int = 0  # Number of training samples
    n_features: int = 0  # Number of features
    confidence_level: float = 0.95  # Confidence level for prediction intervals
    feature_names: Optional[List[str]] = None  # Exact feature names used during training


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, config: ModelConfig, logger: Optional[MLLogger] = None):
        self.config = config
        self.logger = logger
        self.model: Any = None
        self.imputer: Optional[Any] = None  # Fitted imputer for models that require imputation
        
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
        
        # Handle NaN values for models that don't support them (RandomForest, ExtraTrees)
        # LightGBM handles NaN natively, but RandomForest and ExtraTrees require imputation
        needs_imputation = isinstance(self.model, (RandomForestRegressor, ExtraTreesRegressor))
        if needs_imputation:
            # Check for NaN values and impute if necessary
            if np.isnan(X).any() or (X_val is not None and np.isnan(X_val).any()):
                from sklearn.impute import SimpleImputer
                # Create and fit imputer on training data only (prevents data leakage)
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
                if X_val is not None:
                    X_val = imputer.transform(X_val)
                # Store fitted imputer for use during inference
                self.imputer = imputer
                if self.logger:
                    self.logger.debug(f"Applied median imputation for {self.get_model_type()} model and stored imputer for inference")
        
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
        """
        Make predictions using the trained model.
        
        Uses the fitted imputer from training to transform inference data,
        preventing data leakage and ensuring consistency with training.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Handle NaN values for models that don't support them (RandomForest, ExtraTrees)
        # LightGBM handles NaN natively, but RandomForest and ExtraTrees require imputation
        needs_imputation = isinstance(self.model, (RandomForestRegressor, ExtraTreesRegressor))
        if needs_imputation:
            # Use the fitted imputer from training (stored in self.imputer)
            # This prevents data leakage and ensures consistency with training
            if hasattr(self, 'imputer') and self.imputer is not None:
                X = self.imputer.transform(X)
            elif np.isnan(X).any():
                # Fallback: if imputer not found but NaNs exist, raise error
                # This should not happen if training was done correctly
                raise ValueError(
                    f"Model requires imputation but no fitted imputer found. "
                    f"This indicates the model was not properly trained or the imputer was not saved."
                )
        
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
    """LightGBM model implementation with configurable threading."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> lgb.LGBMRegressor:
        """Create LightGBM model with configured parameters."""
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
        
        # Use configured model_n_jobs from environment variable (MODEL_N_JOBS)
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': self.config.model_n_jobs,
            'force_row_wise': True,  # Better for multi-threading
            'verbose': -1
        })
        
        return lgb.LGBMRegressor(**params)  # type: ignore[arg-type]
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "lightgbm"


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> RandomForestRegressor:
        """Create Random Forest model."""
        if trial is not None:
            # Hyperparameter optimization
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            }
        else:
            # Default parameters
            params = {
                'n_estimators': 1000,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
            }
        
        # Use configured model_n_jobs from environment variable (MODEL_N_JOBS)
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': self.config.model_n_jobs,
        })
        
        return RandomForestRegressor(**params)
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "random_forest"


class ExtraTreesModel(BaseModel):
    """Extra Trees (Extremely Randomized Trees) model implementation."""
    
    def _create_model(self, trial: Optional[optuna.Trial] = None) -> ExtraTreesRegressor:
        """Create Extra Trees model."""
        if trial is not None:
            # Hyperparameter optimization
            params = {
                'n_estimators': trial.suggest_int('et_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('et_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', None]),
            }
        else:
            # Default parameters - Extra Trees often work well with more trees
            params = {
                'n_estimators': 1000,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
            }
        
        # Use configured model_n_jobs from environment variable (MODEL_N_JOBS)
        params.update({
            'random_state': self.config.random_state,
            'n_jobs': self.config.model_n_jobs,
        })
        
        return ExtraTreesRegressor(**params)
    
    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "extra_trees"


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
        
        # Log weights if logger is available
        if self.logger and len(self.models) > 0:
            model_types = [model.get_model_type() for model in self.models]
            weight_str = ', '.join([f"{mt}: {w:.3f}" for mt, w in zip(model_types, self.weights)])
            self.logger.debug(f"Ensemble weights: {weight_str}")

    
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
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        confidence_level: float = 0.60
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make ensemble predictions with uncertainty estimates using ensemble spread.
        
        Uses the variance across ensemble model predictions as a measure of uncertainty.
        This is a robust method that adapts to prediction difficulty.
        
        Args:
            X: Input features
            confidence_level: Confidence level for prediction intervals (default 0.80 for narrower ranges)
        
        Returns:
            Tuple of (mean_prediction, lower_bound, upper_bound)
        """
        if not self.models or any(model.model is None for model in self.models):
            raise ValueError("All models must be trained before making predictions")
        
        if len(self.models) < 2:
            raise ValueError("Ensemble spread requires at least 2 models in the ensemble")
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate weighted mean prediction
        mean_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Calculate weighted standard deviation (ensemble spread)
        # This measures disagreement between models, which indicates uncertainty
        variance = np.average((predictions - mean_pred) ** 2, axis=0, weights=self.weights)
        std_pred = np.sqrt(variance)
        
        # Use appropriate multiplier for prediction intervals
        # For small ensembles (<=2 models), t-distribution has very fat tails (df=1 gives t_crit=12.7)
        # Use a more conservative approach: normal distribution approximation or fixed multiplier
        from scipy import stats
        n_models = len(self.models)
        
        if n_models <= 2:
            # For very small ensembles, use normal distribution (z-score) with a conservative multiplier
            # This prevents unreasonably wide intervals from t-distribution with low df
            # Use 2 standard deviations for 95% confidence (normal approximation)
            multiplier = 2.0
            if self.logger:
                self.logger.debug(
                    f"Using normal approximation for small ensemble (n={n_models}), "
                    f"multiplier={multiplier} for {confidence_level*100}% confidence"
                )
        elif n_models <= 5:
            # For small ensembles, use t-distribution but cap the multiplier
            # This balances statistical rigor with practical bounds
            degrees_of_freedom = n_models - 1
            t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
            # Cap at 3.0 to prevent excessive widening
            multiplier = min(t_critical, 3.0)
            if self.logger:
                self.logger.debug(
                    f"Using capped t-distribution (n={n_models}, df={degrees_of_freedom}), "
                    f"t_critical={t_critical:.2f}, capped to {multiplier:.2f}"
                )
        else:
            # For larger ensembles, use full t-distribution
            degrees_of_freedom = n_models - 1
            multiplier = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
            if self.logger:
                self.logger.debug(
                    f"Using t-distribution (n={n_models}, df={degrees_of_freedom}), "
                    f"t_critical={multiplier:.2f}"
                )
        
        # Calculate bounds
        lower = mean_pred - multiplier * std_pred
        upper = mean_pred + multiplier * std_pred
        
        # Ensure bounds are reasonable for price predictions
        # Lower bound cannot be negative
        lower = np.maximum(lower, 0.0)
        
        # For very small predictions, ensure minimum interval width
        min_interval_width = 0.01  # 0.01c minimum
        interval_width = upper - lower
        too_narrow = interval_width < min_interval_width
        if np.any(too_narrow):
            center = (upper + lower) / 2
            lower = np.where(too_narrow, np.maximum(0.0, center - min_interval_width / 2), lower)
            upper = np.where(too_narrow, center + min_interval_width / 2, upper)
        
        # Cap unreasonably large ranges (more than 5x the predicted price suggests model disagreement or error)
        # This prevents extremely wide intervals that aren't useful
        max_range_multiplier = 5.0
        interval_width = upper - lower
        max_reasonable_range = np.maximum(mean_pred * max_range_multiplier, 10.0)  # At least 10c range allowed
        too_wide = interval_width > max_reasonable_range
        if np.any(too_wide):
            # Cap the range while keeping the mean prediction centered
            center = (upper + lower) / 2
            half_range = max_reasonable_range / 2
            lower = np.where(too_wide, np.maximum(0.0, center - half_range), lower)
            upper = np.where(too_wide, center + half_range, upper)
        
        return mean_pred, lower, upper
    
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
        
        # Use in-memory storage to avoid excessive disk I/O
        storage_file = None
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(),
            study_name=study_name,
            load_if_exists=True
        )
        if self.logger:
            self.logger.debug(f"Using in-memory storage for {currency} (reduces disk I/O)")
        
        # Use Optuna workers from environment variable (MAX_OPTUNA_WORKERS)
        # This is already configured and should not be recalculated
        optuna_workers = self.config.max_optuna_workers
        if self.logger:
            self.logger.debug(f"Using {optuna_workers} Optuna workers (from MAX_OPTUNA_WORKERS environment variable)")
            
        study.optimize(objective, n_trials=self.config.n_hyperparameter_trials, n_jobs=optuna_workers)
        
        if self.logger:
            self.logger.debug(f"Optimization completed. Best MAE: {study.best_value:.4f}")
        
        return study.best_params


class EnsembleWeightOptimizer:
    """Optimizer for ensemble weights using Optuna."""
    
    def __init__(
        self,
        config: ModelConfig,
        logger: Optional[MLLogger] = None
    ):
        self.config = config
        self.logger = logger
        
        # Configure Optuna to be less verbose
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def optimize_weights(
        self,
        models: List[BaseModel],
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency: str,
        n_trials: int = 50
    ) -> List[float]:
        """
        Optimize ensemble weights using Optuna.
        
        Args:
            models: List of trained base models
            X_val: Validation features
            y_val: Validation targets
            currency: Currency name for logging
            n_trials: Number of Optuna trials
            
        Returns:
            List of optimized weights (normalized to sum to 1.0)
        """
        if len(models) < 2:
            if self.logger:
                self.logger.warning(f"Need at least 2 models for weight optimization, got {len(models)}. Using equal weights.")
            return [1.0 / len(models)] * len(models)
        
        if len(X_val) < 10:
            if self.logger:
                self.logger.warning(f"Insufficient validation data ({len(X_val)} samples) for weight optimization. Using equal weights.")
            return [1.0 / len(models)] * len(models)
        
        def objective(trial: Any) -> float:
            # Suggest weights for each model
            weights = []
            for i in range(len(models)):
                weight = trial.suggest_float(f'weight_{i}', 0.01, 1.0)
                weights.append(weight)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Get predictions from each model
            predictions = []
            for model in models:
                if model.model is None:
                    raise ValueError("Model must be trained before weight optimization")
                pred = model.predict(X_val)
                predictions.append(pred)
            
            # Calculate weighted ensemble prediction
            weighted_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, normalized_weights):
                weighted_pred += weight * pred
            
            # Calculate MAE as objective
            mae = mean_absolute_error(y_val, weighted_pred)
            return float(mae)
        
        # Create study
        study_name = f"ensemble_weights_{currency}_{self.config.random_state}"
        
        # Initialize storage_file variable for cleanup
        storage_file = None
        
        # Use in-memory storage to avoid excessive disk I/O
        # Weight optimization is fast and doesn't need persistent storage
        storage_file = None
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(),
            study_name=study_name,
            load_if_exists=True
        )
        if self.logger:
            self.logger.debug(f"Using in-memory storage for weight optimization (reduces disk I/O)")
        
        # Use more workers for weight optimization to utilize CPU
        # Use Optuna workers from environment variable (MAX_OPTUNA_WORKERS)
        optuna_workers = self.config.max_optuna_workers
        
        study.optimize(objective, n_trials=n_trials, n_jobs=optuna_workers)
        
        if self.logger:
            self.logger.debug(f"Weight optimization completed. Best MAE: {study.best_value:.4f}")
        
        # Extract optimized weights
        best_weights = []
        for i in range(len(models)):
            weight = study.best_params[f'weight_{i}']
            best_weights.append(weight)
        
        # Normalize to sum to 1.0
        total_weight = sum(best_weights)
        normalized_weights = [w / total_weight for w in best_weights]
        
        if self.logger:
            model_types = [model.get_model_type() for model in models]
            weight_str = ', '.join([f"{mt}: {w:.3f}" for mt, w in zip(model_types, normalized_weights)])
            self.logger.debug(f"Optimized weights for {currency}: {weight_str}")
        
        return normalized_weights


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
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """Train a single model with configured parameters."""
        start_time = time.time()
        
        if self.logger:
            self.logger.debug(f"Training {model_type} model for {currency}")
        
        # Use provided validation data from DynamoDB if available, otherwise use train/test split
        validation_data_source = None
        if X_val is None or y_val is None:
            # No validation data provided - use train/test split from training data
            if self.logger:
                self.logger.warning(
                    f"Validation data not available for {currency}. "
                    "Using train/test split from training data instead."
                )
            
            # Use 80/20 split for train/validation (maintaining temporal order for time series)
            split_idx = int(len(X) * 0.8)
            if split_idx < 10:
                # If we have very few samples, use at least 10 for training
                split_idx = max(10, len(X) - 5) if len(X) > 15 else len(X) - 1
            
            X_train_split = X[:split_idx]
            y_train_split = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            validation_data_source = 'train_test_split'
            
            if self.logger:
                self.logger.debug(f"Using train/test split: {len(X_train_split)} training, {len(X_val)} validation samples")
        else:
            # Use provided validation data from DynamoDB
            X_train_split = X  # All historical data for training
            y_train_split = y
            validation_data_source = 'dynamodb_daily_prices'
            if self.logger:
                self.logger.debug(f"Using DynamoDB validation data: {len(X_val)} samples")
        
        # Scale features if needed
        scaler = None
        if self.processing_config.robust_scaling:
            scaler = RobustScaler()
            X_train_split = scaler.fit_transform(X_train_split)
            if X_val is not None:
                X_val = scaler.transform(X_val)
        
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
        
        # Make predictions and calculate metrics on validation set
        y_pred = model.predict(X_val)
        metrics = ModelMetrics.from_predictions(y_val, y_pred, target_names)
        
        # Get number of validation samples and features
        n_test = len(y_val)
        n_features = X_val.shape[1]
        
        # Get feature importance with feature names
        feature_importance = model.get_feature_importance(feature_names=feature_names)
        
        # Calculate cross-validation scores
        cv_scores = self._calculate_cv_scores(model, X_train_split, y_train_split)
        training_time = time.time() - start_time
        
        # Get hyperparameters
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            hyperparameters = model.model.get_params()
        elif hasattr(model, 'models'):
            hyperparameters = {f"model_{i}": m.model.get_params() if m.model else {} for i, m in enumerate(model.models)}
        else:
            hyperparameters = {}
        
        # Extract imputer from model if it exists
        # For ensemble models, check if any base model has an imputer
        imputer = None
        if hasattr(model, 'imputer') and model.imputer is not None:
            imputer = model.imputer
        elif hasattr(model, 'models'):
            # For ensemble models, extract imputer from first model that has one
            # All models in ensemble should use the same imputer if they need one
            for base_model in model.models:
                if hasattr(base_model, 'imputer') and base_model.imputer is not None:
                    imputer = base_model.imputer
                    break
        
        return TrainingResult(
            model=model,
            scaler=scaler,
            imputer=imputer,
            metrics=metrics,
            training_time=training_time,
            hyperparameters=hyperparameters,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            model_type=model_type,
            currency=currency,
            training_history=training_history,
            training_samples=n_test,
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
                    self.logger.debug(f"Optimizing LightGBM hyperparameters for {currency}")
                
                try:
                    optimized_params = self.hyperparameter_optimizer.optimize(
                        LightGBMModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                    )
                    lgb_model.model = lgb_model._create_model_with_params(optimized_params)
                    if self.logger:
                        self.logger.debug(f"Using optimized LightGBM params for {currency}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"LightGBM optimization failed for {currency}: {e}. Using default parameters.")
                    lgb_model.model = lgb_model._create_model()
                
                models.append(lgb_model)
            
            if self.config.use_random_forest:
                rf_model = RandomForestModel(self.config, self.logger)
                
                # Perform individual optimization for this currency
                if self.logger:
                    self.logger.debug(f"Optimizing RandomForest hyperparameters for {currency}")
                
                try:
                    optimized_params = self.hyperparameter_optimizer.optimize(
                        RandomForestModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                    )
                    rf_model.model = rf_model._create_model_with_params(optimized_params)
                    if self.logger:
                        self.logger.debug(f"Using optimized RandomForest params for {currency}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"RandomForest optimization failed for {currency}: {e}. Using default parameters.")
                    rf_model.model = rf_model._create_model()
                
                models.append(rf_model)  # type: ignore[arg-type]
            
            if self.config.use_extra_trees:
                et_model = ExtraTreesModel(self.config, self.logger)
                
                # Perform individual optimization for this currency
                if self.logger:
                    self.logger.debug(f"Optimizing ExtraTrees hyperparameters for {currency}")
                
                try:
                    optimized_params = self.hyperparameter_optimizer.optimize(
                        ExtraTreesModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                    )
                    et_model.model = et_model._create_model_with_params(optimized_params)
                    if self.logger:
                        self.logger.debug(f"Using optimized ExtraTrees params for {currency}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"ExtraTrees optimization failed for {currency}: {e}. Using default parameters.")
                    et_model.model = et_model._create_model()
                
                models.append(et_model)  # type: ignore[arg-type]
        else:
            # Use default parameters if optimization is disabled
            if self.logger:
                self.logger.warning(f"Hyperparameter optimization disabled for {currency}. Using default parameters.")
            
            if self.config.use_lightgbm:
                lgb_model = LightGBMModel(self.config, self.logger)
                lgb_model.model = lgb_model._create_model()
                models.append(lgb_model)
            
            if self.config.use_random_forest:
                rf_model = RandomForestModel(self.config, self.logger)
                rf_model.model = rf_model._create_model()
                models.append(rf_model)  # type: ignore[arg-type]
            
            if self.config.use_extra_trees:
                et_model = ExtraTreesModel(self.config, self.logger)
                et_model.model = et_model._create_model()
                models.append(et_model)  # type: ignore[arg-type]
        
        # Ensure we have the three required models
        if len(models) == 0:
            raise ValueError("No models were created for ensemble. Ensure use_lightgbm, use_random_forest, and use_extra_trees are enabled.")
        
        if self.logger:
            model_types = [model.get_model_type() for model in models]
            self.logger.debug(f"Training ensemble for {currency} with models: {', '.join(model_types)}")
        
        # Train all models
        for model in models:
            if model.model is None:
                model.model = model._create_model()
            model.fit(X_train, y_train, X_val, y_val)
        
        # Optimize ensemble weights if enabled
        optimized_weights = None
        if self.config.optimize_ensemble_weights and len(models) >= 2:
            if self.logger:
                self.logger.debug(f"Optimizing ensemble weights for {currency}")
            
            try:
                weight_optimizer = EnsembleWeightOptimizer(self.config, self.logger)
                optimized_weights = weight_optimizer.optimize_weights(
                    models, X_val, y_val, currency, 
                    n_trials=self.config.ensemble_weight_optimization_trials
                )
                if self.logger:
                    model_types = [model.get_model_type() for model in models]
                    weight_str = ', '.join([f"{mt}: {w:.3f}" for mt, w in zip(model_types, optimized_weights)])
                    self.logger.debug(f"Optimized ensemble weights for {currency}: {weight_str}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Ensemble weight optimization failed for {currency}: {e}. Using equal weights.")
                optimized_weights = None
        
        # Create ensemble with optimized or equal weights
        ensemble = EnsembleModel(models, weights=optimized_weights, logger=self.logger)  # type: ignore[arg-type]
        
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
            elif model_type == "random_forest":
                params = optimizer.optimize(
                    RandomForestModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                )
                model = RandomForestModel(self.config, self.logger)
                model.model = model._create_model_with_params(params)
            elif model_type == "extra_trees":
                params = optimizer.optimize(
                    ExtraTreesModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                )
                model = ExtraTreesModel(self.config, self.logger)
                model.model = model._create_model_with_params(params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if self.logger:
                self.logger.debug(f"{model_type} optimized params: {params}")
        else:
            # Use default parameters for fast training
            if model_type == "lightgbm":
                model: BaseModel = LightGBMModel(self.config, self.logger)
            elif model_type == "random_forest":
                model = RandomForestModel(self.config, self.logger)
            elif model_type == "extra_trees":
                model = ExtraTreesModel(self.config, self.logger)
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
                        RandomForestModel(self.config, self.logger),
                        ExtraTreesModel(self.config, self.logger)
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
    
    # Save imputer without compression if exists (for RandomForest/ExtraTrees models)
    if result.imputer is not None:
        imputer_path = output_dir / "imputer.pkl"
        joblib.dump(result.imputer, imputer_path, compress=0)
        saved_files['imputer'] = str(imputer_path)
    
    # Save minimal metadata (only what's needed for inference)
    metadata = {
        'model_type': result.model_type,
        'currency': currency,
        'training_timestamp': datetime.now().isoformat(),
        'training_time': result.training_time,
        'metrics': result.metrics.to_dict(),
        'training_samples': result.training_samples,
        'n_features': result.n_features,
        'confidence_level': result.confidence_level,
    }
    
    # Store feature names if available (for exact feature matching during inference)
    if result.feature_names:
        metadata['feature_names'] = result.feature_names
    
    # Store feature importance if available (for analysis)
    if result.feature_importance:
        metadata['feature_importance'] = result.feature_importance
    
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