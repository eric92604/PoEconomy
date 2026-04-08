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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)
import optuna
from optuna.samplers import TPESampler

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

    @staticmethod
    def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric MAPE with small-value guard, returns inf when no valid pairs."""
        epsilon = 1e-8
        values = [
            abs(t - p) / ((abs(t) + abs(p)) / 2 + epsilon) * 100
            for t, p in zip(y_true, y_pred)
            if abs(t) > epsilon
        ]
        return float(np.mean(values)) if values else float('inf')

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray,
                         target_names: Optional[List[str]] = None) -> 'ModelMetrics':
        """Calculate metrics from predictions."""
        if y_true.ndim > 1 and y_true.shape[1] > 1 and target_names is not None:
            return cls._from_multi_output_predictions(y_true, y_pred, target_names)

        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = cls._compute_mape(y_true, y_pred)

        if len(y_true) > 1:
            directional_accuracy = float(
                np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)) * 100
            )
        else:
            directional_accuracy = 0.0

        return cls(mae=mae, rmse=rmse, mape=mape, r2=r2,
                   directional_accuracy=directional_accuracy)

    @classmethod
    def _from_multi_output_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray,
                                       target_names: List[str]) -> 'ModelMetrics':
        """Calculate metrics for multi-output predictions."""
        mae_dict: Dict[str, float] = {}
        rmse_dict: Dict[str, float] = {}
        mape_dict: Dict[str, float] = {}
        r2_dict: Dict[str, float] = {}
        directional_accuracy_dict: Dict[str, float] = {}

        for i, target_name in enumerate(target_names):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]

            valid_mask = ~(np.isnan(y_t) | np.isnan(y_p))
            if not np.any(valid_mask):
                mae_dict[target_name] = float('inf')
                rmse_dict[target_name] = float('inf')
                mape_dict[target_name] = float('inf')
                r2_dict[target_name] = -1.0
                directional_accuracy_dict[target_name] = 0.0
                continue

            y_tv = y_t[valid_mask]
            y_pv = y_p[valid_mask]

            mae_dict[target_name] = mean_absolute_error(y_tv, y_pv)
            rmse_dict[target_name] = float(np.sqrt(mean_squared_error(y_tv, y_pv)))
            r2_dict[target_name] = r2_score(y_tv, y_pv)
            mape_dict[target_name] = cls._compute_mape(y_tv, y_pv)
            directional_accuracy_dict[target_name] = (
                float(np.mean((np.diff(y_tv) > 0) == (np.diff(y_pv) > 0)) * 100)
                if len(y_tv) > 1 else 0.0
            )

        return cls(mae=mae_dict, rmse=rmse_dict, mape=mape_dict,
                   r2=r2_dict, directional_accuracy=directional_accuracy_dict)


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
    model_type: str = ""
    currency: str = ""
    training_history: Optional[Dict[str, Any]] = None  # Training losses per epoch/trial
    training_samples: int = 0  # Number of training samples
    n_features: int = 0  # Number of features
    confidence_level: float = 0.95  # Confidence level for prediction intervals
    feature_names: Optional[List[str]] = None  # Exact feature names used during training


class _MockTrial:
    """
    Wraps a pre-computed parameter dict to satisfy the Optuna Trial interface
    used by each model's ``_create_model`` method.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def _lookup(self, name: str) -> Any:
        if name in self.params:
            return self.params[name]
        for key, value in self.params.items():
            if key.endswith(f"_{name}"):
                return value
        raise ValueError(
            f"Parameter '{name}' not found in optimized parameters. "
            "This indicates hyperparameter optimization failure."
        )

    def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
        return self._lookup(name)

    def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
        return self._lookup(name)

    def suggest_categorical(self, name: str, choices: List[Any], **kwargs) -> Any:
        return self._lookup(name)


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
        return self._create_model(_MockTrial(params))
    
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
                'n_estimators': trial.suggest_int('lgb_n_estimators', 100, self.config.n_model_trials),
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
            'verbose': -1,
            'importance_type': 'gain',
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
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            }
        else:
            params = {
                'n_estimators': self.config.n_model_trials,
                'max_depth': self.config.max_depth,
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
            params = {
                'n_estimators': trial.suggest_int('et_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('et_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', None]),
            }
        else:
            params = {
                'n_estimators': self.config.n_model_trials,
                'max_depth': self.config.max_depth,
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


class CatBoostModel(BaseModel):
    """
    CatBoost gradient boosting model implementation.

    CatBoost with ordered boosting resists overfitting on small tabular datasets,
    making it a better fit than ExtraTrees for per-currency training sets that
    typically have 60–200 samples. Replaces ExtraTrees in the ensemble.
    """

    def _create_model(self, trial: Optional[optuna.Trial] = None):
        """Create CatBoost model with optional Optuna hyperparameter search."""
        from catboost import CatBoostRegressor  # lazy import — optional dependency

        if trial is not None:
            params = {
                'iterations': trial.suggest_int('cb_iterations', 200, 1000),
                'depth': trial.suggest_int('cb_depth', 4, 10),
                'learning_rate': trial.suggest_float('cb_lr', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('cb_l2', 1.0, 10.0),
                'bagging_temperature': trial.suggest_float('cb_bagging_temp', 0.0, 1.0),
                'random_strength': trial.suggest_float('cb_random_strength', 0.0, 1.0),
            }
        else:
            params = {
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 0.5,
                'random_strength': 0.5,
            }

        params.update({
            'cat_features': [],  # all features are numeric; update if categoricals are added
            'random_seed': self.config.random_state,
            'verbose': 0,
            'thread_count': self.config.model_n_jobs,
        })

        return CatBoostRegressor(**params)

    def get_model_type(self) -> str:
        """Get model type identifier."""
        return "catboost"


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


class LeagueWalkForwardSplit:
    """
    Walk-forward cross-validation splits by league boundary.

    For N historical leagues (sorted chronologically), generates up to N-2 folds:
        fold 0: train=league_0,           val=league_1
        fold 1: train=league_0+league_1,  val=league_2
        ...

    The most recent (current) league is excluded from folds — it is reserved for
    early stopping and ensemble weight optimisation. This prevents future-leakage
    into the HP search objective.

    Falls back gracefully: if fewer than min_leagues distinct leagues are found in
    the training DataFrame, is_usable() returns False and TimeSeriesSplit is used.

    Requires a 'league_name' column in the DataFrame and uses the first observed
    date per league to determine chronological order.
    """

    def __init__(self, df: pd.DataFrame, min_leagues: int = 3) -> None:
        self.df = df.reset_index(drop=True)
        self.min_leagues = min_leagues
        if 'league_name' in df.columns and 'date' in df.columns:
            league_first_dates = df.groupby('league_name')['date'].min().sort_values()
            self._leagues: List[str] = league_first_dates.index.tolist()
        else:
            self._leagues = []

    def is_usable(self) -> bool:
        """Return True when enough leagues exist to generate at least one fold."""
        return len(self._leagues) >= self.min_leagues

    def split(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_idx, val_idx) pairs.

        The last league in self._leagues is treated as the current league and is
        excluded from folds. Only historical leagues are used for walk-forward CV.
        """
        historical = self._leagues[:-1]  # exclude current (most recent) league
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(1, len(historical)):
            train_leagues = set(historical[:i])
            val_league = historical[i]
            train_idx = self.df.index[self.df['league_name'].isin(train_leagues)].to_numpy()
            val_idx = self.df.index[self.df['league_name'] == val_league].to_numpy()
            if len(train_idx) >= 10 and len(val_idx) >= 5:
                folds.append((train_idx, val_idx))
        return folds


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
        cv_folds: int = 5,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.

        When train_df is provided and contains enough distinct leagues, the Optuna
        objective uses LeagueWalkForwardSplit for cross-validation instead of the
        single current-league validation set. This prevents HP overfitting to one
        league's price dynamics and generalises better across league resets.
        """
        use_walk_forward = getattr(self.config, 'use_league_walk_forward', True)
        min_wf_leagues = getattr(self.config, 'min_leagues_for_walk_forward', 3)

        # Pre-compute walk-forward folds outside the trial function (expensive).
        wf_folds: List[Tuple[np.ndarray, np.ndarray]] = []
        if use_walk_forward and train_df is not None:
            splitter = LeagueWalkForwardSplit(train_df, min_leagues=min_wf_leagues)
            if splitter.is_usable():
                wf_folds = splitter.split()
                if self.logger and wf_folds:
                    self.logger.debug(
                        f"Using LeagueWalkForwardSplit for {currency}: "
                        f"{len(wf_folds)} fold(s)"
                    )

        def objective(trial: Any) -> float:
            # Create model with trial parameters
            model = model_class(self.config, self.logger)
            model._is_optuna_trial = True  # Mark as Optuna trial to suppress logging
            model.model = model._create_model(trial)

            # Priority 1: League walk-forward CV across historical leagues.
            # Evaluates HP on held-out leagues → better generalisation than single-league val.
            if wf_folds:
                scores = []
                for train_idx, val_idx in wf_folds:
                    X_train_fold = X[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X[val_idx]
                    y_val_fold = y[val_idx]
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    scores.append(float(mean_absolute_error(y_val_fold, y_pred)))
                return float(np.mean(scores))

            # Priority 2: Use current-league validation data directly.
            if X_val is not None and y_val is not None and len(X_val) >= 5:
                model.fit(X, y, X_val, y_val)
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                return float(mae)

            # Priority 3: Fall back to time series cross-validation.
            min_samples_per_fold = 10
            max_folds = max(2, len(X) // min_samples_per_fold)
            n_splits = min(cv_folds, max_folds)

            if n_splits < 2:
                if self.logger:
                    self.logger.warning(
                        f"Insufficient data for cross-validation: {len(X)} samples, "
                        f"need at least {min_samples_per_fold * 2}. "
                        "Using simple train/validation split."
                    )
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
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                mae = mean_absolute_error(y_val_fold, y_pred)
                scores.append(mae)

            return float(np.mean(scores))
        
        # Make study name unique per currency and model type to avoid conflicts
        study_name = f"{model_class.__name__}_{currency}_{self.config.random_state}"
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
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
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
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
        y_val: Optional[np.ndarray] = None,
        train_df: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """Train a single model with configured parameters."""
        start_time = time.time()
        
        if self.logger:
            self.logger.debug(f"Training {model_type} model for {currency}")
        
        if X_val is None or y_val is None:
            # No validation data provided - use train/test split from training data
            if self.logger:
                self.logger.warning(
                    f"Validation data not available for {currency}. "
                    "Using train/test split from training data instead."
                )

            # Use 60/20/20 split: train / early-stop val / weight-opt val
            # Temporal order is preserved so later slices are always more recent
            train_end = int(len(X) * 0.6)
            val_end = int(len(X) * 0.8)
            if train_end < 10:
                # Fall back to 80/20 with no weight-opt split for very small datasets
                train_end = max(10, len(X) - 5) if len(X) > 15 else len(X) - 1
                val_end = len(X)

            X_train_split = X[:train_end]
            y_train_split = y[:train_end]
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            X_weight_opt = X[val_end:]
            y_weight_opt = y[val_end:]

            if self.logger:
                self.logger.debug(
                    f"Using 60/20/20 temporal split: {len(X_train_split)} train, "
                    f"{len(X_val)} val, {len(X_weight_opt)} weight-opt samples"
                )
        else:
            # Use provided validation data from DynamoDB
            X_train_split = X  # All historical data for training
            y_train_split = y
            # Split DynamoDB val data 50/50 into early-stop val and weight-opt val
            val_mid = len(X_val) // 2
            X_weight_opt = X_val[val_mid:]
            y_weight_opt = y_val[val_mid:]
            X_val = X_val[:val_mid]
            y_val = y_val[:val_mid]
            if self.logger:
                self.logger.debug(
                    f"Using DynamoDB validation data: {len(X_val)} early-stop val, "
                    f"{len(X_weight_opt)} weight-opt samples"
                )
        
        # Scale features if needed — scaler is fit on train only to prevent leakage.
        # All downstream splits (val and weight-opt) must use the same transform.
        scaler = None
        if self.processing_config.robust_scaling:
            scaler = RobustScaler()
            X_train_split = scaler.fit_transform(X_train_split)
            if X_val is not None:
                X_val = scaler.transform(X_val)
            if X_weight_opt is not None and len(X_weight_opt) > 0:
                X_weight_opt = scaler.transform(X_weight_opt)
        
        # Train model (validation data is always available now)
        training_history = None
        if model_type == "ensemble":
            model: Any = self._train_ensemble_model(X_train_split, y_train_split, X_val, y_val, currency, X_weight_opt, y_weight_opt, train_df=train_df)
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
        
        # Evaluate on the held-out weight-opt split so metrics are not contaminated
        # by the data used for Optuna hyperparameter selection and LightGBM early
        # stopping.  Fall back to X_val only when the weight-opt split is empty
        # (very small datasets that hit the 60/20/20 fallback path).
        eval_X = X_weight_opt if len(X_weight_opt) > 0 else X_val
        eval_y = y_weight_opt if len(X_weight_opt) > 0 else y_val
        y_pred = model.predict(eval_X)
        metrics = ModelMetrics.from_predictions(eval_y, y_pred, target_names)

        n_train = len(y_train_split)
        n_features = X_train_split.shape[1]

        feature_importance = model.get_feature_importance(feature_names=feature_names)
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
            model_type=model_type,
            currency=currency,
            training_history=training_history,
            training_samples=n_train,
            n_features=n_features,
            confidence_level=0.95,
            feature_names=feature_names,
        )
    
    def _build_base_model(
        self,
        model_class: type,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency: str,
        train_df: Optional[pd.DataFrame] = None,
    ) -> BaseModel:
        """
        Instantiate *model_class* and, when optimisation is enabled, tune its
        hyperparameters.  Falls back to default parameters on any error.

        train_df: Optional full training DataFrame with league_name and date
            columns used by LeagueWalkForwardSplit in the Optuna objective.
        """
        model = model_class(self.config, self.logger)
        if self.config.n_hyperparameter_trials > 1:
            if self.logger:
                self.logger.debug(
                    f"Optimizing {model_class.__name__} hyperparameters for {currency}"
                )
            try:
                params = self.hyperparameter_optimizer.optimize(
                    model_class, X_train, y_train, X_val, y_val,
                    currency, self.config.cv_folds,
                    train_df=train_df,
                )
                model.model = model._create_model_with_params(params)
                if self.logger:
                    self.logger.debug(
                        f"Using optimized {model_class.__name__} params for {currency}"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"{model_class.__name__} optimisation failed for {currency}: "
                        f"{e}. Using default parameters."
                    )
                model.model = model._create_model()
        else:
            model.model = model._create_model()
        return model

    def _train_ensemble_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        currency: str,
        X_weight_opt: Optional[np.ndarray] = None,
        y_weight_opt: Optional[np.ndarray] = None,
        train_df: Optional[pd.DataFrame] = None,
    ) -> EnsembleModel:
        """Train ensemble model using per-currency hyperparameter optimization."""
        if self.config.n_hyperparameter_trials <= 1 and self.logger:
            self.logger.warning(
                f"Hyperparameter optimization disabled for {currency}. Using default parameters."
            )

        enabled_classes = [
            (self.config.use_lightgbm, LightGBMModel),
            (self.config.use_random_forest, RandomForestModel),
            (getattr(self.config, 'use_catboost', False), CatBoostModel),
            (self.config.use_extra_trees, ExtraTreesModel),
        ]
        models: List[BaseModel] = [
            self._build_base_model(cls, X_train, y_train, X_val, y_val, currency, train_df=train_df)
            for enabled, cls in enabled_classes
            if enabled
        ]

        # Ensure we have at least one model
        if len(models) == 0:
            raise ValueError(
                "No models were created for ensemble. "
                "Enable at least one of: use_lightgbm, use_random_forest, use_catboost."
            )
        
        if self.logger:
            model_types = [model.get_model_type() for model in models]
            self.logger.debug(f"Training ensemble for {currency} with models: {', '.join(model_types)}")
        
        # Train all models
        for model in models:
            if model.model is None:
                model.model = model._create_model()
            model.fit(X_train, y_train, X_val, y_val)
        
        # Optimize ensemble weights if enabled, using the held-out weight-opt split
        optimized_weights = None
        if self.config.optimize_ensemble_weights and len(models) >= 2:
            X_wo = X_weight_opt if X_weight_opt is not None and len(X_weight_opt) > 0 else X_val
            y_wo = y_weight_opt if y_weight_opt is not None and len(y_weight_opt) > 0 else y_val
            if self.logger:
                self.logger.debug(
                    f"Optimizing ensemble weights for {currency} on "
                    f"{'weight-opt' if X_weight_opt is not None and len(X_weight_opt) > 0 else 'val (fallback)'} set "
                    f"({len(X_wo)} samples)"
                )

            try:
                weight_optimizer = EnsembleWeightOptimizer(self.config, self.logger)
                optimized_weights = weight_optimizer.optimize_weights(
                    models, X_wo, y_wo, currency,
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
            elif model_type == "catboost":
                params = optimizer.optimize(
                    CatBoostModel, X_train, y_train, X_val, y_val, currency, self.config.cv_folds
                )
                model = CatBoostModel(self.config, self.logger)
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
            elif model_type == "catboost":
                model = CatBoostModel(self.config, self.logger)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.model = model._create_model()
        
        model.fit(X_train, y_train, X_val, y_val)
        return model
    


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