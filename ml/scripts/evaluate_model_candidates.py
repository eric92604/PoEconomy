#!/usr/bin/env python3
"""
Model Candidate Evaluation Script

Evaluates 20 diverse model candidates across different algorithm families
for currency price time-series forecasting. Uses proper temporal cross-validation
to ensure fair comparison.
"""

import sys
import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Model imports
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Optional imports with fallbacks
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
script_dir = Path(__file__).parent
ml_dir = script_dir.parent
project_root = ml_dir.parent
sys.path.insert(0, str(ml_dir))
sys.path.insert(0, str(project_root))

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

from ml.config.training_config import MLConfig, get_default_config
from ml.utils.common_utils import setup_ml_logging, MLLogger


@dataclass
class ModelResult:
    """Results for a single model evaluation."""
    model_name: str
    model_family: str
    horizon: str
    mae: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float
    training_time: float
    n_samples: int
    n_features: int
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BaseModelWrapper:
    """Base wrapper for all models with standardized interface."""
    
    def __init__(self, name: str, family: str):
        self.name = name
        self.family = family
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit model. Most models don't support eval_set, so it's ignored here.
        Models that support early stopping (LightGBM, XGBoost, CatBoost) override this method."""
        start_time = time.time()
        self.model.fit(X, y)
        return time.time() - start_time
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X).reshape(-1, 1) if self.model else np.array([])
    
    def create_model(self) -> Any:
        """Create the underlying model. Override in subclasses."""
        raise NotImplementedError


# Gradient Boosting Family
class LightGBMWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("LightGBM", "Gradient Boosting")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        # Priority: if trial is provided (during optimization), use it first
        # Otherwise use optimized_params if available, otherwise defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            # This takes priority over optimized_params to allow re-optimization
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42,
                'n_jobs': 2,
                'verbose': -1
            }
            # Don't store params during optimization - Optuna will track best params
            self.model = lgb.LGBMRegressor(**params)
        elif self.optimized_params:
            # Use pre-optimized parameters (from previous optimization run)
            params = self.optimized_params.copy()
            params.update({
                'random_state': 42,
                'n_jobs': 2,
                'verbose': -1
            })
            self.model = lgb.LGBMRegressor(**params)
        else:
            # Default parameters with early stopping support
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=100,
                random_state=42,
                n_jobs=2,
                verbose=-1
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit LightGBM with early stopping if validation set provided."""
        start_time = time.time()
        if X_val is not None and y_val is not None:
            # Use early stopping with validation set
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                verbose=False
            )
        else:
            self.model.fit(X, y, verbose=False)
        return time.time() - start_time


class XGBoostWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("XGBoost", "Gradient Boosting")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        # Priority: trial first (during optimization), then optimized_params, then defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42,
                'n_jobs': 2,
                'verbosity': 0,
                'early_stopping_rounds': 50  # Enable early stopping for optimization
            }
            # Don't store params during optimization - Optuna will track best params
            self.model = xgb.XGBRegressor(**params)
        elif self.optimized_params:
            # Use pre-optimized parameters
            params = self.optimized_params.copy()
            params.update({
                'random_state': 42,
                'n_jobs': 2,
                'verbosity': 0,
                'early_stopping_rounds': 50
            })
            self.model = xgb.XGBRegressor(**params)
        else:
            # Default parameters with early stopping support
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=2,
                verbosity=0,
                early_stopping_rounds=50
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit XGBoost with early stopping if validation set provided."""
        start_time = time.time()
        if X_val is not None and y_val is not None:
            # Use early stopping with validation set
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y, verbose=False)
        return time.time() - start_time


class CatBoostWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("CatBoost", "Gradient Boosting")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        # Priority: trial first (during optimization), then optimized_params, then defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50  # Enable early stopping for optimization
            }
            # Don't store params during optimization - Optuna will track best params
            self.model = cb.CatBoostRegressor(**params)
        elif self.optimized_params:
            # Use pre-optimized parameters
            params = self.optimized_params.copy()
            params.update({
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50
            })
            self.model = cb.CatBoostRegressor(**params)
        else:
            # Default parameters with early stopping support
            self.model = cb.CatBoostRegressor(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=50
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit CatBoost with early stopping if validation set provided."""
        start_time = time.time()
        if X_val is not None and y_val is not None:
            # Use early stopping with validation set
            self.model.fit(
                X, y,
                eval_set=(X_val, y_val),
                verbose=False
            )
        else:
            self.model.fit(X, y, verbose=False)
        return time.time() - start_time


class HistGradientBoostingWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("HistGradientBoosting", "Gradient Boosting")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        # Priority: trial first (during optimization), then optimized_params, then defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            params = {
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
                'random_state': 42
            }
            # Don't store params during optimization - Optuna will track best params
            self.model = HistGradientBoostingRegressor(**params)
        elif self.optimized_params:
            # Use pre-optimized parameters
            params = self.optimized_params.copy()
            params.update({'random_state': 42})
            self.model = HistGradientBoostingRegressor(**params)
        else:
            # Default parameters with early stopping support
            self.model = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50
            )


# Neural Network Family
class MLPWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("MLP", "Neural Network")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        # Priority: trial first (during optimization), then optimized_params, then defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            params = {
                'hidden_layer_sizes': (trial.suggest_int('hidden_size1', 50, 200), trial.suggest_int('hidden_size2', 25, 100)),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
                'max_iter': 200,
                'solver': 'adam',
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
            # Don't store params during optimization - Optuna will track best params
            self.model = MLPRegressor(**params)
        elif self.optimized_params:
            # Use pre-optimized parameters
            params = self.optimized_params.copy()
            
            # Optuna stores trial parameter names (hidden_size1, hidden_size2) not the constructed parameter
            # Convert hidden_size1/hidden_size2 to hidden_layer_sizes tuple if needed
            if 'hidden_size1' in params and 'hidden_size2' in params:
                params['hidden_layer_sizes'] = (params.pop('hidden_size1'), params.pop('hidden_size2'))
            # If hidden_layer_sizes is already present, ensure it's a tuple
            elif 'hidden_layer_sizes' in params:
                if isinstance(params['hidden_layer_sizes'], list):
                    params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
                elif not isinstance(params['hidden_layer_sizes'], tuple):
                    # If it's neither list nor tuple, try to convert (shouldn't happen but be safe)
                    try:
                        params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
                    except:
                        # Fallback to default if conversion fails
                        params['hidden_layer_sizes'] = (100, 50)
            
            # Remove any remaining individual size parameters that might cause issues
            params.pop('hidden_size1', None)
            params.pop('hidden_size2', None)
            
            params.update({
                'solver': 'adam',
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            })
            self.model = MLPRegressor(**params)
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )


class LSTMWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("LSTM", "Neural Network")
        self.scaler = None
        self.y_scaler = None
        self.device = None
        self.input_size = None
    
    def create_model(self, input_size: Optional[int] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        if input_size is None:
            input_size = self.input_size if self.input_size else 1
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit LSTM model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        start_time = time.time()
        
        # Store input size for model creation
        self.input_size = X.shape[1]
        
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Reshape for LSTM: (samples, timesteps, features)
        # For simplicity, use sequence length of 1 (each sample is independent)
        X_seq = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
        y_seq = torch.FloatTensor(y_scaled).to(self.device)
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(50):  # Back to 50 epochs
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X).squeeze()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        
        return time.time() - start_time
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with LSTM."""
        if self.scaler is None:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        X_seq = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_seq).cpu().numpy()
        
        # Inverse transform predictions
        if self.y_scaler is not None:
            predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions.reshape(-1, 1)


class GRUWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("GRU", "Neural Network")
        self.scaler = None
        self.y_scaler = None
        self.device = None
        self.input_size = None
    
    def create_model(self, input_size: Optional[int] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        if input_size is None:
            input_size = self.input_size if self.input_size else 1
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GRUModel(input_size=input_size, hidden_size=64, num_layers=2).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit GRU model (similar to LSTM)."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        start_time = time.time()
        
        # Store input size for model creation
        self.input_size = X.shape[1]
        
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Reshape for GRU: (samples, timesteps, features)
        X_seq = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
        y_seq = torch.FloatTensor(y_scaled).to(self.device)
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(50):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X).squeeze()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        
        return time.time() - start_time
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with GRU."""
        if self.scaler is None:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        X_seq = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_seq).cpu().numpy()
        
        # Inverse transform predictions
        if self.y_scaler is not None:
            predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions.reshape(-1, 1)


# PyTorch model definitions
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    class GRUModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])
    
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


# Tree-Based Ensemble Family
class RandomForestWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("Random Forest", "Tree Ensemble")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        # Priority: trial first (during optimization), then optimized_params, then defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': 2
            }
            # Don't store params during optimization - Optuna will track best params
            self.model = RandomForestRegressor(**params)
        elif self.optimized_params:
            # Use pre-optimized parameters
            params = self.optimized_params.copy()
            params.update({
                'random_state': 42,
                'n_jobs': 2
            })
            self.model = RandomForestRegressor(**params)
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=2
            )


class ExtraTreesWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("Extra Trees", "Tree Ensemble")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        # Priority: trial first (during optimization), then optimized_params, then defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': 2
            }
            # Don't store params during optimization - Optuna will track best params
            self.model = ExtraTreesRegressor(**params)
        elif self.optimized_params:
            # Use pre-optimized parameters
            params = self.optimized_params.copy()
            params.update({
                'random_state': 42,
                'n_jobs': 2
            })
            self.model = ExtraTreesRegressor(**params)
        else:
            self.model = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=2
            )


class BaggingWrapper(BaseModelWrapper):
    def __init__(self, use_optimization: bool = False):
        super().__init__("Bagging Regressor", "Tree Ensemble")
        self.use_optimization = use_optimization
        self.optimized_params = None
    
    def create_model(self, trial=None):
        from sklearn.tree import DecisionTreeRegressor
        # Priority: trial first (during optimization), then optimized_params, then defaults
        if trial is not None:
            # During optimization: use trial to suggest parameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
            }
            # Don't store params during optimization - Optuna will track best params
            base_estimator = DecisionTreeRegressor(max_depth=params.get('max_depth', 10), random_state=42)
            self.model = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=params.get('n_estimators', 100),
                random_state=42,
                n_jobs=2
            )
        elif self.optimized_params:
            # Use pre-optimized parameters
            params = self.optimized_params.copy()
            base_estimator = DecisionTreeRegressor(max_depth=params.get('max_depth', 10), random_state=42)
            self.model = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=params.get('n_estimators', 100),
                random_state=42,
                n_jobs=2
            )
        else:
            base_estimator = DecisionTreeRegressor(max_depth=10, random_state=42)
            self.model = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=100,
                random_state=42,
                n_jobs=2
            )


# Linear Model Family
class RidgeWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("Ridge Regression", "Linear")
    
    def create_model(self):
        self.model = Ridge(alpha=1.0, random_state=42)


class ElasticNetWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("ElasticNet", "Linear")
    
    def create_model(self):
        self.model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=1000)


class HuberWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("Huber Regressor", "Linear")
    
    def create_model(self):
        self.model = HuberRegressor(epsilon=1.35, max_iter=200)


# Support Vector Family
class SVRWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("SVR (RBF)", "Support Vector")
    
    def create_model(self):
        self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)


class NuSVRWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("NuSVR", "Support Vector")
    
    def create_model(self):
        self.model = NuSVR(kernel='rbf', C=1.0, nu=0.5)


# Nearest Neighbor Family
class KNNWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("KNN Regressor", "Nearest Neighbor")
    
    def create_model(self):
        self.model = KNeighborsRegressor(n_neighbors=5, weights='distance')


class RadiusNeighborsWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("Radius Neighbors", "Nearest Neighbor")
        self.imputer = None
    
    def create_model(self):
        self.model = RadiusNeighborsRegressor(radius=1.0, weights='distance')
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit Radius Neighbors model with NaN handling."""
        start_time = time.time()
        # Handle NaN/inf values - always use imputer to ensure no NaNs
        from sklearn.impute import SimpleImputer
        # Replace inf with nan first, then impute
        X_clean = np.where(np.isfinite(X), X, np.nan)
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X_clean)
        # Final safety check - ensure no NaN or inf remain
        if np.isnan(X_imputed).any() or np.isinf(X_imputed).any():
            X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
        self.model.fit(X_imputed, y.flatten())
        return time.time() - start_time
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with NaN handling."""
        if self.imputer is not None:
            # Replace inf with nan first, then impute
            X_clean = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
            X_imputed = self.imputer.transform(X_clean)
            # Final safety check
            X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            X_imputed = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.model.predict(X_imputed).reshape(-1, 1) if self.model else np.array([])


# Specialized Time-Series Family
class ProphetWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("Prophet", "Time-Series")
        self.df_prophet = None
        self.dates = None
    
    def create_model(self):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
    
    def set_dates(self, dates: Optional[pd.Series]):
        """Set dates for Prophet model."""
        self.dates = dates
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")
        
        start_time = time.time()
        
        # Prophet requires a DataFrame with 'ds' (date) and 'y' (target) columns
        dates = self.dates
        if dates is None:
            # Create dummy dates if not provided
            dates = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
        
        # Remove timezone if present (Prophet doesn't support timezones)
        if isinstance(dates, pd.Series):
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
        elif hasattr(dates, 'tz') and dates.tz is not None:
            dates = dates.tz_localize(None)
        
        self.df_prophet = pd.DataFrame({
            'ds': dates,
            'y': y.flatten()
        })
        
        # Add features as regressors (first 5 features to avoid overfitting)
        for i in range(min(5, X.shape[1])):
            self.df_prophet[f'feature_{i}'] = X[:, i]
            self.model.add_regressor(f'feature_{i}')
        
        self.model.fit(self.df_prophet)
        return time.time() - start_time
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with Prophet."""
        dates = self.dates
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(X), freq='D')
        
        # Remove timezone if present (Prophet doesn't support timezones)
        if isinstance(dates, pd.Series):
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
        elif hasattr(dates, 'tz') and dates.tz is not None:
            dates = dates.tz_localize(None)
        
        future_df = pd.DataFrame({'ds': dates})
        for i in range(min(5, X.shape[1])):
            future_df[f'feature_{i}'] = X[:, i]
        
        forecast = self.model.predict(future_df)
        return forecast['yhat'].values.reshape(-1, 1)


class ARIMAWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("ARIMA", "Time-Series")
    
    def create_model(self):
        # Use simple ARIMA-like approach with sklearn
        # For full ARIMA, would need statsmodels, but keeping it simple
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> float:
        """Fit ARIMA-like model."""
        # Simple autoregressive approach: use lagged values as features
        # This is a simplified version - full ARIMA would use statsmodels
        start_time = time.time()
        self.model.fit(X, y.flatten())
        return time.time() - start_time


class ModelCandidateEvaluator:
    """Evaluates multiple model candidates on time-series data."""
    
    def __init__(self, config: Optional[MLConfig] = None, logger: Optional[MLLogger] = None, use_optimization: bool = False, n_trials: int = 50):
        self.config = config or get_default_config()
        self.logger = logger or MLLogger("ModelEvaluator")
        self.use_optimization = use_optimization and OPTUNA_AVAILABLE
        self.n_trials = n_trials
        
        # Initialize all model wrappers
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> List[BaseModelWrapper]:
        """Initialize all model candidates."""
        models = [
            # Gradient Boosting
            LightGBMWrapper(use_optimization=self.use_optimization),
            XGBoostWrapper(use_optimization=self.use_optimization),
            HistGradientBoostingWrapper(use_optimization=self.use_optimization),
            # Tree Ensembles
            RandomForestWrapper(use_optimization=self.use_optimization),
            ExtraTreesWrapper(use_optimization=self.use_optimization),
            BaggingWrapper(use_optimization=self.use_optimization),
            # Linear
            RidgeWrapper(),
            ElasticNetWrapper(),
            HuberWrapper(),
            # Support Vector
            SVRWrapper(),
            NuSVRWrapper(),
            # Nearest Neighbor
            KNNWrapper(),
            RadiusNeighborsWrapper(),
            # Neural Network
            MLPWrapper(use_optimization=self.use_optimization),
            # Time-Series
            ARIMAWrapper(),
        ]
        
        # Add optional models if available
        if CATBOOST_AVAILABLE:
            models.insert(2, CatBoostWrapper(use_optimization=self.use_optimization))  # Insert after XGBoost
        
        if TORCH_AVAILABLE:
            models.append(LSTMWrapper())
            models.append(GRUWrapper())
        
        if PROPHET_AVAILABLE:
            models.append(ProphetWrapper())
        
        return models
    
    def load_data_with_currency(self, data_path: Optional[str] = None, currency: Optional[str] = None, min_avg_value: float = 1000.0) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str], pd.Series]:
        """
        Load and prepare data for evaluation, preserving currency information.
        
        Returns:
            df: DataFrame with currency column preserved
            targets: Dictionary of target arrays by horizon
            feature_names: List of feature names
            dates: Date series for time-series models
        """
        if data_path:
            # Load from local file
            self.logger.info(f"Loading data from {data_path}")
            df = pd.read_parquet(data_path)
        else:
            # Try to load from S3 or default location
            from ml.utils.data_sources import create_s3_data_source
            import os
            
            data_lake_bucket = os.getenv('DATA_LAKE_BUCKET', '')
            s3_config = {
                'data_lake_bucket': data_lake_bucket,
                'processed_data_prefix': 'processed_data/'
            }
            data_source = create_s3_data_source(s3_config, self.logger)
            
            processed_data, _ = data_source.load_processed_parquet_data_with_experiment_id(
                data_lake_bucket=data_lake_bucket,
                experiment_id=None
            )
            
            if processed_data is None:
                raise ValueError("Could not load processed data. Provide --data-path or ensure S3 is configured.")
            
            df = processed_data
        
        # Filter by currency value (high-value currencies)
        if 'currency' in df.columns and 'price' in df.columns:
            # Calculate average price per currency
            currency_avg_prices = df.groupby('currency')['price'].mean()
            high_value_currencies = currency_avg_prices[currency_avg_prices >= min_avg_value].index.tolist()
            
            if high_value_currencies:
                df = df[df['currency'].isin(high_value_currencies)].copy()
                self.logger.info(
                    f"Filtered to high-value currencies (>= {min_avg_value}c): "
                    f"{len(high_value_currencies)} currencies, {len(df)} records"
                )
                self.logger.info(f"Currencies: {', '.join(high_value_currencies[:10])}{'...' if len(high_value_currencies) > 10 else ''}")
            else:
                self.logger.warning(f"No currencies found with avg price >= {min_avg_value}c")
        
        # Filter by specific currency if specified (overrides value filter)
        if currency and 'currency' in df.columns:
            df = df[df['currency'] == currency].copy()
            self.logger.info(f"Filtered to currency: {currency} ({len(df)} records)")
        
        # Get feature columns (exclude targets and metadata)
        exclude_patterns = [
            'target_', 'date', 'league_name', 'currency', 'id', 'league_start', 
            'league_end', 'league_active', 'get_currency', 'pay_currency', 
            '_multi_output_targets', 'timestamp'
        ]
        feature_cols = [
            col for col in df.columns 
            if not any(pattern in col for pattern in exclude_patterns)
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if not feature_cols:
            raise ValueError("No feature columns found in data")
        
        # Handle missing values in feature columns only
        imputer = SimpleImputer(strategy='median')
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        
        # Get targets - store full arrays, we'll filter per horizon during evaluation
        # Apply log transformation to handle extreme skewness and large price ranges
        targets = {}
        for horizon in ['1d', '3d', '7d']:
            target_col = f'target_price_{horizon}'
            if target_col in df.columns:
                target_values = df[target_col].values
                # Log transform to handle extreme outliers and improve model performance
                # Use log1p(x) = log(1+x) to handle zeros and small values better
                epsilon = 1e-6
                targets[horizon] = np.log1p(target_values + epsilon)
                
                # Log transformation statistics
                if self.logger:
                    valid_targets = target_values[~pd.isna(target_values)]
                    if len(valid_targets) > 0:
                        self.logger.info(
                            f"Target {horizon}: original range [{valid_targets.min():.2f}, {valid_targets.max():.2f}], "
                            f"log-transformed range [{targets[horizon][~pd.isna(targets[horizon])].min():.2f}, "
                            f"{targets[horizon][~pd.isna(targets[horizon])].max():.2f}], "
                            f"skewness: {pd.Series(valid_targets).skew():.2f}"
                        )
            else:
                self.logger.warning(f"Target column {target_col} not found")
        
        if not targets:
            raise ValueError("No target columns found")
        
        # Get dates if available (before filtering)
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            # Remove timezone for Prophet compatibility
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
        else:
            dates = pd.Series(pd.date_range(start='2020-01-01', periods=len(df), freq='D'))
        
        self.logger.info(f"Loaded data: {len(df)} samples, {len(feature_cols)} features")
        self.logger.info(f"Target horizons: {list(targets.keys())}")
        if 'currency' in df.columns:
            self.logger.info(f"Currencies: {df['currency'].nunique()} unique currencies")
        
        return df, targets, feature_cols, dates
    
    def _optimize_hyperparameters(
        self,
        model_wrapper: BaseModelWrapper,
        X: np.ndarray,
        y: np.ndarray,
        horizon: str,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            return {}
        
        # Use a subset of data for faster optimization
        # Use first 80% for training, last 20% for validation
        split_idx = int(len(X) * 0.8)
        X_train_opt = X[:split_idx]
        X_val_opt = X[split_idx:]
        y_train_opt = y[:split_idx]
        y_val_opt = y[split_idx:]
        
        def objective(trial):
            try:
                # Create model with trial parameters
                # Special handling for LSTM/GRU - need input_size
                if isinstance(model_wrapper, (LSTMWrapper, GRUWrapper)):
                    if not hasattr(model_wrapper, 'input_size') or model_wrapper.input_size is None:
                        model_wrapper.input_size = X_train_opt.shape[1]
                    model_wrapper.create_model(input_size=X_train_opt.shape[1])
                else:
                    model_wrapper.create_model(trial=trial)
                
                # Train model
                # Production-level training: Use validation set for early stopping
                split_opt = int(len(X_train_opt) * 0.8)
                X_train_opt_split = X_train_opt[:split_opt]
                X_val_opt_split = X_train_opt[split_opt:]
                y_train_opt_split = y_train_opt[:split_opt]
                y_val_opt_split = y_train_opt[split_opt:]
                model_wrapper.fit(X_train_opt_split, y_train_opt_split, X_val_opt_split, y_val_opt_split)
                
                # Predict and evaluate (transform back to original scale)
                y_pred_log = model_wrapper.predict(X_val_opt)
                
                # Clip log predictions to prevent overflow (same logic as main evaluation)
                max_log_value = 20.0
                min_log_value = -10.0  # More realistic than -50
                y_pred_log_flat = y_pred_log.flatten()
                y_val_opt_log_flat = y_val_opt.flatten()
                
                y_pred_log_clipped = np.clip(y_pred_log_flat, min_log_value, max_log_value)
                # Don't clip true values - they should be in valid range
                y_val_opt_log_clipped = y_val_opt_log_flat
                
                epsilon = 1e-6
                y_pred = np.expm1(y_pred_log_clipped) - epsilon
                y_val_opt_original = np.expm1(y_val_opt_log_clipped) - epsilon
                
                # Clip extreme predictions only (not true values)
                max_reasonable_price = 1e7
                y_pred = np.clip(y_pred, 0, max_reasonable_price)
                # Don't clip true values
                
                mae = mean_absolute_error(y_val_opt_original, y_pred)
                
                return mae
            except Exception as e:
                # Log error but don't fail the trial - let Optuna handle it
                self.logger.debug(f"  Trial {trial.number} failed: {str(e)}")
                raise optuna.TrialPruned()  # Prune failed trials
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except Exception as e:
            self.logger.warning(f"  Optimization failed: {str(e)}")
            return {}
        
        # Check if we have completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            self.logger.warning(f"  No completed trials in optimization")
            return {}
        
        self.logger.info(f"  Best MAE: {study.best_value:.4f}")
        self.logger.info(f"  Completed trials: {len(completed_trials)}/{len(study.trials)}")
        
        # Get best parameters - try multiple methods
        best_params = {}
        try:
            # Method 1: Use study.best_params (preferred)
            best_params = study.best_params
            if best_params:
                self.logger.info(f"  Best parameters: {best_params}")
            else:
                # Method 2: Get from best_trial directly
                if study.best_trial and hasattr(study.best_trial, 'params'):
                    best_params = study.best_trial.params
                    if best_params:
                        self.logger.info(f"  Using params from best_trial: {best_params}")
                    else:
                        self.logger.warning(f"  best_trial.params is also empty")
                else:
                    self.logger.warning(f"  No best_trial available")
        except ValueError as e:
            # Handle case where no trials completed
            if "No trials are completed" in str(e):
                self.logger.warning(f"  No completed trials: {str(e)}")
                best_params = {}
            else:
                self.logger.warning(f"  ValueError getting best_params: {str(e)}")
                # Try to get from best_trial directly
                try:
                    if study.best_trial:
                        best_params = study.best_trial.params
                        self.logger.info(f"  Using params from best_trial (fallback): {best_params}")
                except:
                    best_params = {}
        except Exception as e:
            self.logger.warning(f"  Unexpected error getting best_params: {str(e)}")
            # Try to get from best_trial directly
            try:
                if study.best_trial:
                    best_params = study.best_trial.params
                    self.logger.info(f"  Using params from best_trial (fallback): {best_params}")
            except:
                best_params = {}
        
        if not best_params:
            self.logger.warning(f"  Could not retrieve best parameters despite best_value={study.best_value:.4f}")
            # Log trial states for debugging
            trial_states = {}
            for trial in study.trials:
                state = str(trial.state)
                trial_states[state] = trial_states.get(state, 0) + 1
            self.logger.warning(f"  Trial states: {trial_states}")
            
            # Try to get params from any completed trial
            for trial in completed_trials:
                if trial.params:
                    self.logger.warning(f"  Found params in trial {trial.number}: {trial.params}")
                    best_params = trial.params
                    break
            
            # If still empty, check best_trial more carefully
            if not best_params and study.best_trial:
                self.logger.warning(f"  best_trial number: {study.best_trial.number}")
                self.logger.warning(f"  best_trial state: {study.best_trial.state}")
                self.logger.warning(f"  best_trial has params attr: {hasattr(study.best_trial, 'params')}")
                if hasattr(study.best_trial, 'params'):
                    self.logger.warning(f"  best_trial.params type: {type(study.best_trial.params)}")
                    self.logger.warning(f"  best_trial.params value: {study.best_trial.params}")
        
        return best_params
    
    def evaluate_model(
        self,
        model_wrapper: BaseModelWrapper,
        X: np.ndarray,
        y: np.ndarray,
        horizon: str,
        cv_folds: int = 5,
        dates: Optional[pd.Series] = None
    ) -> ModelResult:
        """Evaluate a single model using time-series cross-validation."""
        model_name = model_wrapper.name
        model_family = model_wrapper.family
        
        try:
            # CRITICAL FIX: Do NOT fit scaler on all data - this causes data leakage!
            # Only do basic imputation on raw data, then fit scaler per fold on training data only
            from sklearn.impute import SimpleImputer
            
            # Basic imputation on raw data (median imputation doesn't leak future info)
            # But we'll refit per fold to be extra safe
            X_imputed = X.copy()
            X_imputed = np.nan_to_num(X_imputed, nan=np.nan, posinf=np.nan, neginf=np.nan)
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            mae_scores = []
            rmse_scores = []
            mape_scores = []
            r2_scores = []
            directional_accuracies = []
            training_times = []
            
            # Hyperparameter optimization if enabled
            # NOTE: Optimization should ideally be done per fold, but for efficiency we do it once
            # on a time-series split. This is acceptable as long as we use proper time-series splits.
            if self.use_optimization and hasattr(model_wrapper, 'use_optimization') and model_wrapper.use_optimization:
                # Clear any previous optimized_params to allow fresh optimization
                model_wrapper.optimized_params = None
                
                self.logger.info(f"  Optimizing hyperparameters for {model_name} ({self.n_trials} trials)...")
                # Use first CV fold's training data for optimization to avoid data leakage
                # Get the first fold's training indices
                train_idx_opt, _ = next(tscv.split(X_imputed))
                X_opt = X_imputed[train_idx_opt]
                y_opt = y[train_idx_opt]
                
                # Fit imputer and scaler on optimization training data only
                imputer_opt = SimpleImputer(strategy='median')
                X_opt_imputed = imputer_opt.fit_transform(X_opt)
                X_opt_imputed = np.nan_to_num(X_opt_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
                
                scaler_opt = RobustScaler()
                X_opt_scaled = scaler_opt.fit_transform(X_opt_imputed)
                X_opt_scaled = np.nan_to_num(X_opt_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
                
                best_params = self._optimize_hyperparameters(
                    model_wrapper, X_opt_scaled, y_opt, horizon, n_trials=self.n_trials
                )
                if best_params:
                    model_wrapper.optimized_params = best_params
                    self.logger.info(f"  Using optimized parameters: {list(best_params.keys())}")
                else:
                    self.logger.warning(f"  Optimization returned no parameters, using defaults")
                    model_wrapper.optimized_params = None  # Ensure it's cleared
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_imputed)):
                # CRITICAL FIX: Fit imputer and scaler ONLY on training data for this fold
                # This prevents data leakage from validation set
                X_train_raw = X_imputed[train_idx]
                X_val_raw = X_imputed[val_idx]
                
                # Fit imputer on training data only, then transform both train and val
                imputer = SimpleImputer(strategy='median')
                X_train_imputed = imputer.fit_transform(X_train_raw)
                X_val_imputed = imputer.transform(X_val_raw)
                
                # Replace any remaining inf/nan values
                X_train_imputed = np.nan_to_num(X_train_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
                X_val_imputed = np.nan_to_num(X_val_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Fit scaler on training data only, then transform both train and val
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train_imputed)
                X_val = scaler.transform(X_val_imputed)
                
                # Final safety check
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
                
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create and train model
                # Special handling for LSTM/GRU - need input_size
                if isinstance(model_wrapper, (LSTMWrapper, GRUWrapper)):
                    if not hasattr(model_wrapper, 'input_size') or model_wrapper.input_size is None:
                        model_wrapper.input_size = X_train.shape[1]
                    model_wrapper.create_model(input_size=X_train.shape[1])
                else:
                    model_wrapper.create_model()
                
                # Special handling for Prophet (set dates before fit)
                if isinstance(model_wrapper, ProphetWrapper):
                    train_dates = dates.iloc[train_idx].reset_index(drop=True) if dates is not None and len(dates) > max(train_idx) else None
                    model_wrapper.set_dates(train_dates)
                    train_time = model_wrapper.fit(X_train, y_train)
                    val_dates = dates.iloc[val_idx].reset_index(drop=True) if dates is not None and len(dates) > max(val_idx) else None
                    model_wrapper.set_dates(val_dates)
                else:
                    # Production-level training: Use validation set for early stopping
                    # Further split training data: 80% train, 20% validation for early stopping
                    train_split_idx = int(len(X_train) * 0.8)
                    X_train_split = X_train[:train_split_idx]
                    X_val_split = X_train[train_split_idx:]
                    y_train_split = y_train[:train_split_idx]
                    y_val_split = y_train[train_split_idx:]
                    
                    # Fit with validation set for early stopping (production-level)
                    train_time = model_wrapper.fit(X_train_split, y_train_split, X_val_split, y_val_split)
                
                training_times.append(train_time)
                
                # Predict (model predicts in log space)
                y_pred_log = model_wrapper.predict(X_val)
                
                # Transform predictions back to original scale for evaluation
                # Clip log predictions to prevent overflow when exponentiating
                # For prices in range [100, 10M], log1p values are [4.6, 16.1], so [-50, 20] is safe
                max_log_value = 20.0  # exp(20) ≈ 485 million, reasonable upper bound
                min_log_value = -10.0  # exp(-10) ≈ 0.000045, reasonable lower bound (more realistic than -50)
                
                y_pred_log_flat = y_pred_log.flatten()
                y_val_log_flat = y_val.flatten()
                
                # Clip predictions (but log if clipping occurs to detect model issues)
                n_clipped_pred = np.sum((y_pred_log_flat < min_log_value) | (y_pred_log_flat > max_log_value))
                if n_clipped_pred > 0:
                    self.logger.debug(f"  Fold {fold_idx}: Clipped {n_clipped_pred}/{len(y_pred_log_flat)} predictions")
                
                y_pred_log_clipped = np.clip(y_pred_log_flat, min_log_value, max_log_value)
                # Don't clip true values - they should be in valid range if data is clean
                # If they're not, we want to know about it
                y_val_log_clipped = y_val_log_flat
                
                epsilon = 1e-6
                y_pred = np.expm1(y_pred_log_clipped) - epsilon  # expm1(x) = exp(x) - 1, inverse of log1p
                y_val_original = np.expm1(y_val_log_clipped) - epsilon  # Transform validation targets back too
                
                # Additional safety: clip extreme predictions only (not true values)
                max_reasonable_price = 1e7  # 10 million as upper bound
                y_pred = np.clip(y_pred, 0, max_reasonable_price)
                # Don't clip true values - if they're extreme, we should know
                # y_val_original = np.clip(y_val_original, 0, max_reasonable_price)  # REMOVED: Don't clip true values
                
                # Calculate metrics on original scale
                mae = mean_absolute_error(y_val_original, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
                r2 = r2_score(y_val_original, y_pred)
                
                # MAPE (on original scale)
                epsilon_mape = 1e-8
                mape = np.mean(np.abs((y_val_original - y_pred) / (y_val_original + epsilon_mape))) * 100
                
                # Directional accuracy (on original scale)
                if len(y_val_original) > 1:
                    true_direction = np.diff(y_val_original) > 0
                    pred_direction = np.diff(y_pred) > 0
                    directional_accuracy = np.mean(true_direction == pred_direction) * 100
                else:
                    directional_accuracy = 0.0
                
                mae_scores.append(mae)
                rmse_scores.append(rmse)
                mape_scores.append(mape)
                r2_scores.append(r2)
                directional_accuracies.append(directional_accuracy)
            
            return ModelResult(
                model_name=model_name,
                model_family=model_family,
                horizon=horizon,
                mae=np.mean(mae_scores),
                rmse=np.mean(rmse_scores),
                mape=np.mean(mape_scores),
                r2=np.mean(r2_scores),
                directional_accuracy=np.mean(directional_accuracies),
                training_time=np.mean(training_times),
                n_samples=len(X),
                n_features=X.shape[1]
            )
        
        except Exception as e:
            self.logger.warning(f"Model {model_name} failed: {str(e)}")
            return ModelResult(
                model_name=model_name,
                model_family=model_family,
                horizon=horizon,
                mae=float('inf'),
                rmse=float('inf'),
                mape=float('inf'),
                r2=-1.0,
                directional_accuracy=0.0,
                training_time=0.0,
                n_samples=len(X),
                n_features=X.shape[1],
                error=str(e)
            )
    
    def evaluate_all_models(
        self,
        data_path: Optional[str] = None,
        currency: Optional[str] = None,
        horizons: List[str] = ['1d'],
        cv_folds: int = 5,  # Production standard: 5 folds
        min_avg_value: float = 1000.0
    ) -> List[ModelResult]:
        """Evaluate all model candidates. Trains per-currency to match production pipeline."""
        self.logger.info("Starting model candidate evaluation")
        self.logger.info(f"Evaluating {len(self.models)} models across {len(horizons)} horizons")
        
        # Load data with currency information preserved
        df, targets_dict, feature_names, dates = self.load_data_with_currency(data_path, currency, min_avg_value)
        
        results = []
        
        # Get unique currencies
        if 'currency' in df.columns:
            currencies = df['currency'].unique().tolist()
            self.logger.info(f"Training per-currency models for {len(currencies)} currencies")
        else:
            currencies = ['all']
            self.logger.warning("No currency column found, training on all data together")
        
        for horizon in horizons:
            if horizon not in targets_dict:
                self.logger.warning(f"Skipping horizon {horizon} - target not available")
                continue
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Evaluating models for {horizon} horizon")
            self.logger.info(f"{'='*80}")
            
            # Evaluate each model across all currencies
            for i, model_wrapper in enumerate(self.models):
                self.logger.info(f"[{i+1}/{len(self.models)}] Evaluating {model_wrapper.name}...")
                
                # Aggregate results across currencies
                currency_results = []
                
                for curr in currencies:
                    # Filter data for this currency
                    if 'currency' in df.columns:
                        currency_mask = df['currency'] == curr
                        X_curr = df.loc[currency_mask, feature_names].values
                        y_curr = targets_dict[horizon][currency_mask]
                        if dates is not None:
                            dates_curr = dates.loc[currency_mask].reset_index(drop=True) if hasattr(dates, 'loc') else dates[currency_mask].reset_index(drop=True)
                        else:
                            dates_curr = None
                    else:
                        X_curr = df[feature_names].values
                        y_curr = targets_dict[horizon]
                        dates_curr = dates
                    
                    # Filter out NaN targets
                    valid_mask = ~pd.isna(y_curr)
                    X_curr = X_curr[valid_mask]
                    y_curr = y_curr[valid_mask]
                    dates_curr = dates_curr[valid_mask].reset_index(drop=True) if dates_curr is not None else None
                    
                    if len(X_curr) < cv_folds * 10:  # Need at least 10 samples per fold
                        self.logger.debug(f"  Skipping {curr}: insufficient data ({len(X_curr)} samples)")
                        continue
                    
                    # Evaluate model on this currency
                    try:
                        result = self.evaluate_model(
                            model_wrapper, X_curr, y_curr, horizon, cv_folds, dates_curr
                        )
                        if not result.error:
                            currency_results.append(result)
                    except Exception as e:
                        self.logger.debug(f"  {model_wrapper.name} failed for {curr}: {str(e)}")
                        continue
                
                # Aggregate results across currencies (weighted by sample count)
                if currency_results:
                    total_samples = sum(r.n_samples for r in currency_results)
                    aggregated_mae = sum(r.mae * r.n_samples for r in currency_results) / total_samples
                    aggregated_rmse = sum(r.rmse * r.n_samples for r in currency_results) / total_samples
                    aggregated_r2 = sum(r.r2 * r.n_samples for r in currency_results) / total_samples
                    aggregated_mape = sum(r.mape * r.n_samples for r in currency_results) / total_samples
                    aggregated_dir_acc = sum(r.directional_accuracy * r.n_samples for r in currency_results) / total_samples
                    aggregated_time = np.mean([r.training_time for r in currency_results])
                    
                    aggregated_result = ModelResult(
                        model_name=model_wrapper.name,
                        model_family=model_wrapper.family,
                        horizon=horizon,
                        mae=aggregated_mae,
                        rmse=aggregated_rmse,
                        mape=aggregated_mape,
                        r2=aggregated_r2,
                        directional_accuracy=aggregated_dir_acc,
                        training_time=aggregated_time,
                        n_samples=total_samples,
                        n_features=currency_results[0].n_features
                    )
                    results.append(aggregated_result)
                    
                    self.logger.info(
                        f"  {model_wrapper.name}: MAE={aggregated_mae:.4f}, "
                        f"RMSE={aggregated_rmse:.4f}, R²={aggregated_r2:.4f} "
                        f"({len(currency_results)} currencies)"
                    )
                else:
                    # All currencies failed
                    results.append(ModelResult(
                        model_name=model_wrapper.name,
                        model_family=model_wrapper.family,
                        horizon=horizon,
                        mae=float('inf'),
                        rmse=float('inf'),
                        mape=float('inf'),
                        r2=-1.0,
                        directional_accuracy=0.0,
                        training_time=0.0,
                        n_samples=0,
                        n_features=len(feature_names),
                        error=f"Failed on all {len(currencies)} currencies"
                    ))
                    self.logger.warning(f"  {model_wrapper.name} failed on all currencies")
        
        return results
    
    def save_results(self, results: List[ModelResult], output_path: str):
        """Save evaluation results to CSV and JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Save CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")
        
        # Save JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, default=str)
        self.logger.info(f"Results saved to {json_path}")
        
        # Print summary
        self._print_summary(df)
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary of results."""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        for horizon in df['horizon'].unique():
            print(f"\n{horizon.upper()} Horizon:")
            print("-" * 80)
            df_h = df[df['horizon'] == horizon].copy()
            df_h = df_h[df_h['error'].isna()]  # Remove failed models
            df_h = df_h.sort_values('mae')
            
            print(f"{'Rank':<5} {'Model':<25} {'Family':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Time(s)':<10}")
            print("-" * 80)
            
            # Show all models, not just top 10
            for idx, (_, row) in enumerate(df_h.iterrows(), 1):
                print(
                    f"{idx:<5} {row['model_name']:<25} {row['model_family']:<20} "
                    f"{row['mae']:<10.4f} {row['rmse']:<10.4f} {row['r2']:<10.4f} "
                    f"{row['training_time']:<10.2f}"
                )
            
            # Also show failed models if any
            failed_models = df[(df['horizon'] == horizon) & (df['error'].notna())]
            if len(failed_models) > 0:
                print(f"\nFailed Models ({len(failed_models)}):")
                print("-" * 80)
                for _, row in failed_models.iterrows():
                    print(f"  {row['model_name']}: {row.get('error', 'Unknown error')}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate 20 model candidates for currency price prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models with default settings
  python evaluate_model_candidates.py
  
  # Evaluate on specific currency
  python evaluate_model_candidates.py --currency "Chaos Orb"
  
  # Use local data file
  python evaluate_model_candidates.py --data-path data/processed.parquet
  
  # Custom horizons and CV folds
  python evaluate_model_candidates.py --horizons 1d 3d --cv-folds 3
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to processed parquet data file (if not provided, loads from S3)'
    )
    
    parser.add_argument(
        '--currency',
        type=str,
        help='Specific currency to evaluate (if not provided, uses all currencies)'
    )
    
    parser.add_argument(
        '--horizons',
        nargs='+',
        default=['1d'],
        help='Prediction horizons to evaluate (default: 1d)'
    )
    
    parser.add_argument(
        '--min-avg-value',
        type=float,
        default=1000.0,
        help='Minimum average currency value (in Chaos Orbs) to include (default: 1000.0)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5, production standard)'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable hyperparameter optimization using Optuna (slower but better results)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials per model when --optimize is enabled (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='model_candidate_results',
        help='Output file path (without extension, default: model_candidate_results)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_ml_logging(
        name="ModelEvaluator",
        level="INFO",
        log_dir=str(Path(__file__).parent.parent / "logs"),
        experiment_id="model_evaluation",
        console_output=True,
        suppress_external=True
    )
    
    # Get config
    config = get_default_config()
    
    # Create evaluator
    evaluator = ModelCandidateEvaluator(
        config, 
        logger, 
        use_optimization=args.optimize,
        n_trials=args.n_trials
    )
    
    # Run evaluation
    try:
        results = evaluator.evaluate_all_models(
            data_path=args.data_path,
            currency=args.currency,
            horizons=args.horizons,
            cv_folds=args.cv_folds,
            min_avg_value=args.min_avg_value
        )
        
        # Save results
        output_path = Path(__file__).parent.parent / args.output
        evaluator.save_results(results, str(output_path))
        
        logger.info(f"\nEvaluation complete! Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exception=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
