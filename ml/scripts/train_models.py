#!/usr/bin/env python3
"""
Improved Currency-Specific Model Training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML imports
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import optuna
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CurrencyModelResults:
    """Results for a single currency model."""
    currency_pair: str
    model_type: str
    data_points: int
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    model_path: str
    scaler_path: str
    training_time: float
    improvement_strategies: List[str]
    # Settlers league evaluation metrics
    settlers_mae: Optional[float] = None
    settlers_rmse: Optional[float] = None
    settlers_mape: Optional[float] = None
    settlers_directional_accuracy: Optional[float] = None
    settlers_sample_count: Optional[int] = None

class ImprovedCurrencyTrainer:
    def __init__(self, data_path: str, output_dir: str = "./models/currency", 
                 verbose: bool = False):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Enhanced configurations for production training
        self.min_samples_required = 150 
        self.test_size = 0.2
        self.random_state = 42
        self.results = []
        self.has_settlers_data = False  # Will be set during data loading
        
        # Improvement strategies flags
        self.strategies = {
            'log_transform': True,
            'robust_scaling': True,
            'feature_selection': True,
            'ensemble_models': True,
            'outlier_removal': True,
            'advanced_cv': True
        }
        
        # Set logging level based on verbosity
        if not self.verbose:
            logging.getLogger('lightgbm').setLevel(logging.WARNING)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load and analyze the training data."""
        logger.info(f"Loading data from {self.data_path}")
        
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        data = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(data)} records with {len(data.columns)} features")
        
        # Validate required columns
        required_cols = ['currency_pair', 'target_price_1d']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for league information
        if 'league_name' in data.columns:
            league_counts = data['league_name'].value_counts()
            logger.info(f"Available leagues: {dict(league_counts)}")
            
            # Check if Settlers league data is available
            if 'Settlers' in league_counts:
                settlers_count = league_counts['Settlers']
                logger.info(f"Settlers league data available: {settlers_count:,} records")
                self.has_settlers_data = True
            else:
                logger.warning("No Settlers league data found - will use standard evaluation")
                self.has_settlers_data = False
        else:
            logger.warning("No league information found - will use standard evaluation")
            self.has_settlers_data = False
        
        return data
    
    def remove_outliers(self, currency_data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        if not self.strategies['outlier_removal']:
            return currency_data
        
        target_col = 'target_price_1d'
        Q1 = currency_data[target_col].quantile(0.25)
        Q3 = currency_data[target_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # More conservative outlier removal
        lower_bound = Q1 - 2.0 * IQR  # Less aggressive than 1.5 * IQR
        upper_bound = Q3 + 2.0 * IQR
        
        original_count = len(currency_data)
        currency_data = currency_data[
            (currency_data[target_col] >= lower_bound) & 
            (currency_data[target_col] <= upper_bound)
        ]
        
        removed_count = original_count - len(currency_data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outliers ({removed_count/original_count*100:.1f}%)")
        
        return currency_data
    
    def apply_log_transformation(self, y: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply log transformation if beneficial."""
        if not self.strategies['log_transform']:
            return y, False
        
        # Only apply if all values are positive and have wide range
        if np.all(y > 0) and (np.max(y) / np.min(y) > 10):
            y_log = np.log1p(y)  # log(1+x) to handle values near zero
            
            # Check if log transformation reduces variance
            original_cv = np.std(y) / np.mean(y)
            log_cv = np.std(y_log) / np.mean(y_log)
            
            if log_cv < original_cv * 0.8:  # 20% improvement required
                logger.info(f"Applied log transformation (CV: {original_cv:.3f} -> {log_cv:.3f})")
                return y_log, True
        
        return y, False
    
    def prepare_currency_features(self, currency_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Enhanced feature preparation with multiple improvements."""
        # Remove currency-specific and target columns
        exclude_cols = [
            'currency_pair', 'get_currency', 'pay_currency', 
            'getCurrencyId', 'payCurrencyId',
            'target_price_1d', 'target_price_3d', 'target_price_7d',
            'target_change_1d', 'target_change_3d', 'target_change_7d',
            'target_change_pct_1d', 'target_change_pct_3d', 'target_change_pct_7d',
            'target_direction_1d', 'target_direction_3d', 'target_direction_7d',
            'target_volatility_1d', 'target_volatility_3d', 'target_volatility_7d',
            'created_at', 'updated_at'
        ]
        
        # Get feature columns, excluding target columns and other problematic columns
        feature_cols = [col for col in currency_data.columns 
                       if col not in exclude_cols and not col.startswith('target_')]
        
        # Additional safety check for problematic columns
        safe_feature_cols = []
        for col in feature_cols:
            if currency_data[col].dtype in ['object', 'category']:
                # Check if it's a simple categorical that can be encoded
                unique_vals = currency_data[col].nunique()
                if unique_vals <= 50:  # Only encode if reasonable number of categories
                    safe_feature_cols.append(col)
                else:
                    logger.warning(f"Excluding high-cardinality categorical column: {col} ({unique_vals} unique values)")
            else:
                safe_feature_cols.append(col)
        
        X = currency_data[safe_feature_cols].copy()
        y = currency_data['target_price_1d'].copy()
        
        logger.info(f"Using {len(safe_feature_cols)} feature columns (excluded {len(currency_data.columns) - len(safe_feature_cols)} columns)")
        
        # Enhanced preprocessing info
        preprocessing_info = {
            'original_features': len(feature_cols),
            'log_transformed': False,
            'features_selected': False,
            'scaler_type': 'standard'
        }
        
        # Handle datetime columns more robustly
        for col in X.columns:
            if X[col].dtype.name.startswith('datetime') or 'datetime' in str(X[col].dtype):
                try:
                    # Convert datetime to timestamp (seconds since epoch)
                    X[col] = pd.to_datetime(X[col]).astype('int64') // 10**9
                except Exception as e:
                    logger.warning(f"Could not convert datetime column {col}: {e}")
                    X = X.drop(columns=[col])
        
        # Handle missing values more intelligently
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Fill numeric with median, categorical with mode
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown')
        
        # Enhanced categorical encoding
        label_encoders = {}
        for col in categorical_cols:
            try:
                # Convert to string first to handle mixed types
                X[col] = X[col].astype(str)
                
                # Skip columns that might be arrays/lists
                sample_value = X[col].iloc[0]
                if '[' in sample_value or '(' in sample_value:
                    logger.warning(f"Skipping complex column: {col}")
                    X = X.drop(columns=[col])
                    continue
                
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
                
            except Exception as e:
                logger.warning(f"Error encoding column {col}: {e}")
                X = X.drop(columns=[col])
        
        # Convert to numpy arrays with better error handling
        try:
            # Handle any remaining NaN or inf values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True))
            
            X = X.astype(float).values
            y = y.astype(float).values
            
        except Exception as e:
            logger.error(f"Error converting to float: {e}")
            logger.info("Checking problematic columns...")
            
            # Find and handle problematic columns one by one
            problematic_cols = []
            for col in X.columns:
                try:
                    # Try to convert each column individually
                    test_series = X[col].replace([np.inf, -np.inf], np.nan)
                    test_series = test_series.fillna(test_series.median() if test_series.dtype in ['int64', 'float64'] else 0)
                    test_series.astype(float)
                except Exception as col_error:
                    logger.warning(f"Marking column for removal {col}: {col_error}")
                    problematic_cols.append(col)
            
            # Drop problematic columns
            if problematic_cols:
                logger.info(f"Dropping {len(problematic_cols)} problematic columns: {problematic_cols}")
                X = X.drop(columns=problematic_cols)
            
            # Try again after cleanup
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True))
            X = X.astype(float).values
            y = y.astype(float).values
            
            logger.info(f"Successfully converted data: X shape {X.shape}, y shape {y.shape}")
        
        # Apply log transformation if beneficial
        y, log_transformed = self.apply_log_transformation(y)
        preprocessing_info['log_transformed'] = log_transformed
        
        # Feature selection for high-dimensional data
        if self.strategies['feature_selection'] and X.shape[1] > 50:
            n_features = min(30, X.shape[1] // 2)  # Select top 30 or half the features
            selector = SelectKBest(score_func=f_regression, k=n_features)
            X = selector.fit_transform(X, y)
            preprocessing_info['features_selected'] = True
            preprocessing_info['final_features'] = X.shape[1]
            logger.info(f"Selected {X.shape[1]} most important features")
        
        return X, y, preprocessing_info
    
    def create_advanced_scaler(self, X: np.ndarray) -> Any:
        """Create robust scaler for better handling of outliers."""
        if self.strategies['robust_scaling']:
            return RobustScaler()  # Less sensitive to outliers
        else:
            return StandardScaler()
    
    def calculate_improved_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  log_transformed: bool = False) -> Dict[str, float]:
        """Enhanced metrics calculation with better MAPE handling."""
        
        # Transform back if log transformed
        if log_transformed:
            y_true_orig = np.expm1(y_true)  # inverse of log1p
            y_pred_orig = np.expm1(y_pred)
        else:
            y_true_orig = y_true
            y_pred_orig = y_pred
        
        # Basic metrics
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        
        # Improved MAPE calculation
        mape_values = []
        epsilon = 1e-3  # Small value to avoid division by zero
        
        for true, pred in zip(y_true_orig, y_pred_orig):
            if abs(true) > epsilon:
                # Symmetric MAPE for better handling of small values
                denominator = (abs(true) + abs(pred)) / 2 + epsilon
                mape_val = abs(true - pred) / denominator * 100
                mape_values.append(mape_val)
        
        mape = np.mean(mape_values) if mape_values else float('inf')
        
        # Enhanced directional accuracy
        if len(y_true_orig) > 1:
            true_direction = np.diff(y_true_orig) > 0
            pred_direction = np.diff(y_pred_orig) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0.0
        
        # Additional metrics
        r2_score = 1 - (np.sum((y_true_orig - y_pred_orig) ** 2) / 
                       (np.sum((y_true_orig - np.mean(y_true_orig)) ** 2) + epsilon))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'r2_score': max(r2_score, -1.0)  # Cap at -1 for very bad predictions
        }
    
    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray, currency_pair: str,
                           log_transformed: bool = False) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Train ensemble of models for better predictions."""
        
        def lgb_objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'verbosity': -1
            }
            
            # Skip advanced CV for very small datasets
            if len(X) < 100:
                # Use simple holdout validation for small datasets
                split_point = int(len(X) * 0.8)
                X_train_cv, X_val_cv = X[:split_point], X[split_point:]
                y_train_cv, y_val_cv = y[:split_point], y[split_point:]
                
                if len(X_val_cv) < 5:
                    # Too small for validation - return simple score
                    return float('inf')
                
                train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
                
                try:
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=100  # Fewer iterations for small data
                    )
                    y_pred = model.predict(X_val_cv)
                    metrics = self.calculate_improved_metrics(y_val_cv, y_pred, log_transformed)
                    return metrics['rmse']
                except Exception:
                    return float('inf')
            else:
                # Advanced time series cross-validation for larger datasets
                if self.strategies['advanced_cv']:
                    tscv = TimeSeriesSplit(n_splits=5, test_size=max(10, len(X) // 10))
                else:
                    tscv = TimeSeriesSplit(n_splits=3)
                
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    if len(train_idx) < 10 or len(val_idx) < 5:
                        continue
                        
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                    
                    train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
                    
                    try:
                        # Use validation set only if we have enough data
                        if len(X_val_cv) >= 10:
                            val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
                            model = lgb.train(
                                params,
                                train_data,
                                valid_sets=[val_data],
                                num_boost_round=200,
                                callbacks=[lgb.early_stopping(20, verbose=False)]
                            )
                        else:
                            # Train without validation if validation set too small
                            model = lgb.train(
                                params,
                                train_data,
                                num_boost_round=100
                            )
                        
                        y_pred = model.predict(X_val_cv)
                        metrics = self.calculate_improved_metrics(y_val_cv, y_pred, log_transformed)
                        scores.append(metrics['rmse'])  # Optimize for RMSE
                    except Exception:
                        continue  # Skip problematic folds
                
                return np.mean(scores) if scores else float('inf')
        
        # Optimize LightGBM
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(lgb_objective, n_trials=200, show_progress_bar=True)
        
        # Train final ensemble
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1
        })
        
        # Split data
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        models = {}
        predictions = []
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Check if we have enough data for validation
        if len(X_test) < 10:
            # Too little test data for early stopping - train without validation
            logger.warning(f"Insufficient test data ({len(X_test)} samples) for early stopping - training without validation")
            lgb_model = lgb.train(
                best_params, 
                train_data, 
                num_boost_round=100  # Reduce iterations to prevent overfitting
            )
        else:
            # Sufficient data for early stopping with validation
            try:
                val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
                lgb_model = lgb.train(
                    best_params, 
                    train_data, 
                    valid_sets=[val_data],
                    num_boost_round=300,
                    callbacks=[lgb.early_stopping(20, verbose=False)]
                )
            except Exception as e:
                logger.warning(f"Early stopping failed ({str(e)}) - training without validation")
                lgb_model = lgb.train(
                    best_params, 
                    train_data, 
                    num_boost_round=100
                )
        
        lgb_pred = lgb_model.predict(X_test)
        models['lightgbm'] = lgb_model
        predictions.append(lgb_pred)
        
        # Train XGBoost if ensemble enabled
        if self.strategies['ensemble_models'] and len(X_train) > 100:
            xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbosity': 0 if not self.verbose else 1
            }
            
            xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=200)
            xgb_model.fit(X_train, y_train, verbose=self.verbose)
            xgb_pred = xgb_model.predict(X_test)
            models['xgboost'] = xgb_model
            predictions.append(xgb_pred)
        
        # Ensemble prediction (simple average)
        if len(predictions) > 1:
            y_pred = np.mean(predictions, axis=0)
            model_type = "LightGBM+XGBoost Ensemble"
        else:
            y_pred = predictions[0]
            model_type = "LightGBM"
        
        # Calculate metrics
        metrics = self.calculate_improved_metrics(y_test, y_pred, log_transformed)
        
        return {'models': models, 'type': model_type}, metrics
    
    def train_currency_model(self, currency_pair: str, currency_data: pd.DataFrame) -> CurrencyModelResults:
        """Train improved model for a specific currency pair."""
        start_time = datetime.now()
        
        logger.info(f"Training improved model for {currency_pair} ({len(currency_data)} samples)")
        
        # Check minimum data requirement before any processing
        if len(currency_data) < 30:
            logger.warning(f"Insufficient data for {currency_pair}: {len(currency_data)} samples (minimum 30 required)")
            raise ValueError(f"Insufficient data: {len(currency_data)} samples (minimum 30 required)")
        
        # Remove outliers
        currency_data = self.remove_outliers(currency_data)
        
        if len(currency_data) < self.min_samples_required:
            logger.warning(f"Insufficient data after outlier removal for {currency_pair}")
            raise ValueError("Insufficient data after preprocessing")
        
        # Prepare features with improvements
        X, y, preprocessing_info = self.prepare_currency_features(currency_data)
        
        # Create advanced scaler
        scaler = self.create_advanced_scaler(X)
        X_scaled = scaler.fit_transform(X)
        
        # Train ensemble model
        model_info, metrics = self.train_ensemble_model(
            X_scaled, y, currency_pair, preprocessing_info['log_transformed']
        )
        
        # Evaluate on Settlers league data if available
        settlers_metrics = {}
        if self.has_settlers_data:
            settlers_metrics = self.evaluate_on_settlers_league(
                X_scaled, y, currency_data, model_info, scaler, preprocessing_info
            )
        
        # Create currency-specific directory
        currency_dir = self.output_dir / currency_pair.replace('_', '-').replace(' ', '-')
        currency_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models and scaler
        model_path = currency_dir / "ensemble_model.pkl"
        scaler_path = currency_dir / "scaler.pkl"
        preprocessing_path = currency_dir / "preprocessing_info.json"
        
        joblib.dump(model_info, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(preprocessing_path, 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Track applied improvements
        applied_strategies = [k for k, v in self.strategies.items() if v]
        
        # Create results with Settlers league metrics
        results = CurrencyModelResults(
            currency_pair=currency_pair,
            model_type=model_info['type'],
            data_points=len(currency_data),
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            mape=metrics['mape'],
            directional_accuracy=metrics['directional_accuracy'],
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            training_time=training_time,
            improvement_strategies=applied_strategies,
            # Add Settlers league metrics if available
            settlers_mae=settlers_metrics.get('settlers_mae'),
            settlers_rmse=settlers_metrics.get('settlers_rmse'),
            settlers_mape=settlers_metrics.get('settlers_mape'),
            settlers_directional_accuracy=settlers_metrics.get('settlers_directional_accuracy'),
            settlers_sample_count=settlers_metrics.get('settlers_sample_count')
        )
        
        return results
    
    def train_all_currencies(self) -> List[CurrencyModelResults]:
        """Train improved models for all currency pairs."""
        # Load data
        data = self.load_and_analyze_data()
        
        # Get currency pairs with sufficient data
        currency_counts = data['currency_pair'].value_counts()
        eligible_currencies = currency_counts[currency_counts >= self.min_samples_required]
        
        logger.info(f"Training improved models for {len(eligible_currencies)} currency pairs")
        logger.info(f"Applied strategies: {[k for k, v in self.strategies.items() if v]}")
        
        results = []
        
        for currency_pair in eligible_currencies.index:
            try:
                currency_data = data[data['currency_pair'] == currency_pair].copy()
                
                if len(currency_data) < 30:
                    logger.warning(f"Skipping {currency_pair}: insufficient data ({len(currency_data)} samples)")
                    continue
                
                result = self.train_currency_model(currency_pair, currency_data)
                results.append(result)
                
                # Enhanced logging with Settlers league metrics
                log_msg = f"✓ Completed {currency_pair}: {result.model_type} (RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f})"
                if result.settlers_rmse is not None:
                    log_msg += f" | Settlers: RMSE {result.settlers_rmse:.4f}, MAE {result.settlers_mae:.4f}"
                logger.info(log_msg)
                
            except Exception as e:
                logger.error(f"Failed to train improved model for {currency_pair}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def save_results_summary(self):
        """Save comprehensive results summary with improvements."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Create detailed summary
        summary_data = []
        for result in self.results:
            summary_data.append({
                'currency_pair': result.currency_pair,
                'model_type': result.model_type,
                'data_points': result.data_points,
                'mae': result.mae,
                'rmse': result.rmse,
                'mape': result.mape,
                'directional_accuracy': result.directional_accuracy,
                'training_time': result.training_time,
                'improvement_strategies': ', '.join(result.improvement_strategies),
                'model_path': result.model_path,
                # Settlers league metrics
                'settlers_mae': result.settlers_mae,
                'settlers_rmse': result.settlers_rmse,
                'settlers_mape': result.settlers_mape,
                'settlers_directional_accuracy': result.settlers_directional_accuracy,
                'settlers_sample_count': result.settlers_sample_count
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('rmse')
        
        # Save summary
        summary_path = self.output_dir / "training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Calculate improvement statistics based on RMSE
        rmse_under_threshold = len(summary_df[summary_df['rmse'] < summary_df['rmse'].quantile(0.25)])  # Best 25%
        da_over_65 = len(summary_df[summary_df['directional_accuracy'] > 65])
        
        # Calculate Settlers league statistics
        settlers_available = summary_df['settlers_rmse'].notna().sum()
        settlers_stats = {}
        if settlers_available > 0:
            settlers_df = summary_df[summary_df['settlers_rmse'].notna()]
            settlers_stats = {
                'settlers_models_evaluated': int(settlers_available),
                'settlers_avg_rmse': float(settlers_df['settlers_rmse'].mean()),
                'settlers_median_rmse': float(settlers_df['settlers_rmse'].median()),
                'settlers_avg_mae': float(settlers_df['settlers_mae'].mean()),
                'settlers_avg_da': float(settlers_df['settlers_directional_accuracy'].mean()),
                'settlers_total_samples': int(settlers_df['settlers_sample_count'].sum())
            }
        
        # Save performance report (convert numpy types to native Python types)
        report = {
            'total_models': int(len(self.results)),
            'models_rmse_top_quartile': int(rmse_under_threshold),
            'models_da_over_65': int(da_over_65),
            'rmse_success_rate': float(rmse_under_threshold / len(self.results) * 100),
            'da_success_rate': float(da_over_65 / len(self.results) * 100),
            'average_rmse': float(summary_df['rmse'].mean()),
            'median_rmse': float(summary_df['rmse'].median()),
            'rmse_25th_percentile': float(summary_df['rmse'].quantile(0.25)),
            'average_mae': float(summary_df['mae'].mean()),
            'average_da': float(summary_df['directional_accuracy'].mean()),
            'applied_strategies': [k for k, v in self.strategies.items() if v],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add Settlers league statistics (already converted to native types above)
        report.update(settlers_stats)
        
        with open(self.output_dir / "improvement_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved results to {self.output_dir}")
        logger.info(f"Top quartile RMSE: {rmse_under_threshold}/{len(self.results)} ({rmse_under_threshold/len(self.results)*100:.1f}%)")
        logger.info(f"DA > 65%: {da_over_65}/{len(self.results)} ({da_over_65/len(self.results)*100:.1f}%)")
        logger.info(f"Average RMSE: {summary_df['rmse'].mean():.4f}, Median RMSE: {summary_df['rmse'].median():.4f}")
        
        # Log Settlers league performance if available
        if settlers_available > 0:
            logger.info(f"Settlers League Evaluation:")
            logger.info(f"  Models evaluated: {settlers_available}/{len(self.results)}")
            logger.info(f"  Average RMSE: {settlers_stats['settlers_avg_rmse']:.4f}")
            logger.info(f"  Average MAE: {settlers_stats['settlers_avg_mae']:.4f}")
            logger.info(f"  Average DA: {settlers_stats['settlers_avg_da']:.1f}%")
            logger.info(f"  Total samples: {settlers_stats['settlers_total_samples']:,}")
        else:
            logger.info("No Settlers league evaluation performed")

    def evaluate_on_settlers_league(self, X: np.ndarray, y: np.ndarray, currency_data: pd.DataFrame,
                                   model_info: Dict[str, Any], scaler: Any, 
                                   preprocessing_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the trained model specifically on Settlers league data.
        
        Args:
            X: Full feature matrix
            y: Full target vector
            currency_data: Original currency data with league information
            model_info: Trained model information
            scaler: Fitted scaler
            preprocessing_info: Preprocessing metadata
            
        Returns:
            Dictionary of evaluation metrics on Settlers league data
        """
        if not self.has_settlers_data or 'league_name' not in currency_data.columns:
            logger.warning("No Settlers league data available for evaluation")
            return {}
        
        # Filter for Settlers league data
        settlers_mask = currency_data['league_name'] == 'Settlers'
        settlers_data = currency_data[settlers_mask].copy()
        
        if len(settlers_data) < 10:
            logger.warning(f"Insufficient Settlers league data for evaluation: {len(settlers_data)} samples")
            return {}
        
        # Get corresponding features and targets for Settlers league
        X_settlers = X[settlers_mask]
        y_settlers = y[settlers_mask]
        
        logger.info(f"Evaluating on Settlers league: {len(X_settlers)} samples")
        
        # Get the primary model (LightGBM or ensemble)
        if 'models' in model_info and 'lightgbm' in model_info['models']:
            primary_model = model_info['models']['lightgbm']
            
            # Make predictions on Settlers data
            y_pred_settlers = primary_model.predict(X_settlers)
            
            # If ensemble, also get XGBoost predictions and average
            if 'xgboost' in model_info['models']:
                xgb_pred = model_info['models']['xgboost'].predict(X_settlers)
                y_pred_settlers = (y_pred_settlers + xgb_pred) / 2
                model_type = "Ensemble"
            else:
                model_type = "LightGBM"
            
            # Calculate metrics on Settlers league data
            settlers_metrics = self.calculate_improved_metrics(
                y_settlers, y_pred_settlers, preprocessing_info['log_transformed']
            )
            
            # Add prefix to distinguish from training metrics
            settlers_metrics = {f"settlers_{k}": v for k, v in settlers_metrics.items()}
            settlers_metrics['settlers_sample_count'] = len(X_settlers)
            settlers_metrics['settlers_model_type'] = model_type
            
            logger.info(f"Settlers league evaluation - RMSE: {settlers_metrics['settlers_rmse']:.4f}, "
                       f"MAE: {settlers_metrics['settlers_mae']:.4f}, "
                       f"DA: {settlers_metrics['settlers_directional_accuracy']:.1f}%")
            
            return settlers_metrics
        else:
            logger.error("No valid model found for Settlers league evaluation")
            return {}

def main():
    """Main function to run improved currency-specific training."""
    # Use relative path to work from ml/scripts directory
    data_path = "./ml/training_data/combined_currency_features_exp_20250611_170257.parquet"
    
    trainer = ImprovedCurrencyTrainer(data_path, verbose=False)
    
    logger.info("Starting improved currency-specific training (quiet mode)...")
    results = trainer.train_all_currencies()
    
    if results:
        trainer.save_results_summary()
        logger.info(f"Successfully trained {len(results)} improved models")
    else:
        logger.error("No models were trained successfully")

if __name__ == "__main__":
    main() 