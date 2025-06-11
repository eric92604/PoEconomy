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

class ImprovedCurrencyTrainer:
    def __init__(self, data_path: str, output_dir: str = "../models/improved_currency"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced configurations for production training
        self.min_samples_required = 150 
        self.test_size = 0.2
        self.random_state = 42
        self.results = []
        
        # Improvement strategies flags
        self.strategies = {
            'log_transform': True,
            'robust_scaling': True,
            'feature_selection': True,
            'ensemble_models': True,
            'outlier_removal': True,
            'advanced_cv': True
        }
    
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
            'created_at', 'updated_at'
        ]
        
        feature_cols = [col for col in currency_data.columns if col not in exclude_cols]
        X = currency_data[feature_cols].copy()
        y = currency_data['target_price_1d'].copy()
        
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
            X = X.astype(float).values
            y = y.astype(float).values
        except Exception as e:
            logger.error(f"Error converting to float: {e}")
            logger.info("Checking problematic columns...")
            
            # Find problematic columns
            for col in X.columns:
                try:
                    X[col].astype(float)
                except Exception as col_error:
                    logger.warning(f"Dropping problematic column {col}: {col_error}")
                    X = X.drop(columns=[col])
            
            # Try again after cleanup
            X = X.astype(float).values
            y = y.astype(float).values
        
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
            
            # Advanced time series cross-validation
            if self.strategies['advanced_cv']:
                tscv = TimeSeriesSplit(n_splits=5, test_size=max(10, len(X) // 10))
            else:
                tscv = TimeSeriesSplit(n_splits=3)
            
            scores = []
            for train_idx, val_idx in tscv.split(X):
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                    
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=200,
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val)
                metrics = self.calculate_improved_metrics(y_val, y_pred, log_transformed)
                scores.append(metrics['mae'])  # Optimize for MAE
            
            return np.mean(scores) if scores else float('inf')
        
        # Optimize LightGBM
        study = optuna.create_study(direction='minimize')
        study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
        
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
        lgb_model = lgb.train(best_params, train_data, num_boost_round=300)
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
                'random_state': self.random_state
            }
            
            xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=200)
            xgb_model.fit(X_train, y_train)
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
        
        # Create results
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
            improvement_strategies=applied_strategies
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
                result = self.train_currency_model(currency_pair, currency_data)
                results.append(result)
                
                logger.info(f"{currency_pair}: MAE={result.mae:.4f}, MAPE={result.mape:.2f}%, DA={result.directional_accuracy:.1f}%")
                
            except Exception as e:
                logger.error(f"Failed to train improved model for {currency_pair}: {e}")
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
                'model_path': result.model_path
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('mape')
        
        # Save summary
        summary_path = self.output_dir / "improved_training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Calculate improvement statistics
        mape_under_15 = len(summary_df[summary_df['mape'] < 15])
        da_over_65 = len(summary_df[summary_df['directional_accuracy'] > 65])
        
        # Save performance report
        report = {
            'total_models': len(self.results),
            'models_mape_under_15': mape_under_15,
            'models_da_over_65': da_over_65,
            'mape_success_rate': mape_under_15 / len(self.results) * 100,
            'da_success_rate': da_over_65 / len(self.results) * 100,
            'average_mape': summary_df['mape'].mean(),
            'median_mape': summary_df['mape'].median(),
            'average_da': summary_df['directional_accuracy'].mean(),
            'applied_strategies': [k for k, v in self.strategies.items() if v],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "improvement_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved results to {self.output_dir}")
        logger.info(f"MAPE < 15%: {mape_under_15}/{len(self.results)} ({mape_under_15/len(self.results)*100:.1f}%)")
        logger.info(f"DA > 65%: {da_over_65}/{len(self.results)} ({da_over_65/len(self.results)*100:.1f}%)")

def main():
    """Main function to run improved currency-specific training."""
    # Use relative path to work from ml/scripts directory
    data_path = "./ml/training_data/combined_currency_features_exp_20250611_162923.parquet"
    
    trainer = ImprovedCurrencyTrainer(data_path)
    
    logger.info("Starting improved currency-specific training...")
    results = trainer.train_all_currencies()
    
    if results:
        trainer.save_results_summary()
        logger.info(f"Successfully trained {len(results)} improved models")
    else:
        logger.error("No models were trained successfully")

if __name__ == "__main__":
    main() 