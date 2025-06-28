"""
Model inference utilities for price prediction in current leagues.

This module provides functionality to load trained models and make predictions
on current league data, handling feature engineering and preprocessing requirements.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
import warnings
from dataclasses import dataclass
from scipy import stats
from sklearn.utils import resample

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import MLConfig
from utils.logging_utils import MLLogger
from utils.data_processing import DataProcessor
from utils.feature_engineering import FeatureEngineer
from utils.database import get_db_connection

warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Container for prediction results with prediction intervals."""
    currency: str
    current_price: float
    predicted_price: float
    prediction_horizon_days: int
    confidence_score: float
    price_change_percent: float
    prediction_timestamp: str
    model_type: str
    features_used: int
    data_points_used: int
    # New prediction interval fields
    prediction_lower: float
    prediction_upper: float
    interval_width: float
    confidence_method: str
    uncertainty_components: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'currency': self.currency,
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'prediction_horizon_days': self.prediction_horizon_days,
            'confidence_score': self.confidence_score,
            'price_change_percent': self.price_change_percent,
            'prediction_timestamp': self.prediction_timestamp,
            'model_type': self.model_type,
            'features_used': self.features_used,
            'data_points_used': self.data_points_used,
            'prediction_lower': self.prediction_lower,
            'prediction_upper': self.prediction_upper,
            'interval_width': self.interval_width,
            'confidence_method': self.confidence_method,
            'uncertainty_components': self.uncertainty_components
        }


class ModelPredictor:
    """
    Model predictor for currency price forecasting.
    
    This class handles loading trained models, preprocessing current league data,
    and making predictions while handling preprocessing requirements.
    """
    
    def __init__(
        self,
        models_dir: Union[str, Path],
        config: Optional[MLConfig] = None,
        logger: Optional[MLLogger] = None
    ):
        """
        Initialize the model predictor.
        
        Args:
            models_dir: Directory containing trained models
            config: ML configuration (will create default if None)
            logger: Optional logger instance
        """
        self.models_dir = Path(models_dir)
        self.config = config or MLConfig()
        self.logger = logger or MLLogger("ModelPredictor")
        
        # Initialize data processing components
        self.data_processor = DataProcessor(
            self.config.data, self.config.processing, self.logger
        )
        self.feature_engineer = FeatureEngineer(self.config.data, self.logger)
        
        # Model storage
        self.loaded_models = {}
        self.model_metadata = {}
        self.feature_scalers = {}
        
        # Current league cache
        self._current_league_data = None
        self._last_data_update = None
        self._data_cache_duration = timedelta(hours=1)  # Cache for 1 hour
    
    def load_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all available trained models including multi-horizon models.
        
        Returns:
            Dictionary mapping currencies to model information
        """
        available_models = {}
        
        if not self.models_dir.exists():
            self.logger.warning(f"Models directory not found: {self.models_dir}")
            return available_models
        
        # Find all currency directories (both single and multi-horizon)
        for item in self.models_dir.iterdir():
            if item.is_dir():
                item_name = item.name
                
                # Check if this is a horizon-specific directory (e.g., "Divine Orb_1d")
                if '_' in item_name and item_name.split('_')[-1] in ['1d', '3d', '7d']:
                    currency_name = '_'.join(item_name.split('_')[:-1])
                    horizon = item_name.split('_')[-1]
                    
                    # Initialize currency entry if not exists
                    if currency_name not in available_models:
                        available_models[currency_name] = {
                            'horizons': {},
                            'model_type': 'multi-horizon',
                            'has_multi_horizon': True
                        }
                    
                    # Load horizon-specific model
                    model_file = item / "ensemble_model.pkl"
                    metadata_file = item / "model_metadata.json"
                    
                    if model_file.exists():
                        try:
                            # Load metadata
                            metadata = {}
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                            
                            # Load model
                            model = joblib.load(model_file)
                            
                            # Load scaler if exists
                            scaler_file = item / "scaler.pkl"
                            scaler = None
                            if scaler_file.exists():
                                scaler = joblib.load(scaler_file)
                            
                            # Store horizon-specific model
                            horizon_key = f"{currency_name}_{horizon}"
                            self.loaded_models[horizon_key] = model
                            self.model_metadata[horizon_key] = metadata
                            self.feature_scalers[horizon_key] = scaler
                            
                            # Store horizon information
                            available_models[currency_name]['horizons'][horizon] = {
                                'model_type': metadata.get('model_type', 'ensemble'),
                                'training_metrics': metadata.get('metrics', {}),
                                'training_time': metadata.get('training_time', 0),
                                'features_used': len(metadata.get('feature_importance', {})),
                                'training_timestamp': metadata.get('training_timestamp', 'unknown'),
                                'has_scaler': scaler is not None
                            }
                            
                            self.logger.info(f"Loaded {horizon} model for {currency_name}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to load {horizon} model for {currency_name}: {str(e)}")
                
                else:
                    # Traditional single model directory
                    currency = item_name
                    
                    # Check for model files
                    model_file = item / "ensemble_model.pkl"
                    metadata_file = item / "model_metadata.json"
                    
                    if model_file.exists():
                        try:
                            # Load metadata
                            metadata = {}
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                            
                            # Load model
                            model = joblib.load(model_file)
                            
                            # Load scaler if exists
                            scaler_file = item / "scaler.pkl"
                            scaler = None
                            if scaler_file.exists():
                                scaler = joblib.load(scaler_file)
                            
                            # Store model and metadata
                            self.loaded_models[currency] = model
                            self.model_metadata[currency] = metadata
                            self.feature_scalers[currency] = scaler
                            
                            available_models[currency] = {
                                'model_type': metadata.get('model_type', 'unknown'),
                                'training_metrics': metadata.get('metrics', {}),
                                'training_time': metadata.get('training_time', 0),
                                'features_used': len(metadata.get('feature_importance', {})),
                                'training_timestamp': metadata.get('training_timestamp', 'unknown'),
                                'has_scaler': scaler is not None,
                                'has_multi_horizon': False
                            }
                            
                            self.logger.info(f"Loaded single model for {currency}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to load model for {currency}: {str(e)}")
        
        # Log summary
        multi_horizon_count = sum(1 for model in available_models.values() if model.get('has_multi_horizon', False))
        single_model_count = len(available_models) - multi_horizon_count
        
        self.logger.info(f"Loaded {len(available_models)} currencies: {multi_horizon_count} multi-horizon, {single_model_count} single models")
        return available_models
    
    def get_current_league_data(
        self,
        currencies: Optional[List[str]] = None,
        days_back: int = 30,
        use_live_data: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get current league data for prediction.
        
        Args:
            currencies: Specific currencies to get data for
            days_back: Number of days of historical data to include
            use_live_data: Whether to prioritize live POE Watch data
            
        Returns:
            DataFrame with current league data or None if failed
        """
        # Check cache
        if (self._current_league_data is not None and 
            self._last_data_update is not None and
            datetime.now() - self._last_data_update < self._data_cache_duration):
            
            if currencies:
                # Filter for specific currencies
                filtered_data = self._current_league_data[
                    self._current_league_data['currency'].isin(currencies)
                ]
                return filtered_data if not filtered_data.empty else None
            return self._current_league_data
        
        try:
            conn = get_db_connection()
            
            # Try to get live data first if enabled
            if use_live_data:
                live_data = self._get_live_poe_watch_data(conn, currencies, days_back)
                if live_data is not None and not live_data.empty:
                    self.logger.info("Using live POE Watch data for predictions")
                    self._current_league_data = live_data
                    self._last_data_update = datetime.now()
                    conn.close()
                    return live_data
                else:
                    self.logger.info("Live data not available, falling back to historical data")
            
            # Fallback to historical data
            historical_data = self._get_historical_data(conn, currencies, days_back)
            conn.close()
            
            if historical_data is not None and not historical_data.empty:
                self._current_league_data = historical_data
                self._last_data_update = datetime.now()
                return historical_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get current league data: {str(e)}")
            return None
    
    def _get_live_poe_watch_data(
        self,
        conn,
        currencies: Optional[List[str]] = None,
        days_back: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get live data from POE Watch ingestion table.
        
        Args:
            conn: Database connection
            currencies: Specific currencies to get data for
            days_back: Number of days of historical data to include
            
        Returns:
            DataFrame with live data or None if failed
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query live POE Watch data table with price history
            live_query = """
            SELECT 
                currency_name,
                league,
                mean_price,
                min_price,
                max_price,
                daily_volume,
                price_change_percent,
                price_history,
                CASE WHEN low_confidence THEN 0.5 ELSE 0.8 END as confidence_level,
                fetch_time
            FROM live_poe_watch 
            WHERE fetch_time >= %s
                AND mean_price > 0
                AND league = 'Mercenaries'  -- Filter for current league
            ORDER BY fetch_time DESC
            """
            
            df = pd.read_sql(live_query, conn, params=[cutoff_date])
            
            if df.empty:
                return None
            
            # Expand price history data to create individual price records
            expanded_records = []
            
            for _, row in df.iterrows():
                try:
                    # Parse price history JSON
                    import json
                    if row['price_history'] and row['price_history'] != 'null':
                        price_history = json.loads(row['price_history'])
                        
                        # Extract historical prices (assuming format like [{'date': '...', 'price': ...}, ...])
                        if isinstance(price_history, list):
                            for entry in price_history:
                                if isinstance(entry, dict) and 'date' in entry and 'price' in entry:
                                    expanded_records.append({
                                        'currency_name': row['currency_name'],
                                        'currency': row['currency_name'],  # For compatibility
                                        'league': row['league'],
                                        'league_name': row['league'],
                                        'price': float(entry['price']),
                                        'chaos_equivalent': float(entry['price']),
                                        'date': pd.to_datetime(entry['date']),
                                        'confidence_level': row['confidence_level'],
                                        'total_change': row['price_change_percent'],
                                        'listing_count': row['daily_volume'],
                                        'get_currency': row['currency_name'],
                                        'pay_currency': 'Chaos Orb'
                                    })
                        elif isinstance(price_history, dict):
                            # Handle different price history format
                            for date_str, price_val in price_history.items():
                                try:
                                    expanded_records.append({
                                        'currency_name': row['currency_name'],
                                        'currency': row['currency_name'],
                                        'league': row['league'],
                                        'league_name': row['league'],
                                        'price': float(price_val),
                                        'chaos_equivalent': float(price_val),
                                        'date': pd.to_datetime(date_str),
                                        'confidence_level': row['confidence_level'],
                                        'total_change': row['price_change_percent'],
                                        'listing_count': row['daily_volume'],
                                        'get_currency': row['currency_name'],
                                        'pay_currency': 'Chaos Orb'
                                    })
                                except (ValueError, TypeError):
                                    continue
                    
                    # Always add the current record as well
                    expanded_records.append({
                        'currency_name': row['currency_name'],
                        'currency': row['currency_name'],
                        'league': row['league'],
                        'league_name': row['league'],
                        'price': float(row['mean_price']),
                        'chaos_equivalent': float(row['mean_price']),
                        'date': pd.to_datetime(row['fetch_time']),
                        'confidence_level': row['confidence_level'],
                        'total_change': row['price_change_percent'],
                        'listing_count': row['daily_volume'],
                        'get_currency': row['currency_name'],
                        'pay_currency': 'Chaos Orb'
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error processing price history for {row['currency_name']}: {e}")
                    # Fallback to current price only
                    expanded_records.append({
                        'currency_name': row['currency_name'],
                        'currency': row['currency_name'],
                        'league': row['league'],
                        'league_name': row['league'],
                        'price': float(row['mean_price']),
                        'chaos_equivalent': float(row['mean_price']),
                        'date': pd.to_datetime(row['fetch_time']),
                        'confidence_level': row['confidence_level'],
                        'total_change': row['price_change_percent'],
                        'listing_count': row['daily_volume'],
                        'get_currency': row['currency_name'],
                        'pay_currency': 'Chaos Orb'
                    })
            
            if not expanded_records:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(expanded_records)
            
            # Remove duplicates based on currency and date
            df = df.drop_duplicates(subset=['currency', 'date'], keep='last')
            
            # Sort by currency and date
            df = df.sort_values(['currency', 'date']).reset_index(drop=True)
            
            # Add required columns for training compatibility
            df['league_active'] = True
            df['league_start'] = pd.to_datetime('2025-06-13 15:00:00+00:00')  # Mercenaries start date
            df['league_day'] = (df['date'] - df['league_start']).dt.days
            
            # Add derived columns for compatibility
            df['id'] = range(len(df))
            df['leagueId'] = 1  # Default league ID
            df['getCurrencyId'] = 1  # Default currency ID  
            df['payCurrencyId'] = 2  # Default pay currency ID
            
            # Filter for specific currencies if requested
            if currencies:
                df = df[df['currency'].isin(currencies)]
            
            self.logger.info(f"Loaded {len(df)} live records for {df['currency'].nunique()} currencies")
            return df if not df.empty else None
            
        except Exception as e:
            self.logger.error(f"Failed to get live POE Watch data: {str(e)}")
            return None
    
    def _get_historical_data(
        self,
        conn,
        currencies: Optional[List[str]] = None,
        days_back: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data from the main currency_prices table.
        
        Args:
            conn: Database connection
            currencies: Specific currencies to get data for
            days_back: Number of days of historical data to include
            
        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            # Get current active league
            league_query = """
            SELECT id, name, "startDate", "endDate" 
            FROM leagues 
            WHERE "isActive" = true 
            ORDER BY "startDate" DESC 
            LIMIT 1
            """
            
            league_df = pd.read_sql(league_query, conn)
            if league_df.empty:
                self.logger.warning("No active league found")
                return None
            
            current_league = league_df.iloc[0]
            league_id = int(current_league['id'])  # Convert to Python int to avoid numpy.int64 issues
            league_name = current_league['name']
            
            self.logger.info(f"Getting historical data for league: {league_name}")
            
            # Get recent price data - focus on currencies priced in Chaos Orbs
            # Use more recent data (last 14 days) for better relevance
            cutoff_date = datetime.now() - timedelta(days=min(days_back, 14))
            
            data_query = """
            SELECT 
                cp.id,
                cp."leagueId",
                cp."getCurrencyId", 
                cp."payCurrencyId",
                cp.date AT TIME ZONE 'UTC' as date,
                cp.value as price,
                cp.value as chaos_equivalent,  -- Add for compatibility
                l.name as league_name,
                l."startDate" AT TIME ZONE 'UTC' as league_start,
                l."endDate" AT TIME ZONE 'UTC' as league_end,
                l."isActive" as league_active,
                gc.name as get_currency,
                pc.name as pay_currency,
                gc.name as currency,
                EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) as league_day,
                -- Add default values for compatibility
                1.0 as confidence_level,
                0.0 as total_change,
                0 as listing_count
            FROM currency_prices cp
            JOIN leagues l ON cp."leagueId" = l.id
            JOIN currency gc ON cp."getCurrencyId" = gc.id
            JOIN currency pc ON cp."payCurrencyId" = pc.id
            WHERE cp."leagueId" = %s
                AND cp.value > 0
                AND cp.date >= %s
                AND pc.name = 'Chaos Orb'  -- Focus on currencies priced in Chaos Orbs
            ORDER BY cp.date DESC
            """
            
            df = pd.read_sql(data_query, conn, params=[league_id, cutoff_date])
            
            if df.empty:
                self.logger.warning("No recent historical data found")
                return None
            
            # Basic preprocessing
            df['date'] = pd.to_datetime(df['date'])
            df['league_start'] = pd.to_datetime(df['league_start'])
            df['league_day'] = (df['date'] - df['league_start']).dt.days
            
            self.logger.info(f"Loaded {len(df)} historical records for {df['currency'].nunique()} currencies")
            
            # Filter for specific currencies if requested
            if currencies:
                df = df[df['currency'].isin(currencies)]
                return df if not df.empty else None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {str(e)}")
            return None
    
    def prepare_features_for_prediction(
        self,
        currency: str,
        raw_data: pd.DataFrame,
        prediction_horizon_days: int = 1
    ) -> Optional[np.ndarray]:
        """
        Prepare features for prediction matching training data format.
        
        Args:
            currency: Currency to predict
            raw_data: Raw current league data
            
        Returns:
            Feature matrix ready for prediction or None if failed
        """
        try:
            # Filter data for this currency
            currency_data = raw_data[raw_data['currency'] == currency].copy()
            
            if currency_data.empty:
                self.logger.warning(f"No data found for {currency}")
                return None
            
            # Sort by date to ensure proper time series order
            currency_data = currency_data.sort_values('date').reset_index(drop=True)
            
            # Apply feature engineering (same as training)
            processed_data, _ = self.data_processor.process_currency_data(
                currency_data, currency
            )
            
            if processed_data is None or processed_data.empty:
                self.logger.warning(f"Feature engineering failed for {currency}")
                return None
            
            # Get feature columns using EXACT same logic as training
            exclude_patterns = [
                'target_', 'date', 'league_name', 'currency', 'id', 'league_start', 'league_end',
                'league_active', 'get_currency', 'pay_currency', 'getCurrencyId', 'payCurrencyId'
            ]
            
            feature_cols = [col for col in processed_data.columns 
                           if not any(pattern in col for pattern in exclude_patterns)]
            
            # Filter to only include numeric columns (EXACT same logic as training)
            numeric_feature_cols = []
            for col in feature_cols:
                if processed_data[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'float16']:
                    numeric_feature_cols.append(col)
                else:
                    self.logger.debug(f"Excluding non-numeric column: {col} (dtype: {processed_data[col].dtype})")
            
            feature_cols = numeric_feature_cols
            
            if not feature_cols:
                self.logger.warning(f"No numeric feature columns found for {currency}")
                return None
            
            # Extract features
            X = processed_data[feature_cols].values
            
            # Handle missing values (same strategy as training)
            from sklearn.impute import SimpleImputer
            if np.any(pd.isna(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Check if we need to align features with training data
            model_key = f"{currency}_{prediction_horizon_days}d"
            model = self.loaded_models.get(model_key)
            if model is None:
                # Fallback to base currency name for single-horizon models
                model = self.loaded_models.get(currency)
                
            if model and hasattr(model, 'models') and len(model.models) > 0:
                # For ensemble models, check the first model's expected features
                first_model = model.models[0]
                if hasattr(first_model, 'model') and hasattr(first_model.model, 'n_features_in_'):
                    expected_features = first_model.model.n_features_in_
                    current_features = X.shape[1]
                    
                    if current_features != expected_features:
                        self.logger.warning(f"Feature alignment needed for {currency}: got {current_features}, expected {expected_features}")
                        
                        if current_features < expected_features:
                            # Pad with feature means to maintain distributions
                            if X.shape[0] > 0:
                                feature_means = np.mean(X, axis=0)
                                overall_mean = np.mean(feature_means) if len(feature_means) > 0 else 0.0
                            else:
                                overall_mean = 0.0
                            padding = np.full((X.shape[0], expected_features - current_features), overall_mean)
                            X = np.concatenate([X, padding], axis=1)
                            self.logger.info(f"Padded {currency} features: {current_features} → {expected_features}")
                        elif current_features > expected_features:
                            # Truncate to expected feature count (keep most important/first features)
                            X = X[:, :expected_features]
                            self.logger.info(f"Truncated {currency} features: {current_features} → {expected_features}")
                    else:
                        self.logger.debug(f"Feature count matches for {currency}: {current_features} features")
            
            # Apply scaling if available
            if currency in self.feature_scalers and self.feature_scalers[currency] is not None:
                X = self.feature_scalers[currency].transform(X)
            
            self.logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features for {currency}")
            return X
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features for {currency}: {str(e)}")
            return None
    
    def calculate_prediction_intervals(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_new: np.ndarray,
        prediction: float,
        currency: str,
        alpha: float = 0.05
    ) -> Tuple[float, float, float, str, Dict[str, float]]:
        """
        Calculate prediction intervals using multiple methods based on data quality.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_new: New data point for prediction
            prediction: Point prediction
            currency: Currency name for logging
            alpha: Significance level (0.05 for 95% intervals)
            
        Returns:
            Tuple of (lower_bound, upper_bound, confidence_score, method, uncertainty_components)
        """
        n_samples = len(y_train)
        n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
        
        try:
            # Method selection based on data quality
            if n_samples < 5:
                # Very small dataset - use conservative fallback
                method = "conservative_fallback"
                margin = abs(prediction) * 0.5  # 50% margin
                lower_bound = prediction - margin
                upper_bound = prediction + margin
                confidence_score = 0.2  # Very low confidence
                uncertainty_components = {
                    'method': 'fallback',
                    'sample_size_penalty': 0.8,
                    'margin_percent': 50.0
                }
                
            elif n_samples < 15:
                # Small dataset - use bootstrap method
                method = "bootstrap"
                lower_bound, upper_bound, confidence_score, uncertainty_components = self._bootstrap_intervals(
                    model, X_train, y_train, X_new, prediction, alpha, n_bootstrap=min(50, n_samples*3)
                )
                
            else:
                # Sufficient data - use residual-based method
                method = "residual_based"
                lower_bound, upper_bound, confidence_score, uncertainty_components = self._residual_intervals(
                    model, X_train, y_train, X_new, prediction, alpha
                )
            
            # Apply data quality penalties
            quality_penalty = self._calculate_quality_penalty(n_samples, n_features)
            confidence_score = confidence_score * (1 - quality_penalty)
            confidence_score = max(0.05, min(0.95, confidence_score))  # Clamp between 5% and 95%
            
            # Ensure intervals are reasonable
            interval_width = upper_bound - lower_bound
            if interval_width <= 0:
                # Fix invalid intervals
                margin = abs(prediction) * 0.2
                lower_bound = prediction - margin
                upper_bound = prediction + margin
                interval_width = upper_bound - lower_bound
                method += "_corrected"
            
            uncertainty_components.update({
                'quality_penalty': quality_penalty,
                'final_confidence': confidence_score,
                'interval_width': interval_width,
                'sample_size': n_samples,
                'n_features': n_features
            })
            
            self.logger.debug(f"{currency}: {method} intervals [{lower_bound:.2f}, {upper_bound:.2f}], confidence: {confidence_score:.3f}")
            
            return lower_bound, upper_bound, confidence_score, method, uncertainty_components
            
        except Exception as e:
            self.logger.error(f"Prediction interval calculation failed for {currency}: {e}")
            # Emergency fallback
            margin = abs(prediction) * 0.3
            return (
                prediction - margin,
                prediction + margin,
                0.3,
                "error_fallback",
                {'error': str(e), 'margin_percent': 30.0}
            )
    
    def _bootstrap_intervals(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_new: np.ndarray,
        prediction: float,
        alpha: float,
        n_bootstrap: int = 50
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Calculate prediction intervals using bootstrap method."""
        
        bootstrap_predictions = []
        
        try:
            for i in range(n_bootstrap):
                # Resample training data
                indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices]
                
                # Clone and train model (simplified - works for sklearn models)
                try:
                    from sklearn.base import clone
                    model_boot = clone(model)
                except:
                    # Fallback for non-sklearn models
                    model_boot = model
                
                model_boot.fit(X_boot, y_boot)
                
                # Make prediction
                pred_boot = model_boot.predict(X_new.reshape(1, -1))[0]
                bootstrap_predictions.append(pred_boot)
            
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate intervals using percentiles
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_bound = np.percentile(bootstrap_predictions, lower_percentile)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile)
            
            # Confidence based on prediction stability
            pred_std = np.std(bootstrap_predictions)
            pred_mean = np.mean(bootstrap_predictions)
            relative_std = pred_std / (abs(pred_mean) + 1e-8)
            confidence_score = 1 / (1 + relative_std)
            
            uncertainty_components = {
                'bootstrap_std': float(pred_std),
                'bootstrap_mean': float(pred_mean),
                'relative_std': float(relative_std),
                'n_bootstrap': n_bootstrap
            }
            
            return lower_bound, upper_bound, confidence_score, uncertainty_components
            
        except Exception as e:
            # Fallback to simple margin
            margin = abs(prediction) * 0.25
            return (
                prediction - margin,
                prediction + margin,
                0.4,
                {'error': str(e), 'fallback_margin': 0.25}
            )
    
    def _residual_intervals(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_new: np.ndarray,
        prediction: float,
        alpha: float
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Calculate prediction intervals using residual analysis."""
        
        try:
            # Get training predictions and residuals
            y_pred_train = model.predict(X_train)
            residuals = y_train - y_pred_train
            
            # Calculate residual statistics
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            
            # Degrees of freedom
            n = len(y_train)
            p = X_train.shape[1] if len(X_train.shape) > 1 else 1
            df = max(1, n - p - 1)
            
            # T-distribution critical value
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # Calculate prediction interval
            # For simplicity, we use constant variance assumption
            margin_of_error = t_critical * residual_std
            lower_bound = prediction - margin_of_error
            upper_bound = prediction + margin_of_error
            
            # Confidence based on residual quality and sample size
            cv_residuals = residual_std / (abs(np.mean(y_train)) + 1e-8)
            sample_size_factor = min(1.0, n / 50.0)  # Discount for small samples
            confidence_score = sample_size_factor / (1 + cv_residuals)
            
            uncertainty_components = {
                'residual_std': float(residual_std),
                'residual_mean': float(residual_mean),
                'residual_cv': float(cv_residuals),
                'sample_size_factor': float(sample_size_factor),
                't_critical': float(t_critical),
                'degrees_freedom': int(df)
            }
            
            return lower_bound, upper_bound, confidence_score, uncertainty_components
            
        except Exception as e:
            # Fallback
            margin = abs(prediction) * 0.2
            return (
                prediction - margin,
                prediction + margin,
                0.5,
                {'error': str(e), 'fallback_margin': 0.2}
            )
    
    def _calculate_quality_penalty(self, n_samples: int, n_features: int) -> float:
        """Calculate penalty based on data quality indicators."""
        
        penalties = []
        
        # Sample size penalty
        if n_samples < 5:
            penalties.append(0.7)  # Very high penalty
        elif n_samples < 10:
            penalties.append(0.5)  # High penalty
        elif n_samples < 30:
            penalties.append(0.3)  # Moderate penalty
        elif n_samples < 100:
            penalties.append(0.1)  # Small penalty
        else:
            penalties.append(0.0)  # No penalty
        
        # Curse of dimensionality penalty
        if n_features > n_samples:
            penalties.append(0.6)  # Very high penalty
        elif n_features > n_samples * 0.5:
            penalties.append(0.4)  # High penalty
        elif n_features > n_samples * 0.2:
            penalties.append(0.2)  # Moderate penalty
        else:
            penalties.append(0.0)  # No penalty
        
        # Return maximum penalty (most conservative)
        return max(penalties)
    
    def predict_price(
        self,
        currency: str,
        prediction_horizon_days: int = 1
    ) -> Optional[PredictionResult]:
        """
        Predict future price for a currency using multi-horizon models.
        
        Args:
            currency: Currency to predict (e.g., "Divine Orb")
            prediction_horizon_days: Number of days ahead to predict
            
        Returns:
            Prediction result or None if failed
        """
        # Map prediction horizon to model suffix
        horizon_map = {1: '1d', 3: '3d', 7: '7d'}
        
        # Try to find the appropriate model
        model_key = None
        model = None
        metadata = {}
        
        # First, try to find horizon-specific model
        if prediction_horizon_days in horizon_map:
            horizon_suffix = horizon_map[prediction_horizon_days]
            model_key = f"{currency}_{horizon_suffix}"
            
            if model_key in self.loaded_models:
                model = self.loaded_models[model_key]
                metadata = self.model_metadata.get(model_key, {})
                self.logger.info(f"Using {horizon_suffix} model for {currency}")
            else:
                self.logger.warning(f"No {horizon_suffix} model found for {currency}")
        
        # Fallback to single model if no horizon-specific model found
        if model is None and currency in self.loaded_models:
            model_key = currency
            model = self.loaded_models[currency]
            metadata = self.model_metadata.get(currency, {})
            self.logger.info(f"Using single model for {currency} (requested {prediction_horizon_days}d)")
        
        # If still no model found, try to find any available horizon for this currency
        if model is None:
            available_horizons = []
            for key in self.loaded_models.keys():
                if key.startswith(f"{currency}_") and key.split('_')[-1] in ['1d', '3d', '7d']:
                    available_horizons.append(key)
            
            if available_horizons:
                # Use the closest available horizon
                requested_days = prediction_horizon_days
                best_match = None
                min_diff = float('inf')
                
                for horizon_key in available_horizons:
                    horizon_suffix = horizon_key.split('_')[-1]
                    horizon_days = {'1d': 1, '3d': 3, '7d': 7}.get(horizon_suffix, 999)
                    diff = abs(horizon_days - requested_days)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_match = horizon_key
                
                if best_match:
                    model_key = best_match
                    model = self.loaded_models[best_match]
                    metadata = self.model_metadata.get(best_match, {})
                    horizon_used = best_match.split('_')[-1]
                    self.logger.info(f"Using closest available {horizon_used} model for {currency} (requested {prediction_horizon_days}d)")
        
        if model is None:
            self.logger.error(f"No model available for {currency}")
            return None
        
        try:
            # Get current data - use cached data if available, otherwise load for this currency
            if self._current_league_data is not None and not self._current_league_data.empty:
                # Use cached data and filter for this currency
                current_data = self._current_league_data[self._current_league_data['currency'] == currency].copy()
                if current_data.empty:
                    # Currency not in cached data, load specifically for this currency
                    current_data = self.get_current_league_data([currency], days_back=30)
            else:
                # No cached data, load for this currency
                current_data = self.get_current_league_data([currency], days_back=30)
            
            if current_data is None or current_data.empty:
                self.logger.error(f"No current data available for {currency}")
                return None
            
            # Get current price (most recent)
            latest_data = current_data.sort_values('date').iloc[-1]
            current_price = float(latest_data['price'])
            
            # Prepare features
            X = self.prepare_features_for_prediction(currency, current_data, prediction_horizon_days)
            if X is None:
                return None
            
            # Get model metadata
            model_type = metadata.get('model_type', 'unknown')
            
            # Make prediction
            if len(X) > 0:
                # Use the most recent data point(s) for prediction
                if hasattr(model, 'predict'):
                    prediction = model.predict(X[-1:])  # Use last data point
                    if isinstance(prediction, np.ndarray):
                        predicted_price = float(prediction[0])
                    else:
                        predicted_price = float(prediction)
                else:
                    self.logger.error(f"Model for {currency} doesn't have predict method")
                    return None
            else:
                self.logger.error(f"No data available for prediction")
                return None
            
            # Calculate metrics
            price_change_percent = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate prediction intervals and proper confidence
            try:
                # We need training data for proper intervals - use current data as proxy
                # In production, this should use cached training data
                X_train = X[:-1] if len(X) > 1 else X  # Use all but last point as "training"
                y_train = np.array([current_price] * len(X_train))  # Simplified - should be actual training targets
                X_new = X[-1:][0] if len(X) > 0 else X[0]  # Last point for prediction
                
                # Calculate prediction intervals
                lower_bound, upper_bound, confidence_score, method, uncertainty_components = self.calculate_prediction_intervals(
                    model, X_train, y_train, X_new, predicted_price, currency
                )
                
                interval_width = upper_bound - lower_bound
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate prediction intervals for {currency}: {e}")
                # Fallback to simple confidence calculation
                training_metrics = metadata.get('metrics', {})
                r2_score = training_metrics.get('r2', 0)
                confidence_score = max(0, min(1, (r2_score + 1) / 2))  # Convert R² to 0-1 scale
                
                # Simple interval based on prediction magnitude
                margin = abs(predicted_price) * 0.2
                lower_bound = predicted_price - margin
                upper_bound = predicted_price + margin
                interval_width = upper_bound - lower_bound
                method = "fallback_r2"
                uncertainty_components = {
                    'r2_score': r2_score,
                    'fallback_margin': 0.2,
                    'error': str(e)
                }
            
            result = PredictionResult(
                currency=currency,
                current_price=current_price,
                predicted_price=predicted_price,
                prediction_horizon_days=prediction_horizon_days,
                confidence_score=confidence_score,
                price_change_percent=price_change_percent,
                prediction_timestamp=datetime.now().isoformat(),
                model_type=f"{model_type}_{model_key.split('_')[-1] if '_' in model_key else 'single'}",
                features_used=X.shape[1] if len(X.shape) > 1 else 0,
                data_points_used=len(X),
                prediction_lower=lower_bound,
                prediction_upper=upper_bound,
                interval_width=interval_width,
                confidence_method=method,
                uncertainty_components=uncertainty_components
            )
            
            self.logger.info(f"Prediction for {currency} ({prediction_horizon_days}d): {current_price:.2f} -> {predicted_price:.2f} ({price_change_percent:+.1f}%)")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {currency}: {str(e)}")
            return None
    
    def predict_multiple_currencies(
        self,
        currencies: Optional[List[str]] = None,
        prediction_horizon_days: int = 1
    ) -> List[PredictionResult]:
        """
        Predict prices for multiple currencies using multi-horizon models.
        
        Args:
            currencies: List of currencies to predict (None for all loaded models)
            prediction_horizon_days: Number of days ahead to predict
            
        Returns:
            List of prediction results
        """
        if currencies is None:
            # Extract unique currency names from loaded models
            unique_currencies = set()
            
            for model_key in self.loaded_models.keys():
                if '_' in model_key and model_key.split('_')[-1] in ['1d', '3d', '7d']:
                    # Multi-horizon model: extract currency name
                    currency_name = '_'.join(model_key.split('_')[:-1])
                    unique_currencies.add(currency_name)
                else:
                    # Single model: use as-is
                    unique_currencies.add(model_key)
            
            currencies = list(unique_currencies)
            self.logger.info(f"Predicting for {len(currencies)} unique currencies")
        
        # Pre-load data for all currencies to improve efficiency
        try:
            self.logger.info(f"Pre-loading data for {len(currencies)} currencies")
            all_data = self.get_current_league_data(currencies, days_back=30)
            if all_data is not None and not all_data.empty:
                self.logger.info(f"Loaded data for currencies: {all_data['currency'].unique()}")
        except Exception as e:
            self.logger.warning(f"Failed to pre-load data: {e}")
        
        predictions = []
        for currency in currencies:
            try:
                result = self.predict_price(currency, prediction_horizon_days)
                if result:
                    predictions.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict {currency}: {str(e)}")
                continue
        
        self.logger.info(f"Generated {len(predictions)} predictions for {prediction_horizon_days}d horizon")
        return predictions
    
    def get_top_predictions(
        self,
        top_n: int = None,
        sort_by: str = 'price_change_percent',
        ascending: bool = False
    ) -> List[PredictionResult]:
        """
        Get top predictions sorted by specified criteria.
        
        Args:
            top_n: Number of top predictions to return (None for all)
            sort_by: Field to sort by
            ascending: Sort order
            
        Returns:
            List of top prediction results
        """
        # Get predictions for all loaded models
        all_predictions = self.predict_multiple_currencies()
        
        # Sort by specified criteria
        if sort_by == 'price_change_percent':
            all_predictions.sort(key=lambda x: x.price_change_percent, reverse=not ascending)
        elif sort_by == 'confidence_score':
            all_predictions.sort(key=lambda x: x.confidence_score, reverse=not ascending)
        elif sort_by == 'predicted_price':
            all_predictions.sort(key=lambda x: x.predicted_price, reverse=not ascending)
        elif sort_by == 'current_price':
            all_predictions.sort(key=lambda x: x.current_price, reverse=not ascending)
        
        return all_predictions[:top_n] if top_n is not None else all_predictions
    
    def export_predictions(
        self,
        predictions: List[PredictionResult],
        output_file: str
    ) -> None:
        """
        Export predictions to JSON file.
        
        Args:
            predictions: List of prediction results
            output_file: Output file path
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_predictions': len(predictions),
                'predictions': [pred.to_dict() for pred in predictions]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(predictions)} predictions to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export predictions: {str(e)}")


def main():
    """
    Main function for command-line usage.
    Example: python model_inference.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Currency Price Prediction')
    parser.add_argument('--models-dir', default='models/currency_production', 
                       help='Directory containing trained models')
    parser.add_argument('--currency', help='Specific currency to predict')
    parser.add_argument('--horizon', type=int, default=1, 
                       help='Prediction horizon in days')
    parser.add_argument('--top-n', type=int, default=None, 
                       help='Number of top predictions to show (default: all)')
    parser.add_argument('--output', help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ModelPredictor(args.models_dir)
    
    # Load models
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found! Please train models first.")
        return
    
    print(f"Loaded {len(available_models)} models")
    
    # Make predictions
    if args.currency:
        # Single currency prediction
        result = predictor.predict_price(args.currency, args.horizon)
        if result:
            print(f"\nPrediction for {args.currency}:")
            print(f"Current Price: {result.current_price:.2f}")
            print(f"Predicted Price: {result.predicted_price:.2f}")
            print(f"Expected Change: {result.price_change_percent:+.1f}%")
            print(f"Confidence: {result.confidence_score:.2f}")
        else:
            print(f"Prediction failed for {args.currency}")
    else:
        # Multiple currency predictions
        predictions = predictor.get_top_predictions(args.top_n)
        
        print(f"\nAll {len(predictions)} Currency Predictions:")
        print("-" * 80)
        for pred in predictions:
            print(f"{pred.currency:30} | {pred.current_price:8.2f} -> {pred.predicted_price:8.2f} | {pred.price_change_percent:+6.1f}% | {pred.confidence_score:.2f}")
    
    # Export if requested
    if args.output:
        if args.currency:
            predictions = [result] if result else []
        else:
            predictions = predictor.predict_multiple_currencies()
        
        predictor.export_predictions(predictions, args.output)


if __name__ == "__main__":
    main() 