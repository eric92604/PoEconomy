"""
Model inference utilities for price prediction in current leagues.

This module provides functionality to load trained models and make predictions
on current league data, handling feature engineering and LSTM sequence requirements.
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import MLConfig
from utils.logging_utils import MLLogger
from utils.data_processing import DataProcessor
from utils.feature_engineering import FeatureEngineer
from utils.database import get_db_connection
from utils.model_training import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Container for prediction results."""
    currency_pair: str
    current_price: float
    predicted_price: float
    prediction_horizon_days: int
    confidence_score: float
    price_change_percent: float
    prediction_timestamp: str
    model_type: str
    features_used: int
    data_points_used: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'currency_pair': self.currency_pair,
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'prediction_horizon_days': self.prediction_horizon_days,
            'confidence_score': self.confidence_score,
            'price_change_percent': self.price_change_percent,
            'prediction_timestamp': self.prediction_timestamp,
            'model_type': self.model_type,
            'features_used': self.features_used,
            'data_points_used': self.data_points_used
        }


class ModelPredictor:
    """
    Production model inference system for currency price prediction.
    
    This class handles loading trained models, processing current league data,
    and making predictions while handling LSTM sequence requirements.
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
        Load all available trained models.
        
        Returns:
            Dictionary mapping currency pairs to model information
        """
        available_models = {}
        
        if not self.models_dir.exists():
            self.logger.warning(f"Models directory not found: {self.models_dir}")
            return available_models
        
        # Find all currency pair directories
        for currency_dir in self.models_dir.iterdir():
            if currency_dir.is_dir():
                currency_pair = currency_dir.name.replace('_', ' ')
                
                # Check for model files
                model_file = currency_dir / "ensemble_model.pkl"
                metadata_file = currency_dir / "model_metadata.json"
                
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
                        scaler_file = currency_dir / "scaler.pkl"
                        scaler = None
                        if scaler_file.exists():
                            scaler = joblib.load(scaler_file)
                        
                        # Store model and metadata
                        self.loaded_models[currency_pair] = model
                        self.model_metadata[currency_pair] = metadata
                        self.feature_scalers[currency_pair] = scaler
                        
                        available_models[currency_pair] = {
                            'model_type': metadata.get('model_type', 'unknown'),
                            'training_metrics': metadata.get('metrics', {}),
                            'training_time': metadata.get('training_time', 0),
                            'features_used': (metadata.get('feature_importance') or {}).get('feature_count', 0),
                            'training_timestamp': metadata.get('training_timestamp', 'unknown'),
                            'has_scaler': scaler is not None
                        }
                        
                        self.logger.info(f"Loaded model for {currency_pair}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load model for {currency_pair}: {str(e)}")
        
        self.logger.info(f"Loaded {len(available_models)} models successfully")
        return available_models
    
    def get_current_league_data(
        self,
        currency_pairs: Optional[List[str]] = None,
        days_back: int = 30,
        use_live_data: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get current league data for prediction.
        
        Args:
            currency_pairs: Specific currency pairs to get data for
            days_back: Number of days of historical data to include
            use_live_data: Whether to prioritize live poe.ninja data
            
        Returns:
            DataFrame with current league data or None if failed
        """
        # Check cache
        if (self._current_league_data is not None and 
            self._last_data_update is not None and
            datetime.now() - self._last_data_update < self._data_cache_duration):
            
            if currency_pairs:
                # Filter for specific pairs
                filtered_data = self._current_league_data[
                    self._current_league_data['currency_pair'].isin(currency_pairs)
                ]
                return filtered_data if not filtered_data.empty else None
            return self._current_league_data
        
        try:
            conn = get_db_connection()
            
            # Try to get live data first if enabled
            if use_live_data:
                live_data = self._get_live_ninja_data(conn, currency_pairs, days_back)
                if live_data is not None and not live_data.empty:
                    self.logger.info("Using live poe.ninja data for predictions")
                    self._current_league_data = live_data
                    self._last_data_update = datetime.now()
                    conn.close()
                    return live_data
                else:
                    self.logger.info("Live data not available, falling back to historical data")
            
            # Fallback to historical data
            historical_data = self._get_historical_data(conn, currency_pairs, days_back)
            conn.close()
            
            if historical_data is not None and not historical_data.empty:
                self._current_league_data = historical_data
                self._last_data_update = datetime.now()
                return historical_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get current league data: {str(e)}")
            return None
    
    def _get_live_ninja_data(
        self,
        conn,
        currency_pairs: Optional[List[str]] = None,
        days_back: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get live data from poe.ninja ingestion table.
        
        Args:
            conn: Database connection
            currency_pairs: Specific currency pairs to get data for
            days_back: Number of days of historical data to include
            
        Returns:
            DataFrame with live data or None if failed
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query live currency prices table
            live_query = """
            SELECT 
                currency_name,
                league,
                direction,
                value as price,
                chaos_equivalent,
                sample_time as date,
                confidence_level,
                total_change,
                listing_count,
                CASE 
                    WHEN direction = 'receive' THEN CONCAT(currency_name, ' -> Chaos Orb')
                    ELSE CONCAT('Chaos Orb -> ', currency_name)
                END as currency_pair
            FROM live_currency_prices 
            WHERE sample_time >= %s
                AND value > 0
                AND direction = 'receive'  -- Focus on chaos -> currency direction
            ORDER BY sample_time DESC
            """
            
            df = pd.read_sql(live_query, conn, params=[cutoff_date])
            
            if df.empty:
                return None
            
            # Convert to format compatible with prediction system
            df['date'] = pd.to_datetime(df['date'])
            df['league_name'] = df['league']
            df['get_currency'] = df['currency_name']
            df['pay_currency'] = 'Chaos Orb'
            df['league_active'] = True
            df['league_start'] = df['date'].min()  # Approximate
            df['league_day'] = (df['date'] - df['league_start']).dt.days
            
            # Add derived columns for compatibility
            df['id'] = range(len(df))
            df['leagueId'] = 1  # Default league ID
            df['getCurrencyId'] = 1  # Default currency ID
            df['payCurrencyId'] = 2  # Default pay currency ID
            
            # Filter for specific pairs if requested
            if currency_pairs:
                df = df[df['currency_pair'].isin(currency_pairs)]
            
            self.logger.info(f"Loaded {len(df)} live records for {df['currency_pair'].nunique()} currency pairs")
            return df if not df.empty else None
            
        except Exception as e:
            self.logger.error(f"Failed to get live ninja data: {str(e)}")
            return None
    
    def _get_historical_data(
        self,
        conn,
        currency_pairs: Optional[List[str]] = None,
        days_back: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data from the main currency_prices table.
        
        Args:
            conn: Database connection
            currency_pairs: Specific currency pairs to get data for
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
            league_id = current_league['id']
            league_name = current_league['name']
            
            self.logger.info(f"Getting historical data for league: {league_name}")
            
            # Get recent price data
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            data_query = """
            SELECT 
                cp.id,
                cp."leagueId",
                cp."getCurrencyId", 
                cp."payCurrencyId",
                cp.date AT TIME ZONE 'UTC' as date,
                cp.value as price,
                l.name as league_name,
                l."startDate" AT TIME ZONE 'UTC' as league_start,
                l."endDate" AT TIME ZONE 'UTC' as league_end,
                l."isActive" as league_active,
                gc.name as get_currency,
                pc.name as pay_currency,
                CONCAT(gc.name, ' -> ', pc.name) as currency_pair,
                EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) as league_day
            FROM currency_prices cp
            JOIN leagues l ON cp."leagueId" = l.id
            JOIN currency gc ON cp."getCurrencyId" = gc.id
            JOIN currency pc ON cp."payCurrencyId" = pc.id
            WHERE cp."leagueId" = %s
                AND cp.value > 0
                AND cp.date >= %s
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
            
            self.logger.info(f"Loaded {len(df)} historical records for {df['currency_pair'].nunique()} currency pairs")
            
            # Filter for specific pairs if requested
            if currency_pairs:
                df = df[df['currency_pair'].isin(currency_pairs)]
                return df if not df.empty else None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {str(e)}")
            return None
    
    def prepare_features_for_prediction(
        self,
        currency_pair: str,
        raw_data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        Prepare features for prediction matching training data format.
        
        Args:
            currency_pair: Currency pair to predict
            raw_data: Raw current league data
            
        Returns:
            Feature matrix ready for prediction or None if failed
        """
        try:
            # Filter data for this currency pair
            pair_data = raw_data[raw_data['currency_pair'] == currency_pair].copy()
            
            if pair_data.empty:
                self.logger.warning(f"No data found for {currency_pair}")
                return None
            
            # Sort by date to ensure proper time series order
            pair_data = pair_data.sort_values('date').reset_index(drop=True)
            
            # Apply feature engineering (same as training)
            processed_data, _ = self.data_processor.process_currency_data(
                pair_data, currency_pair
            )
            
            if processed_data is None or processed_data.empty:
                self.logger.warning(f"Feature engineering failed for {currency_pair}")
                return None
            
            # Get feature columns (exclude target and metadata columns)
            exclude_patterns = [
                'target_', 'date', 'league_name', 'currency_pair', 'id', 
                'league_start', 'league_end', 'league_active', 'get_currency', 
                'pay_currency', 'getCurrencyId', 'payCurrencyId'
            ]
            
            feature_cols = [col for col in processed_data.columns 
                           if not any(pattern in col for pattern in exclude_patterns)]
            
            if not feature_cols:
                self.logger.warning(f"No feature columns found for {currency_pair}")
                return None
            
            # Extract features
            X = processed_data[feature_cols].values
            
            # Handle missing values (same strategy as training)
            from sklearn.impute import SimpleImputer
            if np.any(pd.isna(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Apply scaling if scaler exists
            if currency_pair in self.feature_scalers and self.feature_scalers[currency_pair] is not None:
                X = self.feature_scalers[currency_pair].transform(X)
            
            self.logger.info(f"Prepared features for {currency_pair}: shape {X.shape}")
            return X
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features for {currency_pair}: {str(e)}")
            return None
    
    def predict_price(
        self,
        currency_pair: str,
        prediction_horizon_days: int = 1
    ) -> Optional[PredictionResult]:
        """
        Predict future price for a currency pair.
        
        Args:
            currency_pair: Currency pair to predict (e.g., "Divine Orb -> Chaos Orb")
            prediction_horizon_days: Number of days ahead to predict
            
        Returns:
            Prediction result or None if failed
        """
        if currency_pair not in self.loaded_models:
            self.logger.error(f"No model loaded for {currency_pair}")
            return None
        
        try:
            # Get current data
            current_data = self.get_current_league_data([currency_pair], days_back=30)
            if current_data is None or current_data.empty:
                self.logger.error(f"No current data available for {currency_pair}")
                return None
            
            # Get current price (most recent)
            latest_data = current_data.sort_values('date').iloc[-1]
            current_price = float(latest_data['price'])
            
            # Prepare features
            X = self.prepare_features_for_prediction(currency_pair, current_data)
            if X is None:
                return None
            
            # Get model and metadata
            model = self.loaded_models[currency_pair]
            metadata = self.model_metadata.get(currency_pair, {})
            model_type = metadata.get('model_type', 'unknown')
            
            # Handle LSTM sequence requirements
            if 'lstm' in model_type.lower() and hasattr(model, 'models'):
                # For ensemble with LSTM, we need enough data for sequences
                lstm_seq_length = 14  # Default sequence length
                if len(X) < lstm_seq_length:
                    self.logger.warning(f"Insufficient data for LSTM prediction: {len(X)} < {lstm_seq_length}")
                    # Use only the available data
                
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
                    self.logger.error(f"Model for {currency_pair} doesn't have predict method")
                    return None
            else:
                self.logger.error(f"No data available for prediction")
                return None
            
            # Calculate metrics
            price_change_percent = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate confidence score (simplified)
            training_metrics = metadata.get('metrics', {})
            r2_score = training_metrics.get('r2', 0)
            confidence_score = max(0, min(1, (r2_score + 1) / 2))  # Convert R² to 0-1 scale
            
            result = PredictionResult(
                currency_pair=currency_pair,
                current_price=current_price,
                predicted_price=predicted_price,
                prediction_horizon_days=prediction_horizon_days,
                confidence_score=confidence_score,
                price_change_percent=price_change_percent,
                prediction_timestamp=datetime.now().isoformat(),
                model_type=model_type,
                features_used=X.shape[1] if len(X.shape) > 1 else 0,
                data_points_used=len(X)
            )
            
            self.logger.info(f"Prediction for {currency_pair}: {current_price:.2f} -> {predicted_price:.2f} ({price_change_percent:+.1f}%)")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {currency_pair}: {str(e)}")
            return None
    
    def predict_multiple_currencies(
        self,
        currency_pairs: Optional[List[str]] = None,
        prediction_horizon_days: int = 1
    ) -> List[PredictionResult]:
        """
        Predict prices for multiple currency pairs.
        
        Args:
            currency_pairs: List of currency pairs to predict (None for all loaded models)
            prediction_horizon_days: Number of days ahead to predict
            
        Returns:
            List of prediction results
        """
        if currency_pairs is None:
            currency_pairs = list(self.loaded_models.keys())
        
        results = []
        for currency_pair in currency_pairs:
            result = self.predict_price(currency_pair, prediction_horizon_days)
            if result:
                results.append(result)
        
        return results
    
    def get_top_predictions(
        self,
        top_n: int = 10,
        sort_by: str = 'price_change_percent',
        ascending: bool = False
    ) -> List[PredictionResult]:
        """
        Get top N predictions sorted by specified criteria.
        
        Args:
            top_n: Number of top predictions to return
            sort_by: Field to sort by ('price_change_percent', 'confidence_score', etc.)
            ascending: Sort order
            
        Returns:
            List of top predictions
        """
        all_predictions = self.predict_multiple_currencies()
        
        if not all_predictions:
            return []
        
        # Sort predictions
        try:
            sorted_predictions = sorted(
                all_predictions,
                key=lambda x: getattr(x, sort_by),
                reverse=not ascending
            )
            return sorted_predictions[:top_n]
        except AttributeError:
            self.logger.error(f"Invalid sort field: {sort_by}")
            return all_predictions[:top_n]
    
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
    """Example usage of the model predictor."""
    # Initialize predictor
    models_dir = Path("models/currency_production_lstm")  # Adjust path as needed
    predictor = ModelPredictor(models_dir)
    
    print("Loading available models...")
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found!")
        return
    
    print(f"Loaded {len(available_models)} models:")
    for currency_pair, info in available_models.items():
        print(f"  - {currency_pair}: {info['model_type']} (R²: {info['training_metrics'].get('r2', 'N/A'):.3f})")
    
    # Make predictions
    print("\nGenerating predictions...")
    top_predictions = predictor.get_top_predictions(top_n=10, sort_by='price_change_percent')
    
    print(f"\nTop 10 Price Change Predictions:")
    print("-" * 80)
    for i, pred in enumerate(top_predictions, 1):
        print(f"{i:2d}. {pred.currency_pair:<35} {pred.current_price:>8.2f} -> {pred.predicted_price:>8.2f} "
              f"({pred.price_change_percent:>+6.1f}%) [Conf: {pred.confidence_score:.2f}]")
    
    # Export results
    output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    predictor.export_predictions(top_predictions, output_file)
    print(f"\nResults exported to: {output_file}")


if __name__ == "__main__":
    main() 