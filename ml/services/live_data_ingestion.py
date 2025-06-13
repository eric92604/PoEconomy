#!/usr/bin/env python3
"""
Live Data Ingestion Service

This service continuously fetches live currency data from poe.ninja API
and integrates it with the PoEconomy prediction system for real-time analysis.
"""

import sys
import asyncio
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.poe_ninja_client import PoENinjaClient, CurrencyData
from utils.model_inference import ModelPredictor
from utils.logging_utils import MLLogger
from config.training_config import MLConfig


class LiveDataIngestionService:
    """
    Service for continuous live data ingestion and real-time predictions.
    
    This service fetches data from poe.ninja, stores it to the database,
    and optionally generates real-time predictions.
    """
    
    def __init__(
        self,
        config: Optional[MLConfig] = None,
        logger: Optional[MLLogger] = None,
        fetch_interval: int = 300,  # 5 minutes
        prediction_interval: int = 1800,  # 30 minutes
        enable_predictions: bool = True
    ):
        """
        Initialize the live data ingestion service.
        
        Args:
            config: ML configuration
            logger: Logger instance
            fetch_interval: Data fetch interval in seconds
            prediction_interval: Prediction generation interval in seconds
            enable_predictions: Whether to generate predictions
        """
        self.config = config or MLConfig()
        self.logger = logger or MLLogger("LiveDataIngestion")
        self.fetch_interval = fetch_interval
        self.prediction_interval = prediction_interval
        self.enable_predictions = enable_predictions
        
        # Service state
        self.running = False
        self.last_fetch_time = None
        self.last_prediction_time = None
        
        # Components
        self.ninja_client = None
        self.predictor = None
        
        # Leagues to monitor
        self.monitored_leagues = ['Standard', 'Hardcore']
        
        # Price change alerts
        self.alert_thresholds = {
            'Mirror of Kalandra': 5.0,
            'Divine Orb': 10.0,
            'Exalted Orb': 15.0,
            'Mirror Shard': 10.0,
            'Eternal Orb': 20.0
        }
        
        # Statistics tracking
        self.stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'total_predictions': 0,
            'successful_predictions': 0,
            'currencies_tracked': 0,
            'alerts_generated': 0,
            'start_time': None,
            'last_error': None
        }
    
    async def initialize(self):
        """Initialize service components."""
        try:
            self.logger.info("Initializing Live Data Ingestion Service...")
            
            # Initialize PoE Ninja client
            self.ninja_client = PoENinjaClient(
                logger=self.logger,
                rate_limit_delay=1.0,
                timeout=30
            )
            
            # Initialize predictor if predictions are enabled
            if self.enable_predictions:
                try:
                    # Find latest model directory
                    models_base = Path("models")
                    if models_base.exists():
                        currency_dirs = [d for d in models_base.iterdir() 
                                       if d.is_dir() and d.name.startswith('currency_')]
                        if currency_dirs:
                            latest_dir = max(currency_dirs, key=lambda d: d.stat().st_mtime)
                            self.predictor = ModelPredictor(latest_dir, self.config, self.logger)
                            available_models = self.predictor.load_available_models()
                            self.logger.info(f"Loaded {len(available_models)} prediction models")
                        else:
                            self.logger.warning("No trained models found - predictions disabled")
                            self.enable_predictions = False
                    else:
                        self.logger.warning("Models directory not found - predictions disabled")
                        self.enable_predictions = False
                except Exception as e:
                    self.logger.error(f"Failed to initialize predictor: {str(e)}")
                    self.enable_predictions = False
            
            self.stats['start_time'] = datetime.now()
            self.logger.info("Service initialization completed")
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {str(e)}")
            raise
    
    async def fetch_and_store_data(self) -> bool:
        """
        Fetch live data from poe.ninja and store to database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.stats['total_fetches'] += 1
            self.logger.info("Fetching live currency data...")
            
            async with self.ninja_client:
                # Fetch data for all monitored leagues
                all_data = await self.ninja_client.fetch_all_leagues_data(self.monitored_leagues)
                
                if not all_data:
                    self.logger.warning("No data fetched from poe.ninja")
                    return False
                
                total_currencies = 0
                stored_successfully = True
                
                # Process each league
                for league, currency_data in all_data.items():
                    if currency_data:
                        self.logger.info(f"Processing {len(currency_data)} currencies for {league}")
                        
                        # Store to database
                        success = await self.ninja_client.store_to_database(
                            currency_data,
                            table_name='live_currency_prices'
                        )
                        
                        if success:
                            total_currencies += len(currency_data)
                            
                            # Check for significant price changes
                            await self.check_price_alerts(currency_data)
                        else:
                            stored_successfully = False
                            self.logger.error(f"Failed to store data for {league}")
                
                if stored_successfully:
                    self.stats['successful_fetches'] += 1
                    self.stats['currencies_tracked'] = total_currencies
                    self.last_fetch_time = datetime.now()
                    self.logger.info(f"Successfully processed {total_currencies} currency records")
                
                return stored_successfully
                
        except Exception as e:
            self.logger.error(f"Error in fetch_and_store_data: {str(e)}")
            self.stats['last_error'] = str(e)
            return False
    
    async def check_price_alerts(self, currency_data: List[CurrencyData]):
        """
        Check for significant price changes and generate alerts.
        
        Args:
            currency_data: List of currency data to check
        """
        try:
            significant_changes = self.ninja_client.get_price_changes(
                currency_data,
                min_change_threshold=5.0
            )
            
            for change in significant_changes:
                currency_name = change['currency_name']
                total_change = change['total_change']
                
                # Check if this currency has a specific threshold
                threshold = self.alert_thresholds.get(currency_name, 15.0)
                
                if abs(total_change) >= threshold:
                    await self.generate_alert(change)
                    self.stats['alerts_generated'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error checking price alerts: {str(e)}")
    
    async def generate_alert(self, change_info: Dict):
        """
        Generate and log price change alert.
        
        Args:
            change_info: Price change information
        """
        currency_name = change_info['currency_name']
        total_change = change_info['total_change']
        chaos_equivalent = change_info['chaos_equivalent']
        league = change_info['league']
        direction = change_info['direction']
        
        alert_message = (
            f"🚨 PRICE ALERT: {currency_name} ({league}) "
            f"changed {total_change:+.1f}% "
            f"(Current: {chaos_equivalent:.2f} Chaos)"
        )
        
        self.logger.warning(alert_message)
        
        # Here you could add additional alert mechanisms:
        # - Send webhook notifications
        # - Email alerts
        # - Discord/Slack notifications
        # - Database alert logging
        
        # Store alert to database
        try:
            from utils.database import get_db_connection
            
            conn = get_db_connection()
            
            # Create alerts table if it doesn't exist
            create_alert_table = """
            CREATE TABLE IF NOT EXISTS price_alerts (
                id SERIAL PRIMARY KEY,
                currency_name VARCHAR(100) NOT NULL,
                league VARCHAR(50) NOT NULL,
                change_percent DECIMAL(10, 4) NOT NULL,
                chaos_equivalent DECIMAL(20, 8),
                confidence_level VARCHAR(20),
                alert_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                alert_message TEXT
            );
            """
            
            with conn.cursor() as cursor:
                cursor.execute(create_alert_table)
                
                # Insert alert
                insert_alert = """
                INSERT INTO price_alerts 
                (currency_name, league, change_percent, chaos_equivalent, confidence_level, alert_message)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_alert, (
                    currency_name,
                    league,
                    total_change,
                    chaos_equivalent,
                    change_info.get('confidence_level', 'normal'),
                    alert_message
                ))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store alert to database: {str(e)}")
    
    async def generate_predictions(self) -> bool:
        """
        Generate predictions using current data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_predictions or not self.predictor:
            return False
        
        try:
            self.stats['total_predictions'] += 1
            self.logger.info("Generating real-time predictions...")
            
            # Get top predictions
            predictions = self.predictor.get_top_predictions(
                top_n=20,
                sort_by='price_change_percent',
                ascending=False
            )
            
            if predictions:
                self.logger.info(f"Generated {len(predictions)} predictions")
                
                # Log top 5 predictions
                self.logger.info("Top 5 predictions:")
                for i, pred in enumerate(predictions[:5], 1):
                    self.logger.info(
                        f"  {i}. {pred.currency_pair}: "
                        f"{pred.current_price:.2f} -> {pred.predicted_price:.2f} "
                        f"({pred.price_change_percent:+.1f}%)"
                    )
                
                # Store predictions to database
                await self.store_predictions(predictions)
                
                self.stats['successful_predictions'] += 1
                self.last_prediction_time = datetime.now()
                return True
            else:
                self.logger.warning("No predictions generated")
                return False
                
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            self.stats['last_error'] = str(e)
            return False
    
    async def store_predictions(self, predictions):
        """
        Store predictions to database.
        
        Args:
            predictions: List of prediction results
        """
        try:
            from utils.database import get_db_connection
            
            conn = get_db_connection()
            
            # Create predictions table if it doesn't exist
            create_predictions_table = """
            CREATE TABLE IF NOT EXISTS live_predictions (
                id SERIAL PRIMARY KEY,
                currency_pair VARCHAR(100) NOT NULL,
                current_price DECIMAL(20, 8) NOT NULL,
                predicted_price DECIMAL(20, 8) NOT NULL,
                price_change_percent DECIMAL(10, 4),
                confidence_score DECIMAL(5, 4),
                prediction_horizon_days INTEGER,
                model_type VARCHAR(50),
                features_used INTEGER,
                data_points_used INTEGER,
                prediction_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_live_predictions_currency_time 
            ON live_predictions(currency_pair, prediction_time);
            """
            
            with conn.cursor() as cursor:
                cursor.execute(create_predictions_table)
                
                # Insert predictions
                for pred in predictions:
                    insert_prediction = """
                    INSERT INTO live_predictions 
                    (currency_pair, current_price, predicted_price, price_change_percent,
                     confidence_score, prediction_horizon_days, model_type, 
                     features_used, data_points_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(insert_prediction, (
                        pred.currency_pair,
                        pred.current_price,
                        pred.predicted_price,
                        pred.price_change_percent,
                        pred.confidence_score,
                        pred.prediction_horizon_days,
                        pred.model_type,
                        pred.features_used,
                        pred.data_points_used
                    ))
                
                conn.commit()
            
            conn.close()
            self.logger.info(f"Stored {len(predictions)} predictions to database")
            
        except Exception as e:
            self.logger.error(f"Failed to store predictions: {str(e)}")
    
    def get_service_status(self) -> Dict:
        """Get current service status and statistics."""
        uptime = None
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
        
        return {
            'running': self.running,
            'uptime_seconds': uptime.total_seconds() if uptime else 0,
            'last_fetch_time': self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'fetch_interval': self.fetch_interval,
            'prediction_interval': self.prediction_interval,
            'predictions_enabled': self.enable_predictions,
            'monitored_leagues': self.monitored_leagues,
            'statistics': self.stats.copy()
        }
    
    async def run_service_loop(self):
        """Main service loop."""
        self.running = True
        self.logger.info("Starting live data ingestion service loop...")
        
        try:
            while self.running:
                loop_start = datetime.now()
                
                # Check if it's time to fetch data
                should_fetch = (
                    self.last_fetch_time is None or
                    (datetime.now() - self.last_fetch_time).total_seconds() >= self.fetch_interval
                )
                
                if should_fetch:
                    await self.fetch_and_store_data()
                
                # Check if it's time to generate predictions
                should_predict = (
                    self.enable_predictions and
                    (self.last_prediction_time is None or
                     (datetime.now() - self.last_prediction_time).total_seconds() >= self.prediction_interval)
                )
                
                if should_predict:
                    await self.generate_predictions()
                
                # Calculate sleep time to maintain interval
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, min(self.fetch_interval, self.prediction_interval) - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Service loop cancelled")
        except Exception as e:
            self.logger.error(f"Service loop error: {str(e)}")
            self.stats['last_error'] = str(e)
        finally:
            self.running = False
            self.logger.info("Service loop stopped")
    
    async def start(self):
        """Start the service."""
        try:
            await self.initialize()
            await self.run_service_loop()
        except Exception as e:
            self.logger.error(f"Service failed to start: {str(e)}")
            raise
    
    def stop(self):
        """Stop the service."""
        self.logger.info("Stopping live data ingestion service...")
        self.running = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live Data Ingestion Service for PoEconomy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--fetch-interval',
        type=int,
        default=300,
        help='Data fetch interval in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--prediction-interval',
        type=int,
        default=1800,
        help='Prediction generation interval in seconds (default: 1800)'
    )
    
    parser.add_argument(
        '--disable-predictions',
        action='store_true',
        help='Disable prediction generation'
    )
    
    parser.add_argument(
        '--leagues',
        nargs='*',
        default=['Standard', 'Hardcore'],
        help='Leagues to monitor (default: Standard Hardcore)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


async def main():
    """Main function for the service."""
    args = parse_arguments()
    
    # Setup logging
    logger = MLLogger("LiveDataIngestion", level=args.log_level)
    
    # Create service
    service = LiveDataIngestionService(
        logger=logger,
        fetch_interval=args.fetch_interval,
        prediction_interval=args.prediction_interval,
        enable_predictions=not args.disable_predictions
    )
    
    service.monitored_leagues = args.leagues
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        service.stop()
    
    # Register signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        logger.info("Starting Live Data Ingestion Service")
        logger.info(f"Fetch interval: {args.fetch_interval}s")
        logger.info(f"Prediction interval: {args.prediction_interval}s")
        logger.info(f"Predictions enabled: {not args.disable_predictions}")
        logger.info(f"Monitored leagues: {args.leagues}")
        
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {str(e)}")
        return 1
    finally:
        logger.info("Service shutdown complete")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main())) 