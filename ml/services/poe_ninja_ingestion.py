#!/usr/bin/env python3
"""
Live Data Ingestion Service

This service continuously fetches live currency data from poe.ninja API
and stores it to the database for analysis.
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
from utils.logging_utils import MLLogger


class LiveDataIngestionService:
    """
    Service for continuous live data ingestion.
    
    This service fetches data from poe.ninja and stores it to the database.
    """
    
    def __init__(
        self,
        logger: Optional[MLLogger] = None,
        fetch_interval: int = 300,  # 5 minutes
    ):
        """
        Initialize the live data ingestion service.
        
        Args:
            logger: Logger instance
            fetch_interval: Data fetch interval in seconds
        """
        self.logger = logger or MLLogger("LiveDataIngestion")
        self.fetch_interval = fetch_interval
        
        # Service state
        self.running = False
        self.last_fetch_time = None
        
        # Components
        self.ninja_client = None
        
        # Leagues to monitor
        self.monitored_leagues = ['Mercenaries']
        
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
    
    def get_service_status(self) -> Dict:
        """Get current service status and statistics."""
        uptime = None
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
        
        return {
            'running': self.running,
            'uptime_seconds': uptime.total_seconds() if uptime else 0,
            'last_fetch_time': self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            'fetch_interval': self.fetch_interval,
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
                    data_updated = await self.fetch_and_store_data()
                    
                    # If data was successfully updated, trigger investment report generation
                    if data_updated:
                        await self.trigger_investment_report()
                
                # Calculate sleep time to maintain interval
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, self.fetch_interval - loop_duration)
                
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
    
    async def trigger_investment_report(self):
        """Trigger investment report generation when data is updated."""
        try:
            self.logger.info("Data updated - triggering investment report generation...")
            
            # Import and run the investment report generator
            sys.path.append(str(Path(__file__).parent.parent / "scripts"))
            from generate_investment_report import InvestmentReportGenerator
            
            # Run investment report generation in a separate task to avoid blocking
            generator = InvestmentReportGenerator()
            report_path = generator.generate_comprehensive_report()
            
            if report_path:
                self.logger.info(f"Investment report generated: {report_path}")
            else:
                self.logger.warning("Failed to generate investment report")
                
        except Exception as e:
            self.logger.error(f"Error triggering investment report: {str(e)}")
    
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
        '--leagues',
        nargs='*',
        default=['Mercenaries'],
        help='Leagues to monitor (default: Mercenaries)'
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
        fetch_interval=args.fetch_interval
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