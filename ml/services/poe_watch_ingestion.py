#!/usr/bin/env python3
"""
POE Watch Live Data Ingestion Service

This service continuously fetches live currency data from poe.watch API
and stores it to the database for analysis.

Based on POE Watch API documentation: https://docs.poe.watch/
"""

import sys
import asyncio
import aiohttp
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
import json
import argparse
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import MLLogger
from utils.database import get_db_connection


@dataclass
class PoeWatchCurrency:
    """Data structure for POE Watch currency information."""
    id: int
    name: str
    category: str
    group: str
    frame: int
    icon: str
    mean: float
    min: float
    max: float
    exalted: float
    daily: int
    change: float
    history: List[float]
    low_confidence: bool
    league: str
    fetch_time: datetime = field(default_factory=datetime.now)
    
    # Optional fields
    mode: Optional[float] = None
    total: Optional[int] = None
    current: Optional[int] = None
    accepted: Optional[int] = None
    divine: Optional[float] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any], league: str) -> 'PoeWatchCurrency':
        """Create instance from POE Watch API response."""
        try:
            # Validate required fields
            if not isinstance(data, dict):
                raise ValueError("Invalid data format - expected dictionary")
            
            # Safely convert numeric fields with better error handling
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                try:
                    return int(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            return cls(
                id=safe_int(data.get('id'), 0),
                name=str(data.get('name', '')),
                category=str(data.get('category', '')),
                group=str(data.get('group', '')),
                frame=safe_int(data.get('frame'), 0),
                icon=str(data.get('icon', '')),
                mean=safe_float(data.get('mean')),
                min=safe_float(data.get('min')),
                max=safe_float(data.get('max')),
                exalted=safe_float(data.get('exalted')),
                daily=safe_int(data.get('daily')),
                change=safe_float(data.get('change')),
                history=data.get('history', []) if isinstance(data.get('history'), list) else [],
                low_confidence=bool(data.get('lowConfidence', False)),
                league=str(league),
                mode=safe_float(data.get('mode')),
                total=safe_int(data.get('total')),
                current=safe_int(data.get('current')),
                accepted=safe_int(data.get('accepted')),
                divine=safe_float(data.get('divine'))
            )
        except Exception as e:
            raise ValueError(f"Failed to parse API response data: {e}")
    
    def is_valid(self) -> bool:
        """Check if currency data is valid for storage."""
        return (
            self.id > 0 and
            self.name and
            self.mean >= 0 and
            self.league
        )


class PoeWatchAPIClient:
    """Client for POE Watch API interactions."""
    
    def __init__(
        self,
        base_url: str = "https://api.poe.watch",
        timeout: int = 30,
        rate_limit_delay: float = 1.0,
        logger: Optional[MLLogger] = None
    ):
        """Initialize POE Watch API client."""
        self.base_url = base_url
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.logger = logger or MLLogger("PoeWatchAPI")
        
        self.session = None
        self.last_request_time = datetime.min
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Any]:
        """Make a rate-limited request to the POE Watch API with retry logic."""
        for attempt in range(max_retries):
            try:
                # Rate limiting
                time_since_last = (datetime.now() - self.last_request_time).total_seconds()
                if time_since_last < self.rate_limit_delay:
                    await asyncio.sleep(self.rate_limit_delay - time_since_last)
                
                url = f"{self.base_url}/{endpoint.lstrip('/')}"
                
                self.logger.debug(f"Making request to: {url} with params: {params} (attempt {attempt + 1}/{max_retries})")
                
                async with self.session.get(url, params=params) as response:
                    self.last_request_time = datetime.now()
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                            return data
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to decode JSON response: {e}")
                            if attempt == max_retries - 1:
                                return None
                            continue
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status >= 500:  # Server error, retry
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            self.logger.warning(f"Server error {response.status}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API request failed with status {response.status}: {error_text}")
                            return None
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
                        
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request timeout, retrying (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    self.logger.error("Request timeout after all retries")
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request error: {str(e)}, retrying (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    self.logger.error(f"Error making API request after all retries: {str(e)}")
                    return None
        
        return None
    
    async def get_leagues(self) -> List[Dict]:
        """Get available leagues from POE Watch API."""
        try:
            data = await self._make_request("leagues")
            return data if data else []
        except Exception as e:
            self.logger.error(f"Error fetching leagues: {str(e)}")
            return []
    
    async def get_categories(self) -> List[Dict]:
        """Get available categories from POE Watch API."""
        try:
            data = await self._make_request("categories")
            return data if data else []
        except Exception as e:
            self.logger.error(f"Error fetching categories: {str(e)}")
            return []
    
    async def get_currency_data(self, league: str, category: str = 'currency') -> List[PoeWatchCurrency]:
        """Fetch currency/fragment data for a specific league and category."""
        try:
            params = {
                'category': category,
                'league': league
            }
            
            data = await self._make_request("get", params)
            
            if not data:
                self.logger.warning(f"No {category} data received for league: {league}")
                return []
            
            # Convert API response to data objects
            currency_data = []
            skipped_items = 0
            
            for item in data:
                try:
                    currency_obj = PoeWatchCurrency.from_api_response(item, league)
                    # Validate the currency data before adding
                    if currency_obj.is_valid():
                        currency_data.append(currency_obj)
                    else:
                        skipped_items += 1
                        self.logger.debug(f"Skipped invalid {category} item: {item.get('name', 'Unknown')} (ID: {item.get('id', 'Unknown')})")
                except Exception as e:
                    skipped_items += 1
                    self.logger.warning(f"Error parsing {category} item: {str(e)} - Item data: {item}")
                    continue
            
            if skipped_items > 0:
                self.logger.info(f"Fetched {len(currency_data)} valid {category} items for {league} (skipped {skipped_items} invalid items)")
            else:
                self.logger.info(f"Fetched {len(currency_data)} {category} items for {league}")
            
            return currency_data
            
        except Exception as e:
            self.logger.error(f"Error fetching {category} data for {league}: {str(e)}")
            return []


class PoeWatchIngestionService:
    """Service for continuous live data ingestion from POE Watch API."""
    
    def __init__(
        self,
        logger: Optional[MLLogger] = None,
        fetch_interval: int = 300,  # 5 minutes
    ):
        """Initialize the POE Watch data ingestion service."""
        self.logger = logger or MLLogger("PoeWatchIngestion")
        self.fetch_interval = fetch_interval
        
        # Service state
        self.running = False
        self.last_fetch_time = None
        
        # Components
        self.client = PoeWatchAPIClient(logger=self.logger)
        
        # Leagues to monitor
        self.monitored_leagues = ['Mercenaries']
        
        # Categories to monitor
        self.monitored_categories = ['currency', 'fragment']
        
        # Price change alert thresholds (DISABLED)
        self.alert_thresholds = {}
        self.alerts_enabled = False
        
        # Statistics
        self.stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'currencies_tracked': 0,
            'alerts_generated': 0,
            'start_time': None,
            'last_error': None,
            'database_writes': 0,
            'failed_writes': 0
        }
        
        # Price cache for change detection (with cleanup)
        self.price_cache = {}
        self.cache_cleanup_interval = 3600  # Clean cache every hour
        self.last_cache_cleanup = datetime.now()
    
    async def initialize(self):
        """Initialize service components."""
        try:
            self.logger.info("🚀 Initializing POE Watch Data Ingestion Service...")
            
            # Test API connectivity
            async with self.client:
                leagues = await self.client.get_leagues()
                categories = await self.client.get_categories()
                
                if leagues:
                    league_names = [league.get('name', 'Unknown') for league in leagues]
                    self.logger.info(f"📊 Connected to POE Watch API - Found {len(leagues)} leagues: {league_names}")
                else:
                    self.logger.warning("⚠️ No leagues data received from POE Watch API")
                
                if categories:
                    category_names = [cat.get('name', 'Unknown') for cat in categories]
                    self.logger.info(f"📂 Available categories: {category_names}")
            
            # Initialize database tables
            await self.setup_database_tables()
            
            self.stats['start_time'] = datetime.now()
            self.logger.info("✅ POE Watch service initialization completed")
            
        except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {str(e)}")
            raise
    
    async def setup_database_tables(self):
        """Setup database tables for POE Watch data."""
        try:
            conn = get_db_connection()
            
            # Create main currency data table
            create_currency_table = """
            CREATE TABLE IF NOT EXISTS live_poe_watch (
                id SERIAL PRIMARY KEY,
                poe_watch_id INTEGER NOT NULL,
                currency_name VARCHAR(100) NOT NULL,
                category VARCHAR(50),
                group_name VARCHAR(50),
                frame INTEGER,
                icon_url TEXT,
                mean_price DECIMAL(20, 8),
                min_price DECIMAL(20, 8),
                max_price DECIMAL(20, 8),
                exalted_price DECIMAL(20, 8),
                divine_price DECIMAL(20, 8),
                daily_volume INTEGER,
                price_change_percent DECIMAL(10, 4),
                price_history JSONB,
                low_confidence BOOLEAN,
                league VARCHAR(50) NOT NULL,
                mode_price DECIMAL(20, 8),
                total_listings INTEGER,
                current_listings INTEGER,
                accepted_listings INTEGER,
                fetch_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            # Create indices
            create_indices = [
                "CREATE INDEX IF NOT EXISTS idx_poe_watch_league_time ON live_poe_watch(league, fetch_time);",
                "CREATE INDEX IF NOT EXISTS idx_poe_watch_currency_name ON live_poe_watch(currency_name);",
                "CREATE INDEX IF NOT EXISTS idx_poe_watch_price_change ON live_poe_watch(price_change_percent);",
                "CREATE INDEX IF NOT EXISTS idx_poe_watch_id_league ON live_poe_watch(poe_watch_id, league);",
            ]
            
            with conn.cursor() as cursor:
                cursor.execute(create_currency_table)
                
                for index_sql in create_indices:
                    cursor.execute(index_sql)
                
                conn.commit()
            
            conn.close()
            self.logger.info("📊 Database tables setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Database setup failed: {str(e)}")
            raise
    
    async def fetch_and_store_data(self) -> bool:
        """Fetch live data from POE Watch API and store to database."""
        try:
            self.stats['total_fetches'] += 1
            self.logger.info("📡 Fetching currency and fragment data from POE Watch API...")
            
            async with self.client:
                all_currencies = []
                
                # Fetch data for all monitored leagues and categories
                for league in self.monitored_leagues:
                    for category in self.monitored_categories:
                        currency_data = await self.client.get_currency_data(league, category)
                        
                        if currency_data:
                            all_currencies.extend(currency_data)
                            self.logger.info(f"📈 Fetched {len(currency_data)} {category} items for {league}")
                        else:
                            self.logger.warning(f"⚠️ No {category} data fetched for {league}")
                
                if not all_currencies:
                    self.logger.warning("❌ No currency/fragment data fetched from POE Watch API")
                    return False
                
                # Store to database
                success = await self.store_to_database(all_currencies)
                
                if success:
                    self.stats['successful_fetches'] += 1
                    self.stats['currencies_tracked'] = len(all_currencies)
                    self.last_fetch_time = datetime.now()
                    
                    # Price alerts are disabled
                    # await self.check_price_alerts(all_currencies)
                    
                    self.logger.info(f"✅ Successfully processed {len(all_currencies)} currency/fragment records")
                
                return success
                
        except Exception as e:
            self.logger.error(f"❌ Error in fetch_and_store_data: {str(e)}")
            self.stats['last_error'] = str(e)
            return False
    
    async def store_to_database(self, currency_data: List[PoeWatchCurrency]) -> bool:
        """Store currency data to database."""
        if not currency_data:
            self.logger.warning("No currency data to store")
            return False
            
        conn = None
        try:
            conn = get_db_connection()
            
            insert_sql = """
            INSERT INTO live_poe_watch (
                poe_watch_id, currency_name, category, group_name, frame, icon_url,
                mean_price, min_price, max_price, exalted_price, divine_price,
                daily_volume, price_change_percent, price_history, low_confidence,
                league, mode_price, total_listings, current_listings, accepted_listings,
                fetch_time
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            records_inserted = 0
            batch_size = 100  # Process in batches for better performance
            
            with conn.cursor() as cursor:
                for i in range(0, len(currency_data), batch_size):
                    batch = currency_data[i:i+batch_size]
                    batch_records = 0
                    
                    for currency in batch:
                        try:
                            # Validate data before insertion
                            if not currency.is_valid():
                                self.logger.debug(f"Skipping invalid currency data: {currency.name}")
                                continue
                                
                            cursor.execute(insert_sql, (
                                currency.id,
                                currency.name[:100],  # Truncate to prevent overflow
                                currency.category[:50],
                                currency.group[:50],
                                currency.frame,
                                currency.icon,
                                currency.mean,
                                currency.min,
                                currency.max,
                                currency.exalted,
                                currency.divine,
                                currency.daily,
                                currency.change,
                                json.dumps(currency.history) if currency.history else '[]',
                                currency.low_confidence,
                                currency.league[:50],
                                currency.mode,
                                currency.total,
                                currency.current,
                                currency.accepted,
                                currency.fetch_time
                            ))
                            batch_records += 1
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to insert currency {currency.name} (ID: {currency.id}): {str(e)}")
                            self.stats['failed_writes'] += 1
                            continue
                    
                    # Commit batch
                    try:
                        conn.commit()
                        records_inserted += batch_records
                        self.logger.debug(f"Committed batch {i//batch_size + 1}: {batch_records} records")
                    except Exception as e:
                        self.logger.error(f"Failed to commit batch {i//batch_size + 1}: {str(e)}")
                        conn.rollback()
                        self.stats['failed_writes'] += batch_records
            
            self.stats['database_writes'] += records_inserted
            self.logger.info(f"💾 Successfully stored {records_inserted} currency records to database")
            
            return records_inserted > 0
            
        except Exception as e:
            self.logger.error(f"❌ Database storage failed: {str(e)}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            self.stats['failed_writes'] += len(currency_data)
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    async def cleanup_price_cache(self):
        """Clean up old entries from price cache to prevent memory leaks."""
        try:
            current_time = datetime.now()
            if (current_time - self.last_cache_cleanup).total_seconds() >= self.cache_cleanup_interval:
                cache_size_before = len(self.price_cache)
                
                # Keep cache size manageable (max 1000 entries)
                if cache_size_before > 1000:
                    # Keep only the most recent 500 entries
                    cache_items = list(self.price_cache.items())
                    self.price_cache = dict(cache_items[-500:])
                    cache_size_after = len(self.price_cache)
                    self.logger.info(f"Cleaned price cache: {cache_size_before} → {cache_size_after} entries")
                
                self.last_cache_cleanup = current_time
                
        except Exception as e:
            self.logger.warning(f"Error cleaning price cache: {str(e)}")

    def get_service_status(self) -> Dict:
        """Get current service status and statistics."""
        uptime = None
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
        
        return {
            'service_name': 'POE Watch Data Ingestion',
            'api_source': 'https://api.poe.watch',
            'running': self.running,
            'uptime_seconds': uptime.total_seconds() if uptime else 0,
            'last_fetch_time': self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            'fetch_interval': self.fetch_interval,
            'monitored_leagues': self.monitored_leagues,
            'monitored_categories': self.monitored_categories,
            'alerts_enabled': self.alerts_enabled,
            'price_cache_size': len(self.price_cache),
            'statistics': self.stats.copy()
        }
    
    async def run_service_loop(self):
        """Main service loop."""
        self.running = True
        self.logger.info("🔄 Starting POE Watch data ingestion service loop...")
        
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
                    
                    # Clean up price cache periodically
                    await self.cleanup_price_cache()
                    
                    # If data was successfully updated, trigger investment report generation
                    if data_updated:
                        await self.trigger_investment_report()
                
                # Calculate sleep time to maintain interval
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, min(60, self.fetch_interval - loop_duration))
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("🛑 Service loop cancelled")
        except Exception as e:
            self.logger.error(f"❌ Service loop error: {str(e)}")
            self.stats['last_error'] = str(e)
        finally:
            self.running = False
            self.logger.info("🔚 Service loop stopped")
    
    async def trigger_investment_report(self):
        """Trigger POE Watch investment report generation when data is updated."""
        try:
            self.logger.info("📊 POE Watch data updated - triggering POE Watch investment report generation...")
            
            # Import and run the POE Watch investment report generator
            sys.path.append(str(Path(__file__).parent.parent / "scripts"))
            from generate_poe_watch_investment_report import PoeWatchInvestmentReportGenerator
            
            # Run POE Watch investment report generation
            generator = PoeWatchInvestmentReportGenerator("C:/Workspace/PoEconomy/ml/investment_reports")
            report_path = generator.generate_comprehensive_report()
            
            if report_path:
                self.logger.info(f"📄 POE Watch investment report generated: {report_path}")
            else:
                self.logger.warning("⚠️ Failed to generate POE Watch investment report")
                
        except Exception as e:
            self.logger.error(f"❌ Error triggering POE Watch investment report: {str(e)}")
    
    async def start(self):
        """Start the service."""
        try:
            await self.initialize()
            await self.run_service_loop()
        except Exception as e:
            self.logger.error(f"❌ Service failed to start: {str(e)}")
            raise
    
    def stop(self):
        """Stop the service."""
        self.logger.info("🛑 Stopping POE Watch data ingestion service...")
        self.running = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="POE Watch Live Data Ingestion Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (10 minute intervals)
  python poe_watch_ingestion.py
  
  # Run with 2 minute intervals
  python poe_watch_ingestion.py --fetch-interval 120
  
  # Monitor specific leagues
  python poe_watch_ingestion.py --leagues Mercenaries
  
  # Monitor specific categories
  python poe_watch_ingestion.py --categories currency fragment
  
  # Enable debug logging
  python poe_watch_ingestion.py --log-level DEBUG
  
  # Run single test fetch
  python poe_watch_ingestion.py --test-mode
        """
    )
    
    parser.add_argument(
        '--fetch-interval',
        type=int,
        default=600,
        help='Data fetch interval in seconds (default: 600)'
    )
    
    parser.add_argument(
        '--leagues',
        nargs='*',
        default=['Mercenaries'],
        help='Leagues to monitor (default: Mercenaries)'
    )
    
    parser.add_argument(
        '--categories',
        nargs='*',
        default=['currency', 'fragment'],
        help='Categories to monitor (default: currency fragment)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode (single fetch then exit)'
    )
    
    return parser.parse_args()


async def main():
    """Main function for the service."""
    args = parse_arguments()
    
    # Setup logging
    logger = MLLogger("PoeWatchIngestion", level=args.log_level)
    
    # Create service
    service = PoeWatchIngestionService(
        logger=logger,
        fetch_interval=args.fetch_interval
    )
    
    service.monitored_leagues = args.leagues
    service.monitored_categories = args.categories
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        service.stop()
    
    # Register signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        logger.info("🚀 Starting POE Watch Data Ingestion Service")
        logger.info(f"⏱️ Fetch interval: {args.fetch_interval}s")
        logger.info(f"🏆 Monitored leagues: {args.leagues}")
        logger.info(f"📂 Monitored categories: {service.monitored_categories}")
        logger.info(f"🔗 API Base URL: https://api.poe.watch")
        
        if args.test_mode:
            logger.info("🧪 Running in test mode")
            await service.initialize()
            success = await service.fetch_and_store_data()
            if success:
                # Trigger investment report generation in test mode too
                await service.trigger_investment_report()
                logger.info("✅ Test completed successfully")
                return 0
            else:
                logger.error("❌ Test failed")
                return 1
        else:
            await service.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Service interrupted by user")
    except Exception as e:
        logger.error(f"❌ Service failed: {str(e)}")
        return 1
    finally:
        logger.info("🔚 Service shutdown complete")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main())) 