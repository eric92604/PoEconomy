"""
PoE Ninja API Client for live currency data ingestion.

This module provides functionality to fetch real-time currency prices from poe.ninja
and integrate them with the PoEconomy prediction system.
"""

import sys
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import MLLogger
from utils.database import get_db_connection


@dataclass
class CurrencyData:
    """Data structure for currency information from poe.ninja."""
    currency_name: str
    currency_id: int
    chaos_equivalent: float
    pay_value: Optional[float]
    receive_value: Optional[float]
    pay_count: Optional[int]
    receive_count: Optional[int]
    listing_count: Optional[int]
    sample_time: datetime
    league: str
    pay_sparkline: Optional[List[float]] = None
    receive_sparkline: Optional[List[float]] = None
    total_change: Optional[float] = None
    confidence_level: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        data['sample_time'] = self.sample_time.isoformat()
        return data


class PoENinjaClient:
    """
    Client for fetching currency data from poe.ninja API.
    
    This class handles API requests, data parsing, and integration with the
    PoEconomy database and prediction system.
    """
    
    BASE_URL = "https://poe.ninja/api/data"
    
    def __init__(
        self,
        logger: Optional[MLLogger] = None,
        rate_limit_delay: float = 1.0,
        timeout: int = 30
    ):
        """
        Initialize the PoE Ninja client.
        
        Args:
            logger: Optional logger instance
            rate_limit_delay: Delay between API requests in seconds
            timeout: Request timeout in seconds
        """
        self.logger = logger or MLLogger("PoENinjaClient")
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.session = None
        
        # Cache for reducing API calls
        self._cache = {}
        self._cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
        # League mapping
        self.league_mapping = {
            'Standard': 'Standard',
            'Hardcore': 'Hardcore',
            'Settlers': 'Settlers',  # Current league example
            'Hardcore Settlers': 'Hardcore Settlers'
        }
    
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
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        cached_time = self._cache[cache_key]['timestamp']
        return datetime.now() - cached_time < self._cache_duration
    
    async def fetch_currency_overview(
        self,
        league: str = "Standard",
        currency_type: str = "Currency"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch currency overview data from poe.ninja.
        
        Args:
            league: League name (e.g., "Standard", "Settlers")
            currency_type: Type of currency data ("Currency", "Fragment")
            
        Returns:
            Raw API response data or None if failed
        """
        cache_key = f"{league}_{currency_type}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Using cached data for {cache_key}")
            return self._cache[cache_key]['data']
        
        url = f"{self.BASE_URL}/currencyoverview"
        params = {
            'league': league,
            'type': currency_type
        }
        
        try:
            if not self.session:
                raise RuntimeError("Client not initialized. Use async context manager.")
            
            self.logger.info(f"Fetching currency data for {league} - {currency_type}")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the response
                    self._cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    
                    self.logger.info(f"Successfully fetched {len(data.get('lines', []))} currency entries")
                    return data
                else:
                    self.logger.error(f"API request failed with status {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            self.logger.error(f"Request timeout for {url}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching currency data: {str(e)}")
            return None
        finally:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
    
    def parse_currency_data(
        self,
        raw_data: Dict[str, Any],
        league: str
    ) -> List[CurrencyData]:
        """
        Parse raw API response into structured currency data.
        
        Args:
            raw_data: Raw API response
            league: League name
            
        Returns:
            List of parsed currency data
        """
        if not raw_data or 'lines' not in raw_data:
            self.logger.warning("No currency lines found in API response")
            return []
        
        parsed_currencies = []
        currency_details = {item['name']: item for item in raw_data.get('currencyDetails', [])}
        
        for line in raw_data['lines']:
            try:
                currency_name = line.get('currencyTypeName', 'Unknown')
                
                # Get currency ID from details
                currency_id = None
                if currency_name in currency_details:
                    currency_id = currency_details[currency_name].get('id')
                
                # Parse sample time
                sample_time = datetime.utcnow()  # Default to now
                if 'pay' in line and 'sample_time_utc' in line['pay']:
                    try:
                        sample_time = datetime.fromisoformat(
                            line['pay']['sample_time_utc'].replace('Z', '+00:00')
                        )
                    except:
                        pass
                
                # Extract price data
                pay_data = line.get('pay', {})
                receive_data = line.get('receive', {})
                
                # Calculate confidence level based on data quality
                confidence_level = "normal"
                if pay_data.get('count', 0) < 10 or receive_data.get('count', 0) < 10:
                    confidence_level = "low"
                elif pay_data.get('count', 0) > 100 and receive_data.get('count', 0) > 100:
                    confidence_level = "high"
                
                # Parse sparkline data for trend analysis
                pay_sparkline = line.get('paySparkLine', {}).get('data', [])
                receive_sparkline = line.get('receiveSparkLine', {}).get('data', [])
                total_change = line.get('paySparkLine', {}).get('totalChange', 0)
                
                currency_data = CurrencyData(
                    currency_name=currency_name,
                    currency_id=currency_id or 0,
                    chaos_equivalent=float(line.get('chaosEquivalent', 0)),
                    pay_value=float(pay_data.get('value', 0)) if pay_data.get('value') else None,
                    receive_value=float(receive_data.get('value', 0)) if receive_data.get('value') else None,
                    pay_count=pay_data.get('count'),
                    receive_count=receive_data.get('count'),
                    listing_count=pay_data.get('listing_count', 0) + receive_data.get('listing_count', 0),
                    sample_time=sample_time,
                    league=league,
                    pay_sparkline=pay_sparkline,
                    receive_sparkline=receive_sparkline,
                    total_change=total_change,
                    confidence_level=confidence_level
                )
                
                parsed_currencies.append(currency_data)
                
            except Exception as e:
                self.logger.error(f"Error parsing currency {line.get('currencyTypeName', 'Unknown')}: {str(e)}")
                continue
        
        self.logger.info(f"Parsed {len(parsed_currencies)} currencies successfully")
        return parsed_currencies
    
    async def fetch_all_leagues_data(
        self,
        leagues: Optional[List[str]] = None
    ) -> Dict[str, List[CurrencyData]]:
        """
        Fetch currency data for multiple leagues.
        
        Args:
            leagues: List of league names (None for default leagues)
            
        Returns:
            Dictionary mapping league names to currency data
        """
        if leagues is None:
            leagues = ['Standard', 'Hardcore']
        
        all_data = {}
        
        for league in leagues:
            try:
                # Fetch currency data
                currency_raw = await self.fetch_currency_overview(league, "Currency")
                if currency_raw:
                    currency_data = self.parse_currency_data(currency_raw, league)
                    all_data[league] = currency_data
                
                # Fetch fragment data
                fragment_raw = await self.fetch_currency_overview(league, "Fragment")
                if fragment_raw:
                    fragment_data = self.parse_currency_data(fragment_raw, league)
                    if league in all_data:
                        all_data[league].extend(fragment_data)
                    else:
                        all_data[league] = fragment_data
                
            except Exception as e:
                self.logger.error(f"Error fetching data for league {league}: {str(e)}")
                continue
        
        return all_data
    
    def convert_to_database_format(
        self,
        currency_data: List[CurrencyData]
    ) -> pd.DataFrame:
        """
        Convert currency data to database-compatible format.
        
        Args:
            currency_data: List of currency data objects
            
        Returns:
            DataFrame ready for database insertion
        """
        if not currency_data:
            return pd.DataFrame()
        
        records = []
        
        for currency in currency_data:
            # Create records for both pay and receive directions
            base_record = {
                'currency_name': currency.currency_name,
                'league': currency.league,
                'sample_time': currency.sample_time,
                'chaos_equivalent': currency.chaos_equivalent,
                'listing_count': currency.listing_count,
                'confidence_level': currency.confidence_level,
                'total_change': currency.total_change
            }
            
            # Pay direction (currency -> chaos)
            if currency.pay_value is not None:
                pay_record = base_record.copy()
                pay_record.update({
                    'direction': 'pay',
                    'value': currency.pay_value,
                    'count': currency.pay_count,
                    'sparkline': json.dumps(currency.pay_sparkline) if currency.pay_sparkline else None
                })
                records.append(pay_record)
            
            # Receive direction (chaos -> currency)
            if currency.receive_value is not None:
                receive_record = base_record.copy()
                receive_record.update({
                    'direction': 'receive',
                    'value': currency.receive_value,
                    'count': currency.receive_count,
                    'sparkline': json.dumps(currency.receive_sparkline) if currency.receive_sparkline else None
                })
                records.append(receive_record)
        
        df = pd.DataFrame(records)
        
        # Add derived columns
        if not df.empty:
            df['created_at'] = datetime.utcnow()
            df['data_source'] = 'poe_ninja'
            df['api_version'] = '1.0'
        
        return df
    
    async def store_to_database(
        self,
        currency_data: List[CurrencyData],
        table_name: str = 'live_currency_prices'
    ) -> bool:
        """
        Store currency data to database.
        
        Args:
            currency_data: List of currency data to store
            table_name: Database table name
            
        Returns:
            True if successful, False otherwise
        """
        if not currency_data:
            self.logger.warning("No currency data to store")
            return False

        try:
            df = self.convert_to_database_format(currency_data)
            
            if df.empty:
                self.logger.warning("No valid records to store")
                return False

            conn = get_db_connection()
            
            # Create table if it doesn't exist
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                currency_name VARCHAR(100) NOT NULL,
                league VARCHAR(50) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                value DECIMAL(20, 8) NOT NULL,
                count INTEGER,
                chaos_equivalent DECIMAL(20, 8),
                listing_count INTEGER,
                confidence_level VARCHAR(20),
                total_change DECIMAL(10, 4),
                sparkline TEXT,
                sample_time TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                data_source VARCHAR(50) DEFAULT 'poe_ninja',
                api_version VARCHAR(10) DEFAULT '1.0',
                UNIQUE(currency_name, league, direction, sample_time)
            );
            
            CREATE INDEX IF NOT EXISTS idx_{table_name}_currency_league 
            ON {table_name}(currency_name, league);
            
            CREATE INDEX IF NOT EXISTS idx_{table_name}_sample_time 
            ON {table_name}(sample_time);
            """
            
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                conn.commit()

            # Use manual INSERT with ON CONFLICT to handle duplicates
            insert_sql = f"""
            INSERT INTO {table_name} (
                currency_name, league, direction, value, count, chaos_equivalent,
                listing_count, confidence_level, total_change, sparkline,
                sample_time, created_at, data_source, api_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (currency_name, league, direction, sample_time) 
            DO UPDATE SET
                value = EXCLUDED.value,
                count = EXCLUDED.count,
                chaos_equivalent = EXCLUDED.chaos_equivalent,
                listing_count = EXCLUDED.listing_count,
                confidence_level = EXCLUDED.confidence_level,
                total_change = EXCLUDED.total_change,
                sparkline = EXCLUDED.sparkline,
                created_at = EXCLUDED.created_at,
                data_source = EXCLUDED.data_source,
                api_version = EXCLUDED.api_version
            """
            
            # Prepare data for batch insert
            records_inserted = 0
            records_updated = 0
            
            with conn.cursor() as cursor:
                for _, row in df.iterrows():
                    try:
                        cursor.execute(insert_sql, (
                            row['currency_name'], row['league'], row['direction'],
                            row['value'], row['count'], row['chaos_equivalent'],
                            row['listing_count'], row['confidence_level'], row['total_change'],
                            row['sparkline'], row['sample_time'], row['created_at'],
                            row['data_source'], row['api_version']
                        ))
                        
                        # Check if it was an insert or update
                        if cursor.rowcount > 0:
                            records_inserted += 1
                        
                    except Exception as row_error:
                        self.logger.warning(f"Failed to insert/update row for {row['currency_name']}: {str(row_error)}")
                        continue
                
                conn.commit()
            
            conn.close()
            
            self.logger.info(f"Successfully processed {records_inserted} records to {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing data to database: {str(e)}")
            if 'conn' in locals():
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass
            return False
    
    def get_price_changes(
        self,
        currency_data: List[CurrencyData],
        min_change_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Identify significant price changes from sparkline data.
        
        Args:
            currency_data: List of currency data
            min_change_threshold: Minimum change percentage to report
            
        Returns:
            List of significant price changes
        """
        significant_changes = []
        
        for currency in currency_data:
            if currency.total_change and abs(currency.total_change) >= min_change_threshold:
                change_info = {
                    'currency_name': currency.currency_name,
                    'league': currency.league,
                    'total_change': currency.total_change,
                    'chaos_equivalent': currency.chaos_equivalent,
                    'confidence_level': currency.confidence_level,
                    'sample_time': currency.sample_time.isoformat(),
                    'direction': 'up' if currency.total_change > 0 else 'down'
                }
                significant_changes.append(change_info)
        
        # Sort by absolute change magnitude
        significant_changes.sort(key=lambda x: abs(x['total_change']), reverse=True)
        
        return significant_changes


async def main():
    """Example usage of the PoE Ninja client."""
    logger = MLLogger("PoENinjaExample")
    
    async with PoENinjaClient(logger) as client:
        # Fetch data for Standard league
        print("Fetching Standard league currency data...")
        
        raw_data = await client.fetch_currency_overview("Standard", "Currency")
        if raw_data:
            currency_data = client.parse_currency_data(raw_data, "Standard")
            
            print(f"Found {len(currency_data)} currencies")
            
            # Show top 10 by chaos equivalent
            top_currencies = sorted(currency_data, key=lambda x: x.chaos_equivalent, reverse=True)[:10]
            
            print("\nTop 10 currencies by chaos equivalent:")
            print(f"{'Currency':<30} {'Chaos Equiv':<12} {'Change %':<10} {'Confidence'}")
            print("-" * 70)
            
            for currency in top_currencies:
                change_str = f"{currency.total_change:+.1f}%" if currency.total_change else "N/A"
                print(f"{currency.currency_name:<30} {currency.chaos_equivalent:<12.2f} "
                      f"{change_str:<10} {currency.confidence_level}")
            
            # Find significant price changes
            changes = client.get_price_changes(currency_data, min_change_threshold=10.0)
            if changes:
                print(f"\nSignificant price changes (>10%):")
                for change in changes[:5]:
                    print(f"  {change['currency_name']}: {change['total_change']:+.1f}%")
            
            # Store to database
            print("\nStoring data to database...")
            success = await client.store_to_database(currency_data)
            print(f"Database storage: {'Success' if success else 'Failed'}")


if __name__ == "__main__":
    asyncio.run(main()) 