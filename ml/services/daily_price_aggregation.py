#!/usr/bin/env python3
"""
Daily Price Aggregation Service

This service aggregates hourly price data from LivePricesTable into daily price data
for the DailyPricesTable. It calculates OHLC (Open, High, Low, Close) prices and
other daily statistics.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

# Add the ml directory to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ml.utils.common_utils import MLLogger
from ml.utils.data_sources import get_current_seasonal_league_from_table
from config.training_config import MLConfig


@dataclass
class DailyPriceData:
    """Daily price aggregation data structure."""
    
    currency: str
    league: str
    date: str  # YYYY-MM-DD format
    high_price: Decimal
    low_price: Decimal
    avg_price: Decimal
    price_change_percent: Decimal


class DailyPriceAggregator:
    """Service for aggregating hourly prices into daily data."""
    
    def __init__(self, region_name: str = None, logger: MLLogger = None):
        """Initialize the daily price aggregator.
        
        Args:
            region_name: AWS region name
            logger: Logger instance
        """
        self.region_name = region_name or os.getenv("AWS_REGION", "us-west-2")
        self.logger = logger or MLLogger("DailyPriceAggregator")
        
        # Initialize DynamoDB
        self.dynamodb = boto3.resource("dynamodb", region_name=self.region_name)
        
        # Get table names from environment or config
        config = MLConfig()
        self.live_prices_table_name = os.getenv("LIVE_PRICES_TABLE", os.getenv("DYNAMO_CURRENCY_PRICES_TABLE", config.dynamo.currency_prices_table))
        self.daily_prices_table_name = os.getenv("DAILY_PRICES_TABLE", "poeconomy-daily-prices")
        self.league_metadata_table_name = os.getenv("LEAGUE_METADATA_TABLE", os.getenv("DYNAMO_LEAGUE_METADATA_TABLE", config.dynamo.league_metadata_table))
        self.currency_metadata_table_name = os.getenv("CURRENCY_METADATA_TABLE", os.getenv("DYNAMO_CURRENCY_METADATA_TABLE", config.dynamo.currency_metadata_table))
        
        self.live_prices_table = self.dynamodb.Table(self.live_prices_table_name)
        self.daily_prices_table = self.dynamodb.Table(self.daily_prices_table_name)
        self.league_metadata_table = self.dynamodb.Table(self.league_metadata_table_name)
        self.currency_metadata_table = self.dynamodb.Table(self.currency_metadata_table_name)
        
        self.logger.info(f"Initialized DailyPriceAggregator with tables: {self.live_prices_table_name}, {self.daily_prices_table_name}, {self.league_metadata_table_name}, {self.currency_metadata_table_name}")
    
    def get_current_seasonal_league(self) -> Optional[str]:
        """Get the current active seasonal league from the league metadata table.
        
        Returns:
            Name of the current seasonal league or None if not found
        """
        return get_current_seasonal_league_from_table(self.league_metadata_table, self.logger)
    
    def get_available_currencies(self, league: str = None) -> List[str]:
        """Get currencies that exist in both metadata and live prices tables.
        
        Args:
            league: League name to filter by (if None, gets currencies from all leagues)
            
        Returns:
            List of currency names that exist in both tables
        """
        try:
            # Get currencies from metadata table
            metadata_currencies = set()
            
            if league:
                # Filter by specific league
                response = self.currency_metadata_table.scan(
                    FilterExpression=Key("league").eq(league)
                )
            else:
                # Get all currencies from metadata
                response = self.currency_metadata_table.scan()
            
            for item in response.get("Items", []):
                currency_name = item.get("currency")
                if currency_name:
                    metadata_currencies.add(currency_name)
            
            # Handle pagination for metadata table
            while "LastEvaluatedKey" in response:
                if league:
                    response = self.currency_metadata_table.scan(
                        FilterExpression=Key("league").eq(league),
                        ExclusiveStartKey=response["LastEvaluatedKey"]
                    )
                else:
                    response = self.currency_metadata_table.scan(
                        ExclusiveStartKey=response["LastEvaluatedKey"]
                    )
                
                for item in response.get("Items", []):
                    currency_name = item.get("currency")
                    if currency_name:
                        metadata_currencies.add(currency_name)
            
            self.logger.info(f"Found {len(metadata_currencies)} currencies in metadata table")
            
            # Verify currencies exist in live prices table
            available_currencies = []
            
            for currency in metadata_currencies:
                if self._currency_exists_in_live_prices(currency, league):
                    available_currencies.append(currency)
                else:
                    self.logger.debug(f"Currency {currency} not found in live prices table")
            
            self.logger.info(f"Found {len(available_currencies)} currencies available in both metadata and live prices tables")
            return available_currencies
            
        except Exception as e:
            self.logger.error(f"Error getting available currencies: {e}")
            # Fallback to default currencies
            return ["Divine Orb", "Chaos Orb", "Exalted Orb"]
    
    def _currency_exists_in_live_prices(self, currency: str, league: str = None) -> bool:
        """Check if a currency exists in the live prices table.
        
        Args:
            currency: Currency name to check
            league: League name to check (if None, checks all leagues)
            
        Returns:
            True if currency exists in live prices table, False otherwise
        """
        try:
            if league:
                # Check specific currency-league combination
                currency_league = f"{currency}#{league}"
                response = self.live_prices_table.query(
                    KeyConditionExpression=Key("currency_league").eq(currency_league),
                    Limit=1
                )
                return len(response.get("Items", [])) > 0
            else:
                # Check if currency exists in any league
                # We'll scan for any currency_league that starts with the currency name
                response = self.live_prices_table.scan(
                    FilterExpression=Key("currency").eq(currency),
                    Limit=1
                )
                return len(response.get("Items", [])) > 0
                
        except Exception as e:
            self.logger.debug(f"Error checking if currency {currency} exists in live prices: {e}")
            return False
    
    def aggregate_daily_prices(
        self, 
        currency: str, 
        league: str, 
        target_date: str = None
    ) -> Optional[DailyPriceData]:
        """Aggregate hourly prices for a specific currency/league/date into daily data.
        
        Args:
            currency: Currency name (e.g., "Divine Orb")
            league: League name (e.g., "Mercenaries")
            target_date: Date in YYYY-MM-DD format (defaults to yesterday)
            
        Returns:
            DailyPriceData object or None if no data found
        """
        if target_date is None:
            # Default to yesterday in UTC
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            target_date = yesterday.strftime("%Y-%m-%d")
        
        self.logger.info(f"Aggregating daily prices for {currency} in {league} on {target_date}")
        
        # Get all hourly data for the target date
        hourly_data = self._get_hourly_data_for_date(currency, league, target_date)
        
        if not hourly_data:
            self.logger.warning(f"No hourly data found for {currency} in {league} on {target_date}")
            return None
        
        # Aggregate the data
        daily_data = self._aggregate_hourly_to_daily(hourly_data, currency, league, target_date)
        
        return daily_data
    
    def _get_hourly_data_for_date(self, currency: str, league: str, date: str) -> List[Dict[str, Any]]:
        """Get all hourly price data for a specific date.
        
        Args:
            currency: Currency name
            league: League name
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of hourly price items
        """
        # Convert date to timestamp range (UTC)
        start_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = start_date + timedelta(days=1)
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        self.logger.debug(f"Querying timestamps from {start_timestamp} to {end_timestamp} for {currency} in {league}")
        
        currency_league = f"{currency}#{league}"
        
        try:
            # Query the live prices table for the date range
            response = self.live_prices_table.query(
                KeyConditionExpression=Key("currency_league").eq(currency_league) & 
                                     Key("timestamp").between(start_timestamp, end_timestamp),
                ScanIndexForward=True  # Oldest first
            )
            
            items = response.get("Items", [])
            
            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self.live_prices_table.query(
                    KeyConditionExpression=Key("currency_league").eq(currency_league) & 
                                         Key("timestamp").between(start_timestamp, end_timestamp),
                    ScanIndexForward=True,
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                items.extend(response.get("Items", []))
            
            self.logger.info(f"Found {len(items)} hourly data points for {currency} in {league} on {date}")
            return items
            
        except ClientError as e:
            self.logger.error(f"Error querying hourly data: {e}")
            return []
    
    def _aggregate_hourly_to_daily(
        self, 
        hourly_data: List[Dict[str, Any]], 
        currency: str, 
        league: str, 
        date: str
    ) -> DailyPriceData:
        """Aggregate hourly data into daily statistics.
        
        Args:
            hourly_data: List of hourly price items
            currency: Currency name
            league: League name
            date: Date in YYYY-MM-DD format
            
        Returns:
            DailyPriceData object
        """
        if not hourly_data:
            raise ValueError("No hourly data provided for aggregation")
        
        # Sort by timestamp to ensure proper order
        hourly_data.sort(key=lambda x: x["timestamp"])
        
        # Extract prices and convert to Decimal
        prices = []
        timestamps = []
        
        for item in hourly_data:
            price = item.get("price")
            if price is not None:
                if isinstance(price, (int, float)):
                    prices.append(Decimal(str(price)))
                elif isinstance(price, Decimal):
                    prices.append(price)
                else:
                    prices.append(Decimal(str(price)))
                timestamps.append(item["timestamp"])
        
        if not prices:
            raise ValueError("No valid prices found in hourly data")
        
        # Calculate price statistics
        high_price = max(prices)
        low_price = min(prices)
        avg_price = sum(prices) / len(prices)
        
        # Calculate price change percentage (using first and last prices)
        if len(prices) > 1 and prices[0] > 0:
            price_change_percent = ((prices[-1] - prices[0]) / prices[0]) * 100
        else:
            price_change_percent = Decimal("0")
        
        return DailyPriceData(
            currency=currency,
            league=league,
            date=date,
            high_price=high_price,
            low_price=low_price,
            avg_price=avg_price,
            price_change_percent=price_change_percent
        )
    
    def save_daily_price(self, daily_data: DailyPriceData) -> bool:
        """Save daily price data to DynamoDB.
        
        Args:
            daily_data: DailyPriceData object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            item = {
                "currency_league": f"{daily_data.currency}#{daily_data.league}",
                "date": daily_data.date,
                "currency": daily_data.currency,
                "league": daily_data.league,
                "high_price": daily_data.high_price,
                "low_price": daily_data.low_price,
                "avg_price": daily_data.avg_price,
                "price_change_percent": daily_data.price_change_percent,
                "created_at": int(time.time())
            }
            
            self.daily_prices_table.put_item(Item=item)
            self.logger.info(f"Saved daily price data for {daily_data.currency} in {daily_data.league} on {daily_data.date}")
            return True
            
        except ClientError as e:
            self.logger.error(f"Error saving daily price data: {e}")
            return False
    
    def aggregate_and_save_daily_prices(
        self, 
        currencies: List[str], 
        leagues: List[str] = None, 
        target_date: str = None,
        use_current_seasonal_only: bool = True
    ) -> Dict[str, Any]:
        """Aggregate and save daily prices for multiple currencies and leagues.
        
        Args:
            currencies: List of currency names
            leagues: List of league names (if None and use_current_seasonal_only=True, uses current seasonal league)
            target_date: Date in YYYY-MM-DD format (defaults to yesterday)
            use_current_seasonal_only: If True, only process the current seasonal league
            
        Returns:
            Dictionary with aggregation results
        """
        results = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Determine which leagues to process
        if use_current_seasonal_only and leagues is None:
            current_league = self.get_current_seasonal_league()
            if current_league:
                leagues = [current_league]
                self.logger.info(f"Processing only current seasonal league: {current_league}")
            else:
                self.logger.error("No current seasonal league found and use_current_seasonal_only=True")
                results["errors"].append("No current seasonal league found")
                return results
        elif leagues is None:
            # Fallback to default leagues if none specified
            leagues = ["Mercenaries", "Standard"]
        
        # Get available currencies from metadata and live prices tables
        if not currencies:
            # Get currencies that exist in both metadata and live prices tables
            available_currencies = set()
            for league in leagues:
                league_currencies = self.get_available_currencies(league)
                available_currencies.update(league_currencies)
            
            currencies = list(available_currencies)
            self.logger.info(f"Found {len(currencies)} currencies to process from metadata and live prices tables")
        
        for currency in currencies:
            for league in leagues:
                try:
                    results["processed"] += 1
                    
                    daily_data = self.aggregate_daily_prices(currency, league, target_date)
                    
                    if daily_data:
                        success = self.save_daily_price(daily_data)
                        if success:
                            results["successful"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"Failed to save {currency} in {league}")
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"No data found for {currency} in {league}")
                        
                except Exception as e:
                    results["failed"] += 1
                    error_msg = f"Error processing {currency} in {league}: {e}"
                    results["errors"].append(error_msg)
                    self.logger.error(error_msg)
        
        self.logger.info(f"Daily aggregation complete: {results['successful']}/{results['processed']} successful")
        return results
    
    def get_daily_prices(
        self, 
        currency: str, 
        league: str, 
        start_date: str, 
        end_date: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get daily price data for a currency/league within a date range.
        
        Args:
            currency: Currency name
            league: League name
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to start_date)
            limit: Maximum number of results
            
        Returns:
            List of daily price items
        """
        if end_date is None:
            end_date = start_date
        
        currency_league = f"{currency}#{league}"
        
        try:
            response = self.daily_prices_table.query(
                KeyConditionExpression=Key("currency_league").eq(currency_league) & 
                                     Key("date").between(start_date, end_date),
                ScanIndexForward=True,  # Oldest first
                Limit=limit
            )
            
            return response.get("Items", [])
            
        except ClientError as e:
            self.logger.error(f"Error querying daily prices: {e}")
            return []


def main():
    """Main function for testing the daily price aggregator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Price Aggregation Service")
    parser.add_argument("--currency", help="Currency name (if not provided, uses common currencies)")
    parser.add_argument("--league", help="League name (if not provided, uses current seasonal league)")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format (defaults to yesterday)")
    parser.add_argument("--action", choices=["aggregate", "query", "debug"], default="aggregate", help="Action to perform")
    parser.add_argument("--start-date", help="Start date for query (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for query (YYYY-MM-DD)")
    parser.add_argument("--use-current-seasonal-only", action="store_true", default=True, help="Only process current seasonal league")
    parser.add_argument("--debug-timestamps", action="store_true", help="Debug timestamp ranges for queries")
    
    args = parser.parse_args()
    
    logger = MLLogger("DailyPriceAggregator", level="INFO")
    aggregator = DailyPriceAggregator(logger=logger)
    
    if args.action == "debug":
        # Debug current seasonal league and timestamp ranges
        current_league = aggregator.get_current_seasonal_league()
        print(f"Current seasonal league: {current_league}")
        
        if args.date:
            target_date = args.date
        else:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            target_date = yesterday.strftime("%Y-%m-%d")
        
        print(f"Target date: {target_date}")
        
        # Show timestamp range
        start_date = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = start_date + timedelta(days=1)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        print(f"Timestamp range: {start_timestamp} to {end_timestamp}")
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
        
        # Show available currencies
        available_currencies = aggregator.get_available_currencies(current_league)
        print(f"Available currencies in {current_league}: {len(available_currencies)}")
        if available_currencies:
            print("Sample currencies:", available_currencies[:10])  # Show first 10
        
        # Test query for a specific currency
        if args.currency and current_league:
            test_data = aggregator._get_hourly_data_for_date(args.currency, current_league, target_date)
            print(f"Found {len(test_data)} data points for {args.currency} in {current_league}")
            if test_data:
                print("Sample data points:")
                for i, item in enumerate(test_data[:3]):  # Show first 3 items
                    print(f"  {i+1}: timestamp={item.get('timestamp')}, price={item.get('price')}")
    
    elif args.action == "aggregate":
        # Determine currencies and leagues
        currencies = [args.currency] if args.currency else None  # None means get from metadata
        leagues = [args.league] if args.league else None
        
        # Use the improved aggregation method
        results = aggregator.aggregate_and_save_daily_prices(
            currencies=currencies,
            leagues=leagues,
            target_date=args.date,
            use_current_seasonal_only=args.use_current_seasonal_only
        )
        
        print(f"Aggregation results: {results['successful']}/{results['processed']} successful")
        if results["errors"]:
            print("Errors:")
            for error in results["errors"]:
                print(f"  - {error}")
    
    elif args.action == "query":
        if not args.start_date:
            print("--start-date is required for query action")
            return
        
        daily_prices = aggregator.get_daily_prices(
            args.currency or "Divine Orb", 
            args.league or "Mercenaries", 
            args.start_date, 
            args.end_date
        )
        
        print(f"Found {len(daily_prices)} daily price records:")
        for price in daily_prices:
            print(f"  {price['date']}: H:{price['high_price']} L:{price['low_price']} Avg:{price['avg_price']} Change:{price['price_change_percent']}%")


if __name__ == "__main__":
    main()
