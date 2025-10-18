#!/usr/bin/env python3
"""
Daily Price Aggregation Service

This service aggregates hourly price data from LivePricesTable into daily price data
for the DailyPricesTable. It calculates OHLC (Open, High, Low, Close) prices and
other daily statistics.
"""

import os
import time
from datetime import datetime, timedelta
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

from utils.logging_config import MLLogger
from config.training_config import MLConfig


@dataclass
class DailyPriceData:
    """Daily price aggregation data structure."""
    
    currency: str
    league: str
    date: str  # YYYY-MM-DD format
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    avg_price: Decimal
    volume: int  # Number of hourly data points
    price_change_percent: Decimal
    first_timestamp: int
    last_timestamp: int


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
        self.live_prices_table_name = os.getenv("DYNAMO_LIVE_PRICES_TABLE", config.dynamo.currency_prices_table)
        self.daily_prices_table_name = os.getenv("DYNAMO_DAILY_PRICES_TABLE", "poeconomy-daily-prices")
        
        self.live_prices_table = self.dynamodb.Table(self.live_prices_table_name)
        self.daily_prices_table = self.dynamodb.Table(self.daily_prices_table_name)
        
        self.logger.info(f"Initialized DailyPriceAggregator with tables: {self.live_prices_table_name}, {self.daily_prices_table_name}")
    
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
            # Default to yesterday
            yesterday = datetime.now() - timedelta(days=1)
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
        # Convert date to timestamp range
        start_date = datetime.strptime(date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
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
        
        # Calculate OHLC and other statistics
        open_price = prices[0]
        close_price = prices[-1]
        high_price = max(prices)
        low_price = min(prices)
        avg_price = sum(prices) / len(prices)
        
        # Calculate price change percentage
        if open_price > 0:
            price_change_percent = ((close_price - open_price) / open_price) * 100
        else:
            price_change_percent = Decimal("0")
        
        return DailyPriceData(
            currency=currency,
            league=league,
            date=date,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            avg_price=avg_price,
            volume=len(prices),
            price_change_percent=price_change_percent,
            first_timestamp=timestamps[0],
            last_timestamp=timestamps[-1]
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
                "open_price": daily_data.open_price,
                "high_price": daily_data.high_price,
                "low_price": daily_data.low_price,
                "close_price": daily_data.close_price,
                "avg_price": daily_data.avg_price,
                "volume": daily_data.volume,
                "price_change_percent": daily_data.price_change_percent,
                "first_timestamp": daily_data.first_timestamp,
                "last_timestamp": daily_data.last_timestamp,
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
        leagues: List[str], 
        target_date: str = None
    ) -> Dict[str, Any]:
        """Aggregate and save daily prices for multiple currencies and leagues.
        
        Args:
            currencies: List of currency names
            leagues: List of league names
            target_date: Date in YYYY-MM-DD format (defaults to yesterday)
            
        Returns:
            Dictionary with aggregation results
        """
        results = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
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
    parser.add_argument("--currency", required=True, help="Currency name")
    parser.add_argument("--league", required=True, help="League name")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format (defaults to yesterday)")
    parser.add_argument("--action", choices=["aggregate", "query"], default="aggregate", help="Action to perform")
    parser.add_argument("--start-date", help="Start date for query (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for query (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    logger = MLLogger("DailyPriceAggregator", level="INFO")
    aggregator = DailyPriceAggregator(logger=logger)
    
    if args.action == "aggregate":
        daily_data = aggregator.aggregate_daily_prices(args.currency, args.league, args.date)
        if daily_data:
            print(f"Daily data for {args.currency} in {args.league} on {daily_data.date}:")
            print(f"  Open: {daily_data.open_price}")
            print(f"  High: {daily_data.high_price}")
            print(f"  Low: {daily_data.low_price}")
            print(f"  Close: {daily_data.close_price}")
            print(f"  Average: {daily_data.avg_price}")
            print(f"  Volume: {daily_data.volume}")
            print(f"  Change: {daily_data.price_change_percent}%")
            
            # Save the data
            success = aggregator.save_daily_price(daily_data)
            print(f"Saved to database: {success}")
        else:
            print("No data found for aggregation")
    
    elif args.action == "query":
        if not args.start_date:
            print("--start-date is required for query action")
            return
        
        daily_prices = aggregator.get_daily_prices(
            args.currency, 
            args.league, 
            args.start_date, 
            args.end_date
        )
        
        print(f"Found {len(daily_prices)} daily price records:")
        for price in daily_prices:
            print(f"  {price['date']}: O:{price['open_price']} H:{price['high_price']} L:{price['low_price']} C:{price['close_price']}")


if __name__ == "__main__":
    main()
