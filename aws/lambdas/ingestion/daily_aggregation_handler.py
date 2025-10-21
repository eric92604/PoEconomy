#!/usr/bin/env python3
"""
Daily Price Aggregation Lambda Handler

This Lambda function aggregates hourly price data into daily price data.
It can be triggered by EventBridge (daily) or manually via API.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import boto3

from ml.services.daily_price_aggregation import DailyPriceAggregator
from ml.utils.common_utils import MLLogger, setup_standard_logging
from ..config import AppEnvironment, load_environment

# Set up standardized logging
LOGGER = setup_standard_logging(
    name="DailyAggregationHandler",
    level=os.getenv("LOG_LEVEL", "INFO"),
    console_output=True,
    suppress_external=True
)


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Lambda handler for daily price aggregation.
    
    Expected event format:
    {
        "currencies": ["Divine Orb", "Chaos Orb"],  # Optional, defaults to all
        "leagues": ["Mercenaries", "Standard"],     # Optional, defaults to all
        "target_date": "2024-01-15",               # Optional, defaults to yesterday
        "action": "aggregate"                      # Optional, defaults to "aggregate"
    }
    
    Returns:
        Dictionary with aggregation results
    """
    LOGGER.debug("Received event: %s", json.dumps(event or {}))
    
    app_env = load_environment()
    session = boto3.session.Session(region_name=app_env.region_name)
    dynamodb = session.resource("dynamodb")
    
    logger = MLLogger("DailyAggregationLambda", level=os.getenv("LOG_LEVEL", "INFO"))
    
    try:
        # Parse event parameters
        currencies = event.get("currencies")
        leagues = event.get("leagues")
        target_date = event.get("target_date")
        action = event.get("action", "aggregate")
        
        # Default to yesterday if no date specified
        if not target_date:
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.strftime("%Y-%m-%d")
        
        logger.info(f"Starting daily aggregation for date: {target_date}")
        logger.info(f"Event: {json.dumps(event)}")
        
        # Initialize aggregator
        aggregator = DailyPriceAggregator(region_name=app_env.region_name, logger=logger)
        
        if action == "aggregate":
            # Use the improved aggregation method that automatically selects currencies from metadata
            # and current seasonal league
            results = aggregator.aggregate_and_save_daily_prices(
                currencies=currencies,  # Will be None if not specified, so gets from metadata
                leagues=leagues,  # Will be None, so it uses current seasonal league
                target_date=target_date,
                use_current_seasonal_only=True
            )
            
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Daily aggregation completed",
                    "target_date": target_date,
                    "results": results
                })
            }
        
        elif action == "query":
            # Query existing daily prices
            currency = event.get("currency")
            league = event.get("league")
            start_date = event.get("start_date")
            end_date = event.get("end_date")
            
            if not all([currency, league, start_date]):
                return {
                    "statusCode": 400,
                    "body": json.dumps({
                        "error": "currency, league, and start_date are required for query action"
                    })
                }
            
            daily_prices = aggregator.get_daily_prices(
                currency=currency,
                league=league,
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "currency": currency,
                    "league": league,
                    "start_date": start_date,
                    "end_date": end_date,
                    "count": len(daily_prices),
                    "data": daily_prices
                })
            }
        
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": f"Unknown action: {action}. Supported actions: aggregate, query"
                })
            }
    
    except Exception as e:
        logger.error(f"Error in daily aggregation lambda: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }


def _get_default_currencies_and_leagues(app_env: AppEnvironment, dynamodb, logger: MLLogger) -> tuple[List[str], List[str]]:
    """Get default currencies and leagues from configuration.
    
    Args:
        app_env: Application environment configuration
        dynamodb: DynamoDB resource
        logger: Logger instance
        
    Returns:
        Tuple of (currencies, leagues)
    """
    try:
        # Get currencies from metadata table
        metadata_table = dynamodb.Table(app_env.currency_metadata_table)
        
        # Get unique currencies and leagues
        response = metadata_table.scan(
            ProjectionExpression="currency, league"
        )
        
        currencies = set()
        leagues = set()
        
        for item in response.get("Items", []):
            if item.get("currency"):
                currencies.add(item["currency"])
            if item.get("league"):
                leagues.add(item["league"])
        
        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = metadata_table.scan(
                ProjectionExpression="currency, league",
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            
            for item in response.get("Items", []):
                if item.get("currency"):
                    currencies.add(item["currency"])
                if item.get("league"):
                    leagues.add(item["league"])
        
        currencies_list = sorted(list(currencies))
        leagues_list = sorted(list(leagues))
        
        logger.info(f"Found {len(currencies_list)} currencies and {len(leagues_list)} leagues")
        
        return currencies_list, leagues_list
        
    except Exception as e:
        logger.error(f"Error getting default currencies and leagues: {e}")
        # Fallback to common currencies and leagues
        return ["Divine Orb", "Chaos Orb", "Exalted Orb"], ["Mercenaries", "Standard"]


if __name__ == "__main__":
    # Test the lambda handler
    test_event = {
        "currencies": ["Divine Orb"],
        "leagues": ["Mercenaries"],
        "target_date": "2024-01-15"
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))


__all__ = ["lambda_handler"]