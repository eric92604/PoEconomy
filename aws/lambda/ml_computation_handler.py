#!/usr/bin/env python3
"""
PoEconomy ML Computation Handler - Event-Driven Architecture

This Lambda function is triggered by DynamoDB Streams when price data changes.
It performs ML computations and stores predictions in the cache table.
- Triggered by: DynamoDB Stream (LivePricesTable)
- Memory: 1024MB (configured for ML workloads)
- Timeout: 5 minutes (ML computation requirements)
- Output: Pre-computed predictions stored in PredictionsCacheTable
"""

import json
import boto3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

# Environment variables
METADATA_TABLE_NAME = os.environ.get('METADATA_TABLE')
LIVE_PRICES_TABLE_NAME = os.environ.get('LIVE_PRICES_TABLE')
PREDICTIONS_CACHE_TABLE_NAME = os.environ.get('PREDICTIONS_CACHE_TABLE')
DATA_LAKE_BUCKET = os.environ.get('DATA_LAKE_BUCKET')

# DynamoDB tables (lazy initialization)
metadata_table = None
live_prices_table = None
predictions_cache_table = None

def get_tables():
    """Initialize DynamoDB tables"""
    global metadata_table, live_prices_table, predictions_cache_table
    if not metadata_table:
        metadata_table = dynamodb.Table(METADATA_TABLE_NAME)
        live_prices_table = dynamodb.Table(LIVE_PRICES_TABLE_NAME)
        predictions_cache_table = dynamodb.Table(PREDICTIONS_CACHE_TABLE_NAME)
    return metadata_table, live_prices_table, predictions_cache_table

def lambda_handler(event, context):
    """ML Computation Handler - Triggered by DynamoDB Stream"""
    start_time = datetime.utcnow()
    
    try:
        # Initialize tables
        metadata_table, live_prices_table, predictions_cache_table = get_tables()
        
        # Process DynamoDB Stream records
        records = event.get('Records', [])
        logger.info(f"Processing {len(records)} DynamoDB stream records")
        
        processed_predictions = 0
        currencies_updated = set()
        
        for record in records:
            try:
                # Extract currency and league from the stream record
                if record['eventName'] in ['INSERT', 'MODIFY']:
                    currency_info = extract_currency_info(record)
                    if currency_info:
                        currencies_updated.add((currency_info['currency'], currency_info['league']))
            except Exception as e:
                logger.warning(f"Failed to process record: {str(e)}")
                continue
        
        # Generate predictions for all affected currencies
        for currency, league in currencies_updated:
            try:
                predictions = generate_predictions_for_currency(currency, league)
                processed_predictions += len(predictions)
                logger.info(f"Generated {len(predictions)} predictions for {currency} in {league}")
            except Exception as e:
                logger.error(f"Failed to generate predictions for {currency}: {str(e)}")
                continue
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'ML computation completed successfully',
                'records_processed': len(records),
                'currencies_updated': len(currencies_updated),
                'predictions_generated': processed_predictions,
                'execution_time_seconds': round(execution_time, 2),
                'timestamp': start_time.isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"ML computation error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'ML computation failed',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def extract_currency_info(record: Dict) -> Optional[Dict]:
    """Extract currency and league information from DynamoDB stream record"""
    try:
        if 'dynamodb' in record and 'NewImage' in record['dynamodb']:
            new_image = record['dynamodb']['NewImage']
            currency_league = new_image.get('currency_league', {}).get('S', '')
            
            if '#' in currency_league:
                currency, league = currency_league.split('#', 1)
                return {
                    'currency': currency,
                    'league': league,
                    'timestamp': new_image.get('timestamp', {}).get('S', ''),
                    'value': new_image.get('value', {}).get('S', '')
                }
        return None
    except Exception as e:
        logger.warning(f"Failed to extract currency info: {str(e)}")
        return None

def generate_predictions_for_currency(currency: str, league: str) -> List[Dict]:
    """Generate ML predictions for a specific currency and league"""
    try:
        # Get recent price data for this currency
        recent_prices = get_recent_prices(currency, league)
        
        if len(recent_prices) < 3:
            logger.warning(f"Insufficient data for {currency} in {league}")
            return []
        
        predictions = []
        prediction_horizons = [1, 3, 7, 14]  # 1 day, 3 days, 1 week, 2 weeks
        
        for horizon_days in prediction_horizons:
            prediction = calculate_prediction(recent_prices, horizon_days)
            
            if prediction:
                # Store prediction in cache
                cache_key = f"{currency}#{league}#{horizon_days}"
                store_prediction(cache_key, currency, league, prediction)
                predictions.append(prediction)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction generation failed for {currency}: {str(e)}")
        return []

def get_recent_prices(currency: str, league: str, limit: int = 100) -> List[Dict]:
    """Get recent prices for a currency from DynamoDB"""
    try:
        currency_league_key = f"{currency}#{league}"
        
        response = live_prices_table.query(
            KeyConditionExpression='currency_league = :pk',
            ExpressionAttributeValues={':pk': currency_league_key},
            ScanIndexForward=False,  # Most recent first
            Limit=limit
        )
        
        prices = []
        for item in response.get('Items', []):
            try:
                prices.append({
                    'timestamp': item['timestamp'],
                    'value': float(item['value']),
                    'source': 'dynamodb'
                })
            except (ValueError, KeyError):
                continue
                
        return sorted(prices, key=lambda x: x['timestamp'])
        
    except Exception as e:
        logger.error(f"Failed to get recent prices: {str(e)}")
        return []

def calculate_prediction(prices: List[Dict], horizon_days: int) -> Optional[Dict]:
    """Calculate ML prediction using trend analysis"""
    try:
        if len(prices) < 2:
            return None
        
        current_price = prices[-1]['value']
        
        # Trend analysis (can be replaced with more sophisticated ML models)
        if len(prices) >= 10:
            short_term_avg = sum(p['value'] for p in prices[-5:]) / 5
            medium_term_avg = sum(p['value'] for p in prices[-10:]) / 10
        else:
            short_term_avg = sum(p['value'] for p in prices[-3:]) / min(3, len(prices))
            medium_term_avg = sum(p['value'] for p in prices) / len(prices)
        
        # Calculate trend
        trend_strength = abs(short_term_avg - medium_term_avg) / medium_term_avg
        
        if short_term_avg > medium_term_avg * 1.02:
            trend = 'bullish'
            trend_multiplier = 1 + (trend_strength * horizon_days * 0.1)
        elif short_term_avg < medium_term_avg * 0.98:
            trend = 'bearish'
            trend_multiplier = 1 - (trend_strength * horizon_days * 0.1)
        else:
            trend = 'stable'
            trend_multiplier = 1.0
        
        predicted_price = current_price * trend_multiplier
        price_change_percent = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate confidence
        confidence = min(0.9, max(0.1, len(prices) / 50.0))
        
        return {
            'predicted_price': round(predicted_price, 2),
            'current_price': round(current_price, 2),
            'price_change_percent': round(price_change_percent, 2),
            'trend': trend,
            'confidence': round(confidence, 2),
            'horizon_days': horizon_days,
            'data_points_used': len(prices),
            'computed_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction calculation failed: {str(e)}")
        return None

def store_prediction(cache_key: str, currency: str, league: str, prediction: Dict) -> None:
    """Store prediction in DynamoDB cache table"""
    try:
        ttl = int((datetime.utcnow() + timedelta(hours=24)).timestamp())  # 24-hour cache
        
        predictions_cache_table.put_item(Item={
            'prediction_key': cache_key,
            'currency_league': f"{currency}#{league}",
            'prediction_data': json.dumps(prediction, default=str),
            'created_at': datetime.utcnow().isoformat(),
            'ttl': ttl,
            'currency': currency,
            'league': league,
            'horizon_days': prediction['horizon_days']
        })
        
        logger.info(f"Stored prediction for {cache_key}")
        
    except Exception as e:
        logger.error(f"Failed to store prediction: {str(e)}") 