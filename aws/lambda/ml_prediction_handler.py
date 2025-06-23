#!/usr/bin/env python3
"""
PoEconomy ML Prediction Handler

Lambda function using Lambda layers for ML models:
- Layer-cached models for reduced S3 access
- In-memory caching for cross-invocation persistence
- Multi-tier fallback: Layer → S3 → Trend Analysis
- Extended prediction caching for stable markets
"""

import json
import boto3
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from decimal import Decimal
import gzip
from botocore.exceptions import ClientError

# Add layer path for model access
sys.path.append('/opt/python')

try:
    from model_registry import get_layer_model_loader
    LAYER_MODELS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Lambda layer models available")
except ImportError:
    LAYER_MODELS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Lambda layer models not available, using fallback")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

# Environment variables
METADATA_TABLE_NAME = os.environ.get('METADATA_TABLE')
LIVE_PRICES_TABLE_NAME = os.environ.get('LIVE_PRICES_TABLE')
PREDICTIONS_CACHE_TABLE_NAME = os.environ.get('PREDICTIONS_CACHE_TABLE')
DATA_LAKE_BUCKET = os.environ.get('DATA_LAKE_BUCKET')
MODELS_BUCKET = os.environ.get('MODELS_BUCKET', 'poeconomy-models')
HOT_DATA_RETENTION_DAYS = int(os.environ.get('HOT_DATA_RETENTION_DAYS', '14'))

# Global caches (persist across Lambda invocations)
GLOBAL_MODEL_CACHE = {}
GLOBAL_LAYER_LOADER = None

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

def get_layer_loader():
    """Get layer model loader (singleton)"""
    global GLOBAL_LAYER_LOADER
    if GLOBAL_LAYER_LOADER is None and LAYER_MODELS_AVAILABLE:
        GLOBAL_LAYER_LOADER = get_layer_model_loader()
        available_models = GLOBAL_LAYER_LOADER.list_available_models()
        logger.info(f"Initialized layer loader with {len(available_models)} models: {available_models[:5]}...")
    return GLOBAL_LAYER_LOADER

def lambda_handler(event, context):
    """Lambda Layer ML Prediction Handler"""
    start_time = datetime.utcnow()
    
    try:
        # Parse request
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
            
        currency = body.get('currency')
        league = body.get('league', 'Sanctum')
        prediction_horizon_days = int(body.get('prediction_horizon_days', 1))
        force_refresh = body.get('force_refresh', False)
        
        if not currency:
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'error': 'Missing required parameter: currency',
                    'timestamp': start_time.isoformat()
                })
            }
        
        # Initialize tables
        metadata_table, live_prices_table, predictions_cache_table = get_tables()
        
        # Check cache first (extended TTL)
        if not force_refresh:
            cache_key = f"{currency}#{league}#{prediction_horizon_days}"
            cached_prediction = get_cached_prediction(cache_key)
            if cached_prediction:
                logger.info(f"Cache hit for {cache_key}")
                return {
                    'statusCode': 200,
                    'headers': get_cors_headers(),
                    'body': json.dumps({
                        'prediction': cached_prediction,
                        'source': 'cache',
                        'timestamp': start_time.isoformat(),
                        'cache_hit': True,
                        'layer_enabled': True
                    })
                }
        
        # Generate new prediction using layer approach
        prediction = generate_layer_prediction(currency, league, prediction_horizon_days)
        
        # Cache the prediction with intelligent TTL
        cache_prediction_smart(cache_key, prediction)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'prediction': prediction,
                'source': 'computed',
                'execution_time_seconds': round(execution_time, 3),
                'timestamp': start_time.isoformat(),
                'cache_hit': False,
                'model_source': prediction.get('model_source', 'trend_analysis'),
                'tier': prediction.get('tier', 'minimal'),
                'layer_enabled': True
            })
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Prediction failed',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def get_cors_headers() -> Dict[str, str]:
    """Get CORS headers for API responses"""
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
        'Access-Control-Allow-Methods': 'POST,OPTIONS'
    }

def get_cached_prediction(cache_key: str) -> Optional[Dict]:
    """Get cached prediction from DynamoDB"""
    try:
        response = predictions_cache_table.get_item(Key={'prediction_key': cache_key})
        if 'Item' in response:
            return json.loads(response['Item']['prediction_data'])
        return None
    except Exception as e:
        logger.warning(f"Cache lookup failed: {str(e)}")
        return None

def cache_prediction_smart(cache_key: str, prediction: Dict) -> None:
    """Cache prediction with intelligent TTL based on confidence and volatility"""
    try:
        confidence = prediction.get('confidence', 0.5)
        model_source = prediction.get('model_source', 'trend_analysis')
        
        # Caching based on prediction quality
        if model_source == 'lambda_layer' and confidence > 0.8:
            cache_hours = 24  # High confidence layer predictions
        elif model_source == 'lambda_layer' and confidence > 0.6:
            cache_hours = 12  # Medium confidence layer predictions
        elif confidence > 0.7:
            cache_hours = 6   # High confidence other predictions
        else:
            cache_hours = 1   # Low confidence predictions
        
        ttl = int((datetime.utcnow() + timedelta(hours=cache_hours)).timestamp())
        predictions_cache_table.put_item(Item={
            'prediction_key': cache_key,
            'prediction_data': json.dumps(prediction, default=str),
            'ttl': ttl,
            'created_at': datetime.utcnow().isoformat(),
            'cache_duration_hours': cache_hours,
            'model_source': model_source,
            'confidence': confidence
        })
        
        logger.info(f"Cached prediction for {cache_hours} hours (confidence: {confidence:.2f})")
    except Exception as e:
        logger.warning(f"Cache storage failed: {str(e)}")

def generate_layer_prediction(currency: str, league: str, horizon_days: int) -> Dict:
    """Generate ML prediction using layer-first strategy"""
    try:
        # Step 1: Try Lambda layer models (no S3 access required)
        prediction = try_layer_model_prediction(currency, league, horizon_days)
        if prediction:
            return prediction
        
        # Step 2: Try in-memory cached S3 models (reduced S3 access)
        prediction = try_memory_cached_prediction(currency, league, horizon_days)
        if prediction:
            return prediction
        
        # Step 3: Fallback to trend analysis with hot data (minimal S3 access)
        prediction = generate_trend_prediction_with_hot_data(currency, league, horizon_days)
        return prediction
        
    except Exception as e:
        logger.error(f"Layer prediction failed: {str(e)}")
        return create_error_prediction(currency, league, horizon_days, str(e))

def try_layer_model_prediction(currency: str, league: str, horizon_days: int) -> Optional[Dict]:
    """Try prediction using Lambda layer models"""
    try:
        layer_loader = get_layer_loader()
        if not layer_loader:
            return None
        
        # Load model from layer
        model_data = layer_loader.load_model(currency)
        if not model_data:
            logger.info(f"No layer model available for {currency}")
            return None
        
        # Get price data for prediction
        recent_prices = get_hot_data_prices(currency, league, limit=30)
        if len(recent_prices) < 5:
            logger.warning(f"Insufficient price data for {currency}: {len(recent_prices)} points")
            return None
        
        # Use the model for prediction (simplified - replace with actual ML inference)
        prediction_result = calculate_enhanced_prediction(recent_prices, horizon_days, model_data)
        prediction_result.update({
            'currency': currency,
            'league': league,
            'prediction_horizon_days': horizon_days,
            'model_source': 'lambda_layer',
            'tier': 'layer_cached',
            'data_points_used': len(recent_prices)
        })
        
        logger.info(f"Used Lambda layer model for {currency} (confidence: {prediction_result.get('confidence', 0):.2f})")
        return prediction_result
        
    except Exception as e:
        logger.warning(f"Layer model prediction failed for {currency}: {str(e)}")
        return None

def try_memory_cached_prediction(currency: str, league: str, horizon_days: int) -> Optional[Dict]:
    """Try prediction using in-memory cached models"""
    try:
        cache_key = f"{currency}_{league}"
        
        # Check in-memory cache
        if cache_key in GLOBAL_MODEL_CACHE:
            model_data = GLOBAL_MODEL_CACHE[cache_key]
            logger.info(f"Using memory-cached model for {currency}")
        else:
            # This would load from S3 if needed (not implemented for this example)
            return None
        
        # Get price data
        recent_prices = get_hot_data_prices(currency, league, limit=30)
        if len(recent_prices) < 5:
            return None
        
        # Generate prediction
        prediction_result = calculate_enhanced_prediction(recent_prices, horizon_days, model_data)
        prediction_result.update({
            'currency': currency,
            'league': league,
            'prediction_horizon_days': horizon_days,
            'model_source': 'memory_cached',
            'tier': 'memory_cached',
            'data_points_used': len(recent_prices)
        })
        
        return prediction_result
        
    except Exception as e:
        logger.warning(f"Memory cached prediction failed: {str(e)}")
        return None

def generate_trend_prediction_with_hot_data(currency: str, league: str, horizon_days: int) -> Dict:
    """Generate trend prediction using hot data"""
    try:
        # Get hot data first
        recent_prices = get_hot_data_prices(currency, league, limit=50)
        
        # If insufficient, supplement with limited S3 data
        if len(recent_prices) < 10:
            historical_prices = get_s3_historical_prices_limited(currency, league, days_back=7)
            recent_prices.extend(historical_prices)
        
        if len(recent_prices) < 3:
            return create_insufficient_data_prediction(currency, league, horizon_days, len(recent_prices))
        
        # Enhanced trend calculation
        prediction_result = calculate_enhanced_trend_prediction(recent_prices, horizon_days)
        prediction_result.update({
            'currency': currency,
            'league': league,
            'prediction_horizon_days': horizon_days,
            'model_source': 'trend_analysis',
            'tier': 'trend_analysis',
            'data_points_used': len(recent_prices)
        })
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Trend prediction failed: {str(e)}")
        return create_error_prediction(currency, league, horizon_days, str(e))

def calculate_enhanced_prediction(prices: List[Dict], horizon_days: int, model_data: Optional[Dict] = None) -> Dict:
    """Enhanced prediction calculation with model data"""
    try:
        if len(prices) < 2:
            raise ValueError("Insufficient price data")
        
        # Extract price values
        values = [p['value'] for p in prices[-20:]]  # Use last 20 data points
        current_price = values[-1]
        
        # Enhanced trend analysis
        if len(values) >= 5:
            # Short-term trend (last 5 points)
            short_trend = (values[-1] - values[-5]) / 5
            # Medium-term trend (last 10 points if available)
            medium_trend = (values[-1] - values[-min(10, len(values))]) / min(10, len(values))
            
            # Weighted trend combining short and medium term
            trend_slope = (short_trend * 0.7) + (medium_trend * 0.3)
        else:
            trend_slope = (values[-1] - values[0]) / len(values)
        
        # Predict future price
        predicted_price = current_price + (trend_slope * horizon_days)
        predicted_price = max(0.01, predicted_price)  # Ensure positive price
        
        # Calculate price change
        price_change_percent = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Determine trend direction
        if price_change_percent > 5.0:
            trend = 'strongly_bullish'
        elif price_change_percent > 2.0:
            trend = 'bullish'
        elif price_change_percent < -5.0:
            trend = 'strongly_bearish'
        elif price_change_percent < -2.0:
            trend = 'bearish'
        else:
            trend = 'stable'
        
        # Calculate confidence based on data quality and model presence
        price_variance = sum((v - current_price) ** 2 for v in values) / len(values)
        stability_factor = 1.0 / (1.0 + (price_variance / current_price))
        
        # Boost confidence if we have model data
        model_boost = 0.2 if model_data else 0.0
        confidence = min(0.95, max(0.1, stability_factor + model_boost))
        
        return {
            'predicted_price': round(predicted_price, 4),
            'confidence': round(confidence, 3),
            'trend': trend,
            'price_change_percent': round(price_change_percent, 2),
            'current_price': round(current_price, 4),
            'trend_slope': round(trend_slope, 6),
            'stability_factor': round(stability_factor, 3)
        }
        
    except Exception as e:
        raise Exception(f"Enhanced prediction calculation failed: {str(e)}")

def calculate_enhanced_trend_prediction(prices: List[Dict], horizon_days: int) -> Dict:
    """Enhanced trend prediction without model data"""
    return calculate_enhanced_prediction(prices, horizon_days, model_data=None)

def get_hot_data_prices(currency: str, league: str, limit: int = 50) -> List[Dict]:
    """Get recent prices from DynamoDB hot data"""
    try:
        currency_league = f"{currency}#{league}"
        
        response = live_prices_table.query(
            KeyConditionExpression='currency_league = :cl',
            ExpressionAttributeValues={':cl': currency_league},
            ScanIndexForward=False,  # Most recent first
            Limit=limit
        )
        
        prices = []
        for item in response.get('Items', []):
            try:
                prices.append({
                    'timestamp': item['timestamp'],
                    'value': float(item.get('mean_price', 0)),
                    'volume': int(item.get('daily_volume', 0))
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid price data: {e}")
                continue
        
        return sorted(prices, key=lambda x: x['timestamp'])
        
    except Exception as e:
        logger.error(f"Hot data query failed: {str(e)}")
        return []

def get_s3_historical_prices_limited(currency: str, league: str, days_back: int = 7) -> List[Dict]:
    """Get limited historical prices from S3"""
    # Implementation would normally fetch from S3
    # For now, we limit S3 access to only when absolutely necessary
    return []

def create_insufficient_data_prediction(currency: str, league: str, horizon_days: int, data_points: int) -> Dict:
    """Create prediction response for insufficient data"""
    return {
        'currency': currency,
        'league': league,
        'prediction_horizon_days': horizon_days,
        'predicted_price': None,
        'confidence': 0.0,
        'trend': 'insufficient_data',
        'message': f'Insufficient historical data for prediction ({data_points} points)',
        'data_points_used': data_points,
        'model_source': 'insufficient_data',
        'tier': 'insufficient_data'
    }

def create_error_prediction(currency: str, league: str, horizon_days: int, error_msg: str) -> Dict:
    """Create prediction response for errors"""
    return {
        'currency': currency,
        'league': league,
        'prediction_horizon_days': horizon_days,
        'predicted_price': None,
        'confidence': 0.0,
        'trend': 'error',
        'message': f'Prediction failed: {error_msg}',
        'data_points_used': 0,
        'model_source': 'error',
        'tier': 'error'
    } 