#!/usr/bin/env python3
"""
PoEconomy API Serving Handler

This Lambda function serves pre-computed ML predictions from the cache table.
It provides API responses without performing computations.
- Triggered by: API Gateway requests
- Memory: 256MB (sufficient for read operations)
- Timeout: 10 seconds (API response requirements)
- Input: API requests for predictions
- Output: Responses from cached predictions
"""

import json
import boto3
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')

# Environment variables
PREDICTIONS_CACHE_TABLE_NAME = os.environ.get('PREDICTIONS_CACHE_TABLE')
METADATA_TABLE_NAME = os.environ.get('METADATA_TABLE')

# DynamoDB tables (lazy initialization)
predictions_cache_table = None
metadata_table = None

def get_tables():
    """Initialize DynamoDB tables"""
    global predictions_cache_table, metadata_table
    if not predictions_cache_table:
        predictions_cache_table = dynamodb.Table(PREDICTIONS_CACHE_TABLE_NAME)
        metadata_table = dynamodb.Table(METADATA_TABLE_NAME)
    return predictions_cache_table, metadata_table

def lambda_handler(event, context):
    """API Serving Handler - Read-only operations"""
    start_time = datetime.utcnow()
    
    try:
        # Parse request
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '')
        
        if 'body' in event and event['body']:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = {}
        
        # Route requests
        if path.endswith('/predict/single') and http_method == 'POST':
            return handle_single_prediction(body, start_time)
        elif path.endswith('/predict/batch') and http_method == 'POST':
            return handle_batch_predictions(body, start_time)
        elif path.endswith('/predict/currencies') and http_method == 'GET':
            return handle_list_currencies(start_time)
        elif path.endswith('/predict/leagues') and http_method == 'GET':
            return handle_list_leagues(start_time)
        else:
            return {
                'statusCode': 404,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'error': 'Endpoint not found',
                    'available_endpoints': [
                        'POST /predict/single',
                        'POST /predict/batch',
                        'GET /predict/currencies',
                        'GET /predict/leagues'
                    ]
                })
            }
        
    except Exception as e:
        logger.error(f"API serving error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def handle_single_prediction(body: Dict, start_time: datetime) -> Dict:
    """Handle single prediction request"""
    try:
        predictions_cache_table, metadata_table = get_tables()
        
        # Validate request
        currency = body.get('currency')
        league = body.get('league', 'Sanctum')
        prediction_horizon_days = int(body.get('prediction_horizon_days', 1))
        
        if not currency:
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'error': 'Missing required parameter: currency',
                    'timestamp': start_time.isoformat()
                })
            }
        
        # Get cached prediction
        cache_key = f"{currency}#{league}#{prediction_horizon_days}"
        cached_prediction = get_cached_prediction(cache_key)
        
        if not cached_prediction:
            return {
                'statusCode': 404,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'error': 'Prediction not available',
                    'message': f'No cached prediction found for {currency} in {league} for {prediction_horizon_days} days',
                    'suggestion': 'Try a different currency, league, or prediction horizon',
                    'timestamp': start_time.isoformat()
                })
            }
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'prediction': cached_prediction,
                'currency': currency,
                'league': league,
                'prediction_horizon_days': prediction_horizon_days,
                'source': 'cache',
                'execution_time_seconds': round(execution_time, 3),
                'timestamp': start_time.isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Single prediction error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Failed to get prediction',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def handle_batch_predictions(body: Dict, start_time: datetime) -> Dict:
    """Handle batch prediction requests"""
    try:
        predictions_cache_table, metadata_table = get_tables()
        
        # Validate request
        requests = body.get('requests', [])
        if not requests or not isinstance(requests, list):
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'error': 'Missing or invalid requests array',
                    'example': {
                        'requests': [
                            {'currency': 'Divine Orb', 'league': 'Sanctum', 'prediction_horizon_days': 1},
                            {'currency': 'Chaos Orb', 'league': 'Sanctum', 'prediction_horizon_days': 7}
                        ]
                    }
                })
            }
        
        # Process batch requests
        results = []
        for req in requests[:20]:  # Limit to 20 requests per batch
            try:
                currency = req.get('currency')
                league = req.get('league', 'Sanctum')
                horizon_days = int(req.get('prediction_horizon_days', 1))
                
                if currency:
                    cache_key = f"{currency}#{league}#{horizon_days}"
                    prediction = get_cached_prediction(cache_key)
                    
                    results.append({
                        'currency': currency,
                        'league': league,
                        'prediction_horizon_days': horizon_days,
                        'prediction': prediction,
                        'available': prediction is not None
                    })
            except Exception as e:
                logger.warning(f"Failed to process batch request: {str(e)}")
                continue
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'results': results,
                'total_requests': len(requests),
                'successful_results': len(results),
                'execution_time_seconds': round(execution_time, 3),
                'timestamp': start_time.isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Failed to process batch predictions',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def handle_list_currencies(start_time: datetime) -> Dict:
    """List available currencies with predictions"""
    try:
        predictions_cache_table, metadata_table = get_tables()
        
        # Get unique currencies from cache table
        response = predictions_cache_table.scan(
            ProjectionExpression='currency, league, horizon_days, created_at',
            Limit=1000
        )
        
        currencies = {}
        for item in response.get('Items', []):
            currency = item.get('currency')
            league = item.get('league')
            horizon_days = item.get('horizon_days')
            
            if currency and league:
                key = f"{currency}#{league}"
                if key not in currencies:
                    currencies[key] = {
                        'currency': currency,
                        'league': league,
                        'available_horizons': [],
                        'last_updated': item.get('created_at')
                    }
                
                if horizon_days and horizon_days not in currencies[key]['available_horizons']:
                    currencies[key]['available_horizons'].append(horizon_days)
        
        # Sort currencies by name
        currency_list = sorted(currencies.values(), key=lambda x: x['currency'])
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'currencies': currency_list,
                'total_count': len(currency_list),
                'execution_time_seconds': round(execution_time, 3),
                'timestamp': start_time.isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"List currencies error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Failed to list currencies',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def handle_list_leagues(start_time: datetime) -> Dict:
    """List available leagues"""
    try:
        predictions_cache_table, metadata_table = get_tables()
        
        # Get unique leagues from cache table
        response = predictions_cache_table.scan(
            ProjectionExpression='league',
            Limit=1000
        )
        
        leagues = set()
        for item in response.get('Items', []):
            league = item.get('league')
            if league:
                leagues.add(league)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'leagues': sorted(list(leagues)),
                'total_count': len(leagues),
                'execution_time_seconds': round(execution_time, 3),
                'timestamp': start_time.isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"List leagues error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Failed to list leagues',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def get_cached_prediction(cache_key: str) -> Optional[Dict]:
    """Get cached prediction from DynamoDB"""
    try:
        predictions_cache_table, metadata_table = get_tables()
        
        response = predictions_cache_table.get_item(Key={'prediction_key': cache_key})
        if 'Item' in response:
            prediction_data = response['Item'].get('prediction_data')
            if prediction_data:
                return json.loads(prediction_data)
        return None
        
    except Exception as e:
        logger.warning(f"Cache lookup failed for {cache_key}: {str(e)}")
        return None

def get_cors_headers() -> Dict[str, str]:
    """Get CORS headers for API responses"""
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Cache-Control': 'max-age=300'  # 5 minutes cache for API responses
    } 