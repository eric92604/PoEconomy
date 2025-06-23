#!/usr/bin/env python3
"""
PoEconomy Data Ingestion Handler - DynamoDB + S3 Architecture

Lambda function for data processing:
- 256MB memory (configured for data processing)
- 60s timeout (batch processing requirements)
- Minimal dependencies
- Compressed data storage
- Batch operations
"""

import json
import boto3
import os
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize AWS clients (reuse connections)
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
DATA_LAKE_BUCKET = os.environ.get('DATA_LAKE_BUCKET')
METADATA_TABLE_NAME = os.environ.get('METADATA_TABLE')
LIVE_PRICES_TABLE_NAME = os.environ.get('LIVE_PRICES_TABLE')
CLOUDFLARE_API_TOKEN = os.environ.get('CLOUDFLARE_API_TOKEN', '')

# DynamoDB tables (lazy initialization)
metadata_table = None
live_prices_table = None

def get_tables():
    """Initialize DynamoDB tables"""
    global metadata_table, live_prices_table
    if not metadata_table:
        metadata_table = dynamodb.Table(METADATA_TABLE_NAME)
        live_prices_table = dynamodb.Table(LIVE_PRICES_TABLE_NAME)
    return metadata_table, live_prices_table

def lambda_handler(event, context):
    """PoEconomy Data Ingestion Handler"""
    start_time = datetime.utcnow()
    
    try:
        # Get remaining execution time (stay within timeout limit)
        remaining_time = context.get_remaining_time_in_millis() / 1000
        if remaining_time < 5:  # Reserve 5s for cleanup
            logger.warning("Insufficient time remaining, skipping execution")
            return {'statusCode': 200, 'body': 'Skipped - insufficient time'}
        
        # Initialize tables
        metadata_table, live_prices_table = get_tables()
        
        # POE Watch API call (mock for development)
        currency_data = get_mock_currency_data()
        
        # Process data in batches to stay within limits
        processed_count = process_currency_data_batch(
            currency_data, 
            max_items=50,  # Limit batch size
            remaining_time=remaining_time - 5
        )
        
        # Store compressed data in S3
        s3_key = store_compressed_data(currency_data)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'PoEconomy data ingestion completed',
                'processed_items': processed_count,
                'execution_time_seconds': round(execution_time, 2),
                's3_key': s3_key,
                'timestamp': start_time.isoformat(),
                'architecture': 'poeconomy'
            })
        }
        
    except Exception as e:
        logger.error(f"Data ingestion error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Data ingestion failed',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def get_mock_currency_data() -> List[Dict]:
    """Mock currency data for testing"""
    return [
        {
            "currency": "Divine Orb",
            "league": "Sanctum",
            "value": 180.5,
            "timestamp": datetime.utcnow().isoformat()
        },
        {
            "currency": "Chaos Orb", 
            "league": "Sanctum",
            "value": 1.0,
            "timestamp": datetime.utcnow().isoformat()
        },
        {
            "currency": "Exalted Orb",
            "league": "Sanctum", 
            "value": 12.8,
            "timestamp": datetime.utcnow().isoformat()
        }
    ]

def process_currency_data_batch(data: List[Dict], max_items: int, remaining_time: float) -> int:
    """Process currency data in batches with time limits"""
    metadata_table, live_prices_table = get_tables()
    processed = 0
    start_time = datetime.utcnow()
    
    # Batch write items to DynamoDB
    with metadata_table.batch_writer() as metadata_batch:
        with live_prices_table.batch_writer() as prices_batch:
            
            for item in data[:max_items]:
                # Check time limit
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > remaining_time - 2:  # Reserve 2s for cleanup
                    break
                
                try:
                    # Update metadata (with TTL)
                    metadata_batch.put_item(Item={
                        'currency_name': item['currency'],
                        'league': item['league'],
                        'last_updated': item['timestamp'],
                        'ttl': int((datetime.utcnow() + timedelta(days=30)).timestamp())
                    })
                    
                    # Store live price (with 14-day TTL for hot data)
                    prices_batch.put_item(Item={
                        'currency_league': f"{item['currency']}#{item['league']}",
                        'timestamp': item['timestamp'],
                        'value': str(item['value']),  # Store as string to avoid Decimal issues
                        'ttl': int((datetime.utcnow() + timedelta(days=14)).timestamp())
                    })
                    
                    processed += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process item {item['currency']}: {str(e)}")
                    continue
    
    return processed

def store_compressed_data(data: List[Dict]) -> str:
    """Store compressed data in S3 with lifecycle management"""
    try:
        # Generate S3 key with partitioning
        now = datetime.utcnow()
        s3_key = f"raw-data/currency-prices/{now.strftime('%Y/%m/%d')}/hour={now.hour:02d}/{now.strftime('%H%M%S')}.json.gz"
        
        # Compress data (reduces size significantly)
        json_data = json.dumps(data, separators=(',', ':'))  # Minimal whitespace
        compressed_data = gzip.compress(json_data.encode('utf-8'))
        
        # Store in S3 with Standard-IA storage class
        s3_client.put_object(
            Bucket=DATA_LAKE_BUCKET,
            Key=s3_key,
            Body=compressed_data,
            ContentType='application/json',
            ContentEncoding='gzip',
            StorageClass='STANDARD_IA',  # Storage class for infrequent access
            Metadata={
                'source': 'poeconomy-ingestion',
                'compressed': 'true',
                'item_count': str(len(data))
            }
        )
        
        return s3_key
        
    except Exception as e:
        logger.error(f"S3 storage failed: {str(e)}")
        return f"error: {str(e)}"

def purge_cloudflare_cache():
    """Purge Cloudflare cache if token provided"""
    if not CLOUDFLARE_API_TOKEN:
        return "No Cloudflare token provided"
    
    # Cache purge implementation (would require actual implementation)
    return "Cache purge skipped in current mode"

# Pre-compile frequently used objects
MOCK_RESPONSE = {
    'statusCode': 200,
    'headers': {
        'Content-Type': 'application/json',
        'Cache-Control': 'max-age=900'  # 15 minutes cache
    }
}

# Memory management: Reuse objects
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ' 