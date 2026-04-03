#!/usr/bin/env python3
"""
Feature Engineering entry point for ECS Fargate container.
This script runs the feature engineering pipeline to generate parquet files from S3 historical data.
"""

import os
import sys
import boto3
import tempfile
from pathlib import Path
import logging
import signal
import time
from datetime import datetime, timedelta

# Add ML directory to path
sys.path.append('/var/task/ml')

from ml.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from ml.config.training_config import get_config_by_mode

# Set up standardized logging
from ml.utils.common_utils import setup_standard_logging
logger = setup_standard_logging(
    name="FeatureEngineeringEntrypoint",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    console_output=True,
    suppress_external=True
)

# Global timeout tracking
timeout_seconds = None
start_time = None
timeout_exceeded = False

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    global timeout_exceeded
    timeout_exceeded = True
    logger.error("Task timeout exceeded! Initiating graceful shutdown...")
    sys.exit(1)

def setup_timeout():
    """Setup timeout mechanism."""
    global timeout_seconds, start_time
    
    # TASK_TIMEOUT_MINUTES should be set by CloudFormation (default: 60 for feature engineering)
    # Using 60 as fallback to match CloudFormation default
    timeout_minutes = int(os.getenv('TASK_TIMEOUT_MINUTES', '60'))
    timeout_seconds = timeout_minutes * 60
    start_time = datetime.now()
    
    logger.info(f"Task timeout set to {timeout_minutes} minutes ({timeout_seconds} seconds)")
    
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    return timeout_seconds

def check_timeout():
    """Check if timeout is approaching and log progress."""
    if start_time is None:
        return
    
    elapsed = (datetime.now() - start_time).total_seconds()
    remaining = timeout_seconds - elapsed
    
    if remaining <= 300:  # 5 minutes remaining
        logger.warning(f"Task timeout approaching: {remaining:.0f} seconds remaining")
    elif elapsed % 600 == 0:  # Every 10 minutes
        logger.info(f"Task progress: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
    
    if timeout_exceeded:
        logger.error("Task timeout exceeded!")
        sys.exit(1)

def main():
    """Main feature engineering entry point."""
    try:
        # Setup timeout mechanism
        setup_timeout()
        
        logger.info("Starting PoEconomy feature engineering...")
        check_timeout()
        
        # Determine configuration mode from environment (production/development/test)
        mode = os.getenv('MODE', 'production')
        config = get_config_by_mode(mode)
        
        # Override config with environment variables
        if os.getenv('EXPERIMENT_ID'):
            config.experiment.experiment_id = os.getenv('EXPERIMENT_ID')
        if os.getenv('DESCRIPTION'):
            config.experiment.description = os.getenv('DESCRIPTION')
        
        # Add Fargate-specific tags
        config.experiment.tags.extend(['fargate', 'feature-engineering', 'spot'])
        
        # S3 data source configuration
        s3_data_source_config = {
            'data_source_type': 's3',
            'region_name': os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
            'data_lake_bucket': os.getenv('DATA_LAKE_BUCKET'),
            'historical_data_prefix': 'training-data/',
        }
        
        if not s3_data_source_config['data_lake_bucket']:
            raise ValueError("DATA_LAKE_BUCKET environment variable not set")
        
        logger.info(f"Using S3 data source: s3://{s3_data_source_config['data_lake_bucket']}/{s3_data_source_config['historical_data_prefix']}")
        
        # Initialize feature engineering pipeline with S3 data source
        pipeline = FeatureEngineeringPipeline(config, s3_data_source_config)
        
        # Run feature engineering
        logger.info("Starting feature engineering pipeline...")
        check_timeout()
        results = pipeline.run_feature_engineering_experiment()
        check_timeout()
        
        # S3 upload is handled by the pipeline's _upload_combined_dataset_to_s3 method
        
        # Cancel timeout alarm since we're done
        signal.alarm(0)
        logger.info("Feature engineering completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
