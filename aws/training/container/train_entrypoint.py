#!/usr/bin/env python3
"""
Training entry point for Fargate container.
This script handles data input, model training, and model output for ECS Fargate.
"""

import os
import sys
import boto3
import zipfile
import tempfile
from pathlib import Path
import logging
import signal
import time
from datetime import datetime, timedelta

# Add ML directory to path
sys.path.append('/app/ml')

from ml.scripts.train_models import main as train_main
from ml.config.training_config import get_production_config

# Set up standardized logging
from ml.utils.common_utils import setup_standard_logging
logger = setup_standard_logging(
    name="TrainingEntrypoint",
    level="INFO",
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


# S3 upload functions are implemented below

def main():
    """Main training entry point."""
    try:
        # Setup timeout mechanism
        setup_timeout()
        
        logger.info("Starting PoEconomy model training...")
        check_timeout()
        
        training_mode = os.getenv('TRAINING_MODE', 'production')
        min_records = os.getenv('MIN_RECORDS', '50')
        max_currencies = os.getenv('MAX_CURRENCIES_TO_TRAIN', '0')
        
        logger.info(f"Training configuration:")
        logger.info(f"  Mode: {training_mode}")
        logger.info(f"  Min Records: {min_records}")
        logger.info(f"  Max Currencies: {max_currencies} (0 = no limit)")
        logger.info(f"  Data Lake Bucket: {os.getenv('DATA_LAKE_BUCKET')}")
        logger.info(f"  Models Bucket: {os.getenv('MODELS_BUCKET')}")
        
        # Training pipeline will load processed parquet data from S3
        # Data is loaded directly from S3 data lake bucket
        logger.info("Training pipeline will use processed parquet data from S3 data lake")
        
        sys.argv = [
            'train_models.py',
            '--mode', training_mode,
            '--min-records', min_records
        ]
        
        # Add max currencies parameter if specified
        if max_currencies and max_currencies != '0':
            sys.argv.extend(['--max-currencies', max_currencies])
        
        logger.info("Starting model training...")
        check_timeout()
        train_main()
        check_timeout()
        
        # Models are saved to local directory and then uploaded to S3
        models_output_dir = '/app/ml/models'
        models_base_dir = '/app/ml/models'
        
        # Check if models were created in the working directory
        if os.path.exists(models_base_dir):
            experiment_dirs = [d for d in os.listdir(models_base_dir) 
                             if os.path.isdir(os.path.join(models_base_dir, d)) 
                             and d.startswith('currency_')]
            if experiment_dirs:
                experiment_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_base_dir, d)), reverse=True)
                models_dir = os.path.join(models_base_dir, experiment_dirs[0])
                logger.info(f"Found models directory: {models_dir}")
                
                # Extract experiment ID first
                experiment_id = os.path.basename(models_dir)
                logger.info(f"Extracted experiment ID: {experiment_id}")
                
                # Upload models to S3
                import shutil
                import boto3
                s3_client = boto3.client('s3')
                models_bucket = os.getenv('MODELS_BUCKET')
                
                if models_bucket:
                    logger.info(f"Uploading models to S3 bucket: {models_bucket}")
                    for root, dirs, files in os.walk(models_dir):
                        for file in files:
                            local_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_path, models_dir)
                            s3_key = f"models/currency/{experiment_id}/{relative_path}"
                            s3_client.upload_file(local_path, models_bucket, s3_key)
                            logger.info(f"Uploaded: s3://{models_bucket}/{s3_key}")
                else:
                    logger.warning("MODELS_BUCKET not set - models not uploaded to S3")
            else:
                logger.warning(f"No experiment directories found in {models_base_dir}")
        else:
            logger.warning(f"Models directory not found in {models_base_dir}")
        
        # Cancel timeout alarm since we're done
        signal.alarm(0)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
