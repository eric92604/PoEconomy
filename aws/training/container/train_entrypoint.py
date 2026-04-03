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

# Set up standardized logging
from ml.utils.common_utils import setup_standard_logging
logger = setup_standard_logging(
    name="TrainingEntrypoint",
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
    
    timeout_minutes = int(os.getenv('TASK_TIMEOUT_MINUTES', '1440'))
    timeout_seconds = timeout_minutes * 60
    start_time = datetime.now()
    
    logger.debug(f"Task timeout set to {timeout_minutes} minutes ({timeout_seconds} seconds)")
    
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
        logger.debug(f"Task progress: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
    
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
        
        training_mode = os.getenv('TRAINING_MODE', 'production')  # CloudFormation default: production
        min_records = os.getenv('MIN_RECORDS_THRESHOLD', '50')  # CloudFormation default: 50
        max_currencies = os.getenv('MAX_CURRENCIES_TO_TRAIN', '0')  # CloudFormation default: 0
        
        logger.info(f"Training configuration:")
        logger.debug(f"  Mode: {training_mode}")
        logger.debug(f"  Min Records: {min_records}")
        logger.debug(f"  Max Currencies: {max_currencies} (0 = no limit)")
        logger.debug(f"  Data Lake Bucket: {os.getenv('DATA_LAKE_BUCKET')}")
        logger.debug(f"  Models Bucket: {os.getenv('MODELS_BUCKET')}")
        
        # Training pipeline will load processed parquet data from S3
        # Data is loaded directly from S3 data lake bucket
        logger.debug("Training pipeline will use processed parquet data from S3 data lake")
        
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
        
        # Models are now uploaded incrementally after each currency is trained
        # This ensures models are saved even if training fails partway through
        # We still perform a verification/fallback upload to catch any missed models
        logger.info("Training completed. Models should have been uploaded incrementally during training.")
        logger.info("Performing verification upload for any remaining models...")
        
        # Simplified verification upload (only uploads files that don't already exist in S3)
        models_base_dir = '/app/ml/models'
        models_currency_dir = '/app/ml/models/currency'
        
        # Try to get experiment ID from environment or use a default
        experiment_id = os.getenv('EXPERIMENT_ID')
        
        # Check if models were created in the currency directory (standard location)
        logger.debug(f"Checking for models in: {models_currency_dir}")
        models_dir = None
        
        if os.path.exists(models_currency_dir):
            # Models are saved directly to /app/ml/models/currency/{currency_name}_{horizon}/
            # or /app/ml/models/currency/{currency_name}/
            models_dir = models_currency_dir
            logger.debug(f"Found models directory: {models_dir}")
            
            # If no experiment_id from env, try to extract it from model metadata files
            if not experiment_id:
                # Try to find experiment_id from model metadata files
                all_items = os.listdir(models_currency_dir)
                model_dirs = [d for d in all_items 
                             if os.path.isdir(os.path.join(models_currency_dir, d))]
                
                # Check first few model directories for metadata files
                for model_dir in model_dirs[:5]:
                    metadata_path = os.path.join(models_currency_dir, model_dir, "model_metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            # Check if metadata contains experiment_id or training_timestamp
                            # We can derive experiment_id from training_timestamp if needed
                            training_timestamp = metadata.get('training_timestamp', '')
                            if training_timestamp:
                                from datetime import datetime
                                # Parse timestamp and create experiment_id
                                try:
                                    dt = datetime.fromisoformat(training_timestamp.replace('Z', '+00:00'))
                                    experiment_id = f"xp_{dt.strftime('%Y%m%d_%H%M%S')}"
                                    logger.debug(f"Extracted experiment_id from model metadata: {experiment_id}")
                                    break
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.debug(f"Could not read metadata from {metadata_path}: {e}")
                
                # If still no experiment_id, check for experiment-specific subdirectory (old format)
                if not experiment_id:
                    experiment_dirs = [d for d in all_items 
                                      if os.path.isdir(os.path.join(models_currency_dir, d)) 
                                      and d.startswith('currency_')]
                    if experiment_dirs:
                        # Use the most recent experiment directory
                        experiment_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_currency_dir, d)), reverse=True)
                        experiment_id = experiment_dirs[0]
                        models_dir = os.path.join(models_currency_dir, experiment_id)
                        logger.debug(f"Found experiment-specific directory: {models_dir}")
                
                # Last resort: generate timestamp-based ID
                if not experiment_id:
                    from datetime import datetime
                    experiment_id = f"xp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    logger.warning(f"No experiment_id found, using generated ID: {experiment_id}")
        elif os.path.exists(models_base_dir):
            # Fallback: check if there are experiment directories in the base directory
            logger.debug(f"Currency directory not found, checking base directory: {models_base_dir}")
            all_dirs = [d for d in os.listdir(models_base_dir) 
                       if os.path.isdir(os.path.join(models_base_dir, d))]
            logger.debug(f"Found {len(all_dirs)} directories in models base directory")
            
            experiment_dirs = [d for d in all_dirs if d.startswith('currency_')]
            logger.debug(f"Found {len(experiment_dirs)} experiment directories matching 'currency_' pattern")
            
            if experiment_dirs:
                experiment_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_base_dir, d)), reverse=True)
                models_dir = os.path.join(models_base_dir, experiment_dirs[0])
                experiment_id = experiment_dirs[0]
                logger.debug(f"Found models directory: {models_dir}")
                logger.debug(f"Extracted experiment ID: {experiment_id}")
        
        # Models are saved as subdirectories in models_currency_dir (e.g., currency_name_1d/, currency_name_3d/, etc.)
        # If models_dir is set to models_currency_dir, we should proceed with upload
        # The os.walk() will find all subdirectories and files
        
        if not models_dir or not os.path.exists(models_dir):
            logger.error(f"Models directory not found. Checked:")
            logger.error(f"  - {models_currency_dir}: {os.path.exists(models_currency_dir)}")
            logger.error(f"  - {models_base_dir}: {os.path.exists(models_base_dir)}")
            if os.path.exists(models_currency_dir):
                try:
                    items = os.listdir(models_currency_dir)
                    logger.error(f"  - Contents of {models_currency_dir}: {items[:10]}{'...' if len(items) > 10 else ''}")
                except Exception as e:
                    logger.error(f"  - Error listing directory: {e}")
            logger.warning("Skipping model upload - no models directory found")
        else:
            logger.debug(f"Using models directory: {models_dir}")
            logger.debug(f"Using experiment ID: {experiment_id}")
            
            # Upload models to S3
            import boto3
            from botocore.exceptions import ClientError, BotoCoreError
            
            s3_client = boto3.client('s3')
            models_bucket = os.getenv('MODELS_BUCKET')
            
            if models_bucket:
                logger.info(f"Uploading models to S3 bucket: {models_bucket}")
                
                # Debug: Log directory structure
                logger.debug(f"Scanning directory structure in: {models_dir}")
                try:
                    # List all items in the models directory
                    items = os.listdir(models_dir)
                    logger.debug(f"Found {len(items)} items in models directory: {items[:10]}{'...' if len(items) > 10 else ''}")
                    
                    # Check subdirectories
                    subdirs = [d for d in items if os.path.isdir(os.path.join(models_dir, d))]
                    logger.debug(f"Found {len(subdirs)} subdirectories: {subdirs[:10]}{'...' if len(subdirs) > 10 else ''}")
                    
                    # Check files in root
                    root_files = [f for f in items if os.path.isfile(os.path.join(models_dir, f))]
                    logger.debug(f"Found {len(root_files)} files in root: {root_files}")
                    
                    # For each subdirectory, check what's inside
                    for subdir in subdirs[:5]:  # Check first 5 subdirectories
                        subdir_path = os.path.join(models_dir, subdir)
                        subdir_items = os.listdir(subdir_path)
                        subdir_files = [f for f in subdir_items if os.path.isfile(os.path.join(subdir_path, f))]
                        logger.debug(f"  Subdirectory '{subdir}' contains {len(subdir_files)} files: {subdir_files}")
                except Exception as e:
                    logger.warning(f"Error inspecting directory structure: {e}")
                
                # First, collect all files to upload
                files_to_upload = []
                walk_count = 0
                for root, dirs, files in os.walk(models_dir):
                    walk_count += 1
                    if walk_count <= 3:  # Log first 3 directories walked
                        logger.debug(f"Walking directory: {root}, found {len(files)} files, {len(dirs)} subdirectories")
                    
                    for file in files:
                        local_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_path, models_dir)
                        # Normalize path separators for S3 (use forward slashes)
                        relative_path = relative_path.replace(os.sep, '/')
                        s3_key = f"models/currency/{experiment_id}/{relative_path}"
                        files_to_upload.append((local_path, s3_key))
                
                logger.debug(f"Completed directory walk, found {len(files_to_upload)} total files")
                
                if not files_to_upload:
                    logger.warning(f"No model files found in {models_dir}")
                    # Try to diagnose why
                    if os.path.exists(models_dir):
                        try:
                            total_items = len(os.listdir(models_dir))
                            logger.warning(f"Directory exists and contains {total_items} items, but no files were found during walk")
                        except Exception as e:
                            logger.error(f"Error listing directory contents: {e}")
                    else:
                        logger.error(f"Models directory does not exist: {models_dir}")
                else:
                    logger.debug(f"Found {len(files_to_upload)} model files to upload")
                    
                    # Upload each file with error handling (verification/fallback upload)
                    uploaded_count = 0
                    skipped_count = 0
                    failed_count = 0
                    for local_path, s3_key in files_to_upload:
                        try:
                            # Verify file exists before uploading
                            if not os.path.exists(local_path):
                                logger.warning(f"File not found, skipping: {local_path}")
                                failed_count += 1
                                continue
                            
                            # Check if file already exists in S3 (models should have been uploaded incrementally)
                            try:
                                s3_client.head_object(Bucket=models_bucket, Key=s3_key)
                                # File already exists, skip
                                skipped_count += 1
                                continue
                            except ClientError as e:
                                if e.response['Error']['Code'] != '404':
                                    # Error other than "not found", log and continue
                                    logger.warning(f"Error checking S3 for {s3_key}: {e}")
                            
                            # File doesn't exist, upload it
                            s3_client.upload_file(local_path, models_bucket, s3_key)
                            uploaded_count += 1
                            logger.debug(f"Uploaded missing file ({uploaded_count}/{len(files_to_upload)}): s3://{models_bucket}/{s3_key}")
                            
                        except ClientError as e:
                            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                            error_msg = e.response.get('Error', {}).get('Message', str(e))
                            logger.error(f"Failed to upload {local_path} to s3://{models_bucket}/{s3_key}: {error_code} - {error_msg}")
                            failed_count += 1
                        except BotoCoreError as e:
                            logger.error(f"Boto3 error uploading {local_path} to s3://{models_bucket}/{s3_key}: {str(e)}")
                            failed_count += 1
                        except Exception as e:
                            logger.error(f"Unexpected error uploading {local_path} to s3://{models_bucket}/{s3_key}: {str(e)}")
                            failed_count += 1
                    
                    # Log summary
                    if uploaded_count > 0:
                        logger.info(f"Verification upload: Successfully uploaded {uploaded_count} missing model files to S3")
                    if skipped_count > 0:
                        logger.debug(f"Verification upload: {skipped_count} files already exist in S3 (uploaded incrementally)")
                    if failed_count > 0:
                        logger.warning(f"Verification upload: Failed to upload {failed_count} model files to S3")
                    if uploaded_count == 0 and skipped_count == 0 and failed_count == 0:
                        logger.warning("Verification upload: No files were processed for upload")
            else:
                logger.warning("MODELS_BUCKET not set - models not uploaded to S3")
        
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
