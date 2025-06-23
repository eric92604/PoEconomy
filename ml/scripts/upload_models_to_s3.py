#!/usr/bin/env python3
"""
Upload Trained ML Models to S3

Script to upload locally trained ML models to S3 for AWS Lambda access.
Handles model files, metadata, and scalers with proper S3 organization.
"""

import os
import boto3
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import logging
from botocore.exceptions import ClientError
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelUploader:
    """Uploads trained ML models to S3 for Lambda access."""
    
    def __init__(self, 
                 models_bucket: str,
                 local_models_path: str = "ml/models/currency_production",
                 aws_profile: Optional[str] = None):
        """
        Initialize model uploader.
        
        Args:
            models_bucket: S3 bucket for storing models
            local_models_path: Local path to trained models
            aws_profile: Optional AWS profile name
        """
        # Initialize AWS session
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client('s3')
        else:
            self.s3_client = boto3.client('s3')
        
        self.models_bucket = models_bucket
        self.local_models_path = Path(local_models_path)
        
        # Verify bucket exists
        self._verify_bucket()
    
    def _verify_bucket(self):
        """Verify S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.models_bucket)
            logger.info(f"Verified access to S3 bucket: {self.models_bucket}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket not found: {self.models_bucket}")
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket: {self.models_bucket}")
            else:
                logger.error(f"Error accessing S3 bucket: {str(e)}")
            raise
    
    def upload_all_models(self, league: str = "production") -> Dict[str, any]:
        """
        Upload all trained models to S3.
        
        Args:
            league: League name for S3 organization
            
        Returns:
            Upload results summary
        """
        results = {
            'uploaded': [],
            'failed': [],
            'skipped': [],
            'total_size_mb': 0
        }
        
        if not self.local_models_path.exists():
            logger.error(f"Local models path not found: {self.local_models_path}")
            return results
        
        # Find all currency directories
        currency_dirs = [d for d in self.local_models_path.iterdir() 
                        if d.is_dir() and not d.name.startswith('.')]
        
        logger.info(f"Found {len(currency_dirs)} currency model directories")
        
        for currency_dir in currency_dirs:
            currency_name = currency_dir.name
            try:
                upload_result = self.upload_currency_model(currency_name, league)
                if upload_result['success']:
                    results['uploaded'].append(upload_result)
                    results['total_size_mb'] += upload_result['size_mb']
                else:
                    results['failed'].append(upload_result)
                    
            except Exception as e:
                logger.error(f"Failed to upload model for {currency_name}: {str(e)}")
                results['failed'].append({
                    'currency': currency_name,
                    'error': str(e),
                    'success': False
                })
        
        # Print summary
        logger.info(f"Upload Summary:")
        logger.info(f"  Uploaded: {len(results['uploaded'])} models")
        logger.info(f"  Failed: {len(results['failed'])} models")
        logger.info(f"  Total size: {results['total_size_mb']:.2f} MB")
        
        return results
    
    def upload_currency_model(self, currency: str, league: str = "production") -> Dict[str, any]:
        """
        Upload model files for a specific currency.
        
        Args:
            currency: Currency name
            league: League name
            
        Returns:
            Upload result details
        """
        currency_dir = self.local_models_path / currency
        s3_prefix = f"models/{league}/{currency}/"
        
        result = {
            'currency': currency,
            'league': league,
            'files_uploaded': [],
            'size_mb': 0,
            'success': False
        }
        
        if not currency_dir.exists():
            result['error'] = f"Currency directory not found: {currency_dir}"
            return result
        
        try:
            # Upload model metadata
            metadata_file = currency_dir / "model_metadata.json"
            if metadata_file.exists():
                s3_key = f"{s3_prefix}model_metadata.json"
                self._upload_file(metadata_file, s3_key)
                result['files_uploaded'].append('model_metadata.json')
                result['size_mb'] += metadata_file.stat().st_size / (1024 * 1024)
                logger.info(f"Uploaded metadata for {currency}")
            else:
                logger.warning(f"No metadata file found for {currency}")
            
            # Upload model file
            model_file = currency_dir / "ensemble_model.pkl"
            if model_file.exists():
                s3_key = f"{s3_prefix}ensemble_model.pkl"
                self._upload_file(model_file, s3_key)
                result['files_uploaded'].append('ensemble_model.pkl')
                result['size_mb'] += model_file.stat().st_size / (1024 * 1024)
                logger.info(f"Uploaded model file for {currency} ({result['size_mb']:.2f} MB)")
            else:
                result['error'] = f"No model file found for {currency}"
                return result
            
            # Upload scaler if exists
            scaler_file = currency_dir / "scaler.pkl"
            if scaler_file.exists():
                s3_key = f"{s3_prefix}scaler.pkl"
                self._upload_file(scaler_file, s3_key)
                result['files_uploaded'].append('scaler.pkl')
                result['size_mb'] += scaler_file.stat().st_size / (1024 * 1024)
                logger.info(f"Uploaded scaler for {currency}")
            
            # Check if model is valid by loading metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    result['model_type'] = metadata.get('model_type', 'unknown')
                    result['metrics'] = metadata.get('metrics', {})
                    result['training_timestamp'] = metadata.get('training_timestamp')
            
            result['success'] = True
            logger.info(f"Successfully uploaded model for {currency}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to upload model for {currency}: {str(e)}")
        
        return result
    
    def _upload_file(self, local_file: Path, s3_key: str):
        """Upload a single file to S3."""
        try:
            self.s3_client.upload_file(
                str(local_file),
                self.models_bucket,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'STANDARD'
                }
            )
        except Exception as e:
            logger.error(f"Failed to upload {local_file} to {s3_key}: {str(e)}")
            raise
    
    def list_s3_models(self, league: str = "production") -> List[Dict[str, any]]:
        """List models currently in S3."""
        models = []
        prefix = f"models/{league}/"
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.models_bucket, Prefix=prefix)
            
            for page in pages:
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('/model_metadata.json'):
                        # Extract currency name
                        path_parts = obj['Key'].split('/')
                        if len(path_parts) >= 3:
                            currency = path_parts[2]
                            models.append({
                                'currency': currency,
                                'league': league,
                                'last_modified': obj['LastModified'].isoformat(),
                                'size_bytes': obj['Size']
                            })
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list S3 models: {str(e)}")
            return []
    
    def verify_upload(self, currency: str, league: str = "production") -> Dict[str, any]:
        """Verify that a currency model was uploaded correctly."""
        s3_prefix = f"models/{league}/{currency}/"
        
        verification = {
            'currency': currency,
            'league': league,
            'files_found': [],
            'files_missing': [],
            'valid': False
        }
        
        required_files = ['model_metadata.json', 'ensemble_model.pkl']
        optional_files = ['scaler.pkl']
        
        try:
            # Check required files
            for filename in required_files:
                s3_key = f"{s3_prefix}{filename}"
                try:
                    self.s3_client.head_object(Bucket=self.models_bucket, Key=s3_key)
                    verification['files_found'].append(filename)
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        verification['files_missing'].append(filename)
                    else:
                        raise
            
            # Check optional files
            for filename in optional_files:
                s3_key = f"{s3_prefix}{filename}"
                try:
                    self.s3_client.head_object(Bucket=self.models_bucket, Key=s3_key)
                    verification['files_found'].append(filename)
                except ClientError as e:
                    if e.response['Error']['Code'] != '404':
                        raise
            
            # Verify metadata can be loaded
            if 'model_metadata.json' in verification['files_found']:
                try:
                    metadata_key = f"{s3_prefix}model_metadata.json"
                    response = self.s3_client.get_object(Bucket=self.models_bucket, Key=metadata_key)
                    metadata = json.loads(response['Body'].read().decode('utf-8'))
                    verification['metadata'] = metadata
                    verification['valid'] = len(verification['files_missing']) == 0
                except Exception as e:
                    verification['metadata_error'] = str(e)
            
        except Exception as e:
            verification['error'] = str(e)
        
        return verification


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Upload ML models to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name for models')
    parser.add_argument('--models-path', default='ml/models/currency_production', 
                       help='Local path to trained models')
    parser.add_argument('--league', default='production', help='League name for organization')
    parser.add_argument('--profile', help='AWS profile name')
    parser.add_argument('--currency', help='Upload specific currency only')
    parser.add_argument('--verify', action='store_true', help='Verify uploads after completion')
    parser.add_argument('--list', action='store_true', help='List existing models in S3')
    
    args = parser.parse_args()
    
    # Initialize uploader
    uploader = ModelUploader(
        models_bucket=args.bucket,
        local_models_path=args.models_path,
        aws_profile=args.profile
    )
    
    if args.list:
        # List existing models
        models = uploader.list_s3_models(args.league)
        logger.info(f"Found {len(models)} models in S3:")
        for model in models:
            logger.info(f"  {model['currency']} - {model['last_modified']}")
        return
    
    if args.currency:
        # Upload specific currency
        result = uploader.upload_currency_model(args.currency, args.league)
        if result['success']:
            logger.info(f"Successfully uploaded {args.currency}")
            if args.verify:
                verification = uploader.verify_upload(args.currency, args.league)
                logger.info(f"Verification: {verification['valid']}")
        else:
            logger.error(f"Failed to upload {args.currency}: {result.get('error', 'Unknown error')}")
    else:
        # Upload all models
        results = uploader.upload_all_models(args.league)
        
        if args.verify and results['uploaded']:
            logger.info("Verifying uploads...")
            for upload_result in results['uploaded']:
                verification = uploader.verify_upload(upload_result['currency'], args.league)
                status = "✓" if verification['valid'] else "✗"
                logger.info(f"  {status} {upload_result['currency']}")


if __name__ == "__main__":
    main() 