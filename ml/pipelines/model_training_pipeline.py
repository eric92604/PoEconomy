"""
Model training pipeline module.

This module contains the pipeline orchestration for model training,
separating the orchestration logic from the core model training functionality.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
import argparse
from datetime import datetime
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.impute import SimpleImputer

sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from ml.config.training_config import MLConfig, get_default_config
from ml.utils.common_utils import setup_ml_logging, MLLogger, ProgressLogger
from ml.utils.model_training import ModelTrainer, save_model_artifacts, TrainingResult
from ml.utils.data_processing import generate_target_currency_list

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


class ModelTrainingPipeline:
    """
    Currency-specific model trainer.
    
    This class orchestrates the entire ML pipeline from data loading
    to model training and evaluation, with comprehensive logging,
    error handling, and experiment tracking.
    """
    
    def __init__(self, config: MLConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Complete ML configuration
        """
        self.config = config
        
        # Get S3 bucket configuration
        data_lake_bucket = os.getenv('DATA_LAKE_BUCKET', '')
        
        # Setup logging
        self.logger = setup_ml_logging(
            name="CurrencyTrainer",
            level=config.logging.level,
            log_dir=str(config.paths.logs_dir),
            experiment_id=config.experiment.experiment_id,
            console_output=config.logging.console_logging,
            suppress_external=config.logging.suppress_external
        )
        
        # Initialize components
        self.model_trainer = ModelTrainer(
            config.model, config.processing, self.logger
        )
        # Initialize data source for loading processed data
        from ml.utils.data_sources import create_s3_data_source
        s3_config = {
            'data_lake_bucket': data_lake_bucket,
            'processed_data_prefix': 'processed_data/'
        }
        self.data_source = create_s3_data_source(s3_config, self.logger)
        self.logger.info(f"Using S3 data source for processed data from bucket: {data_lake_bucket}")
        
        # Store S3 configuration for log uploads
        self.data_lake_bucket = data_lake_bucket
        
        # Get models bucket for model uploads
        self.models_bucket = os.getenv('MODELS_BUCKET', '')
        
        # Note: Models directory will be created after experiment_id is potentially updated from processed data
        
        # Initialize tracking
        self.results: List[Any] = []
        self.failed_currencies: List[Dict[str, str]] = []
        self.processing_stats = {
            'total_currencies': 0,
            'successful_training': 0,
            'failed_training': 0,
            'insufficient_data': 0,
            'validation_failures': 0
        }
    
    def train_all_currencies(self, data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Train models for all target currencies.
        
        Args:
            data_path: Optional path to training data file
            
        Returns:
            List of training results
        """
        # Start experiment logging
        self.logger.log_experiment_start(
            self.config.experiment.experiment_id or "unknown",
            self.config.to_dict()
        )
        
        # Log training start
        self.logger.info("Starting training pipeline")
        
        # Initialize variables for tracking
        target_currencies: List[str] = []
        self._training_start_time: float = time.time()
        
        try:
            # Load processed data first to get available currencies
            processed_data = self._load_processed_data()
            if processed_data is None:
                data_lake_bucket = os.getenv('DATA_LAKE_BUCKET', 'NOT_SET')
                self.logger.error(f"Failed to load processed data - cannot determine currencies to train")
                self.logger.error(f"DATA_LAKE_BUCKET environment variable: {data_lake_bucket}")
                self.logger.error("This usually means:")
                self.logger.error("1. The S3 bucket doesn't exist or is not accessible")
                self.logger.error("2. No processed data files exist in the bucket")
                self.logger.error("3. The DATA_LAKE_BUCKET environment variable is not set correctly")
                self.logger.error("Please ensure the feature engineering pipeline has run and created processed data")
                return []
            
            # Use the models_dir directly (which is already ml/models/currency)
            # This aligns with what the prediction lambda expects: models/currency/{currency_name}_{horizon}/
            # The experiment_id is still tracked in model metadata for reference
            self.config.paths.models_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Models will be saved to: {self.config.paths.models_dir} (experiment_id: {self.config.experiment.experiment_id})")
            
            # Extract currencies from processed data
            target_currencies = self._extract_currencies_from_processed_data(processed_data)
            self.logger.info(
                f"Training ALL currencies mode: {len(target_currencies)} currencies selected from processed data"
            )
            
            # Apply threshold filtering
            original_count = len(target_currencies)
            target_currencies = self._filter_currencies_by_thresholds(target_currencies, processed_data)
            filtered_count = len(target_currencies)
            
            if filtered_count < original_count:
                self.logger.info(
                    f"Filtered currencies by thresholds",
                    extra={
                        "original_count": original_count,
                        "filtered_count": filtered_count,
                        "min_records_threshold": int(self.config.data.min_records_threshold),
                        "min_avg_value_threshold": float(self.config.data.min_avg_value_threshold)
                    }
                )
            
            # Apply currency filter if specified
            if self.config.pipeline.currencies_to_train:
                requested = {c.lower() for c in self.config.pipeline.currencies_to_train}
                filtered = [
                    currency for currency in target_currencies
                    if currency.get('get_currency', '').lower() in requested
                ]
                missing = sorted(requested - {currency.get('get_currency', '').lower() for currency in filtered})
                
                self.logger.info(
                    "Filtering currencies based on --currencies argument",
                    extra={
                        "requested": sorted(self.config.pipeline.currencies_to_train),
                        "matched": [currency.get('get_currency') for currency in filtered],
                        "missing": missing
                    }
                )
                
                target_currencies = filtered
            
            # Apply explicit limit when requested
            if (
                self.config.data.max_currencies_to_train is not None and 
                len(target_currencies) > self.config.data.max_currencies_to_train
            ):
                original_count = len(target_currencies)
                target_currencies = target_currencies[:self.config.data.max_currencies_to_train]
                self.logger.info(
                    "Limiting currencies for training",
                    extra={
                        "requested_limit": self.config.data.max_currencies_to_train,
                        "original_total": original_count,
                        "truncated_total": len(target_currencies)
                    }
                )
            
            # Apply currency limit
            self.processing_stats['total_currencies'] = len(target_currencies)
            
            # Log selection results
            self.logger.info(f"Training all currencies with sufficient data: {len(target_currencies)} currencies selected")
            
            self.logger.info(f"Training models for {len(target_currencies)} currencies")
            
            # Setup progress tracking
            progress = ProgressLogger(
                self.logger, len(target_currencies), "Currency Model Training"
            )
            
            # Store processed data as instance variable for use in training methods
            self._processed_data = processed_data
            
            # Hyperparameter optimization
            if self.config.model.n_hyperparameter_trials <= 1:
                self.logger.warning(f"Hyperparameter optimization is disabled (n_hyperparameter_trials={self.config.model.n_hyperparameter_trials}). This will result in poor model performance. Set n_hyperparameter_trials > 1 to enable optimization.")
            else:
                self.logger.info(f"Per-currency hyperparameter optimization enabled with n_hyperparameter_trials={self.config.model.n_hyperparameter_trials}")
            
            # Use parallel training
            max_workers = self.config.model.max_currency_workers
            
            # Adjust workers based on currency count
            if len(target_currencies) == 1:
                max_workers = 1
                self.logger.info("Single currency training - using 1 worker")
            else:
                self.logger.info(f"Parallel currency training with {max_workers} workers")
            
            self._train_currencies_parallel(target_currencies, progress, max_workers)
            
            progress.complete()
            
            # Training completed
            
            # Log experiment end
            self.logger.log_experiment_end(
                self.config.experiment.experiment_id or "unknown",
                {
                    'total_models_trained': len(self.results),
                    'success_rate': len(self.results) / len(target_currencies) if target_currencies else 0,
                    'processing_stats': self.processing_stats,
                    'parallel_workers_used': max_workers
                }
            )
            
            # Upload logs to S3 after training completion
            self._upload_logs_to_s3()
            
            return self.results
            
        except Exception as e:
            self.logger.error("Training pipeline failed", exception=e)
            # Still try to upload logs even if training failed
            try:
                self._upload_logs_to_s3()
            except Exception as upload_error:
                self.logger.error(f"Failed to upload logs to S3: {upload_error}")
            raise
    
    def _train_currencies_parallel(self, target_currencies: List[Dict[str, Any]], 
                                 progress: ProgressLogger, max_workers: int) -> None:
        """
        Train currencies in parallel using ProcessPoolExecutor.
        
        Args:
            target_currencies: List of currency dictionaries
            progress: Progress logger instance
            max_workers: Maximum number of parallel workers
        """
        if max_workers == 1:
            # Single worker processing
            for currency in target_currencies:
                try:
                    currency_name = currency.get('get_currency', 'Unknown')
                    self.logger.info(f"🔄 Starting training for {currency_name}...")
                    
                    result = self._train_currency_model(currency)
                    if result:
                        self.results.append(result)
                        self.processing_stats['successful_training'] += 1
                        self.logger.info(f"✅ {currency_name} training completed successfully")
                        
                        # Upload models for this currency immediately after training
                        try:
                            self._upload_currency_models(currency_name, result)
                        except Exception as upload_error:
                            # Log error but don't fail the training
                            self.logger.warning(f"Failed to upload models for {currency_name}: {upload_error}")
                    else:
                        self.processing_stats['failed_training'] += 1
                        self.logger.warning(f"❌ {currency_name} training failed")
                        
                except Exception as e:
                    currency_name = currency.get('get_currency', 'Unknown')
                    self.logger.error(f"💥 {currency_name} training crashed", exception=e)
                    self.failed_currencies.append({
                        'currency': currency,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.processing_stats['failed_training'] += 1
                finally:
                    progress.update()
            return
        
        # Parallel processing
        processed_data_path = self._get_processed_data_path()
        shared_data = {
            'config_dict': self.config.to_dict(),
            'experiment_id': self.config.experiment.experiment_id,
            'processed_data_path': processed_data_path
        }
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit training tasks with timeout
                future_to_currency = {
                    executor.submit(
                        self._train_currency_worker,
                        currency,
                        shared_data
                    ): currency for currency in target_currencies
                }
                
                # Collect results
                for future in as_completed(future_to_currency):
                    currency = future_to_currency[future]
                    currency_name = currency.get('get_currency', 'Unknown')
                    
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                            self.processing_stats['successful_training'] += 1
                            
                            # Log detailed completion statistics for parallel training
                            self._log_currency_completion_stats(currency_name, result)
                            
                            # Upload models for this currency immediately after training
                            try:
                                self._upload_currency_models(currency_name, result)
                            except Exception as upload_error:
                                # Log error but don't fail the training
                                self.logger.warning(f"Failed to upload models for {currency_name}: {upload_error}")
                        else:
                            self.processing_stats['failed_training'] += 1
                            self.logger.warning(f"❌ {currency_name}: Training returned None")
                    
                    except Exception as e:
                        error_msg = str(e)
                        if "process pool was terminated" in error_msg:
                            self.logger.error(f"💥 {currency_name}: Process pool terminated - likely due to resource exhaustion")
                        else:
                            self.logger.error(f"💥 {currency_name}: Training crashed - {error_msg}")
                        
                        self.failed_currencies.append({
                            'currency': currency,
                            'error': error_msg,
                            'timestamp': datetime.now().isoformat()
                        })
                        self.processing_stats['failed_training'] += 1
                    
                    finally:
                        progress.update()
                        
        except Exception as e:
            self.logger.error(f"💥 Process pool executor failed: {str(e)}")
            # Cancel remaining futures
            for future in future_to_currency:
                future.cancel()
        finally:
            # Clean up temporary file
            if processed_data_path and os.path.exists(processed_data_path):
                try:
                    os.remove(processed_data_path)
                    # Also remove the parent directory if it's empty
                    temp_dir = os.path.dirname(processed_data_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file {processed_data_path}: {e}")
    
    @staticmethod
    def _train_currency_worker(currency: Dict[str, Any], shared_data: Dict[str, Any]) -> Optional[TrainingResult]:
        """
        Worker function for parallel currency training with robust error handling.
        
        Args:
            currency: Currency dictionary
            shared_data: Shared data containing config and processed data path
            
        Returns:
            Training result dictionary or None if failed
        """
        try:
            # Set process name for debugging
            import os
            currency_name = currency.get('get_currency', 'Unknown')
            os.setpgrp()  # Create new process group to prevent signal propagation
            
            # Reconstruct configuration from dict
            config = MLConfig.from_dict(shared_data['config_dict'])
            
            # Calculate optimal threading based on system resources
            import multiprocessing
            total_cores = multiprocessing.cpu_count()
            max_currency_workers = config.model.max_currency_workers
            optimal_model_threads = min(4, max(2, total_cores // max_currency_workers))
            
            # Set optimal threading for this worker
            config.model.model_n_jobs = optimal_model_threads
            config.model.max_optuna_workers = max(1, min(config.model.max_optuna_workers, optimal_model_threads))
            
            # Create pipeline instance for this worker
            pipeline = ModelTrainingPipeline(config)
            
            # Load processed data in the worker
            processed_data_path = shared_data.get('processed_data_path')
            if processed_data_path:
                pipeline._processed_data = pipeline._load_processed_data_from_path(processed_data_path)
            
            # Train the currency model
            result = pipeline._train_currency_model(currency)
            
            return result
            
        except MemoryError as e:
            # Handle memory exhaustion gracefully
            import sys
            sys.stderr.write(f"Memory error in worker for currency {currency_name}: {str(e)}\n")
            return None
            
        except Exception as e:
            # Use the main pipeline logger for error reporting
            # Note: This will be handled by the main process logger
            # Log error to stderr for worker process debugging
            import sys
            sys.stderr.write(f"Worker failed for currency {currency_name}: {str(e)}\n")
            return None
    
    def _get_processed_data_path(self) -> Optional[str]:
        """
        Get the path to the processed data file.
        
        Returns:
            Path to processed data file or None if not available
        """
        if hasattr(self, '_processed_data') and self._processed_data is not None:
            # Save to temporary file for sharing with workers
            return self._save_temp_dataframe(self._processed_data)
        return None
    
    def _load_processed_data_from_path(self, data_path: str) -> Optional[pd.DataFrame]:
        """
        Load processed data from file path.
        
        Args:
            data_path: Path to the processed data file
            
        Returns:
            Loaded dataframe or None if failed
        """
        try:
            return pd.read_parquet(data_path)
        except Exception as e:
            # Use stderr for worker process error reporting
            # Note: This will be handled by the main process logger
            import sys
            sys.stderr.write(f"Failed to load processed data from {data_path}: {str(e)}\n")
            return None
    
    def _save_temp_dataframe(self, df: pd.DataFrame) -> str:
        """
        Save dataframe to temporary file for parallel processing.
        
        Args:
            df: Dataframe to save
            
        Returns:
            Path to temporary file
        """
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"training_data_{self.config.experiment.experiment_id}.parquet")
        df.to_parquet(temp_path, index=False)
        
        self.logger.debug(f"Saved temporary dataframe to {temp_path}")
        return temp_path
    
    def _cleanup_temp_dataframe(self, temp_path: str) -> None:
        """
        Clean up temporary dataframe file.
        
        Args:
            temp_path: Path to temporary file
        """
        try:
            import os
            import shutil
            temp_dir = os.path.dirname(temp_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.debug(f"Cleaned up temporary dataframe at {temp_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
    
    def _extract_currencies_from_processed_data(self, processed_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract unique currency pairs from processed data.
        
        Args:
            processed_data: Processed dataframe containing currency data
            
        Returns:
            List of currency dictionaries with get_currency and avg_value
        """
        try:
            # Get unique currencies from processed data
            unique_currencies = processed_data['currency'].unique()
            
            # Convert to list of dictionaries with avg_value calculation
            currencies = []
            for currency in unique_currencies:
                # Calculate average price for this currency
                currency_data = processed_data[processed_data['currency'] == currency]
                avg_value = currency_data['price'].mean() if 'price' in currency_data.columns else 0.0
                
                currencies.append({
                    'get_currency': currency,
                    'avg_value': avg_value
                })
            
            # Sort by average value (descending) for consistent ordering
            currencies.sort(key=lambda x: x['avg_value'], reverse=True)
            
            self.logger.info(f"Extracted {len(currencies)} unique currencies from processed data")
            
            # Log some examples with their average values
            if currencies:
                examples = currencies[:5]
                for example in examples:
                    self.logger.debug(f"  - {example['get_currency']} (avg_value: {example['avg_value']:.2f})")
            
            return currencies
            
        except Exception as e:
            self.logger.error(f"Failed to extract currencies from processed data: {e}")
            return []
    
    def _filter_currencies_by_thresholds(self, currencies: List[Dict[str, Any]], processed_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Filter currencies based on min_records_threshold and min_avg_value_threshold.
        
        Args:
            currencies: List of currency dictionaries
            processed_data: Processed dataframe for record counting
            
        Returns:
            Filtered list of currencies that meet the thresholds
        """
        try:
            filtered_currencies = []
            
            for currency in currencies:
                currency_name = currency['get_currency']
                
                # Count records for this currency
                currency_data = processed_data[processed_data['currency'] == currency_name]
                record_count = len(currency_data)
                
                # Check if currency meets both thresholds
                meets_records_threshold = bool(record_count >= self.config.data.min_records_threshold)
                meets_value_threshold = bool(currency['avg_value'] >= self.config.data.min_avg_value_threshold)
                
                if meets_records_threshold and meets_value_threshold:
                    filtered_currencies.append(currency)
                else:
                    self.logger.debug(
                        f"Filtered out {currency_name}",
                        extra={
                            "record_count": record_count,
                            "avg_value": currency['avg_value'],
                            "meets_records": meets_records_threshold,
                            "meets_value": meets_value_threshold
                        }
                    )
            
            self.logger.info(
                f"Currency filtering completed",
                extra={
                    "original_count": len(currencies),
                    "filtered_count": len(filtered_currencies),
                    "min_records_threshold": int(self.config.data.min_records_threshold),
                    "min_avg_value_threshold": float(self.config.data.min_avg_value_threshold)
                }
            )
            
            return filtered_currencies
            
        except Exception as e:
            self.logger.error(f"Failed to filter currencies by thresholds: {e}")
            return currencies  # Return original list if filtering fails
    
    
    def _load_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Load processed parquet data from S3.
        
        Returns:
            Processed dataframe or None if not available
        """
        try:
            # Load processed data using S3 data source
            data_lake_bucket = os.getenv('DATA_LAKE_BUCKET', '')
            processed_data, experiment_id = self.data_source.load_processed_parquet_data_with_experiment_id(
                data_lake_bucket=data_lake_bucket,
                experiment_id=None  # Get the most recent
            )
            
            if processed_data is not None:
                self.logger.info(f"Successfully loaded processed data: {processed_data.shape}")
                
                # Update the experiment ID in the config to match the processed data
                if experiment_id:
                    self.config.experiment.experiment_id = experiment_id
                    self.logger.debug(f"Using experiment ID from processed data: {experiment_id}")
                
                return processed_data
            else:
                self.logger.info("No processed parquet files found in S3")
                return None
            
        except Exception as e:
            data_lake_bucket = os.getenv('DATA_LAKE_BUCKET', 'NOT_SET')
            self.logger.error(f"Failed to load processed data: {e}")
            self.logger.error(f"Attempted to load from bucket: {data_lake_bucket}")
            self.logger.error("Please check:")
            self.logger.error("1. Bucket exists and is accessible")
            self.logger.error("2. AWS credentials are properly configured")
            self.logger.error("3. Feature engineering pipeline has run successfully")
            return None
    
    def _train_currency_model(self, currency: Dict[str, Any]) -> Optional[TrainingResult]:
        """
        Train model for a single currency using processed data from data source.
        
        Args:
            currency: Currency dictionary with get_currency
            
        Returns:
            Training result dictionary or None if failed
        """
        currency_name = currency['get_currency']
        with self.logger.log_operation(f"Training model for {currency_name}"):
            # Use stored processed data
            processed_data = self._processed_data
            
            if processed_data is None:
                self.logger.error("No processed data available")
                self.processing_stats['insufficient_data'] += 1
                return None
            
            # Filter processed data for this specific currency
            currency_data = processed_data[
                processed_data['currency'] == currency_name
            ].copy()
            
            if currency_data.empty:
                self.logger.warning(f"No processed data found for {currency_name}")
                self.processing_stats['insufficient_data'] += 1
                return None
            
            self.logger.debug(f"Using processed data for {currency_name}: {len(currency_data)} records")
            
            # Log overall data statistics for this currency
            self.logger.debug(f"Currency data overview for {currency_name}:")
            self.logger.debug(f"  Total records: {len(currency_data):,}")
            if 'timestamp' in currency_data.columns:
                date_range = currency_data['timestamp'].agg(['min', 'max'])
                self.logger.debug(f"  Date range: {date_range['min']} to {date_range['max']}")
            
            # Log league distribution for this currency
            if 'league_name' in currency_data.columns:
                league_dist = currency_data['league_name'].value_counts()
                self.logger.debug(f"League distribution for {currency_name}:")
                for league, count in league_dist.items():
                    self.logger.debug(f"  {league}: {count:,} records")
                
                # Check for included leagues data
                if self.config.data.included_leagues:
                    included_leagues = list(self.config.data.included_leagues)
                    for league in included_leagues:
                        league_data = currency_data[currency_data['league_name'].str.contains(league, case=False, na=False)]
                        if not league_data.empty:
                            self.logger.debug(f"{league} data available for training: {len(league_data):,} records")
                        else:
                            self.logger.warning(f"No {league} data found for {currency_name}")
            
            # Use the already processed data directly (no additional feature engineering needed)
            processed_data = currency_data
            processing_metadata = {
                'currency': currency_name,
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'validation_result': 'passed',
                'feature_engineering_result': 'data_already_processed',
                'success': True,
                'leagues_included': list(currency_data['league_name'].unique()) if 'league_name' in currency_data.columns else []
            }
            
            # Prepare features and targets
            feature_columns = self._get_feature_columns(processed_data)
            target_column = self._get_target_column(processed_data)
            
            if not feature_columns:
                self.logger.warning(f"No feature columns found for {currency_name}")
                self.processing_stats['validation_failures'] += 1
                return None
            
            if target_column is None:
                self.logger.warning(f"No target columns found for {currency_name}")
                self.processing_stats['validation_failures'] += 1
                return None
            
            # Log target column information
            if isinstance(target_column, list):
                self.logger.debug(f"Found {len(target_column)} target horizons: {target_column}")
                # Log NaN statistics for each target column
                for target_col in target_column:
                    if target_col in processed_data.columns:
                        nan_count = processed_data[target_col].isna().sum()
                        total_count = len(processed_data[target_col])
                        nan_pct = (nan_count / total_count) * 100 if total_count > 0 else 0
                        self.logger.debug(f"  {target_col}: {nan_count:,}/{total_count:,} NaN ({nan_pct:.1f}%)")
            else:
                self.logger.debug(f"Found single target: {target_column}")
                if target_column in processed_data.columns:
                    nan_count = processed_data[target_column].isna().sum()
                    total_count = len(processed_data[target_column])
                    nan_pct = (nan_count / total_count) * 100 if total_count > 0 else 0
                    self.logger.debug(f"  {target_column}: {nan_count:,}/{total_count:,} NaN ({nan_pct:.1f}%)")
            
            # Validate target columns exist
            if isinstance(target_column, list):
                missing_targets = [col for col in target_column if col not in processed_data.columns]
                if missing_targets:
                    self.logger.warning(f"Missing target columns for {currency_name}: {missing_targets}")
                    self.processing_stats['validation_failures'] += 1
                    return None
                self.logger.debug(f"Found {len(target_column)} target horizons: {target_column}")
            else:
                if target_column not in processed_data.columns:
                    self.logger.warning(f"Target column {target_column} not found for {currency_name}")
                    self.processing_stats['validation_failures'] += 1
                    return None
                self.logger.debug(f"Found single target: {target_column}")
            
            # Prepare feature matrix
            X = processed_data[feature_columns].values
            self.logger.debug(f"Initial feature matrix shape: X={X.shape}")
            
            # Check for NaN values in features
            if X.dtype.kind in 'biufc':  # binary, integer, unsigned, float, complex
                nan_count = np.isnan(X).sum()
                self.logger.debug(f"Initial NaN count in features: {nan_count}")
            else:
                self.logger.debug(f"Features contain non-numeric data (dtype: {X.dtype})")
                try:
                    X = X.astype(float)
                    nan_count = np.isnan(X).sum()
                    self.logger.debug(f"Converted to numeric. NaN count in features: {nan_count}")
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Cannot convert features to numeric: {e}")
                    return None
            
            # Basic data quality filtering - more lenient for minimal data
            # Remove rows with too many NaN values (>90% of features are NaN)
            row_nan_ratio = np.isnan(X).sum(axis=1) / X.shape[1]
            valid_rows = row_nan_ratio <= 0.9  # More lenient threshold
            X = X[valid_rows]
            processed_data = processed_data[valid_rows].reset_index(drop=True)
            
            self.logger.debug(
                f"Feature NaN filtering results:",
                extra={
                    'samples_before': len(row_nan_ratio),
                    'samples_after': len(X),
                    'removed_samples': len(row_nan_ratio) - len(X),
                    'max_nan_ratio': float(row_nan_ratio.max()) if len(row_nan_ratio) > 0 else 0,
                    'mean_nan_ratio': float(row_nan_ratio.mean()) if len(row_nan_ratio) > 0 else 0
                }
            )
            
            # Impute remaining NaN values with median (column-wise)
            if np.isnan(X).any():
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
                self.logger.debug(f"Applied median imputation for remaining NaN values")

            if len(X) < 5:  # Reduced minimum from 50 to 5
                self.logger.warning(f"Insufficient data for {currency_name}: {len(X)} samples")
                self.processing_stats['insufficient_data'] += 1
                return None

            # Train model(s) for different prediction horizons
            if isinstance(target_column, list) and len(target_column) > 1:
                # Multi-horizon training: train separate models for each horizon
                self.logger.debug(f"Multi-horizon training for {len(target_column)} horizons: {target_column}")
                
                horizon_results = {}
                horizon_models = {}
                
                for target_col in target_column:
                    horizon = target_col.replace('target_price_', '')
                    self.logger.debug(f"Training model for {horizon} horizon")
                    
                    # Use all available features for this horizon
                    X_horizon_filtered = processed_data[feature_columns].values
                    
                    # Get target for this horizon
                    y_horizon = processed_data[target_col].values
                    
                    # Log data filtering statistics
                    total_samples = len(y_horizon)
                    nan_count = pd.isna(y_horizon).sum()
                    valid_count = total_samples - nan_count
                    nan_percentage = (nan_count / total_samples) * 100 if total_samples > 0 else 0
                    
                    self.logger.debug(f"Data filtering for {currency_name} {horizon}:")
                    self.logger.debug(f"  Total samples: {total_samples:,}")
                    self.logger.debug(f"  NaN values: {nan_count:,} ({nan_percentage:.1f}%)")
                    self.logger.debug(f"  Valid samples: {valid_count:,}")
                    
                    # Filter out NaN values for this specific target
                    target_valid_mask = ~pd.isna(y_horizon)
                    X_horizon = X_horizon_filtered[target_valid_mask]
                    y_horizon = y_horizon[target_valid_mask]
                    
                    self.logger.debug(f"  After filtering: {len(X_horizon):,} samples available for training")
                    
                    # If we have very few samples, try a more lenient approach
                    if len(X_horizon) < 20:
                        # Try using forward-fill for NaN values to preserve more data
                        y_horizon_ffill = processed_data[target_col].fillna(method='ffill').values
                        target_valid_mask_ffill = ~pd.isna(y_horizon_ffill)
                        X_horizon_ffill = X_horizon_filtered[target_valid_mask_ffill]
                        y_horizon_ffill = y_horizon_ffill[target_valid_mask_ffill]
                        
                        if len(X_horizon_ffill) > len(X_horizon):
                            self.logger.debug(f"  Using forward-fill strategy: {len(X_horizon_ffill):,} samples (vs {len(X_horizon):,} without)")
                            X_horizon = X_horizon_ffill
                            y_horizon = y_horizon_ffill
                    
                    if len(X_horizon) < 10:  # Absolute minimum threshold
                        self.logger.warning(f"Insufficient data for {currency_name} {horizon} horizon: {len(X_horizon)} samples (need at least 10)")
                        continue
                    
                    # Train model for this horizon
                    training_result = self.model_trainer.train_single_model(
                        X_horizon, y_horizon, f"{currency_name}_{horizon}", 
                        target_names=[target_col],
                        feature_names=feature_columns
                    )
                    
                    if training_result is not None:
                        horizon_results[horizon] = training_result
                        horizon_models[horizon] = training_result.model
                        self.logger.debug(f"Successfully trained {horizon} model for {currency_name}")
                    else:
                        self.logger.warning(f"Failed to train {horizon} model for {currency_name}")
                
                if not horizon_results:
                    self.logger.error(f"All horizon models failed for {currency_name}")
                    return None
                
                # Create combined result for multi-horizon models
                primary_horizon = '1d' if '1d' in horizon_results else list(horizon_results.keys())[0]
                primary_result = horizon_results[primary_horizon]
                
                # Save models for each horizon
                model_info = {}
                for horizon, result in horizon_results.items():
                    try:
                        horizon_model_dir = self.config.paths.models_dir / f"{currency_name}_{horizon}"
                        horizon_model_info = save_model_artifacts(result, horizon_model_dir, f"{currency_name}_{horizon}")
                        model_info[f"{horizon}_model"] = horizon_model_info
                    except Exception as e:
                        self.logger.error(f"Failed to save {horizon} model artifacts for {currency_name}: {e}")
                        return None
                
                # Also save the primary model in the main directory for compatibility
                try:
                    main_model_info = save_model_artifacts(
                        primary_result, self.config.paths.models_dir, currency_name
                    )
                    model_info['primary_model'] = main_model_info
                except Exception as e:
                    self.logger.error(f"Failed to save primary model artifacts for {currency_name}: {e}")
                    return None
                
                # Compile results
                result = {
                    'currency': currency_name,
                    'get_currency': currency['get_currency'],
                    'training_samples': {horizon: len(processed_data[~pd.isna(processed_data[f'target_price_{horizon}'])]) 
                                       for horizon in horizon_results.keys()},
                    'training_metrics': {horizon: result.metrics for horizon, result in horizon_results.items()},
                    'model_info': model_info,
                    'processing_metadata': processing_metadata,
                    'feature_count': len(feature_columns),
                    'leagues_in_training': processing_metadata.get('leagues_included', []),
                    'horizons_trained': list(horizon_results.keys()),
                    'primary_horizon': primary_horizon
                }
                
            else:
                # Single-horizon training (should not happen with proper multi-horizon setup)
                self.logger.warning(f"Using single-horizon training fallback for {currency_name}. This indicates a configuration issue.")
                target_col = target_column if isinstance(target_column, str) else target_column[0]
                
                # Extract horizon from target column name (e.g., "target_price_3d" -> "3d")
                horizon = target_col.replace('target_price_', '') if 'target_price_' in target_col else '1d'
                
                # Use all available features
                X_horizon_filtered = processed_data[feature_columns].values
                y = processed_data[target_col].values
                
                # Filter out NaN values
                target_valid_mask = ~pd.isna(y)
                X = X_horizon_filtered[target_valid_mask]
                y = y[target_valid_mask]
                
                if len(X) < 5:  # Reduced minimum from 50 to 5
                    self.logger.warning(f"Insufficient data for {currency_name}: {len(X)} samples")
                    self.processing_stats['insufficient_data'] += 1
                    return None
                
                # Train single model
                training_result = self.model_trainer.train_single_model(
                    X, y, currency_name, 
                    target_names=[target_col],
                    feature_names=feature_columns
                )
                
                if training_result is None:
                    self.logger.error(f"Model training failed for {currency_name}")
                    return None
                
                # Save model artifacts
                try:
                    model_info = save_model_artifacts(
                        training_result, self.config.paths.models_dir, currency_name
                    )
                except Exception as e:
                    self.logger.error(f"Failed to save model artifacts for {currency_name}: {e}")
                    return None
                
                # Compile results
                result = {
                    'currency': currency_name,
                    'get_currency': currency['get_currency'],
                    'training_samples': len(X),
                    'training_metrics': training_result.metrics,
                    'model_info': model_info,
                    'processing_metadata': processing_metadata,
                    'feature_count': len(feature_columns),
                    'leagues_in_training': processing_metadata.get('leagues_included', [])
                }
            
            # Log detailed completion statistics
            self._log_currency_completion_stats(currency_name, result)
            
            # Log training result
            self.logger.debug(f"Training completed for {currency_name}")
            
            return result
    
    def _log_currency_completion_stats(self, currency_name: str, result: Dict[str, Any]) -> None:
        """
        Log detailed completion statistics for a currency.
        
        Args:
            currency_name: Name of the currency
            result: Training result dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info(f"✅ CURRENCY TRAINING COMPLETED: {currency_name}")
        self.logger.info("=" * 60)
        
        # Basic statistics
        training_samples = result.get('training_samples', 0)
        feature_count = result.get('feature_count', 0)
        leagues = result.get('leagues_in_training', [])
        
        # Ensure training_samples is a number, not a dict
        if isinstance(training_samples, dict):
            training_samples = sum(training_samples.values()) if training_samples else 0
        elif not isinstance(training_samples, (int, float)):
            training_samples = 0
            
        # Ensure feature_count is a number
        if isinstance(feature_count, dict):
            feature_count = len(feature_count) if feature_count else 0
        elif not isinstance(feature_count, (int, float)):
            feature_count = 0
        
        self.logger.info(f"📊 DATA STATISTICS:")
        self.logger.info(f"   • Training samples: {int(training_samples):,}")
        self.logger.info(f"   • Feature count: {int(feature_count)}")
        self.logger.info(f"   • Leagues included: {len(leagues)} ({', '.join(leagues[:3])}{'...' if len(leagues) > 3 else ''})")
        
        # Training metrics
        training_metrics = result.get('training_metrics', {})
        if training_metrics:
            self.logger.info(f"📈 MODEL PERFORMANCE:")
            
            # Handle multi-horizon results
            if 'horizons_trained' in result:
                horizons = result.get('horizons_trained', [])
                primary_horizon = result.get('primary_horizon', horizons[0] if horizons else 'unknown')
                
                self.logger.info(f"   • Horizons trained: {', '.join(horizons)}")
                self.logger.info(f"   • Primary horizon: {primary_horizon}")
                
                # Log metrics for primary horizon
                if primary_horizon in training_metrics:
                    metrics = training_metrics[primary_horizon]
                    self._log_metrics_details(metrics, f"   • {primary_horizon.upper()} HORIZON:")
                
                # Log summary for other horizons
                for horizon in horizons:
                    if horizon != primary_horizon and horizon in training_metrics:
                        metrics = training_metrics[horizon]
                        rmse = getattr(metrics, 'rmse', 'N/A')
                        r2 = getattr(metrics, 'r2', 'N/A')
                        self.logger.info(f"   • {horizon.upper()}: RMSE={rmse:.4f}, R²={r2:.4f}")
            else:
                # Single horizon results
                self._log_metrics_details(training_metrics, "   • PERFORMANCE:")
        
        # Model info
        model_info = result.get('model_info', {})
        if model_info:
            self.logger.info(f"💾 MODEL ARTIFACTS:")
            
            # Handle multi-horizon models (nested structure)
            if 'primary_model' in model_info:
                # Multi-horizon: use primary_model info
                primary_info = model_info.get('primary_model', {})
                model_size = primary_info.get('model_size_mb', 0)
                model_dir = primary_info.get('model_dir', 'N/A')
            else:
                # Single-horizon: direct structure
                model_size = model_info.get('model_size_mb', 0)
                model_dir = model_info.get('model_dir', 'N/A')
            
            self.logger.info(f"   • Model size: {model_size:.2f} MB")
            self.logger.info(f"   • Saved to: {model_dir}")
        
        # Processing metadata
        processing_metadata = result.get('processing_metadata', {})
        if processing_metadata:
            timestamp = processing_metadata.get('processing_timestamp', 'N/A')
            self.logger.info(f"⏱️  PROCESSING INFO:")
            self.logger.info(f"   • Completed at: {timestamp}")
            self.logger.info(f"   • Status: {processing_metadata.get('validation_result', 'N/A')}")
        
        self.logger.info("=" * 60)
    
    def _log_metrics_details(self, metrics: Dict[str, Any], prefix: str) -> None:
        """
        Log detailed metrics information.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for log messages
        """
        # Core regression metrics
        rmse = getattr(metrics, 'rmse', 'N/A')
        mae = getattr(metrics, 'mae', 'N/A')
        r2 = getattr(metrics, 'r2', 'N/A')
        
        self.logger.info(f"{prefix}")
        self.logger.info(f"     - RMSE: {rmse:.4f}" if isinstance(rmse, (int, float)) else f"     - RMSE: {rmse}")
        self.logger.info(f"     - MAE:  {mae:.4f}" if isinstance(mae, (int, float)) else f"     - MAE:  {mae}")
        self.logger.info(f"     - R²:   {r2:.4f}" if isinstance(r2, (int, float)) else f"     - R²:   {r2}")
        
        # Additional metrics if available
        mse = getattr(metrics, 'mse', None)
        if mse is not None:
            self.logger.info(f"     - MSE:  {mse:.4f}" if isinstance(mse, (int, float)) else f"     - MSE:  {mse}")
        
        ev = getattr(metrics, 'explained_variance', None)
        if ev is not None:
            self.logger.info(f"     - EV:   {ev:.4f}" if isinstance(ev, (int, float)) else f"     - EV:   {ev}")
        
        # Model-specific metrics
        importance = getattr(metrics, 'feature_importance', None)
        if importance is not None and isinstance(importance, dict) and importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                self.logger.info(f"     - Top features: {', '.join([f'{f}({v:.3f})' for f, v in top_features])}")
    
    def _find_latest_training_data(self) -> str:
        """Find the latest training data file."""
        data_dir = self.config.paths.data_dir
        pattern = self.config.paths.combined_data_pattern.format(
            experiment_id='*'
        )
        
        files = list(data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No training data files found matching {pattern}")
        
        # Sort by modification time
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)
    
    def _get_feature_columns(self, df: pd.DataFrame) -> Optional[List[str]]:
        """Get feature columns from dataframe.
        
        This method must match the feature extraction logic in ModelPredictor._prepare_features
        to ensure training and inference use the same feature sets.
        """
        exclude_patterns = [
            'target_', 'date', 'league_name', 'currency', 'id', 'league_start', 'league_end',
            'league_active', 'get_currency', 'pay_currency', '_multi_output_targets'
        ]
        feature_cols = [col for col in df.columns 
                       if not any(pattern in col for pattern in exclude_patterns)]
        
        # Filter to only include numeric columns - use same method as inference
        numeric_feature_cols = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_feature_cols.append(col)
            else:
                self.logger.debug(f"Excluding non-numeric column: {col} (dtype: {df[col].dtype})")
        
        if not numeric_feature_cols:
            self.logger.warning("No numeric feature columns found after filtering")
            return None
            
        return numeric_feature_cols
    
    def _get_target_column(self, df: pd.DataFrame) -> Optional[Union[str, List[str]]]:
        """Get target column(s) from dataframe."""
        target_cols = [col for col in df.columns if col.startswith('target_price_')]
        if not target_cols:
            return None
        
        # Return the primary prediction horizons for separate model training
        # We'll train separate models for 1d, 3d, and 7d predictions
        available_horizons = []
        for horizon in ['1d', '3d', '7d']:
            target_col = f'target_price_{horizon}'
            if target_col in target_cols:
                available_horizons.append(target_col)
        
        if available_horizons:
            return available_horizons  # Return list for multi-horizon training
        else:
            # Fallback to any available target
            return sorted(target_cols, key=lambda x: int(''.join(filter(str.isdigit, x))))[0]  # type: ignore[no-any-return]
    
    def _evaluate_model_comprehensive(
        self, 
        df: pd.DataFrame, 
        training_result: Any, 
        currency: str,
        feature_columns: List[str],
        target_column: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation including league-specific metrics.
        
        Args:
            df: Complete dataframe with league information
            training_result: Trained model result
            currency: Currency name
            feature_columns: List of feature column names
            target_column: Target column name(s) - can be single string or list for multi-output
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        evaluation_results: Dict[str, Any] = {
            'overall_metrics': None,
            'league_specific_metrics': {},
            'settlers_detailed_metrics': None,
            'data_quality_metrics': {}
        }
        
        try:
            # Handle both single and multi-output cases
            is_multi_output = isinstance(target_column, list)
            
            # Overall evaluation
            X = df[feature_columns].values
            
            if is_multi_output:
                y = df[target_column].values
                # For multi-output, remove rows where ALL targets are NaN
                valid_mask = ~(pd.isna(X).any(axis=1) | pd.isna(y).all(axis=1))
            else:
                y = df[target_column].values
                # For single output, remove rows where target is NaN
                valid_mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            df_valid = df[valid_mask].reset_index(drop=True)
            
            if len(X_valid) > 0:
                # Impute any remaining NaN values
                # Check if X_valid is numeric before using np.isnan
                if X_valid.dtype.kind in 'biufc' and np.isnan(X_valid).any():
                    imputer = SimpleImputer(strategy='median')
                    X_valid = imputer.fit_transform(X_valid)
                
                # Overall predictions
                y_pred = training_result.model.predict(X_valid)
                
                # Calculate metrics based on output type
                if is_multi_output:
                    # Multi-output metrics
                    target_names = target_column
                    overall_metrics = {}
                    
                    for i, target_name in enumerate(target_names):
                        y_true_single = y_valid[:, i]
                        y_pred_single = y_pred[:, i]
                        
                        # Skip if all values are NaN
                        valid_mask_single = ~(np.isnan(y_true_single) | np.isnan(y_pred_single))
                        if not np.any(valid_mask_single):
                            continue
                        
                        y_true_valid = y_true_single[valid_mask_single]
                        y_pred_valid = y_pred_single[valid_mask_single]
                        
                        overall_metrics[target_name] = {
                            'mae': float(mean_absolute_error(y_true_valid, y_pred_valid)),
                            'rmse': float(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))),
                            'r2': float(r2_score(y_true_valid, y_pred_valid)),
                            'samples': len(y_true_valid),
                            'mape': float(np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100) if np.all(y_true_valid != 0) else None
                        }
                    
                    evaluation_results['overall_metrics'] = overall_metrics
                else:
                    # Single output metrics
                    evaluation_results['overall_metrics'] = {
                        'mae': float(mean_absolute_error(y_valid, y_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_valid, y_pred))),
                        'r2': float(r2_score(y_valid, y_pred)),
                        'samples': len(y_valid),
                        'mape': float(np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100) if np.all(y_valid != 0) else None
                    }
                
                # League-specific evaluation
                if 'league_name' in df.columns:
                    league_names = df_valid['league_name'].unique()
                    
                    for league in league_names:
                        league_mask = df_valid['league_name'] == league
                        if league_mask.sum() > 5:  # At least 5 samples for evaluation
                            X_league = X_valid[league_mask]
                            y_league = y_valid[league_mask]
                            
                            y_pred_league = training_result.model.predict(X_league)
                            
                            league_metrics = {
                                'mae': float(mean_absolute_error(y_league, y_pred_league)),
                                'rmse': float(np.sqrt(mean_squared_error(y_league, y_pred_league))),
                                'r2': float(r2_score(y_league, y_pred_league)),
                                'samples': len(y_league),
                                'mape': float(np.mean(np.abs((y_league - y_pred_league) / y_league)) * 100) if np.all(y_league != 0) else None
                            }
                            
                            evaluation_results['league_specific_metrics'][league] = league_metrics
                            
                            # Detailed Settlers evaluation
                            if 'Settlers' in league:
                                evaluation_results['settlers_detailed_metrics'] = {
                                    **league_metrics,
                                    'prediction_bias': float(np.mean(y_pred_league - y_league)),
                                    'prediction_std': float(np.std(y_pred_league - y_league)),
                                    'price_range': {
                                        'min': float(y_league.min()),
                                        'max': float(y_league.max()),
                                        'median': float(np.median(y_league))
                                    },
                                    'temporal_coverage': {
                                        'start_date': df_valid[league_mask]['date'].min().isoformat() if 'date' in df_valid.columns else None,
                                        'end_date': df_valid[league_mask]['date'].max().isoformat() if 'date' in df_valid.columns else None,
                                        'days_covered': (df_valid[league_mask]['date'].max() - df_valid[league_mask]['date'].min()).days if 'date' in df_valid.columns else None
                                    }
                                }
                
                # Data quality metrics
                data_quality_metrics = {
                    'total_samples': len(df),
                    'valid_samples': len(X_valid),
                    'missing_data_ratio': 1 - (len(X_valid) / len(df)) if len(df) > 0 else 0,
                    'feature_completeness': {
                        col: 1 - (df[col].isna().sum() / len(df)) 
                        for col in feature_columns[:5]  # Sample of first 5 features
                    },
                    'leagues_represented': list(df['league_name'].unique()) if 'league_name' in df.columns else []
                }
                evaluation_results['data_quality_metrics'] = data_quality_metrics
                
            # Log data quality metrics
            self.logger.debug(f"Data quality metrics calculated for {currency}")
                
            self.logger.debug(f"Comprehensive evaluation completed for {currency}")
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed for {currency}: {str(e)}")
            evaluation_results['error'] = str(e)
        
        return evaluation_results
    

    def _upload_currency_models(self, currency_name: str, result: Dict[str, Any]) -> None:
        """
        Upload models for a specific currency to S3 immediately after training.
        
        Args:
            currency_name: Name of the currency
            result: Training result dictionary containing model_info
        """
        if not self.models_bucket:
            self.logger.debug(f"MODELS_BUCKET not configured, skipping model upload for {currency_name}")
            return
        
        try:
            import boto3
            from botocore.exceptions import ClientError, BotoCoreError
            
            # Get experiment ID
            experiment_id = self.config.experiment.experiment_id
            if not experiment_id:
                self.logger.warning(f"No experiment_id configured, skipping model upload for {currency_name}")
                return
            
            # Get model info from result
            model_info = result.get('model_info', {})
            if not model_info:
                self.logger.warning(f"No model_info in result for {currency_name}, skipping upload")
                return
            
            # Create S3 client
            s3_client = boto3.client('s3')
            
            # Models are saved in models_dir (which is /app/ml/models/currency/)
            models_dir = self.config.paths.models_dir
            
            # Find all model directories for this currency
            # Models are saved as {currency_name}_{horizon}/ or just {currency_name}/
            currency_model_dirs = []
            if models_dir.exists():
                for item in models_dir.iterdir():
                    if item.is_dir():
                        # Check if this directory belongs to this currency
                        dir_name = item.name
                        if dir_name.startswith(f"{currency_name}_") or dir_name == currency_name:
                            currency_model_dirs.append(item)
            
            if not currency_model_dirs:
                self.logger.warning(f"No model directories found for {currency_name} in {models_dir}")
                return
            
            # Upload all files from currency model directories
            uploaded_count = 0
            failed_count = 0
            
            for model_dir in currency_model_dirs:
                try:
                    # Walk through the model directory and upload all files
                    for file_path in model_dir.rglob('*'):
                        if file_path.is_file():
                            # Calculate relative path from models_dir
                            relative_path = file_path.relative_to(models_dir)
                            # Normalize path separators for S3 (use forward slashes)
                            relative_path_str = str(relative_path).replace(os.sep, '/')
                            
                            # Create S3 key with experiment_id
                            s3_key = f"models/currency/{experiment_id}/{relative_path_str}"
                            
                            try:
                                # Upload file
                                s3_client.upload_file(str(file_path), self.models_bucket, s3_key)
                                uploaded_count += 1
                                self.logger.debug(f"Uploaded {file_path.name} to s3://{self.models_bucket}/{s3_key}")
                            except ClientError as e:
                                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                                error_msg = e.response.get('Error', {}).get('Message', str(e))
                                self.logger.warning(f"Failed to upload {file_path} to s3://{self.models_bucket}/{s3_key}: {error_code} - {error_msg}")
                                failed_count += 1
                            except BotoCoreError as e:
                                self.logger.warning(f"Boto3 error uploading {file_path} to s3://{self.models_bucket}/{s3_key}: {str(e)}")
                                failed_count += 1
                            except Exception as e:
                                self.logger.warning(f"Unexpected error uploading {file_path} to s3://{self.models_bucket}/{s3_key}: {str(e)}")
                                failed_count += 1
                
                except Exception as e:
                    self.logger.warning(f"Error processing model directory {model_dir} for {currency_name}: {e}")
                    failed_count += 1
            
            if uploaded_count > 0:
                self.logger.info(f"Uploaded {uploaded_count} model files for {currency_name} to S3 (experiment: {experiment_id})")
            if failed_count > 0:
                self.logger.warning(f"Failed to upload {failed_count} files for {currency_name}")
                
        except ImportError:
            self.logger.warning("boto3 not available, cannot upload models to S3")
        except Exception as e:
            self.logger.warning(f"Failed to upload models for {currency_name} to S3: {e}")
    
    def _upload_logs_to_s3(self) -> None:
        """
        Upload training logs to S3 bucket.
        
        This method uploads all log files generated during training to the S3 data lake bucket
        for persistent storage and analysis.
        """
        if not self.config.logging.upload_logs_to_s3:
            self.logger.info("S3 log upload disabled in configuration")
            return
            
        if not self.data_lake_bucket:
            self.logger.warning("DATA_LAKE_BUCKET not configured, skipping log upload to S3")
            return
        
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Create S3 client
            s3_client = boto3.client('s3')
            
            # Get all log files from the logs directory
            logs_dir = Path(self.config.paths.logs_dir)
            if not logs_dir.exists():
                self.logger.warning(f"Logs directory does not exist: {logs_dir}")
                return
            
            log_files = list(logs_dir.glob("*.log"))
            if not log_files:
                self.logger.warning("No log files found to upload")
                return
            
            # Upload each log file
            uploaded_count = 0
            for log_file in log_files:
                try:
                    # Create S3 key with experiment ID and timestamp
                    s3_key = f"{self.config.logging.s3_logs_prefix}/{self.config.experiment.experiment_id}/{log_file.name}"
                    
                    # Upload file
                    s3_client.upload_file(str(log_file), self.data_lake_bucket, s3_key)
                    self.logger.debug(f"Uploaded log file to s3://{self.data_lake_bucket}/{s3_key}")
                    uploaded_count += 1
                    
                except ClientError as e:
                    self.logger.error(f"Failed to upload {log_file.name} to S3: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error uploading {log_file.name}: {e}")
            
            self.logger.info(f"Successfully uploaded {uploaded_count} log files to S3")
            
        except ImportError:
            self.logger.error("boto3 not available, cannot upload logs to S3")
        except Exception as e:
            self.logger.error(f"Failed to upload logs to S3: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Currency-Specific Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training with full optimization
  python train_models.py
  
  # Train ALL currencies with sufficient data
  python train_models.py --train-all-currencies
  
  # Train ALL currencies with custom thresholds
  python train_models.py --train-all-currencies --min-avg-value 2.0 --min-records 75
  
  # Custom data file
  python train_models.py --data-path /path/to/data.parquet
  
  # Custom configuration
  python train_models.py --config /path/to/config.json
        """
    )
    
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to training data file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        help='Custom experiment ID'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Experiment description'
    )
    
    parser.add_argument(
        '--tags',
        nargs='*',
        default=[],
        help='Experiment tags'
    )
    
    parser.add_argument(
        '--train-all-currencies',
        action='store_true',
        help='Train models for all currencies with sufficient data (not just high-value ones)'
    )
    
    parser.add_argument(
        '--min-avg-value',
        type=float,
        help='Minimum average value (in Chaos Orbs) for currency inclusion when using --train-all-currencies'
    )
    
    parser.add_argument(
        '--min-records',
        type=int,
        help='Minimum number of historical records required when using --train-all-currencies'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run currency-specific training."""
    args = parse_arguments()
    
    try:
        # Get configuration
        if args.config:
            config = MLConfig.from_file(args.config)
        else:
            config = get_default_config()
        
        # Override configuration with command line arguments
        if args.experiment_id:
            config.experiment.experiment_id = args.experiment_id
        if args.description:
            config.experiment.description = args.description
        if args.tags:
            config.experiment.tags.extend(args.tags)
        if args.train_all_currencies:
            config.data.train_all_currencies = True
        if args.min_avg_value:
            config.data.min_avg_value_threshold = args.min_avg_value
        if args.min_records:
            config.data.min_records_threshold = args.min_records
        
        # Add mode tag
        config.experiment.tags.append(args.mode)
        
        # Initialize trainer
        trainer = ModelTrainingPipeline(config)
        
        print(f"Starting currency training in {args.mode} mode...")
        print(f"Experiment ID: {config.experiment.experiment_id}")
        
        # Run training
        results = trainer.train_all_currencies(args.data_path)
        
        if results:
            print(f"\nSuccessfully trained {len(results)} models!")
        else:
            print(f"\nNo models were trained successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
