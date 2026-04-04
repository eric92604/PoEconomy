"""
Model training pipeline module.

This module contains the pipeline orchestration for model training,
separating the orchestration logic from the core model training functionality.
"""

import sys
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from ml.config.training_config import MLConfig
from ml.utils.common_utils import setup_ml_logging, ProgressLogger
from ml.utils.model_training import ModelTrainer, save_model_artifacts

from concurrent.futures import ProcessPoolExecutor, as_completed


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
        # Only enable file logging if configured (disable in Fargate to reduce storage writes)
        log_dir = str(config.paths.logs_dir) if config.logging.file_logging else None
        self.logger = setup_ml_logging(
            name="CurrencyTrainer",
            level=config.logging.level,
            log_dir=log_dir,
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
        
        # Initialize DynamoDB data source for validation data (required)
        self.dynamo_data_source = None
        try:
            from ml.utils.data_sources import create_data_source, DataSourceConfig
            dynamo_config_obj = DataSourceConfig.from_dynamo_config(config.dynamo)
            self.dynamo_data_source = create_data_source(dynamo_config_obj, self.logger)
            self.logger.info("Initialized DynamoDB data source for validation data")
        except Exception as e:
            error_msg = (
                f"Failed to initialize DynamoDB data source for validation: {e}. "
                "Validation data is required for training. Please check your DynamoDB configuration "
                "and ensure the tables are accessible."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Initialize data processor for processing validation data
        from ml.utils.data_processing import DataProcessor
        self.data_processor = DataProcessor(
            config.data, config.processing, self.logger
        )
        
        # Store S3 configuration for log uploads
        self.data_lake_bucket = data_lake_bucket
        
        # Get models bucket for model uploads
        self.models_bucket = os.getenv('MODELS_BUCKET', '')
        
        # Note: Models directory will be created after experiment_id is potentially updated from processed data
        
        # Initialize tracking
        self.results: List[Dict[str, Any]] = []
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
    
    def _on_currency_complete(self, currency_name: str, result: Optional[Dict[str, Any]]) -> None:
        """
        Handle the outcome of a completed currency training run.

        Centralises result collection, stat tracking, logging, and model upload so
        the logic is identical for both sequential and parallel training paths.
        """
        if result:
            self.results.append(result)
            self.processing_stats['successful_training'] += 1
            self._log_currency_completion_stats(currency_name, result)
            try:
                self._upload_currency_models(currency_name, result)
            except Exception as upload_error:
                self.logger.warning(f"Failed to upload models for {currency_name}: {upload_error}")
        else:
            self.processing_stats['failed_training'] += 1
            self.logger.warning(f"Training failed for {currency_name}")

    def _train_currencies_parallel(self, target_currencies: List[Dict[str, Any]],
                                   progress: ProgressLogger, max_workers: int) -> None:
        """
        Train currencies sequentially (max_workers=1) or in parallel (max_workers>1).

        Args:
            target_currencies: List of currency dictionaries
            progress: Progress logger instance
            max_workers: Maximum number of parallel workers
        """
        if max_workers == 1:
            for currency in target_currencies:
                currency_name = currency.get('get_currency', 'Unknown')
                try:
                    self.logger.info(f"Starting training for {currency_name}...")
                    result = self._train_currency_model(currency)
                    self._on_currency_complete(currency_name, result)
                except Exception as e:
                    self.logger.error(f"{currency_name} training crashed", exception=e)
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
            'processed_data_path': processed_data_path
        }

        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
            future_to_currency = {
                executor.submit(self._train_currency_worker, currency, shared_data): currency
                for currency in target_currencies
            }

            self.logger.info(f"Submitted {len(future_to_currency)} training tasks to process pool")

            completed_count = 0
            total_tasks = len(future_to_currency)

            for future in as_completed(future_to_currency):
                currency = future_to_currency[future]
                currency_name = currency.get('get_currency', 'Unknown')
                completed_count += 1

                try:
                    result = future.result()
                    self._on_currency_complete(currency_name, result)
                except Exception as e:
                    error_msg = str(e)
                    if "process pool was terminated" in error_msg:
                        self.logger.error(f"{currency_name}: Process pool terminated - likely due to resource exhaustion")
                    else:
                        self.logger.error(f"{currency_name}: Training crashed - {error_msg}")
                    self.failed_currencies.append({
                        'currency': currency,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.processing_stats['failed_training'] += 1
                finally:
                    progress.update()
                    if completed_count % 10 == 0 or completed_count == total_tasks:
                        self.logger.debug(f"Completed {completed_count}/{total_tasks} currencies")

            remaining_futures = [f for f in future_to_currency.keys() if not f.done()]
            if remaining_futures:
                self.logger.warning(f"{len(remaining_futures)} futures still pending after as_completed loop")
            else:
                self.logger.info(f"All {completed_count} training tasks completed successfully")

        except Exception as e:
            self.logger.error(f"Process pool executor failed: {str(e)}")
            if executor and 'future_to_currency' in locals():
                for future in future_to_currency:
                    if not future.done():
                        future.cancel()
        finally:
            if executor:
                try:
                    self.logger.debug("Shutting down ProcessPoolExecutor...")
                    executor.shutdown(wait=False)
                    self.logger.debug("ProcessPoolExecutor shutdown completed")
                except Exception as shutdown_error:
                    self.logger.warning(f"Error during executor shutdown: {shutdown_error}")

            if processed_data_path and os.path.exists(processed_data_path):
                try:
                    os.remove(processed_data_path)
                    temp_dir = os.path.dirname(processed_data_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file {processed_data_path}: {e}")
    
    @staticmethod
    def _train_currency_worker(currency: Dict[str, Any], shared_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Worker function for parallel currency training with robust error handling.
        
        Args:
            currency: Currency dictionary
            shared_data: Shared data containing config and processed data path
            
        Returns:
            Training result dictionary or None if failed
        """
        try:
            currency_name = currency.get('get_currency', 'Unknown')
            os.setpgrp()  # Create new process group to prevent signal propagation
            
            # Reconstruct configuration from dict
            config = MLConfig.from_dict(shared_data['config_dict'])
            
            # Configuration values are set via environment variables (MODEL_N_JOBS, MAX_OPTUNA_WORKERS)
            # No need to calculate optimal values - they're already configured
            
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
            sys.stderr.write(f"Memory error in worker for currency {currency_name}: {str(e)}\n")
            return None
        except Exception as e:
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
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"training_data_{self.config.experiment.experiment_id}.parquet")
        df.to_parquet(temp_path, index=False)
        
        self.logger.debug(f"Saved temporary dataframe to {temp_path}")
        return temp_path
    
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
    
    def _filter_currency_data(
        self, currency_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Slice processed data for a single currency and build processing metadata.

        Returns a dict with keys 'data' and 'metadata', or None if data is
        absent or insufficient.
        """
        processed_data = self._processed_data
        if processed_data is None:
            self.logger.error("No processed data available")
            self.processing_stats['insufficient_data'] += 1
            return None

        currency_data = processed_data[processed_data['currency'] == currency_name].copy()
        if currency_data.empty:
            self.logger.warning(f"No processed data found for {currency_name}")
            self.processing_stats['insufficient_data'] += 1
            return None

        self.logger.debug(f"Currency data for {currency_name}: {len(currency_data):,} records")

        if 'timestamp' in currency_data.columns:
            date_range = currency_data['timestamp'].agg(['min', 'max'])
            self.logger.debug(f"  Date range: {date_range['min']} to {date_range['max']}")

        if 'league_name' in currency_data.columns:
            league_dist = currency_data['league_name'].value_counts()
            self.logger.debug(f"League distribution for {currency_name}:")
            for league, count in league_dist.items():
                self.logger.debug(f"  {league}: {count:,} records")
            if self.config.data.included_leagues:
                for league in list(self.config.data.included_leagues):
                    subset = currency_data[
                        currency_data['league_name'].str.contains(league, case=False, na=False)
                    ]
                    if subset.empty:
                        self.logger.warning(f"No {league} data found for {currency_name}")
                    else:
                        self.logger.debug(f"{league} data available: {len(subset):,} records")

        metadata = {
            'currency': currency_name,
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'validation_result': 'passed',
            'feature_engineering_result': 'data_already_processed',
            'success': True,
            'leagues_included': list(currency_data['league_name'].unique())
            if 'league_name' in currency_data.columns else [],
        }
        return {'data': currency_data, 'metadata': metadata}

    def _build_feature_matrix(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        currency_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Build numpy feature matrix from *data*, cast to float, and drop rows
        whose NaN ratio exceeds the configured threshold.

        Returns a dict with keys 'X' (ndarray) and 'data' (filtered DataFrame),
        or None on failure.
        """
        X = data[feature_cols].values
        self.logger.debug(f"Initial feature matrix shape: {X.shape}")

        if X.dtype.kind not in 'biufc':
            try:
                X = X.astype(float)
                self.logger.debug(f"Converted features to float.")
            except (ValueError, TypeError) as e:
                self.logger.error(f"Cannot convert features to float for {currency_name}: {e}")
                return None

        max_nan_ratio = self.config.processing.max_nan_ratio
        row_nan_ratio = np.isnan(X).sum(axis=1) / X.shape[1]
        valid_rows = row_nan_ratio <= max_nan_ratio
        X = X[valid_rows]
        data = data[valid_rows].reset_index(drop=True)

        self.logger.debug(
            f"NaN row filter: {valid_rows.sum():,}/{len(valid_rows):,} rows kept "
            f"(threshold={max_nan_ratio})"
        )

        if len(X) < 5:
            self.logger.warning(
                f"Insufficient samples for {currency_name} after NaN filtering: {len(X)}"
            )
            self.processing_stats['insufficient_data'] += 1
            return None

        return {'X': X, 'data': data}

    def _fetch_validation_set(
        self,
        currency_name: str,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Fetch and process current-league validation data from DynamoDB.

        Returns a dict with keys 'features' (ndarray or None) and
        'targets' (DataFrame or None).
        """
        empty: Dict[str, Any] = {'features': None, 'targets': None}

        if not self.dynamo_data_source:
            self.logger.warning(
                f"DynamoDB not configured for {currency_name}; skipping validation data."
            )
            self.processing_stats['validation_failures'] += 1
            return empty

        try:
            current_league = None
            if (
                hasattr(self.dynamo_data_source, '_league_table')
                and self.dynamo_data_source._league_table
            ):
                from ml.utils.data_sources import get_current_seasonal_league_from_table
                current_league = get_current_seasonal_league_from_table(
                    self.dynamo_data_source._league_table, self.logger
                )
            else:
                self.logger.warning(
                    f"League metadata table not available for {currency_name}; "
                    "skipping validation data."
                )
                self.processing_stats['validation_failures'] += 1
                return empty

            if not current_league:
                self.logger.warning(
                    f"No active league found for {currency_name}; skipping validation data."
                )
                self.processing_stats['validation_failures'] += 1
                return empty

            self.logger.info(f"Fetching validation data for {currency_name} from league: {current_league}")
            validation_df = self._fetch_validation_data(currency_name, current_league)

            if validation_df is None or validation_df.empty:
                self.logger.warning(
                    f"No validation data returned for {currency_name} from {current_league}."
                )
                self.processing_stats['validation_failures'] += 1
                return empty

            # Align columns — pad any training features missing from validation with NaN.
            missing_cols = [c for c in feature_cols if c not in validation_df.columns]
            if missing_cols:
                self.logger.warning(
                    f"Validation data missing {len(missing_cols)} feature(s) for "
                    f"{currency_name}; padding with NaN: {missing_cols}"
                )
                for col in missing_cols:
                    validation_df[col] = np.nan

            val_X = validation_df[feature_cols].values
            max_nan_ratio = self.config.processing.max_nan_ratio
            valid_mask = np.isnan(val_X).sum(axis=1) / val_X.shape[1] <= max_nan_ratio
            val_X = val_X[valid_mask]
            validation_df = validation_df[valid_mask].reset_index(drop=True)

            self.logger.info(
                f"Validation set ready for {currency_name}: {len(val_X)} samples"
            )
            return {'features': val_X, 'targets': validation_df}

        except Exception as e:
            self.logger.warning(
                f"Error fetching validation data for {currency_name}: {e}. "
                "Continuing without validation data."
            )
            self.processing_stats['validation_failures'] += 1
            return empty

    def _extract_horizon_val(
        self,
        target_col: str,
        horizon: str,
        val_features: Optional[np.ndarray],
        val_targets: Optional[pd.DataFrame],
        currency_name: str,
    ) -> Dict[str, Any]:
        """
        Extract (X_val, y_val) for a single horizon from the validation set.

        Returns a dict with keys 'X_val' and 'y_val' (both None if unavailable).
        """
        if val_features is None or val_targets is None:
            self.logger.warning(
                f"Validation data not available for {currency_name} {horizon}; "
                "using train/test split."
            )
            return {'X_val': None, 'y_val': None}

        try:
            if target_col not in val_targets.columns:
                self.logger.warning(
                    f"Target column {target_col} missing in validation data for "
                    f"{currency_name} {horizon}; using train/test split."
                )
                return {'X_val': None, 'y_val': None}

            y_val_raw = val_targets[target_col].values
            valid_mask = ~pd.isna(y_val_raw)
            X_val = val_features[valid_mask]
            y_val = y_val_raw[valid_mask]

            if len(X_val) == 0:
                self.logger.warning(
                    f"No valid validation samples for {currency_name} {horizon}; "
                    "using train/test split."
                )
                return {'X_val': None, 'y_val': None}

            self.logger.debug(f"Validation samples for {currency_name} {horizon}: {len(X_val)}")
            return {'X_val': X_val, 'y_val': y_val}

        except Exception as e:
            self.logger.warning(
                f"Error extracting validation data for {currency_name} {horizon}: {e}; "
                "using train/test split."
            )
            return {'X_val': None, 'y_val': None}

    def _train_all_horizons(
        self,
        currency_name: str,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        val_features: Optional[np.ndarray],
        val_targets: Optional[pd.DataFrame],
    ) -> Optional[Dict[str, Any]]:
        """
        Train one model per prediction horizon and save each to its own directory.

        Returns a result dict on success, None if every horizon fails.
        """
        max_nan_ratio = self.config.processing.max_nan_ratio
        horizon_results = {}

        for target_col in target_cols:
            horizon = target_col.replace('target_price_', '')
            self.logger.debug(f"Training {horizon} model for {currency_name}")

            # Per-horizon NaN row filter on features
            X_h = data[feature_cols].values
            row_nan = np.isnan(X_h).sum(axis=1) / X_h.shape[1]
            keep = row_nan <= max_nan_ratio
            X_h = X_h[keep]
            data_h = data[keep].reset_index(drop=True)

            y_h = data_h[target_col].values
            total = len(y_h)
            nan_count = pd.isna(y_h).sum()
            self.logger.debug(
                f"  {currency_name} {horizon}: {total - nan_count:,}/{total:,} valid target rows"
            )

            target_mask = ~pd.isna(y_h)
            X_h = X_h[target_mask]
            y_h = y_h[target_mask]

            if len(X_h) < 10:
                self.logger.warning(
                    f"Skipping {currency_name} {horizon}: only {len(X_h)} samples "
                    "(minimum 10 required)"
                )
                continue

            val = self._extract_horizon_val(
                target_col, horizon, val_features, val_targets, currency_name
            )

            training_result = self.model_trainer.train_single_model(
                X_h, y_h, f"{currency_name}_{horizon}",
                target_names=[target_col],
                feature_names=feature_cols,
                X_val=val['X_val'],
                y_val=val['y_val'],
            )

            if training_result is not None:
                horizon_results[horizon] = training_result
                self.logger.debug(f"Trained {horizon} model for {currency_name}")
            else:
                self.logger.warning(f"Failed to train {horizon} model for {currency_name}")

        if not horizon_results:
            self.logger.error(f"All horizon models failed for {currency_name}")
            return None

        model_info = self._save_horizon_models(currency_name, horizon_results)
        if model_info is None:
            return None

        primary_horizon = '1d' if '1d' in horizon_results else next(iter(horizon_results))
        return {
            'training_samples': {
                h: int((~pd.isna(data[f'target_price_{h}'])).sum())
                for h in horizon_results
            },
            'training_metrics': {h: r.metrics for h, r in horizon_results.items()},
            'model_info': model_info,
            'feature_count': len(feature_cols),
            'horizons_trained': list(horizon_results.keys()),
            'primary_horizon': primary_horizon,
        }

    def _save_horizon_models(
        self, currency_name: str, horizon_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Persist each horizon's TrainingResult to its own subdirectory.

        Returns a model_info dict keyed by '{horizon}_model', or None on error.
        """
        model_info: Dict[str, Any] = {}
        for horizon, result in horizon_results.items():
            try:
                model_dir = self.config.paths.models_dir / f"{currency_name}_{horizon}"
                info = save_model_artifacts(result, model_dir, f"{currency_name}_{horizon}")
                model_info[f"{horizon}_model"] = info
            except Exception as e:
                self.logger.error(
                    f"Failed to save {horizon} model for {currency_name}: {e}"
                )
                return None
        return model_info

    def _train_currency_model(self, currency: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Orchestrate training for a single currency.

        Delegates data filtering, feature matrix construction, validation fetch,
        and per-horizon training to focused helper methods.
        """
        currency_name = currency['get_currency']
        with self.logger.log_operation(f"Training model for {currency_name}"):
            # 1. Filter and validate raw data for this currency.
            filtered = self._filter_currency_data(currency_name)
            if filtered is None:
                return None
            data, processing_metadata = filtered['data'], filtered['metadata']

            # 2. Identify feature and target columns.
            feature_columns = self._get_feature_columns(data)
            target_column = self._get_target_column(data)

            if not feature_columns:
                self.logger.warning(f"No feature columns found for {currency_name}")
                self.processing_stats['validation_failures'] += 1
                return None

            if target_column is None:
                self.logger.warning(f"No target columns found for {currency_name}")
                self.processing_stats['validation_failures'] += 1
                return None

            target_cols = target_column if isinstance(target_column, list) else [target_column]
            missing = [c for c in target_cols if c not in data.columns]
            if missing:
                self.logger.warning(f"Missing target columns for {currency_name}: {missing}")
                self.processing_stats['validation_failures'] += 1
                return None

            # 3. Build feature matrix, apply NaN row filter.
            matrix = self._build_feature_matrix(data, feature_columns, currency_name)
            if matrix is None:
                return None
            data = matrix['data']

            # 4. Fetch current-league validation set (optional).
            val = self._fetch_validation_set(currency_name, feature_columns)

            # 5. Train per-horizon models and save artifacts.
            horizon_result = self._train_all_horizons(
                currency_name, data, feature_columns, target_cols,
                val['features'], val['targets'],
            )
            if horizon_result is None:
                return None

            self.logger.debug(f"Training completed for {currency_name}")
            return {
                'currency': currency_name,
                'get_currency': currency['get_currency'],
                'processing_metadata': processing_metadata,
                'leagues_in_training': processing_metadata.get('leagues_included', []),
                **horizon_result,
            }
    
    def _log_currency_completion_stats(self, currency_name: str, result: Dict[str, Any]) -> None:
        """
        Log detailed completion statistics for a currency.
        
        Args:
            currency_name: Name of the currency
            result: Training result dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info(f"CURRENCY TRAINING COMPLETED: {currency_name}")
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
        
        self.logger.info(f"DATA STATISTICS:")
        self.logger.info(f"   • Training samples: {int(training_samples):,}")
        self.logger.info(f"   • Feature count: {int(feature_count)}")
        self.logger.info(f"   • Leagues included: {len(leagues)} ({', '.join(leagues[:3])}{'...' if len(leagues) > 3 else ''})")
        
        # Training metrics
        training_metrics = result.get('training_metrics', {})
        if training_metrics:
            self.logger.info(f"MODEL PERFORMANCE:")
            
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
            self.logger.info(f"MODEL ARTIFACTS:")
            
            # For multi-horizon training, report size/location of the primary horizon model.
            # For single-horizon training, model_info has the size/dir directly.
            primary_h = result.get('primary_horizon', '1d')
            horizon_key = f"{primary_h}_model"
            if horizon_key in model_info:
                primary_info = model_info[horizon_key]
                model_size = primary_info.get('model_size_mb', 0)
                model_dir = primary_info.get('model_dir', 'N/A')
            else:
                model_size = model_info.get('model_size_mb', 0)
                model_dir = model_info.get('model_dir', 'N/A')
            
            self.logger.info(f"   • Model size: {model_size:.2f} MB")
            self.logger.info(f"   • Saved to: {model_dir}")
        
        # Processing metadata
        processing_metadata = result.get('processing_metadata', {})
        if processing_metadata:
            timestamp = processing_metadata.get('processing_timestamp', 'N/A')
            self.logger.info(f"PROCESSING INFO:")
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
    
    def _fetch_validation_data(
        self,
        currency: str,
        current_league: str,
        pay_currency: str = "Chaos Orb"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch current league validation data from DynamoDB.
        
        Args:
            currency: Currency name
            current_league: Current active league name
            pay_currency: Pay currency (default: Chaos Orb)
            
        Returns:
            DataFrame with validation data, or None if not available
        """
        if not self.dynamo_data_source:
            self.logger.warning("DynamoDB data source not available for validation data")
            return None
        
        try:
            # Fetch extended range to ensure future targets are available
            # Features are backward-looking, so we only need future data for targets
            max_horizon = max(self.config.data.prediction_horizons) if self.config.data.prediction_horizons else 7
            buffer_days = 7  # Extra buffer for safety
            days_to_fetch = self.config.data.validation_max_days + max_horizon + buffer_days
            
            self.logger.debug(f"Fetching validation data: {days_to_fetch} days (validation={self.config.data.validation_max_days}, horizon={max_horizon}, buffer={buffer_days})")
            
            # Fetch raw daily prices from DynamoDB
            validation_df = self.dynamo_data_source.build_validation_dataframe(
                currencies=[currency],
                current_league=current_league,
                pay_currency=pay_currency,
                max_days=days_to_fetch
            )
            
            if validation_df is None or validation_df.empty:
                self.logger.warning(f"No validation data fetched for {currency} in league {current_league}")
                return None
            
            self.logger.info(f"Fetched {len(validation_df)} raw validation records for {currency}")
            
            # Process through SAME feature engineering pipeline as training data
            validation_processed, validation_metadata = self.data_processor.process_currency_data(
                validation_df, 
                currency
            )
            
            if validation_processed is None or validation_processed.empty:
                self.logger.warning(f"Feature engineering failed for validation data for {currency}")
                return None
            
            self.logger.debug(f"Processed validation data: {len(validation_processed)} records after feature engineering")

            if len(validation_processed) < 10:
                self.logger.warning(f"Insufficient validation data: {len(validation_processed)} samples (minimum: 10)")
                return None

            self.logger.info(f"Validation data ready: {len(validation_processed)} samples for {currency}")
            return validation_processed
            
        except Exception as e:
            self.logger.error(f"Error fetching validation data for {currency}: {e}", exception=e)
            return None
    
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
            
            # Find all model directories for this currency.
            # Models are saved as {currency_name}_{horizon}/ (one directory per horizon).
            currency_model_dirs = []
            if models_dir.exists():
                for item in models_dir.iterdir():
                    if item.is_dir() and item.name.startswith(f"{currency_name}_"):
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
                
                # Clean up model artifacts from disk after successful upload to save storage
                try:
                    import shutil
                    for model_dir in currency_model_dirs:
                        if model_dir.exists():
                            shutil.rmtree(model_dir, ignore_errors=True)
                            self.logger.debug(f"Cleaned up local model directory: {model_dir}")
                    self.logger.info(f"Cleaned up local model artifacts for {currency_name} after S3 upload")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up local model artifacts for {currency_name}: {cleanup_error}")
            
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