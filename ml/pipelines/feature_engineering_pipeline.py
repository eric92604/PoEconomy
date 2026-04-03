"""
Feature engineering pipeline module.

This module contains the pipeline orchestration for feature engineering,
separating the orchestration logic from the core feature engineering functionality.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import utilities
from ml.config.training_config import MLConfig, DataConfig, ProcessingConfig, get_default_config, get_all_currencies_config, get_high_value_config
from ml.utils.common_utils import setup_ml_logging, MLLogger, ProgressLogger
from ml.utils.data_processing import DataProcessor
from ml.utils.feature_engineering import FeatureEngineer
from ml.utils.data_processing import generate_target_currency_list, generate_all_currencies_list
from ml.utils.data_sources import create_data_source, DataSourceConfig


class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline.
    
    This class orchestrates the entire feature engineering process with
    comprehensive validation, parallel processing, and experiment tracking.
    """
    
    def __init__(self, config: MLConfig, data_source_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Complete ML configuration
            data_source_config: Optional configuration for data source (S3 or DynamoDB)
        """
        self.config = config
        
        # Setup logging
        self.logger = setup_ml_logging(
            name="FeatureEngineer",
            level=config.logging.level,
            log_dir=str(config.paths.logs_dir),
            experiment_id=config.experiment.experiment_id,
            console_output=config.logging.console_logging,
            suppress_external=config.logging.suppress_external,
            file_logging=config.logging.file_logging
        )
        
        # Initialize components
        self.data_processor = DataProcessor(
            config.data, config.processing, self.logger
        )
        
        # Initialize data source based on configuration
        if data_source_config and data_source_config.get('data_source_type') == 's3':
            data_source_config_obj = DataSourceConfig.from_s3_config(data_source_config)
            self.logger.info("Using S3 data source for feature engineering")
        else:
            data_source_config_obj = DataSourceConfig.from_dynamo_config(config.dynamo)
            self.logger.info("Using DynamoDB data source for feature engineering")
        
        self.data_source = create_data_source(data_source_config_obj, self.logger)
        
        # Processing statistics
        self.processing_stats = {
            'total_currencies': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'insufficient_data': 0,
            'validation_failures': 0,
            'total_records_processed': 0,
            'total_features_created': 0
        }
        
        self.processed_datasets: List[Any] = []
        self.failed_currencies: List[Dict[str, str]] = []
    
    def run_feature_engineering_experiment(self) -> Dict[str, Any]:
        """
        Run complete feature engineering experiment.
        
        Returns:
            Dictionary containing experiment results and metadata
        """
        # Log experiment start
        self.logger.log_experiment_start(
            self.config.experiment.experiment_id or "unknown",
            self.config.to_dict()
        )
        
        try:
            # Get target currencies - always use all currencies with sufficient data
            target_currency_data = generate_all_currencies_list(
                data_source=self.data_source,
                config=self.config,
                min_avg_value=0.0,
                min_records=self.config.data.min_records_threshold,
                filter_by_availability=False,
                only_available_currencies=False,
                availability_check_days=0,
            )
            self.logger.info(
                f"Feature engineering for ALL currencies mode: {len(target_currency_data)} currencies selected"
            )

            # Apply explicit limit when requested (useful for smoke tests / quick iterations)
            if (
                self.config.data.max_currencies_to_train is not None
                and len(target_currency_data) > self.config.data.max_currencies_to_train
            ):
                original_count = len(target_currency_data)
                target_currency_data = target_currency_data[:self.config.data.max_currencies_to_train]
                self.logger.info(
                    "Limiting currencies for feature engineering",
                    extra={
                        "requested_limit": self.config.data.max_currencies_to_train,
                        "original_total": original_count,
                        "truncated_total": len(target_currency_data)
                    }
                )
                
            target_currencies = [pair['get_currency'] for pair in target_currency_data]
            self.processing_stats['total_currencies'] = len(target_currencies)

            # Load raw data for selected currencies
            raw_data = self._load_raw_data(target_currency_data)
            if raw_data is None or raw_data.empty:
                raise ValueError("Failed to load raw data from DynamoDB for selected currencies")
            
            self.logger.info(f"Loaded raw data: {raw_data.shape}")
            
            self.logger.info(f"Processing {len(target_currencies)} currencies")
            
            # Process currencies
            if hasattr(self.config.processing, 'use_parallel_processing') and self.config.processing.use_parallel_processing:
                self._process_currencies_parallel(raw_data, target_currencies)
            else:
                self._process_currencies_sequential(raw_data, target_currencies)
            
            # Create combined dataset
            combined_dataset = None
            combined_dataset_path = None
            if self.config.experiment.create_combined_dataset and self.processed_datasets:
                combined_dataset, combined_dataset_path = self._create_combined_dataset()
            
            # Generate experiment report
            experiment_results = self._generate_experiment_report(combined_dataset)
            
            # Add the file path to results for S3 upload
            if combined_dataset_path:
                experiment_results['combined_dataset_path'] = combined_dataset_path
            
            # Log experiment end
            self.logger.log_experiment_end(
                self.config.experiment.experiment_id or "unknown",
                experiment_results
            )
            
            return experiment_results
            
        except Exception as e:
            self.logger.error("Feature engineering experiment failed", exception=e)
            raise
    
    def _load_raw_data(self, target_currency_data: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        Load raw currency price data from DynamoDB for the specified currencies.
        """
        with self.logger.log_operation("Loading raw data from DynamoDB"):
            try:
                included_leagues = list(self.config.data.included_leagues) if self.config.data.included_leagues else None
                # Note: include_settlers_league attribute removed from DataConfig
                if included_leagues and self.config.data.excluded_leagues:
                    included_leagues = [league for league in included_leagues if league not in self.config.data.excluded_leagues]

                df = self.data_source.build_price_dataframe(
                    currencies=[pair['get_currency'] for pair in target_currency_data],
                    included_leagues=included_leagues,
                    max_league_days=self.config.data.max_league_days,
                    min_league_days=self.config.data.min_league_days,
                )

                if df is None or df.empty:
                    self.logger.warning("No data returned from DynamoDB query")
                    return None

                # Apply excluded leagues filter if needed
                if self.config.data.excluded_leagues:
                    df = df[~df['league_name'].isin(self.config.data.excluded_leagues)].reset_index(drop=True)

                # Log league distribution
                league_counts = df["league_name"].value_counts()
                self.logger.info("League data distribution:")
                for league, count in league_counts.items():
                    self.logger.info(f"  {league}: {count:,} records")

                # Basic preprocessing (ensure datetime types)
                df["date"] = pd.to_datetime(df["date"], utc=True)
                
                # Handle missing league_start/league_end columns (for S3DataSource)
                if "league_start" not in df.columns:
                    # For S3 data, use a default league start date (first date in the data)
                    df["league_start"] = df["date"].min()
                else:
                    df["league_start"] = pd.to_datetime(df["league_start"], utc=True)
                    
                if "league_end" not in df.columns:
                    # For S3 data, use a default league end date (last date in the data)
                    df["league_end"] = df["date"].max()
                else:
                    df["league_end"] = pd.to_datetime(df["league_end"], utc=True)

                if "league_day" not in df.columns:
                    df["league_day"] = (df["date"] - df["league_start"]).dt.days

                self.logger.info(f"Loaded {len(df)} records for {df['currency'].nunique()} currencies")
                return df

            except Exception as e:
                self.logger.error("Failed to load raw data from DynamoDB", exception=e)
                return None
    
    def _process_currencies_sequential(self, raw_data: pd.DataFrame, target_currencies: List[str]) -> None:
        """
        Process currencies sequentially.
        
        Args:
            raw_data: Raw dataframe
            target_currencies: List of currencies to process
        """
        progress = ProgressLogger(
            self.logger, len(target_currencies), "Sequential Currency Processing"
        )
        
        for currency in target_currencies:
            try:
                result = self._process_currency(raw_data, currency)
                if result is not None:
                    self.processed_datasets.append(result)
                    self.processing_stats['successful_processing'] += 1
                    self.processing_stats['total_records_processed'] += len(result)
                    self.processing_stats['total_features_created'] += len(result.columns)
                
            except Exception as e:
                self.logger.error(f"Failed to process {currency}", exception=e)
                self.failed_currencies.append({
                    'currency': currency,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                self.processing_stats['failed_processing'] += 1
            
            finally:
                progress.update()
        
        progress.complete()
    
    def _process_currencies_parallel(self, raw_data: pd.DataFrame, target_currencies: List[str]) -> None:
        """
        Process currencies in parallel.
        
        Args:
            raw_data: Raw dataframe
            target_currencies: List of currencies to process
        """
        # Determine number of workers
        max_workers = min(len(target_currencies), self.config.model.max_currency_workers)
        
        self.logger.info(f"Processing currencies with {max_workers} parallel workers")
        
        progress = ProgressLogger(
            self.logger, len(target_currencies), "Parallel Currency Processing"
        )

        # Serialise config once so every worker gets the same settings as the
        # orchestrator (custom rolling_windows, max_league_days, etc.).
        config_dict = self.config.to_dict()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_currency = {
                executor.submit(
                    self._process_currency_worker, raw_data, currency, config_dict
                ): currency
                for currency in target_currencies
            }
            
            # Collect results
            for future in as_completed(future_to_currency):
                currency = future_to_currency[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        self.processed_datasets.append(result)
                        self.processing_stats['successful_processing'] += 1
                        self.processing_stats['total_records_processed'] += len(result)
                        self.processing_stats['total_features_created'] += len(result.columns)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {currency}", exception=e)
                    self.failed_currencies.append({
                        'currency': currency,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.processing_stats['failed_processing'] += 1
                
                finally:
                    progress.update()
        
        progress.complete()
    
    def _process_currency(self, df: pd.DataFrame, currency: str) -> Optional[pd.DataFrame]:
        """
        Process a single currency.
        
        Args:
            df: Complete dataframe
            currency: Currency name to process
            
        Returns:
            Processed dataframe for this currency or None if failed
        """
        with self.logger.log_operation(f"Processing {currency}"):
            # Filter data for this currency using currency column
            currency_data = df[df['currency'] == currency].copy()
            
            # Log initial data shape
            self.logger.info(f"Initial data shape: {currency_data.shape}")
            
            # Process features
            try:
                processed_data, processing_metadata = self.data_processor.process_currency_data(
                    currency_data,
                    currency
                )
                
                if processed_data is None:
                    self.logger.warning(f"Feature processing failed for {currency}")
                    return None
                
                # Log final data shape
                self.logger.info(f"Final data shape: {processed_data.shape}")
                
                return processed_data
                
            except Exception as e:
                self.logger.error(
                    f"Error processing {currency}",
                    exception=e
                )
                return None
    
    @staticmethod
    def _process_currency_worker(
        raw_data: pd.DataFrame,
        currency: str,
        config_dict: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        """
        Worker function for parallel currency processing.

        Accepts a serialised config dict so that every worker uses exactly
        the same settings as the orchestrating process (e.g. custom
        rolling_windows, max_league_days, outlier thresholds).  Using a
        fresh ``MLConfig()`` default in each worker silently ignores any
        programmatic overrides applied before the parallel run.

        This is a static method to satisfy ``ProcessPoolExecutor`` pickling
        requirements.

        Args:
            raw_data: Complete multi-currency dataframe.
            currency: Currency name to filter and process.
            config_dict: Serialised ``MLConfig.to_dict()`` from the parent process.
        """
        config = MLConfig.from_dict(config_dict)
        processor = DataProcessor(config.data, config.processing)
        currency_data = raw_data[raw_data['currency'] == currency].copy()
        processed_data, _ = processor.process_currency_data(currency_data, currency)
        return processed_data
    
    def _save_individual_dataset(self, data: pd.DataFrame, currency: str) -> None:
        """
        Save individual currency dataset.
        
        Args:
            data: Processed dataframe
            currency: Currency identifier
        """
        try:
            # Create currency-specific directory
            currency_dir = self.config.paths.data_dir / "individual" / currency
            currency_dir.mkdir(parents=True, exist_ok=True)
            
            # Save dataset
            filename = f"{currency}_{self.config.experiment.experiment_id}.parquet"
            filepath = currency_dir / filename
            
            data.to_parquet(filepath, index=False)
            
            self.logger.debug(f"Saved individual dataset: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save individual dataset for {currency}: {str(e)}")
    
    def _create_combined_dataset(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Create combined dataset from all processed currencies.
        
        Returns:
            Tuple of (combined dataframe, file path) or (None, None) if failed
        """
        with self.logger.log_operation("Creating combined dataset"):
            try:
                if not self.processed_datasets:
                    self.logger.warning("No processed datasets to combine")
                    return None, None
                
                # Combine all datasets
                combined_df = pd.concat(self.processed_datasets, ignore_index=True)
                
                # Save combined dataset locally
                filename = f"combined_currency_features_{self.config.experiment.experiment_id}.parquet"
                filepath = self.config.paths.data_dir / filename
                
                combined_df.to_parquet(filepath, index=False)
                
                # Upload to S3 datalake bucket
                self._upload_combined_dataset_to_s3(combined_df, filename)
                
                self.logger.info(
                    f"Created combined dataset: {filepath}",
                    extra={
                        'shape': combined_df.shape,
                        'currencies': combined_df['currency'].nunique(),
                        'leagues': combined_df['league_name'].nunique() if 'league_name' in combined_df.columns else 0
                    }
                )
                
                # Log league distribution
                if 'league_name' in combined_df.columns:
                    league_dist = combined_df['league_name'].value_counts()
                    self.logger.info("League distribution in combined dataset:")
                    for league, count in league_dist.items():
                        self.logger.info(f"  {league}: {count:,} records")
                
                return combined_df, str(filepath)
                
            except Exception as e:
                self.logger.error("Failed to create combined dataset", exception=e)
                return None, None
    
    def _upload_combined_dataset_to_s3(self, combined_df: pd.DataFrame, filename: str) -> None:
        """
        Upload combined dataset to S3 datalake bucket.
        
        Args:
            combined_df: Combined dataframe to upload
            filename: Filename for the parquet file
        """
        try:
            import boto3
            import tempfile
            import os
            
            # Get S3 bucket from environment or config
            s3_bucket = os.getenv('DATA_LAKE_BUCKET')
            
            if not s3_bucket:
                self.logger.warning("DATA_LAKE_BUCKET environment variable not set, skipping S3 upload")
                return
            
            # Create S3 client
            s3_client = boto3.client('s3', region_name=self.config.dynamo.region_name)
            
            # Upload to S3 with processed_data prefix
            s3_key = f"processed_data/{filename}"
            
            # Create temporary file for upload
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Save to temporary file
                combined_df.to_parquet(temp_path, index=False)
                
                # Upload to S3
                s3_client.upload_file(temp_path, s3_bucket, s3_key)
                
                self.logger.info(f"Successfully uploaded combined dataset to s3://{s3_bucket}/{s3_key}")
            finally:
                # Always clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as cleanup_error:
                        self.logger.warning(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")
            
        except Exception as e:
            self.logger.warning(f"Failed to upload combined dataset to S3: {e}")
    
    def _generate_experiment_report(self, combined_dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate comprehensive experiment report.
        
        Args:
            combined_dataset: Combined dataset if created
            
        Returns:
            Experiment results dictionary
        """
        # Calculate success rate
        success_rate = (
            self.processing_stats['successful_processing'] / 
            self.processing_stats['total_currencies']
            if self.processing_stats['total_currencies'] > 0 else 0
        )
        
        # Compile results
        results = {
            'experiment_id': self.config.experiment.experiment_id,
            'status': 'completed',
            'processing_stats': self.processing_stats,
            'success_rate': success_rate,
            'failed_currencies': self.failed_currencies,
            'combined_dataset_shape': combined_dataset.shape if combined_dataset is not None else None,
            'output_directory': str(self.config.paths.data_dir),
            'experiment_timestamp': datetime.now().isoformat()
        }
        
        # Save experiment metadata
        metadata_file = self.config.paths.data_dir / f"experiment_metadata_{self.config.experiment.experiment_id}.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        results['metadata_file'] = str(metadata_file)
        
        # Log summary
        self.logger.info("Feature Engineering Experiment Completed", extra=results)
        
        # Print summary to console
        print(f"\n{'='*80}")
        print(f"FEATURE ENGINEERING SUMMARY - {self.config.experiment.experiment_id}")
        print(f"{'='*80}")
        print(f"Currencies Processed: {self.processing_stats['successful_processing']}/{self.processing_stats['total_currencies']}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Records Processed: {self.processing_stats['total_records_processed']:,}")
        print(f"Average Features per Dataset: {self.processing_stats['total_features_created'] // max(1, self.processing_stats['successful_processing'])}")
        
        if combined_dataset is not None:
            print(f"Combined Dataset Shape: {combined_dataset.shape}")
        
        print(f"Output Directory: {self.config.paths.data_dir}")
        print(f"Metadata File: {metadata_file}")
        print(f"{'='*80}\n")
        
        return results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production feature engineering
  python feature_engineering.py --mode production
  
  # Development feature engineering with faster settings
  python feature_engineering.py --mode development
  
  # Testing with minimal settings
  python feature_engineering.py --mode test
  
  # Train ALL currencies with sufficient data
  python feature_engineering.py --train-all-currencies
  
  # Train ALL currencies with custom thresholds
  python feature_engineering.py --train-all-currencies --min-avg-value 2.0 --min-records 75
  
  # With parallel processing
  python feature_engineering.py --parallel
  
  # Save individual datasets
  python feature_engineering.py --save-individual
  
  # Custom experiment ID
  python feature_engineering.py --experiment-id my_experiment
  
  # Custom configuration
  python feature_engineering.py --config /path/to/config.json
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['production', 'development', 'test'],
        default='production',
        help='Training mode (default: production)'
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
        '--parallel',
        action='store_true',
        help='Use parallel processing'
    )
    
    parser.add_argument(
        '--save-individual',
        action='store_true',
        help='Save individual currency datasets'
    )
    
    parser.add_argument(
        '--max-league-days',
        type=int,
        help='Maximum days into each league to consider'
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
    """Main function to run feature engineering."""
    args = parse_arguments()
    
    try:
        # Get configuration based on mode
        if args.config:
            config = MLConfig.from_file(args.config)
        elif args.mode == 'production':
            config = get_default_config()
        elif args.mode == 'development':
            config = get_default_config()
        else:  # test
            config = get_default_config()
        
        # Override configuration with command line arguments
        if args.experiment_id:
            config.experiment.experiment_id = args.experiment_id
        if args.description:
            config.experiment.description = args.description
        if args.tags:
            config.experiment.tags.extend(args.tags)
        if args.parallel:
            # Add parallel processing flag to config
            config.processing.use_parallel_processing = True
        if args.save_individual:
            config.experiment.save_individual_datasets = True
        if args.max_league_days:
            config.data.max_league_days = args.max_league_days
        if args.train_all_currencies:
            config.data.train_all_currencies = True
        if args.min_avg_value:
            config.data.min_avg_value_threshold = args.min_avg_value
        if args.min_records:
            config.data.min_records_threshold = args.min_records
        
        # Add mode and feature engineering tags
        config.experiment.tags.append(args.mode)
        config.experiment.tags.append('feature_engineering')
        
        # Initialize feature engineer
        engineer = FeatureEngineeringPipeline(config)
        
        print(f"Starting feature engineering in {args.mode} mode...")
        print(f"Experiment ID: {config.experiment.experiment_id}")
        print(f"Parallel Processing: {getattr(config.processing, 'use_parallel_processing', False)}")
        
        # Run feature engineering
        results = engineer.run_feature_engineering_experiment()
        
        if results['processing_stats']['successful_processing'] > 0:
            print(f"\nSuccessfully processed {results['processing_stats']['successful_processing']} currencies!")
        else:
            print(f"\nNo currencies were processed successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nFeature engineering interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFeature engineering failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
