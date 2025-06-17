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
from config.training_config import MLConfig, DataConfig, ProcessingConfig, get_production_config
from utils.logging_utils import setup_ml_logging, MLLogger, ProgressLogger
from utils.data_processing import DataProcessor, DataValidator
from utils.feature_engineering import FeatureEngineer
from utils.database import get_db_connection
from utils.identify_target_currencies import generate_target_currency_list


class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline.
    
    This class orchestrates the entire feature engineering process with
    comprehensive validation, parallel processing, and experiment tracking.
    """
    
    def __init__(self, config: MLConfig):
        """
        Initialize the feature engineer.
        
        Args:
            config: Complete ML configuration
        """
        self.config = config
        
        # Setup logging
        self.logger = setup_ml_logging(
            name="FeatureEngineer",
            level=config.logging.level,
            log_dir=str(config.paths.logs_dir),
            experiment_id=config.experiment.experiment_id,
            console_output=config.logging.console_logging,
            suppress_external=config.logging.suppress_lightgbm
        )
        
        # Initialize components
        self.data_processor = DataProcessor(
            config.data, config.processing, self.logger
        )
        
        # Processing statistics
        self.processing_stats = {
            'total_currency_pairs': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'insufficient_data': 0,
            'validation_failures': 0,
            'total_records_processed': 0,
            'total_features_created': 0
        }
        
        self.processed_datasets = []
        self.failed_currencies = []
    
    def run_feature_engineering_experiment(self) -> Dict[str, Any]:
        """
        Run complete feature engineering experiment.
        
        Returns:
            Dictionary containing experiment results and metadata
        """
        # Log experiment start
        self.logger.log_experiment_start(
            self.config.experiment.experiment_id,
            self.config.to_dict()
        )
        
        try:
            # Load raw data from database
            raw_data = self._load_raw_data()
            if raw_data is None or raw_data.empty:
                raise ValueError("Failed to load raw data from database")
            
            self.logger.info(f"Loaded raw data: {raw_data.shape}")
            
            # Get target currency pairs
            target_currency_data = generate_target_currency_list()
            target_currencies = [f"{pair['get_currency']} -> {pair['pay_currency']}" for pair in target_currency_data]
            self.processing_stats['total_currency_pairs'] = len(target_currencies)
            
            self.logger.info(f"Processing {len(target_currencies)} currency pairs")
            
            # Process currency pairs
            if hasattr(self.config.processing, 'use_parallel_processing') and self.config.processing.use_parallel_processing:
                self._process_currencies_parallel(raw_data, target_currencies)
            else:
                self._process_currencies_sequential(raw_data, target_currencies)
            
            # Create combined dataset
            combined_dataset = None
            if self.config.experiment.create_combined_dataset and self.processed_datasets:
                combined_dataset = self._create_combined_dataset()
            
            # Generate experiment report
            experiment_results = self._generate_experiment_report(combined_dataset)
            
            # Log experiment end
            self.logger.log_experiment_end(
                self.config.experiment.experiment_id,
                experiment_results
            )
            
            return experiment_results
            
        except Exception as e:
            self.logger.error("Feature engineering experiment failed", exception=e)
            raise
    
    def _load_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Load raw currency price data from database.
        
        Returns:
            Raw dataframe or None if failed
        """
        with self.logger.log_operation("Loading raw data from database"):
            try:
                conn = get_db_connection()
                
                # Build league filtering conditions
                league_conditions = []
                params = []
                
                # Add included leagues filter
                if self.config.data.included_leagues:
                    placeholders = ','.join(['%s'] * len(self.config.data.included_leagues))
                    league_conditions.append(f"l.name IN ({placeholders})")
                    params.extend(self.config.data.included_leagues)
                
                # Add excluded leagues filter
                if self.config.data.excluded_leagues:
                    placeholders = ','.join(['%s'] * len(self.config.data.excluded_leagues))
                    league_conditions.append(f"l.name NOT IN ({placeholders})")
                    params.extend(self.config.data.excluded_leagues)
                
                # Ensure Settlers is included if the flag is set
                if self.config.data.include_settlers_league and 'Settlers' not in self.config.data.included_leagues:
                    if not self.config.data.included_leagues:
                        # If no included leagues specified, just ensure Settlers is not excluded
                        if 'Settlers' not in self.config.data.excluded_leagues:
                            self.config.data.excluded_leagues.append('Settlers')
                            self.config.data.excluded_leagues.remove('Settlers')  # Remove to not exclude it
                    else:
                        # Add Settlers to included leagues
                        league_conditions[0] = f"l.name IN ({','.join(['%s'] * (len(self.config.data.included_leagues) + 1))})"
                        params = self.config.data.included_leagues + ['Settlers'] + params[len(self.config.data.included_leagues):]
                
                # Build the WHERE clause
                where_clause = "WHERE cp.value > 0"
                if league_conditions:
                    where_clause += " AND " + " AND ".join(league_conditions)
                
                # Query to get currency price data with league information
                query = f"""
                SELECT 
                    cp.id,
                    cp."leagueId",
                    cp."getCurrencyId", 
                    cp."payCurrencyId",
                    cp.date AT TIME ZONE 'UTC' as date,
                    cp.value as price,
                    l.name as league_name,
                    l."startDate" AT TIME ZONE 'UTC' as league_start,
                    l."endDate" AT TIME ZONE 'UTC' as league_end,
                    l."isActive" as league_active,
                    gc.name as get_currency,
                    pc.name as pay_currency,
                    -- Create currency_pair column
                    CONCAT(gc.name, ' -> ', pc.name) as currency_pair,
                    -- Calculate days into league
                    EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) as league_day
                FROM currency_prices cp
                JOIN leagues l ON cp."leagueId" = l.id
                JOIN currency gc ON cp."getCurrencyId" = gc.id
                JOIN currency pc ON cp."payCurrencyId" = pc.id
                {where_clause}
                    -- Only include data from the first X days of each league
                    AND EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) <= %s
                    AND EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) >= 0
                    -- Only include leagues that lasted at least min_league_days
                    AND (l."endDate" IS NULL OR 
                        EXTRACT(DAY FROM (l."endDate" AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) >= %s)
                ORDER BY l."startDate" DESC, cp.date ASC
                """
                
                # Add the day filtering parameters
                params.extend([
                    self.config.data.max_league_days,
                    self.config.data.min_league_days
                ])
                
                # Pass the required parameters
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()
                
                if df.empty:
                    self.logger.warning("No data returned from database query")
                    return None
                
                # Log league distribution
                league_counts = df['league_name'].value_counts()
                self.logger.info(f"League data distribution:")
                for league, count in league_counts.items():
                    self.logger.info(f"  {league}: {count:,} records")
                
                # Verify Settlers data is included
                settlers_count = df[df['league_name'].str.contains('Settlers', case=False, na=False)]['league_name'].value_counts().sum()
                if self.config.data.include_settlers_league and settlers_count == 0:
                    self.logger.warning("Settlers league data requested but not found in loaded data")
                elif settlers_count > 0:
                    self.logger.info(f"Settlers league data included: {settlers_count:,} records")
                
                # Basic preprocessing
                df['date'] = pd.to_datetime(df['date'])
                df['league_start'] = pd.to_datetime(df['league_start'])
                
                # League day filtering is already done in SQL query
                # Just ensure league_day column exists for other processing
                if 'league_day' not in df.columns:
                    df['league_day'] = (df['date'] - df['league_start']).dt.days
                
                self.logger.info(f"Loaded {len(df)} records for {df['currency_pair'].nunique()} currency pairs")
                
                return df
                
            except Exception as e:
                self.logger.error("Failed to load raw data", exception=e)
                return None
    
    def _process_currencies_sequential(self, raw_data: pd.DataFrame, target_currencies: List[str]) -> None:
        """
        Process currency pairs sequentially.
        
        Args:
            raw_data: Raw dataframe
            target_currencies: List of currency pairs to process
        """
        progress = ProgressLogger(
            self.logger, len(target_currencies), "Sequential Currency Processing"
        )
        
        for currency_pair in target_currencies:
            try:
                result = self._process_currency_pair(raw_data, currency_pair)
                if result is not None:
                    self.processed_datasets.append(result)
                    self.processing_stats['successful_processing'] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process {currency_pair}", exception=e)
                self.failed_currencies.append({
                    'currency_pair': currency_pair,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                self.processing_stats['failed_processing'] += 1
            
            finally:
                progress.update()
        
        progress.complete()
    
    def _process_currencies_parallel(self, raw_data: pd.DataFrame, target_currencies: List[str]) -> None:
        """
        Process currency pairs in parallel.
        
        Args:
            raw_data: Raw dataframe
            target_currencies: List of currency pairs to process
        """
        # Determine number of workers
        max_workers = min(mp.cpu_count(), len(target_currencies), 8)  # Cap at 8 workers
        
        self.logger.info(f"Processing currencies with {max_workers} parallel workers")
        
        progress = ProgressLogger(
            self.logger, len(target_currencies), "Parallel Currency Processing"
        )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_currency = {
                executor.submit(self._process_currency_pair_worker, raw_data, currency_pair): currency_pair
                for currency_pair in target_currencies
            }
            
            # Collect results
            for future in as_completed(future_to_currency):
                currency_pair = future_to_currency[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        self.processed_datasets.append(result)
                        self.processing_stats['successful_processing'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {currency_pair}", exception=e)
                    self.failed_currencies.append({
                        'currency_pair': currency_pair,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.processing_stats['failed_processing'] += 1
                
                finally:
                    progress.update()
        
        progress.complete()
    
    def _process_currency_pair(self, df: pd.DataFrame, currency_pair: str) -> Optional[pd.DataFrame]:
        """
        Process a single currency pair.
        
        Args:
            df: Complete dataframe
            currency_pair: Currency pair to process
            
        Returns:
            Processed dataframe for this currency pair or None if failed
        """
        with self.logger.log_operation(f"Processing {currency_pair}"):
            # Filter data for this currency pair
            currency_data = df[df['currency_pair'] == currency_pair].copy()
            
            # Log initial data shape
            self.logger.info(f"Initial data shape: {currency_data.shape}")
            
            # Process features
            try:
                processed_data, processing_metadata = self.data_processor.process_currency_data(
                    currency_data,
                    currency_pair
                )
                
                if processed_data is None:
                    self.logger.warning(f"Feature processing failed for {currency_pair}")
                    return None
                
                # Log final data shape
                self.logger.info(f"Final data shape: {processed_data.shape}")
                
                return processed_data
                
            except Exception as e:
                self.logger.error(
                    f"Error processing {currency_pair}",
                    exception=e
                )
                return None
    
    @staticmethod
    def _process_currency_pair_worker(raw_data: pd.DataFrame, currency_pair: str) -> Optional[pd.DataFrame]:
        """
        Worker function for parallel processing.
        
        This is a static method to avoid pickling issues with multiprocessing.
        """
        # Create minimal configuration for worker
        config = MLConfig()
        
        # Create processor
        processor = DataProcessor(config.data, config.processing)
        
        # Filter and process data
        currency_data = raw_data[raw_data['currency_pair'] == currency_pair].copy()
        
        processed_data, _ = processor.process_currency_data(currency_data, currency_pair)
        return processed_data
    
    def _save_individual_dataset(self, data: pd.DataFrame, currency_pair: str) -> None:
        """
        Save individual currency dataset.
        
        Args:
            data: Processed dataframe
            currency_pair: Currency pair identifier
        """
        try:
            # Create currency-specific directory
            currency_dir = self.config.paths.data_dir / "individual" / currency_pair
            currency_dir.mkdir(parents=True, exist_ok=True)
            
            # Save dataset
            filename = f"{currency_pair}_{self.config.experiment.experiment_id}.parquet"
            filepath = currency_dir / filename
            
            data.to_parquet(filepath, index=False)
            
            self.logger.debug(f"Saved individual dataset: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save individual dataset for {currency_pair}: {str(e)}")
    
    def _create_combined_dataset(self) -> Optional[pd.DataFrame]:
        """
        Create combined dataset from all processed currency pairs.
        
        Returns:
            Combined dataframe or None if failed
        """
        with self.logger.log_operation("Creating combined dataset"):
            try:
                if not self.processed_datasets:
                    self.logger.warning("No processed datasets to combine")
                    return None
                
                # Combine all datasets
                combined_df = pd.concat(self.processed_datasets, ignore_index=True)
                
                # Save combined dataset
                filename = f"combined_currency_features_{self.config.experiment.experiment_id}.parquet"
                filepath = self.config.paths.data_dir / filename
                
                combined_df.to_parquet(filepath, index=False)
                
                self.logger.info(
                    f"Created combined dataset: {filepath}",
                    extra={
                        'shape': combined_df.shape,
                        'currency_pairs': combined_df['currency_pair'].nunique(),
                        'leagues': combined_df['league_name'].nunique() if 'league_name' in combined_df.columns else 0
                    }
                )
                
                # Log league distribution
                if 'league_name' in combined_df.columns:
                    league_dist = combined_df['league_name'].value_counts()
                    self.logger.info("League distribution in combined dataset:")
                    for league, count in league_dist.items():
                        self.logger.info(f"  {league}: {count:,} records")
                
                return combined_df
                
            except Exception as e:
                self.logger.error("Failed to create combined dataset", exception=e)
                return None
    
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
            self.processing_stats['total_currency_pairs']
            if self.processing_stats['total_currency_pairs'] > 0 else 0
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
        print(f"Currency Pairs Processed: {self.processing_stats['successful_processing']}/{self.processing_stats['total_currency_pairs']}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Records Processed: {self.processing_stats['total_records_processed']:,}")
        print(f"Average Features per Dataset: {self.processing_stats['total_features_created'] // max(1, self.processing_stats['successful_processing'])}")
        
        if combined_dataset is not None:
            print(f"Combined Dataset Shape: {combined_dataset.shape}")
        
        print(f"Output Directory: {self.config.paths.data_dir}")
        print(f"Metadata File: {metadata_file}")
        print(f"{'='*80}\n")
        
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard feature engineering
  python feature_engineering.py
  
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
    
    return parser.parse_args()


def main():
    """Main function to run feature engineering."""
    args = parse_arguments()
    
    try:
        # Get configuration
        if args.config:
            config = MLConfig.from_file(args.config)
        else:
            config = get_production_config()
        
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
        
        # Add feature engineering tag
        config.experiment.tags.append('feature_engineering')
        
        # Initialize feature engineer
        engineer = FeatureEngineeringPipeline(config)
        
        print(f"Starting feature engineering...")
        print(f"Experiment ID: {config.experiment.experiment_id}")
        print(f"Parallel Processing: {getattr(config.processing, 'use_parallel_processing', False)}")
        
        # Run feature engineering
        results = engineer.run_feature_engineering_experiment()
        
        if results['processing_stats']['successful_processing'] > 0:
            print(f"\nSuccessfully processed {results['processing_stats']['successful_processing']} currency pairs!")
        else:
            print(f"\nNo currency pairs were processed successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nFeature engineering interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFeature engineering failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 