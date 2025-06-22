"""
Model training pipeline module.

This module contains the pipeline orchestration for model training,
separating the orchestration logic from the core model training functionality.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
import argparse
from datetime import datetime
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.impute import SimpleImputer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import utilities
from config.training_config import MLConfig, get_production_config, get_development_config, get_test_config
from utils.logging_utils import setup_ml_logging, MLLogger, ProgressLogger
from utils.data_processing import DataProcessor, load_and_validate_data
from utils.model_training import ModelTrainer, save_model_artifacts
from utils.identify_target_currencies import generate_target_currency_list, generate_all_currencies_list
from utils.currency_standardizer import CurrencyStandardizer


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
        
        # Setup logging
        self.logger = setup_ml_logging(
            name="CurrencyTrainer",
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
        self.model_trainer = ModelTrainer(
            config.model, config.processing, self.logger
        )
        self.currency_standardizer = CurrencyStandardizer(logger=self.logger)
        
        # Modify models directory to include experiment_id
        experiment_models_dir = config.paths.models_dir.parent / f"currency_{config.experiment.experiment_id}"
        experiment_models_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.models_dir = experiment_models_dir
        
        self.logger.info(f"Models will be saved to: {experiment_models_dir}")
        
        # Results tracking
        self.results = []
        self.failed_currencies = []
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
        # Log experiment start
        self.logger.log_experiment_start(
            self.config.experiment.experiment_id,
            self.config.to_dict()
        )
        
        try:
            # Load and validate data
            if data_path is None:
                data_path = self._find_latest_training_data()
            
            df, load_metadata = load_and_validate_data(data_path, self.logger)
            if df is None:
                raise ValueError(f"Failed to load training data from {data_path}")
            
            self.logger.info(f"Loaded training data: {df.shape}", extra=load_metadata)
            
            # Get target currencies based on configuration
            if self.config.data.train_all_currencies:
                target_currencies = generate_all_currencies_list(
                    min_avg_value=self.config.data.min_avg_value_threshold,
                    min_records=self.config.data.min_records_threshold,
                    filter_by_availability=self.config.data.filter_by_availability,
                    only_available_currencies=self.config.data.only_train_available_currencies,
                    availability_check_days=self.config.data.availability_check_days
                )
                self.logger.info(f"Training ALL currencies mode: {len(target_currencies)} currencies selected")
            else:
                target_currencies = generate_target_currency_list(
                    filter_by_availability=self.config.data.filter_by_availability,
                    only_available_currencies=self.config.data.only_train_available_currencies,
                    availability_check_days=self.config.data.availability_check_days
                )
                self.logger.info(f"Training high-value currencies mode: {len(target_currencies)} currencies selected")
                
            self.processing_stats['total_currencies'] = len(target_currencies)
            
            # Log availability filtering status
            if self.config.data.filter_by_availability:
                available_count = sum(1 for pair in target_currencies if pair.get('is_available', True))
                self.logger.info(f"Availability filtering enabled: {available_count}/{len(target_currencies)} currencies available")
            else:
                self.logger.info("Availability filtering disabled - training all currencies")
            
            self.logger.info(f"Training models for {len(target_currencies)} currencies")
            
            # Initialize progress tracking
            progress = ProgressLogger(
                self.logger, len(target_currencies), "Currency Model Training"
            )
            
            # Process each currency
            for currency in target_currencies:
                try:
                    result = self._train_currency_model(df, currency)
                    if result:
                        self.results.append(result)
                        self.processing_stats['successful_training'] += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to train model for {currency}",
                        exception=e
                    )
                    self.failed_currencies.append({
                        'currency': currency,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.processing_stats['failed_training'] += 1
                
                finally:
                    progress.update()
            
            progress.complete()
            
            # Generate final report
            self._generate_training_report()
            
            # Log experiment end
            self.logger.log_experiment_end(
                self.config.experiment.experiment_id,
                {
                    'total_models_trained': len(self.results),
                    'success_rate': len(self.results) / len(target_currencies) if target_currencies else 0,
                    'processing_stats': self.processing_stats
                }
            )
            
            return self.results
            
        except Exception as e:
            self.logger.error("Training pipeline failed", exception=e)
            raise
    
    def _train_currency_model(self, df: pd.DataFrame, currency: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Train model for a single currency.
        
        Args:
            df: Complete training dataframe (already feature-engineered)
            currency: Currency dictionary with get_currency and pay_currency
            
        Returns:
            Training result dictionary or None if failed
        """
        currency_name = currency['get_currency']  # Payment currency is always Chaos Orb
        with self.logger.log_operation(f"Training model for {currency_name}"):
            # Get currency IDs from database
            currency_ids = self._get_currency_ids(currency['get_currency'], currency['pay_currency'])
            if not currency_ids:
                self.logger.warning(f"Could not find currency IDs for {currency_name}")
                return None
            
            get_currency_id, pay_currency_id = currency_ids
            
            # Filter data for this currency using IDs
            currency_data = df[
                (df['getCurrencyId'] == get_currency_id) & 
                (df['payCurrencyId'] == pay_currency_id)
            ].copy()
            
            # Log league distribution for this currency
            if 'league_name' in currency_data.columns:
                league_dist = currency_data['league_name'].value_counts()
                self.logger.info(f"League distribution for {currency_name}:")
                for league, count in league_dist.items():
                    self.logger.info(f"  {league}: {count:,} records")
                
                # Check for Settlers data
                settlers_data = currency_data[currency_data['league_name'].str.contains('Settlers', case=False, na=False)]
                if not settlers_data.empty:
                    self.logger.info(f"Settlers data available for training: {len(settlers_data):,} records")
                else:
                    self.logger.warning(f"No Settlers data found for {currency_name}")
            
            # Data is already processed by feature engineering - use directly
            processed_data = currency_data
            processing_metadata = {
                'original_samples': len(currency_data),
                'source': 'feature_engineering_pipeline',
                'leagues_included': list(currency_data['league_name'].unique()) if 'league_name' in currency_data.columns else []
            }
            
            # Prepare features and targets
            feature_columns = self._get_feature_columns(processed_data)
            target_column = self._get_target_column(processed_data)
            
            if not feature_columns or target_column not in processed_data.columns:
                self.logger.warning(f"Invalid features/targets for {currency_name}")
                self.processing_stats['validation_failures'] += 1
                return None
            
            X = processed_data[feature_columns].values
            y = processed_data[target_column].values

            self.logger.info(f"Initial data shape: {X.shape}")
            self.logger.info(f"Initial NaN count in features: {np.isnan(X).sum()}")

            target_valid_mask = ~pd.isna(y)
            X = X[target_valid_mask]
            y = y[target_valid_mask]
            
            # Also filter the processed_data to maintain league information
            processed_data = processed_data[target_valid_mask].reset_index(drop=True)

            if len(X) == 0:
                self.logger.warning(f"No valid target values for {currency_name}")
                return None

            self.logger.info(
                f"Target filtering results:",
                extra={
                    'original_samples': len(target_valid_mask),
                    'valid_samples': len(X),
                    'removed_samples': len(target_valid_mask) - len(X)
                }
            )

            # 2. Remove rows with too many NaN values (>70% of features are NaN)
            row_nan_ratio = np.isnan(X).sum(axis=1) / X.shape[1]
            valid_rows = row_nan_ratio <= 0.7
            X = X[valid_rows]
            y = y[valid_rows]
            
            # Also filter processed_data
            processed_data = processed_data[valid_rows].reset_index(drop=True)

            self.logger.info(
                f"Row NaN filtering results:",
                extra={
                    'samples_before': len(row_nan_ratio),
                    'samples_after': len(X),
                    'removed_samples': len(row_nan_ratio) - len(X),
                    'max_nan_ratio': float(row_nan_ratio.max()) if len(row_nan_ratio) > 0 else 0,
                    'mean_nan_ratio': float(row_nan_ratio.mean()) if len(row_nan_ratio) > 0 else 0
                }
            )

            # 3. Impute remaining NaN values with median (column-wise)
            if np.isnan(X).any():
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
                self.logger.info(f"Applied median imputation for remaining NaN values")

            if len(X) < 50:
                self.logger.warning(f"Insufficient data for {currency_name}: {len(X)} samples")
                self.processing_stats['insufficient_data'] += 1
                return None

            # Train model
            training_result = self.model_trainer.train_single_model(
                X, y, currency_name
            )
            
            if training_result is None:
                self.logger.error(f"Model training failed for {currency_name}")
                return None

            # Enhanced evaluation with league-specific metrics
            evaluation_results = self._evaluate_model_comprehensive(
                processed_data, training_result, currency_name, feature_columns, target_column
            )
            
            # Save model artifacts
            model_info = save_model_artifacts(
                training_result,
                self.config.paths.models_dir,
                currency_name
            )

            # Compile final results
            result = {
                'currency': currency_name,
                'get_currency': currency['get_currency'],
                'pay_currency': currency['pay_currency'],
                'training_samples': len(X),
                'training_metrics': training_result.metrics,
                'evaluation_results': evaluation_results,
                'model_info': model_info,
                'processing_metadata': processing_metadata,
                'feature_count': len(feature_columns),
                'leagues_in_training': processing_metadata.get('leagues_included', [])
            }
            
            self.logger.info(f"Successfully trained model for {currency_name}")
            
            return result
    
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
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns from dataframe."""
        exclude_patterns = [
            'target_', 'date', 'league_name', 'currency', 'id', 'league_start', 'league_end',
            'league_active', 'get_currency', 'pay_currency', 'getCurrencyId', 'payCurrencyId'
        ]
        feature_cols = [col for col in df.columns 
                       if not any(pattern in col for pattern in exclude_patterns)]
        
        if not feature_cols:
            self.logger.warning("No feature columns found after filtering")
            return None
            
        return feature_cols
    
    def _get_target_column(self, df: pd.DataFrame) -> str:
        """Get target column from dataframe."""
        # Use the shortest prediction horizon by default
        target_cols = [col for col in df.columns if col.startswith('target_price_')]
        if not target_cols:
            return None
        return min(target_cols, key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    def _get_currency_ids(self, get_currency: str, pay_currency: str) -> Optional[tuple]:
        """Get currency IDs from database using standardizer."""
        try:
            # Standardize currency names to ensure they match database
            std_get_currency = self.currency_standardizer.standardize_currency_name(get_currency)
            std_pay_currency = self.currency_standardizer.standardize_currency_name(pay_currency)
            
            if not std_get_currency:
                self.logger.warning(f"Currency '{get_currency}' not found in database")
                return None
                
            if not std_pay_currency:
                self.logger.warning(f"Currency '{pay_currency}' not found in database")
                return None
            
            # Get IDs using the standardizer
            get_currency_id = self.currency_standardizer.get_currency_id(std_get_currency)
            pay_currency_id = self.currency_standardizer.get_currency_id(std_pay_currency)
            
            if get_currency_id is None or pay_currency_id is None:
                self.logger.warning(f"Failed to get IDs for {std_get_currency} -> {std_pay_currency}")
                return None
            
            return (get_currency_id, pay_currency_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get currency IDs for {get_currency} -> {pay_currency}", exception=e)
            return None
    
    def _evaluate_model_comprehensive(
        self, 
        df: pd.DataFrame, 
        training_result: Any, 
        currency: str,
        feature_columns: List[str],
        target_column: str
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation including league-specific metrics.
        
        Args:
            df: Complete dataframe with league information
            training_result: Trained model result
            currency: Currency name
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        evaluation_results = {
            'overall_metrics': None,
            'league_specific_metrics': {},
            'settlers_detailed_metrics': None,
            'data_quality_metrics': {}
        }
        
        try:
            # Overall evaluation
            X = df[feature_columns].values
            y = df[target_column].values
            
            # Remove NaN values
            valid_mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            df_valid = df[valid_mask].reset_index(drop=True)
            
            if len(X_valid) > 0:
                # Impute any remaining NaN values
                if np.isnan(X_valid).any():
                    imputer = SimpleImputer(strategy='median')
                    X_valid = imputer.fit_transform(X_valid)
                
                # Overall predictions
                y_pred = training_result.model.predict(X_valid)
                
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
                evaluation_results['data_quality_metrics'] = {
                    'total_samples': len(df),
                    'valid_samples': len(X_valid),
                    'missing_data_ratio': 1 - (len(X_valid) / len(df)) if len(df) > 0 else 0,
                    'feature_completeness': {
                        col: 1 - (df[col].isna().sum() / len(df)) 
                        for col in feature_columns[:5]  # Sample of first 5 features
                    },
                    'leagues_represented': list(df['league_name'].unique()) if 'league_name' in df.columns else []
                }
                
            self.logger.info(f"Comprehensive evaluation completed for {currency}")
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed for {currency}: {str(e)}")
            evaluation_results['error'] = str(e)
        
        return evaluation_results
    
    def _generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        report = {
            'experiment_id': self.config.experiment.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_currencies': self.processing_stats['total_currencies'],
            'successful_training': self.processing_stats['successful_training'],
            'failed_training': self.processing_stats['failed_training'],
            'processing_stats': self.processing_stats
        }
        
        # Save report
        report_path = self.config.paths.logs_dir / f"training_report_{self.config.experiment.experiment_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Training report saved to {report_path}")
        
        # Log summary
        self.logger.info("=== TRAINING SUMMARY ===")
        self.logger.info(f"Total currencies: {self.processing_stats['total_currencies']}")
        self.logger.info(f"Successful training: {self.processing_stats['successful_training']}")
        self.logger.info(f"Failed training: {self.processing_stats['failed_training']}")
        
        success_rate = (self.processing_stats['successful_training'] / 
                       max(1, self.processing_stats['total_currencies'])) * 100
        self.logger.info(f"Success rate: {success_rate:.1f}%")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Currency-Specific Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production training with full optimization
  python train_models.py --mode production
  
  # Development training with faster settings
  python train_models.py --mode development
  
  # Testing with minimal settings
  python train_models.py --mode test
  
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
        '--mode',
        choices=['production', 'development', 'test'],
        default='production',
        help='Training mode (default: production)'
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


def main():
    """Main function to run currency-specific training."""
    args = parse_arguments()
    
    try:
        # Get configuration based on mode
        if args.config:
            config = MLConfig.from_file(args.config)
        elif args.mode == 'production':
            config = get_production_config()
        elif args.mode == 'development':
            config = get_development_config()
        else:  # test
            config = get_test_config()
        
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