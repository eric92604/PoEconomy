#!/usr/bin/env python3
"""
Currency-Specific Model Training

This script implements an ML training pipeline with:
- Centralized configuration management
- Comprehensive logging and monitoring
- Modular data processing and validation
- Model training with hyperparameter optimization
- Robust error handling and recovery
- Experiment tracking and reproducibility
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
import argparse
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import utilities
from config.training_config import MLConfig, get_production_config, get_development_config, get_testing_config
from utils.logging_utils import setup_ml_logging, MLLogger, ProgressLogger
from utils.data_processing import DataProcessor, load_and_validate_data
from utils.model_training import ModelTrainer, save_model_artifacts
from utils.identify_target_currencies import generate_target_currency_list


class CurrencyTrainer:
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
            
            # Get target currencies
            target_currencies = generate_target_currency_list()
            self.processing_stats['total_currencies'] = len(target_currencies)
            
            self.logger.info(f"Training models for {len(target_currencies)} currency pairs")
            
            # Initialize progress tracking
            progress = ProgressLogger(
                self.logger, len(target_currencies), "Currency Model Training"
            )
            
            # Process each currency
            for currency_pair in target_currencies:
                try:
                    result = self._train_currency_model(df, currency_pair)
                    if result:
                        self.results.append(result)
                        self.processing_stats['successful_training'] += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to train model for {currency_pair}",
                        exception=e
                    )
                    self.failed_currencies.append({
                        'currency_pair': currency_pair,
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
    
    def _train_currency_model(self, df: pd.DataFrame, currency_pair: str) -> Optional[Dict[str, Any]]:
        """
        Train model for a single currency pair.
        
        Args:
            df: Complete training dataframe
            currency_pair: Currency pair to train
            
        Returns:
            Training result dictionary or None if failed
        """
        with self.logger.log_operation(f"Training model for {currency_pair}"):
            # Filter data for this currency pair
            currency_data = df[df['currency_pair'] == currency_pair].copy()
            
            if len(currency_data) < self.config.model.min_samples_required:
                self.logger.warning(
                    f"Insufficient data for {currency_pair}: {len(currency_data)} samples"
                )
                self.processing_stats['insufficient_data'] += 1
                return None
            
            # Process data
            processed_data, processing_metadata = self.data_processor.process_currency_data(
                currency_data, currency_pair
            )
            
            if processed_data is None:
                self.logger.warning(f"Data processing failed for {currency_pair}")
                self.processing_stats['validation_failures'] += 1
                return None
            
            # Prepare features and targets
            feature_columns = self._get_feature_columns(processed_data)
            target_column = self._get_target_column(processed_data)
            
            if not feature_columns or target_column not in processed_data.columns:
                self.logger.warning(f"Invalid features/targets for {currency_pair}")
                return None
            
            # Extract features and targets
            X = processed_data[feature_columns].values
            y = processed_data[target_column].values
            
            # Remove any remaining NaN values
            valid_mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < self.config.data.min_records_after_cleaning:
                self.logger.warning(
                    f"Insufficient clean data for {currency_pair}: {len(X)} samples"
                )
                return None
            
            # Train model
            training_result = self.model_trainer.train_single_model(
                X, y, currency_pair,
                model_type="ensemble",
                optimize_hyperparameters=True
            )
            
            # Save model artifacts
            if self.config.experiment.save_model_artifacts:
                saved_files = save_model_artifacts(
                    training_result,
                    self.config.paths.models_dir,
                    currency_pair
                )
                self.logger.info(f"Saved model artifacts for {currency_pair}", extra=saved_files)
            
            # Evaluate on Settlers league if available
            settlers_metrics = self._evaluate_on_settlers(
                processed_data, training_result, currency_pair
            )
            
            # Compile result
            result = {
                'currency_pair': currency_pair,
                'model_type': training_result.model_type,
                'data_points': len(X),
                'training_metrics': training_result.metrics.to_dict(),
                'training_time': training_result.training_time,
                'hyperparameters': training_result.hyperparameters,
                'feature_importance': training_result.feature_importance,
                'cross_validation_scores': training_result.cross_validation_scores,
                'processing_metadata': processing_metadata,
                'settlers_evaluation': settlers_metrics,
                'training_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Successfully trained model for {currency_pair}",
                extra={
                    'mae': training_result.metrics.mae,
                    'rmse': training_result.metrics.rmse,
                    'directional_accuracy': training_result.metrics.directional_accuracy
                }
            )
            
            return result
    
    def _evaluate_on_settlers(
        self,
        processed_data: pd.DataFrame,
        training_result,
        currency_pair: str
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate model on Settlers league data if available.
        
        Args:
            processed_data: Processed dataframe
            training_result: Training result object
            currency_pair: Currency pair identifier
            
        Returns:
            Settlers evaluation metrics or None
        """
        try:
            # Check if Settlers data exists
            if 'league_name' not in processed_data.columns:
                return None
            
            settlers_data = processed_data[processed_data['league_name'] == 'Settlers'].copy()
            if len(settlers_data) < 10:  # Need minimum samples
                return None
            
            # Prepare features and targets
            feature_columns = self._get_feature_columns(settlers_data)
            target_column = self._get_target_column(settlers_data)
            
            X_settlers = settlers_data[feature_columns].values
            y_settlers = settlers_data[target_column].values
            
            # Remove NaN values
            valid_mask = ~(pd.isna(X_settlers).any(axis=1) | pd.isna(y_settlers))
            X_settlers = X_settlers[valid_mask]
            y_settlers = y_settlers[valid_mask]
            
            if len(X_settlers) < 5:
                return None
            
            # Make predictions
            y_pred_settlers = training_result.model.predict(X_settlers)
            
            # Calculate metrics
            from utils.model_training import ModelMetrics
            settlers_metrics = ModelMetrics.from_predictions(y_settlers, y_pred_settlers)
            
            self.logger.info(
                f"Settlers evaluation for {currency_pair}",
                extra={
                    'samples': len(X_settlers),
                    'mae': settlers_metrics.mae,
                    'rmse': settlers_metrics.rmse
                }
            )
            
            return {
                'sample_count': len(X_settlers),
                'metrics': settlers_metrics.to_dict()
            }
            
        except Exception as e:
            self.logger.warning(f"Settlers evaluation failed for {currency_pair}: {str(e)}")
            return None
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names from dataframe."""
        # Exclude non-feature columns
        exclude_patterns = [
            'currency_pair', 'date', 'league_name', 'league_start',
            'target_', 'price'  # Exclude target columns and raw price
        ]
        
        feature_columns = [
            col for col in df.columns
            if not any(pattern in col for pattern in exclude_patterns)
            and df[col].dtype in ['int64', 'float64']  # Only numeric columns
        ]
        
        return feature_columns
    
    def _get_target_column(self, df: pd.DataFrame) -> str:
        """Get target column name."""
        # Use 1-day ahead price prediction as default target
        target_candidates = ['target_price_1d', 'target_change_1d', 'price']
        
        for candidate in target_candidates:
            if candidate in df.columns:
                return candidate
        
        raise ValueError("No suitable target column found")
    
    def _find_latest_training_data(self) -> str:
        """Find the latest training data file."""
        data_dir = self.config.paths.data_dir
        
        # Look for parquet files matching the pattern
        pattern = "combined_currency_features_*.parquet"
        data_files = list(data_dir.glob(pattern))
        
        if not data_files:
            raise FileNotFoundError(f"No training data files found in {data_dir}")
        
        # Return the most recent file
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)
    
    def _generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        if not self.results:
            self.logger.warning("No successful training results to report")
            return
        
        # Calculate summary statistics
        training_times = [r['training_time'] for r in self.results]
        maes = [r['training_metrics']['mae'] for r in self.results]
        rmses = [r['training_metrics']['rmse'] for r in self.results]
        directional_accuracies = [r['training_metrics']['directional_accuracy'] for r in self.results]
        
        # Settlers evaluation statistics
        settlers_results = [r for r in self.results if r.get('settlers_evaluation')]
        settlers_maes = [r['settlers_evaluation']['metrics']['mae'] for r in settlers_results]
        settlers_rmses = [r['settlers_evaluation']['metrics']['rmse'] for r in settlers_results]
        
        summary = {
            'experiment_id': self.config.experiment.experiment_id,
            'total_models': len(self.results),
            'success_rate': len(self.results) / self.processing_stats['total_currencies'],
            'average_training_time': sum(training_times) / len(training_times),
            'performance_metrics': {
                'average_mae': sum(maes) / len(maes),
                'median_mae': sorted(maes)[len(maes)//2],
                'average_rmse': sum(rmses) / len(rmses),
                'median_rmse': sorted(rmses)[len(rmses)//2],
                'average_directional_accuracy': sum(directional_accuracies) / len(directional_accuracies)
            },
            'settlers_evaluation': {
                'models_evaluated': len(settlers_results),
                'average_mae': sum(settlers_maes) / len(settlers_maes) if settlers_maes else None,
                'average_rmse': sum(settlers_rmses) / len(settlers_rmses) if settlers_rmses else None
            } if settlers_results else None,
            'processing_stats': self.processing_stats,
            'failed_currencies': self.failed_currencies
        }
        
        # Save report
        report_path = self.config.paths.logs_dir / f"training_report_{self.config.experiment.experiment_id}.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Log summary
        self.logger.info("Training Report Generated", extra=summary)
        
        # Print summary to console
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY - {self.config.experiment.experiment_id}")
        print(f"{'='*80}")
        print(f"Models Trained: {len(self.results)}/{self.processing_stats['total_currencies']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Training Time: {summary['average_training_time']:.1f}s")
        print(f"Average MAE: {summary['performance_metrics']['average_mae']:.2f}")
        print(f"Average RMSE: {summary['performance_metrics']['average_rmse']:.2f}")
        print(f"Average Directional Accuracy: {summary['performance_metrics']['average_directional_accuracy']:.1f}%")
        
        if summary['settlers_evaluation']:
            print(f"\nSettlers League Evaluation:")
            print(f"Models Evaluated: {summary['settlers_evaluation']['models_evaluated']}")
            if summary['settlers_evaluation']['average_mae']:
                print(f"Average MAE: {summary['settlers_evaluation']['average_mae']:.2f}")
                print(f"Average RMSE: {summary['settlers_evaluation']['average_rmse']:.2f}")
        
        print(f"\nReport saved: {report_path}")
        print(f"{'='*80}\n")


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
  python train_models.py --mode testing
  
  # Custom data file
  python train_models.py --data-path /path/to/data.parquet
  
  # Custom configuration
  python train_models.py --config /path/to/config.json
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['production', 'development', 'testing'],
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
        else:  # testing
            config = get_testing_config()
        
        # Override configuration with command line arguments
        if args.experiment_id:
            config.experiment.experiment_id = args.experiment_id
        if args.description:
            config.experiment.description = args.description
        if args.tags:
            config.experiment.tags.extend(args.tags)
        
        # Add mode tag
        config.experiment.tags.append(args.mode)
        
        # Initialize trainer
        trainer = CurrencyTrainer(config)
        
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