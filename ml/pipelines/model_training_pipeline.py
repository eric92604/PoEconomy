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
from utils.identify_target_currencies import generate_target_currency_list


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
    
    def _train_currency_model(self, df: pd.DataFrame, currency_pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Train model for a single currency pair.
        
        Args:
            df: Complete training dataframe (already feature-engineered)
            currency_pair: Currency pair dictionary with get_currency and pay_currency
            
        Returns:
            Training result dictionary or None if failed
        """
        currency_pair_name = f"{currency_pair['get_currency']} -> {currency_pair['pay_currency']}"
        with self.logger.log_operation(f"Training model for {currency_pair_name}"):
            # Get currency IDs from database
            currency_ids = self._get_currency_ids(currency_pair['get_currency'], currency_pair['pay_currency'])
            if not currency_ids:
                self.logger.warning(f"Could not find currency IDs for {currency_pair_name}")
                return None
            
            get_currency_id, pay_currency_id = currency_ids
            
            # Filter data for this currency pair using IDs
            currency_data = df[
                (df['getCurrencyId'] == get_currency_id) & 
                (df['payCurrencyId'] == pay_currency_id)
            ].copy()
            
            # Data is already processed by feature engineering - use directly
            processed_data = currency_data
            processing_metadata = {
                'original_samples': len(currency_data),
                'source': 'feature_engineering_pipeline'
            }
            
            # Prepare features and targets
            feature_columns = self._get_feature_columns(processed_data)
            target_column = self._get_target_column(processed_data)
            
            if not feature_columns or target_column not in processed_data.columns:
                self.logger.warning(f"Invalid features/targets for {currency_pair_name}")
                self.processing_stats['validation_failures'] += 1
                return None
            
            X = processed_data[feature_columns].values
            y = processed_data[target_column].values

            self.logger.info(f"Initial data shape: {X.shape}")
            self.logger.info(f"Initial NaN count in features: {np.isnan(X).sum()}")

            target_valid_mask = ~pd.isna(y)
            X = X[target_valid_mask]
            y = y[target_valid_mask]

            if len(X) == 0:
                self.logger.warning(f"No valid target values for {currency_pair_name}")
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

            self.logger.info(
                f"Row NaN filtering results:",
                extra={
                    'samples_before': len(row_nan_ratio),
                    'samples_after': len(X),
                    'removed_samples': len(row_nan_ratio) - len(X),
                    'max_nan_ratio': float(row_nan_ratio.max()),
                    'mean_nan_ratio': float(row_nan_ratio.mean())
                }
            )

            if len(X) < 10:  # Need at least 10 samples for training
                self.logger.warning(f"Insufficient samples after cleaning: {len(X)} for {currency_pair_name}")
                return None

            self.logger.info(f"Training on {len(X)} samples for {currency_pair_name}")

            # 3. Handle remaining NaN values in features using simple imputation
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)

            self.logger.info(
                f"Imputation results:",
                extra={
                    'final_shape': X.shape,
                    'final_nan_count': np.isnan(X).sum(),
                    'imputation_strategy': 'median'
                }
            )
            
            # 4. Final validation - should be no NaN values after imputation
            if np.any(pd.isna(X)) or np.any(pd.isna(y)):
                self.logger.warning(f"NaN values remain after imputation for {currency_pair_name}")
                return None
            
            # Train model
            training_result = self.model_trainer.train_single_model(
                X, y, currency_pair_name,
                model_type="ensemble",
                optimize_hyperparameters=True
            )
            
            # Save model artifacts
            if self.config.experiment.save_model_artifacts:
                saved_files = save_model_artifacts(
                    training_result,
                    self.config.paths.models_dir,
                    currency_pair_name
                )
                self.logger.info(f"Saved model artifacts for {currency_pair_name}", extra=saved_files)
            
            # Evaluate on Settlers league if available
            settlers_metrics = self._evaluate_on_settlers(
                processed_data, training_result, currency_pair_name
            )
            
            # Compile result
            result = {
                'currency_pair': currency_pair_name,
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
            'target_', 'date', 'league_name', 'currency_pair', 'id', 'league_start', 'league_end',
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
        """Get currency IDs from database."""
        try:
            from utils.database import get_db_connection
            import pandas as pd
            
            conn = get_db_connection()
            query = """
            SELECT id, name FROM currency 
            WHERE name IN (%s, %s)
            """
            
            df = pd.read_sql(query, conn, params=[get_currency, pay_currency])
            conn.close()
            
            if len(df) != 2:
                return None
            
            # Create mapping
            currency_map = dict(zip(df['name'], df['id']))
            
            get_currency_id = currency_map.get(get_currency)
            pay_currency_id = currency_map.get(pay_currency)
            
            if get_currency_id is None or pay_currency_id is None:
                return None
            
            return (get_currency_id, pay_currency_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get currency IDs for {get_currency} -> {pay_currency}", exception=e)
            return None
    
    def _evaluate_on_settlers(
        self,
        df: pd.DataFrame,
        training_result: Any,
        currency_pair: str
    ) -> Optional[Dict[str, Any]]:
        """Evaluate model on Settlers league data if available."""
        if 'league_name' not in df.columns:
            return None
        
        settlers_data = df[df['league_name'] == 'Settlers'].copy()
        
        try:
            # Prepare features and targets
            feature_columns = self._get_feature_columns(settlers_data)
            target_column = self._get_target_column(settlers_data)
            
            X = settlers_data[feature_columns].values
            y = settlers_data[target_column].values
            
            # Remove any NaN values
            valid_mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Make predictions
            y_pred = training_result.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred),
                'samples': len(y)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(
                f"Failed to evaluate on Settlers league for {currency_pair}: {str(e)}"
            )
            return None
    
    def _generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        # Calculate success rate
        success_rate = (
            self.processing_stats['successful_training'] / 
            self.processing_stats['total_currencies']
            if self.processing_stats['total_currencies'] > 0 else 0
        )
        
        # Print summary to console
        print(f"\n{'='*80}")
        print(f"MODEL TRAINING SUMMARY - {self.config.experiment.experiment_id}")
        print(f"{'='*80}")
        print(f"Models Trained: {self.processing_stats['successful_training']}/{self.processing_stats['total_currencies']}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Failed Training: {self.processing_stats['failed_training']}")
        print(f"Insufficient Data: {self.processing_stats['insufficient_data']}")
        print(f"Validation Failures: {self.processing_stats['validation_failures']}")
        print(f"Models Directory: {self.config.paths.models_dir}")
        print(f"{'='*80}\n")
        
        # Save detailed report
        report = {
            'experiment_id': self.config.experiment.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'processing_stats': self.processing_stats,
            'success_rate': success_rate,
            'failed_currencies': self.failed_currencies,
            'models_directory': str(self.config.paths.models_dir),
            'training_results': self.results
        }
        
        report_file = self.config.paths.models_dir / f"training_report_{self.config.experiment.experiment_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Training report saved to: {report_file}")


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