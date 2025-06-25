#!/usr/bin/env python3
"""
Currency Model Training

Standard training script with comprehensive argument support.
Uses parallel processing by default for optimal performance.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.model_training_pipeline import ModelTrainingPipeline
from config.training_config import MLConfig, get_production_config, get_development_config, get_test_config
from utils.model_training import configure_environment_for_parallel_training


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Currency Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production training with full optimization
  python train_models.py --mode production
  
  # Development training with faster settings
  python train_models.py --mode development
  
  # Quick testing with minimal settings
  python train_models.py --mode test
  
  # Train ALL currencies with sufficient data (regardless of value)
  python train_models.py --train-all-currencies
  
  # Train ALL currencies with custom thresholds
  python train_models.py --train-all-currencies --min-avg-value 0.1 --min-records 50
  
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
        help='Train models for ALL currencies with sufficient data (not just high-value ones)'
    )
    
    parser.add_argument(
        '--min-avg-value',
        type=float,
        help='Minimum average value (in Chaos Orbs) for currency inclusion'
    )
    
    parser.add_argument(
        '--min-records',
        type=int,
        help='Minimum number of historical records required'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for model training."""
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
            # When training all currencies, set min value to 0.1 by default
            # This ensures we get ALL currencies that have any meaningful data
            if not args.min_avg_value:
                config.data.min_avg_value_threshold = 0.1
        if args.min_avg_value is not None:
            config.data.min_avg_value_threshold = args.min_avg_value
        if args.min_records:
            config.data.min_records_threshold = args.min_records
        
        # Add mode tag
        config.experiment.tags.append(args.mode)
        
        # Configure environment for optimal parallel performance
        configure_environment_for_parallel_training(config.model)
        
        # Initialize trainer
        trainer = ModelTrainingPipeline(config)
        
        # Log configuration
        trainer.logger.info("=== TRAINING CONFIGURATION ===")
        trainer.logger.info(f"Mode: {args.mode}")
        trainer.logger.info(f"Experiment ID: {config.experiment.experiment_id}")
        trainer.logger.info(f"Train all currencies: {config.data.train_all_currencies}")
        trainer.logger.info(f"Min avg value threshold: {config.data.min_avg_value_threshold}")
        trainer.logger.info(f"Min records threshold: {config.data.min_records_threshold}")
        trainer.logger.info(f"Parallel processing: ENABLED")
        trainer.logger.info("=" * 32)
        
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