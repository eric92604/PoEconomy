#!/usr/bin/env python3
"""
Currency Model Training

Standard training script with comprehensive argument support.
Uses parallel processing by default for optimal performance.
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipelines.model_training_pipeline import ModelTrainingPipeline
from ml.utils.common_utils import create_base_parser, add_currency_arguments, load_config_from_args, apply_args_to_config



def parse_arguments():
    """Parse command line arguments."""
    parser = create_base_parser(
        description="Currency Model Training",
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
    
    # Add currency-specific arguments
    parser = add_currency_arguments(parser)
    
    # Training-specific arguments
    parser.add_argument(
        '--train-all-currencies',
        action='store_true',
        help='Train models for ALL currencies with sufficient data (not just high-value ones)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load and configure
        config = load_config_from_args(args)
        config = apply_args_to_config(config, args)
        
        # Training pipeline now uses S3 data source automatically
        
        # Setup trainer
        trainer = ModelTrainingPipeline(config)
        
        # Log setup
        trainer.logger.info("=== TRAINING CONFIGURATION ===")
        trainer.logger.info(f"Mode: {args.mode}")
        trainer.logger.info(f"Experiment ID: {config.experiment.experiment_id}")
        trainer.logger.info(f"Training mode: All currencies with sufficient data")
        trainer.logger.info(f"Min records threshold: {config.data.min_records_threshold}")
        trainer.logger.info(f"Parallel processing: ENABLED")
        trainer.logger.info("=" * 32)
        
        print(f"Starting currency training in {args.mode} mode...")
        print(f"Experiment ID: {config.experiment.experiment_id}")
        
        # Execute training
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