#!/usr/bin/env python3
"""
Feature engineering script.

This script is the entry point for running feature engineering experiments.
It uses the FeatureEngineeringPipeline to process currency price data.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from config.training_config import MLConfig, get_production_config, get_development_config, get_test_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production feature engineering
  python run_feature_engineering.py --mode production
  
  # Development feature engineering with faster settings
  python run_feature_engineering.py --mode development
  
  # Quick testing with minimal settings
  python run_feature_engineering.py --mode test
  
  # Train ALL currencies with sufficient data (regardless of value)
  python run_feature_engineering.py --train-all-currencies
  
  # Train ALL currencies with custom thresholds
  python run_feature_engineering.py --train-all-currencies --min-avg-value 0.1 --min-records 50
  
  # Custom data file
  python run_feature_engineering.py --data-path /path/to/data.parquet
  
  # Custom configuration
  python run_feature_engineering.py --config /path/to/config.json
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
    
    # Feature engineering specific arguments
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
        if args.parallel:
            # Add parallel processing flag to config
            config.processing.use_parallel_processing = True
        if args.save_individual:
            config.experiment.save_individual_datasets = True
        if args.max_league_days:
            config.data.max_league_days = args.max_league_days
        
        # Add mode and feature engineering tags
        config.experiment.tags.append(args.mode)
        config.experiment.tags.append('feature_engineering')
        
        # Initialize feature engineer
        engineer = FeatureEngineeringPipeline(config)
        
        # Log configuration
        print(f"Starting feature engineering in {args.mode} mode...")
        print(f"Experiment ID: {config.experiment.experiment_id}")
        print(f"Train all currencies: {config.data.train_all_currencies}")
        print(f"Min avg value threshold: {config.data.min_avg_value_threshold}")
        print(f"Min records threshold: {config.data.min_records_threshold}")
        print(f"Parallel processing: {getattr(config.processing, 'use_parallel_processing', False)}")
        
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