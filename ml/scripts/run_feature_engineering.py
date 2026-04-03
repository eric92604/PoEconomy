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

from ml.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from ml.utils.common_utils import create_base_parser, add_currency_arguments, load_config_from_args, apply_args_to_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production feature engineering (full scale)
  python run_feature_engineering.py --mode production

  # Development feature engineering (reduced scale for faster iteration)
  python run_feature_engineering.py --mode development

  # Quick smoke test with minimal settings
  python run_feature_engineering.py --mode test

  # Lower value threshold to include more currencies
  python run_feature_engineering.py --min-avg-value 0.1

  # Save per-currency datasets alongside the combined file
  python run_feature_engineering.py --save-individual

  # Custom data file
  python run_feature_engineering.py --data-path /path/to/data.parquet

  # Custom configuration
  python run_feature_engineering.py --config /path/to/config.json
        """
    )
    
    # Use shared base parser for common arguments
    base_parser = create_base_parser("Feature Engineering Pipeline")
    for action in base_parser._actions:
        if action.dest not in ['help']:
            parser._add_action(action)

    # Add currency-specific arguments
    currency_parser = add_currency_arguments(argparse.ArgumentParser())
    for action in currency_parser._actions:
        if action.dest not in ['help']:
            parser._add_action(action)

    # Feature engineering specific arguments
    parser.add_argument(
        '--save-individual',
        action='store_true',
        help='Save individual currency datasets alongside the combined dataset'
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
        # Load and configure
        config = load_config_from_args(args)
        config = apply_args_to_config(config, args)
        
        # Add feature engineering specific tags
        config.experiment.tags.append('feature_engineering')
        
        # Initialize feature engineer
        engineer = FeatureEngineeringPipeline(config)
        
        # Log configuration
        print(f"Starting feature engineering in {args.mode} mode...")
        print(f"Experiment ID: {config.experiment.experiment_id}")
        print(f"Min avg value threshold: {config.data.min_avg_value_threshold}")
        print(f"Min records threshold: {config.data.min_records_threshold}")
        
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