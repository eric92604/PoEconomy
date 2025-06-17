#!/usr/bin/env python3
"""
Currency Price Prediction Script

This script provides a command-line interface for making price predictions
using trained models in the current league.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_inference import ModelPredictor
from config.training_config import MLConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Currency Price Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all available currency pairs
  python predict_prices.py
  
  # Predict specific currency pairs
  python predict_prices.py --currencies "Divine Orb -> Chaos Orb" "Mirror Shard -> Chaos Orb"
  
  # Get top 20 predictions by price change
  python predict_prices.py --top 20 --sort-by price_change_percent
  
  # Predict with custom model directory
  python predict_prices.py --models-dir models/currency_custom
  
  # Export results to file
  python predict_prices.py --export predictions.json
  
  # Predict for 3 days ahead
  python predict_prices.py --horizon 3
        """
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        help='Directory containing trained models (default: auto-detect latest)'
    )
    
    parser.add_argument(
        '--currencies',
        nargs='*',
        help='Specific currency pairs to predict (default: all available)'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        default=1,
        help='Prediction horizon in days (default: 1)'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top predictions to show (default: 10)'
    )
    
    parser.add_argument(
        '--sort-by',
        choices=['price_change_percent', 'confidence_score', 'predicted_price', 'current_price'],
        default='price_change_percent',
        help='Field to sort predictions by (default: price_change_percent)'
    )
    
    parser.add_argument(
        '--ascending',
        action='store_true',
        help='Sort in ascending order (default: descending)'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export predictions to JSON file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Days of historical data to use (default: 30)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def find_latest_model_directory() -> Path:
    """Find the latest trained model directory."""
    models_base = Path("models")
    
    if not models_base.exists():
        raise FileNotFoundError("Models directory not found")
    
    # Look for currency model directories
    currency_dirs = [d for d in models_base.iterdir() 
                    if d.is_dir() and d.name.startswith('currency_')]
    
    if not currency_dirs:
        raise FileNotFoundError("No currency model directories found")
    
    # Get the most recent one
    latest_dir = max(currency_dirs, key=lambda d: d.stat().st_mtime)
    return latest_dir


def print_model_summary(available_models):
    """Print summary of available models."""
    print(f"\n{'='*80}")
    print(f"AVAILABLE TRAINED MODELS ({len(available_models)} total)")
    print(f"{'='*80}")
    
    if not available_models:
        print("No models found!")
        return
    
    print(f"{'Currency Pair':<35} {'Model Type':<15} {'R²':<6} {'MAE':<8} {'Trained'}")
    print("-" * 80)
    
    for currency_pair, info in available_models.items():
        metrics = info.get('training_metrics', {})
        r2 = metrics.get('r2', 0)
        mae = metrics.get('mae', 0)
        model_type = info.get('model_type', 'unknown')
        timestamp = info.get('training_timestamp', 'unknown')
        
        # Parse timestamp
        if timestamp != 'unknown':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%m/%d %H:%M')
            except:
                time_str = timestamp[:10]
        else:
            time_str = 'unknown'
        
        print(f"{currency_pair:<35} {model_type:<15} {r2:<6.3f} {mae:<8.2f} {time_str}")


def print_predictions(predictions, title="PRICE PREDICTIONS"):
    """Print formatted predictions."""
    print(f"\n{'='*100}")
    print(f"{title} ({len(predictions)} predictions)")
    print(f"{'='*100}")
    
    if not predictions:
        print("No predictions available!")
        return
    
    print(f"{'#':<3} {'Currency Pair':<35} {'Current':<8} {'Predicted':<8} {'Change':<8} {'Conf':<6} {'Model'}")
    print("-" * 100)
    
    for i, pred in enumerate(predictions, 1):
        change_str = f"{pred.price_change_percent:+.1f}%"
        conf_str = f"{pred.confidence_score:.2f}"
        model_type = pred.model_type[:12] + "..." if len(pred.model_type) > 15 else pred.model_type
        
        print(f"{i:<3} {pred.currency:<35} {pred.current_price:<8.2f} "
              f"{pred.predicted_price:<8.2f} {change_str:<8} {conf_str:<6} {model_type}")


def main():
    """Main function for price prediction script."""
    args = parse_arguments()
    
    try:
        # Get configuration
        if args.config:
            config = MLConfig.from_file(args.config)
        else:
            config = MLConfig()
        
        # Find models directory
        if args.models_dir:
            models_dir = Path(args.models_dir)
        else:
            models_dir = find_latest_model_directory()
        
        if not models_dir.exists():
            print(f"Error: Models directory not found: {models_dir}")
            return 1
        
        print(f"Using models from: {models_dir}")
        
        # Initialize predictor
        predictor = ModelPredictor(models_dir, config)
        
        # Load models
        print("Loading trained models...")
        available_models = predictor.load_available_models()
        
        if args.verbose:
            print_model_summary(available_models)
        
        if not available_models:
            print("Error: No trained models found!")
            return 1
        
        print(f"Loaded {len(available_models)} trained models")
        
        # Make predictions
        print(f"Generating predictions for {args.horizon}-day horizon...")
        
        if args.currencies:
            # Predict specific currencies
            # Handle currency pair format variations
            currency_pairs = []
            for currency in args.currencies:
                if currency in available_models:
                    currency_pairs.append(currency)
                else:
                    # Try to find close matches
                    matches = [pair for pair in available_models.keys() 
                             if currency.lower() in pair.lower()]
                    if matches:
                        currency_pairs.extend(matches)
                        print(f"Note: Using {matches} for '{currency}'")
                    else:
                        print(f"Warning: No model found for '{currency}'")
            
            predictions = predictor.predict_multiple_currencies(
                currency_pairs, args.horizon
            )
        else:
            # Predict all available
            if args.top > 0:
                predictions = predictor.get_top_predictions(
                    top_n=args.top,
                    sort_by=args.sort_by,
                    ascending=args.ascending
                )
            else:
                predictions = predictor.predict_multiple_currencies(
                    prediction_horizon_days=args.horizon
                )
        
        if not predictions:
            print("Error: No predictions could be generated!")
            return 1
        
        # Display results
        title = f"{args.horizon}-DAY PRICE PREDICTIONS"
        if args.currencies:
            title += f" (Selected Currencies)"
        elif args.top > 0:
            title += f" (Top {args.top} by {args.sort_by.replace('_', ' ').title()})"
        
        print_predictions(predictions, title)
        
        # Export if requested
        if args.export:
            predictor.export_predictions(predictions, args.export)
            print(f"\nResults exported to: {args.export}")
        
        # Summary statistics
        if predictions:
            avg_change = sum(p.price_change_percent for p in predictions) / len(predictions)
            avg_confidence = sum(p.confidence_score for p in predictions) / len(predictions)
            positive_predictions = sum(1 for p in predictions if p.price_change_percent > 0)
            
            print(f"\nSummary:")
            print(f"  Average predicted change: {avg_change:+.2f}%")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Positive predictions: {positive_predictions}/{len(predictions)} ({positive_predictions/len(predictions)*100:.1f}%)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 