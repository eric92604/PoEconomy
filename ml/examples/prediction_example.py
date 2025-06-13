#!/usr/bin/env python3
"""
PoEconomy Price Prediction Example

This script demonstrates various ways to use the trained models for price prediction
in the current league, including command-line usage, API integration, and batch processing.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import asyncio
import aiohttp

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_inference import ModelPredictor
from config.training_config import MLConfig
from utils.logging_utils import MLLogger


def example_1_basic_prediction():
    """Example 1: Basic single currency prediction."""
    print("=" * 80)
    print("EXAMPLE 1: BASIC SINGLE CURRENCY PREDICTION")
    print("=" * 80)
    
    # Initialize predictor
    models_dir = Path("models/currency_production_lstm")  # Adjust path as needed
    predictor = ModelPredictor(models_dir)
    
    # Load models
    print("Loading models...")
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found! Please train models first.")
        return
    
    print(f"✓ Loaded {len(available_models)} models")
    
    # Make prediction for a specific currency
    currency_pair = list(available_models.keys())[0]  # Get first available
    print(f"\nPredicting price for: {currency_pair}")
    
    result = predictor.predict_price(currency_pair, prediction_horizon_days=1)
    
    if result:
        print(f"✓ Current Price: {result.current_price:.2f}")
        print(f"✓ Predicted Price: {result.predicted_price:.2f}")
        print(f"✓ Expected Change: {result.price_change_percent:+.1f}%")
        print(f"✓ Confidence: {result.confidence_score:.2f}")
        print(f"✓ Model Type: {result.model_type}")
    else:
        print("✗ Prediction failed")


def example_2_batch_predictions():
    """Example 2: Batch predictions for multiple currencies."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: BATCH PREDICTIONS FOR MULTIPLE CURRENCIES")
    print("=" * 80)
    
    # Initialize predictor
    models_dir = Path("models/currency_production_lstm")
    predictor = ModelPredictor(models_dir)
    
    # Load models
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found!")
        return
    
    # Get top 10 predictions by expected price change
    print("Getting top 10 predictions by price change...")
    top_predictions = predictor.get_top_predictions(
        top_n=10,
        sort_by='price_change_percent',
        ascending=False
    )
    
    if top_predictions:
        print(f"\n{'Rank':<4} {'Currency Pair':<35} {'Current':<8} {'Predicted':<8} {'Change':<8} {'Conf':<6}")
        print("-" * 80)
        
        for i, pred in enumerate(top_predictions, 1):
            print(f"{i:<4} {pred.currency_pair:<35} {pred.current_price:<8.2f} "
                  f"{pred.predicted_price:<8.2f} {pred.price_change_percent:+6.1f}% {pred.confidence_score:<6.2f}")
        
        # Calculate summary statistics
        avg_change = sum(p.price_change_percent for p in top_predictions) / len(top_predictions)
        positive_count = sum(1 for p in top_predictions if p.price_change_percent > 0)
        
        print(f"\nSummary:")
        print(f"  Average expected change: {avg_change:+.2f}%")
        print(f"  Positive predictions: {positive_count}/{len(top_predictions)}")
    else:
        print("No predictions generated")


def example_3_specific_currencies():
    """Example 3: Predictions for specific currencies of interest."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: PREDICTIONS FOR SPECIFIC CURRENCIES")
    print("=" * 80)
    
    # Initialize predictor
    models_dir = Path("models/currency_production_lstm")
    predictor = ModelPredictor(models_dir)
    
    # Load models
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found!")
        return
    
    # Define currencies of interest (high-value currencies)
    currencies_of_interest = [
        "Divine Orb -> Chaos Orb",
        "Mirror Shard -> Chaos Orb",
        "Eternal Orb -> Chaos Orb",
        "Exalted Orb -> Chaos Orb",
        "Ancient Orb -> Chaos Orb"
    ]
    
    # Filter to only available currencies
    available_currencies = [c for c in currencies_of_interest if c in available_models]
    
    if not available_currencies:
        print("None of the target currencies are available")
        print("Available currencies:", list(available_models.keys())[:5])
        return
    
    print(f"Predicting prices for {len(available_currencies)} high-value currencies...")
    
    results = predictor.predict_multiple_currencies(
        available_currencies,
        prediction_horizon_days=1
    )
    
    if results:
        print(f"\n{'Currency':<35} {'Current':<10} {'Predicted':<10} {'Change':<10} {'Confidence'}")
        print("-" * 80)
        
        for result in results:
            print(f"{result.currency_pair:<35} {result.current_price:<10.2f} "
                  f"{result.predicted_price:<10.2f} {result.price_change_percent:+8.1f}% {result.confidence_score:<10.3f}")
    else:
        print("No predictions generated")


def example_4_multi_horizon_prediction():
    """Example 4: Multi-horizon predictions (1, 3, 7 days)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: MULTI-HORIZON PREDICTIONS")
    print("=" * 80)
    
    # Initialize predictor
    models_dir = Path("models/currency_production_lstm")
    predictor = ModelPredictor(models_dir)
    
    # Load models
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found!")
        return
    
    # Get a high-value currency for demonstration
    currency_pair = list(available_models.keys())[0]
    horizons = [1, 3, 7]
    
    print(f"Multi-horizon prediction for: {currency_pair}")
    print(f"\n{'Horizon':<8} {'Predicted Price':<15} {'Change %':<10} {'Confidence'}")
    print("-" * 50)
    
    for horizon in horizons:
        result = predictor.predict_price(currency_pair, horizon)
        if result:
            print(f"{horizon}d{'':<5} {result.predicted_price:<15.2f} {result.price_change_percent:+8.1f}% {result.confidence_score:<10.3f}")
        else:
            print(f"{horizon}d{'':<5} {'Failed':<15} {'N/A':<10} {'N/A'}")


def example_5_export_predictions():
    """Example 5: Export predictions to JSON file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: EXPORT PREDICTIONS TO FILE")
    print("=" * 80)
    
    # Initialize predictor
    models_dir = Path("models/currency_production_lstm")
    predictor = ModelPredictor(models_dir)
    
    # Load models
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found!")
        return
    
    # Get all predictions
    print("Generating predictions for all available currencies...")
    all_predictions = predictor.predict_multiple_currencies()
    
    if all_predictions:
        # Export to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.json"
        
        predictor.export_predictions(all_predictions, output_file)
        
        print(f"✓ Exported {len(all_predictions)} predictions to {output_file}")
        
        # Show sample of exported data
        print(f"\nSample of exported data:")
        with open(output_file, 'r') as f:
            data = json.load(f)
            print(f"  Export timestamp: {data['export_timestamp']}")
            print(f"  Total predictions: {data['total_predictions']}")
            print(f"  First prediction: {data['predictions'][0]['currency_pair']}")


async def example_6_api_integration():
    """Example 6: API integration (requires FastAPI server running)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: API INTEGRATION")
    print("=" * 80)
    
    base_url = "http://localhost:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Check API health
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✓ API is healthy: {health_data['status']}")
                    print(f"  Models loaded: {health_data.get('models_loaded', 0)}")
                else:
                    print("✗ API is not responding")
                    return
            
            # Get available currencies
            async with session.get(f"{base_url}/models/currencies") as response:
                if response.status == 200:
                    currencies = await response.json()
                    print(f"✓ Available currencies: {len(currencies)}")
                    
                    if currencies:
                        # Make a prediction via API
                        prediction_data = {
                            "currency_pair": currencies[0],
                            "prediction_horizon_days": 1
                        }
                        
                        async with session.post(f"{base_url}/predict", json=prediction_data) as pred_response:
                            if pred_response.status == 200:
                                result = await pred_response.json()
                                pred = result['predictions'][0]
                                print(f"✓ API Prediction for {pred['currency_pair']}:")
                                print(f"  Current: {pred['current_price']:.2f}")
                                print(f"  Predicted: {pred['predicted_price']:.2f}")
                                print(f"  Change: {pred['price_change_percent']:+.1f}%")
                            else:
                                print("✗ API prediction failed")
                else:
                    print("✗ Failed to get currencies from API")
                    
    except aiohttp.ClientError:
        print("✗ API server not running or not accessible")
        print("  Start the API server with: python ml/api/prediction_api.py")


def example_7_model_comparison():
    """Example 7: Compare different model types if available."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: MODEL COMPARISON")
    print("=" * 80)
    
    # Initialize predictor
    models_dir = Path("models/currency_production_lstm")
    predictor = ModelPredictor(models_dir)
    
    # Load models
    available_models = predictor.load_available_models()
    
    if not available_models:
        print("No models found!")
        return
    
    # Group models by type
    model_types = {}
    for currency_pair, info in available_models.items():
        model_type = info.get('model_type', 'unknown')
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(currency_pair)
    
    print("Model types available:")
    for model_type, pairs in model_types.items():
        avg_r2 = sum(available_models[pair]['training_metrics'].get('r2', 0) for pair in pairs) / len(pairs)
        print(f"  {model_type}: {len(pairs)} models (avg R²: {avg_r2:.3f})")
    
    # Compare performance on a few predictions
    if len(model_types) > 1:
        print("\nPerformance comparison on sample predictions:")
        sample_pairs = list(available_models.keys())[:3]
        
        for pair in sample_pairs:
            result = predictor.predict_price(pair)
            if result:
                info = available_models[pair]
                print(f"  {pair}: {info['model_type']} - R²: {info['training_metrics'].get('r2', 0):.3f}, "
                      f"Predicted change: {result.price_change_percent:+.1f}%")


def main():
    """Run all examples."""
    print("PoEconomy Price Prediction Examples")
    print("=" * 80)
    
    # Run examples
    try:
        example_1_basic_prediction()
        example_2_batch_predictions()
        example_3_specific_currencies()
        example_4_multi_horizon_prediction()
        example_5_export_predictions()
        example_7_model_comparison()
        
        # API example (requires server)
        print("\nRunning API example (requires FastAPI server)...")
        asyncio.run(example_6_api_integration())
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("USAGE SUMMARY")
    print("=" * 80)
    print("""
Available interfaces for price prediction:

1. Command Line Interface:
   python ml/scripts/predict_prices.py --help

2. Python API:
   from ml.utils.model_inference import ModelPredictor
   predictor = ModelPredictor("models/currency_production_lstm")
   result = predictor.predict_price("Divine Orb -> Chaos Orb")

3. REST API:
   python ml/api/prediction_api.py
   # Then access http://localhost:8000/docs

4. Batch Processing:
   python ml/examples/prediction_example.py

Key Features:
- Support for LSTM and ensemble models
- Multi-horizon predictions (1-30 days)
- Confidence scoring
- Export to JSON/CSV
- Real-time current league data
- Automatic feature engineering
- Price change analysis
- Model performance comparison

For production use, consider:
- Model retraining schedule
- Prediction caching
- Error handling and fallbacks
- Performance monitoring
- Data quality checks
""")


if __name__ == "__main__":
    main() 