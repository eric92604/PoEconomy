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
    print("=== Example 1: Basic Currency Prediction ===")
    
    try:
        # Initialize predictor with models directory
        models_dir = Path(__file__).parent.parent / "models" / "currency_production"
        predictor = ModelPredictor(models_dir)
        
        # Load available models
        available_models = predictor.load_available_models()
        print(f"Available models: {list(available_models.keys())}")
        
        if not available_models:
            print("No models found. Please train models first.")
            return
        
        # Pick first available currency for demo
        currency = list(available_models.keys())[0]
        print(f"Predicting price for: {currency}")
        
        # Make prediction
        result = predictor.predict_price(currency, prediction_horizon_days=1)
        
        if result:
            print(f"Current Price: {result.current_price:.2f}c")
            print(f"Predicted Price (1d): {result.predicted_price:.2f}c")
            print(f"Price Change: {result.price_change_percent:+.1f}%")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Model Type: {result.model_type}")
        else:
            print("Prediction failed")
            
    except Exception as e:
        print(f"Error: {e}")


def example_2_batch_predictions():
    """Example 2: Batch predictions for multiple currencies."""
    print("\n=== Example 2: Batch Predictions ===")
    
    try:
        models_dir = Path(__file__).parent.parent / "models" / "currency_production"
        predictor = ModelPredictor(models_dir)
        
        # Load models
        available_models = predictor.load_available_models()
        if not available_models:
            print("No models available")
            return
        
        # Get predictions for all available currencies
        predictions = predictor.predict_multiple_currencies(prediction_horizon_days=1)
        
        print(f"Generated {len(predictions)} predictions:")
        print(f"{'Currency':<20} {'Current':<10} {'Predicted':<10} {'Change %':<10} {'Confidence':<10}")
        print("-" * 70)
        
        for pred in predictions[:10]:  # Show first 10
            print(f"{pred.currency:<20} {pred.current_price:<10.2f} "
                  f"{pred.predicted_price:<10.2f} {pred.price_change_percent:<10.1f} "
                  f"{pred.confidence_score:<10.2f}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_3_specific_currencies():
    """Example 3: Predictions for specific currencies of interest."""
    print("\n=== Example 3: Specific Currency Predictions ===")
    
    try:
        models_dir = Path(__file__).parent.parent / "models" / "currency_production"
        predictor = ModelPredictor(models_dir)
        
        # Load models
        available_models = predictor.load_available_models()
        
        # Currencies of interest
        target_currencies = ["Divine Orb", "Exalted Orb", "Chaos Orb", "Ancient Orb"]
        available_targets = [c for c in target_currencies if c in available_models]
        
        if not available_targets:
            print("None of the target currencies have trained models")
            return
        
        print(f"Predicting for: {available_targets}")
        
        # Get predictions
        predictions = predictor.predict_multiple_currencies(
            currencies=available_targets,
            prediction_horizon_days=1
        )
        
        for pred in predictions:
            print(f"\n{pred.currency}:")
            print(f"  Current: {pred.current_price:.2f}c")
            print(f"  Predicted: {pred.predicted_price:.2f}c")
            print(f"  Change: {pred.price_change_percent:+.1f}%")
            print(f"  Confidence: {pred.confidence_score:.2f}")
            print(f"  Data Points: {pred.data_points_used}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_4_multi_horizon_prediction():
    """Example 4: Multi-horizon predictions (1, 3, 7 days)."""
    print("\n=== Example 4: Multi-Horizon Predictions ===")
    
    try:
        models_dir = Path(__file__).parent.parent / "models" / "currency_production"
        predictor = ModelPredictor(models_dir)
        
        available_models = predictor.load_available_models()
        if not available_models:
            print("No models available")
            return
        
        # Pick first available currency
        currency = list(available_models.keys())[0]
        print(f"Multi-horizon predictions for: {currency}")
        
        horizons = [1, 3, 7]
        for horizon in horizons:
            result = predictor.predict_price(currency, prediction_horizon_days=horizon)
            if result:
                print(f"{horizon}d: {result.predicted_price:.2f}c ({result.price_change_percent:+.1f}%)")
            else:
                print(f"{horizon}d: Prediction failed")
                
    except Exception as e:
        print(f"Error: {e}")


def example_5_export_predictions():
    """Example 5: Export predictions to file."""
    print("\n=== Example 5: Export Predictions ===")
    
    try:
        models_dir = Path(__file__).parent.parent / "models" / "currency_production"
        predictor = ModelPredictor(models_dir)
        
        # Generate predictions
        predictions = predictor.predict_multiple_currencies(prediction_horizon_days=1)
        
        if predictions:
            # Export to JSON
            output_file = "currency_predictions.json"
            predictor.export_predictions(predictions, output_file)
            print(f"Exported {len(predictions)} predictions to {output_file}")
        else:
            print("No predictions to export")
            
    except Exception as e:
        print(f"Error: {e}")


async def example_6_api_integration():
    """Example 6: Integration with prediction API."""
    print("\n=== Example 6: API Integration ===")
    
    try:
        import aiohttp
        
        # API endpoint (adjust URL as needed)
        base_url = "http://localhost:8000"
        
        async with aiohttp.ClientSession() as session:
            # Get model status
            async with session.get(f"{base_url}/models/status") as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"API Status: {status['status']}")
                    print(f"Loaded models: {status['loaded_models']}")
                else:
                    print("API not available")
                    return
            
            # Get top predictions
            async with session.get(f"{base_url}/predict/top?top_n=5") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"\nTop 5 predictions from API:")
                    for pred in data['predictions']:
                        print(f"  {pred['currency']}: {pred['price_change_percent']:+.1f}%")
                else:
                    print("Failed to get predictions from API")
                    
    except ImportError:
        print("aiohttp not available. Install with: pip install aiohttp")
    except Exception as e:
        print(f"API integration error: {e}")


def example_7_model_comparison():
    """Example 7: Compare model performance across currencies."""
    print("\n=== Example 7: Model Performance Comparison ===")
    
    try:
        models_dir = Path(__file__).parent.parent / "models" / "currency_production"
        predictor = ModelPredictor(models_dir)
        
        available_models = predictor.load_available_models()
        if not available_models:
            print("No models available")
            return
        
        print("Model performance comparison:")
        print(f"{'Currency':<25} {'Model Type':<15} {'Confidence':<12} {'Data Points'}")
        print("-" * 70)
        
        # Get sample predictions to show model info
        sample_currencies = list(available_models.keys())[:10]
        for currency in sample_currencies:
            result = predictor.predict_price(currency, prediction_horizon_days=1)
            if result:
                print(f"{currency:<25} {result.model_type:<15} "
                      f"{result.confidence_score:<12.3f} {result.data_points_used}")
                      
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all prediction examples."""
    print("🔮 PoEconomy ML Prediction Examples")
    print("=" * 60)
    print("This script demonstrates various prediction capabilities")
    print("- Single currency predictions")
    print("- Batch predictions")
    print("- Multi-horizon forecasting")
    print("- Data export functionality")
    print("- API integration")
    print("=" * 60)
    
    # Run examples
    example_1_basic_prediction()
    example_2_batch_predictions()
    example_3_specific_currencies()
    example_4_multi_horizon_prediction()
    example_5_export_predictions()
    
    # Async examples
    asyncio.run(example_6_api_integration())
    
    example_7_model_comparison()
    
    print("\n✅ All examples completed!")
    print("\nNext steps:")
    print("- Train models using the training pipeline")
    print("- Start the prediction API server")
    print("- Integrate predictions into your trading strategy")


if __name__ == "__main__":
    main() 