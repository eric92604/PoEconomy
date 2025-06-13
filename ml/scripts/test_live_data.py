#!/usr/bin/env python3
"""
Test script for live data ingestion from poe.ninja.

This script demonstrates fetching live currency data from poe.ninja,
storing it to the database, and using it for predictions.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.poe_ninja_client import PoENinjaClient
from utils.model_inference import ModelPredictor
from utils.logging_utils import MLLogger


async def test_poe_ninja_client():
    """Test the PoE Ninja client functionality."""
    print("=" * 80)
    print("TESTING POE.NINJA CLIENT")
    print("=" * 80)
    
    logger = MLLogger("TestPoENinja")
    
    async with PoENinjaClient(logger) as client:
        # Test fetching Standard league data
        print("\n1. Fetching Standard league currency data...")
        raw_data = await client.fetch_currency_overview("Standard", "Currency")
        
        if raw_data:
            print(f"✓ Successfully fetched raw data with {len(raw_data.get('lines', []))} currencies")
            
            # Parse the data
            currency_data = client.parse_currency_data(raw_data, "Standard")
            print(f"✓ Parsed {len(currency_data)} currency records")
            
            # Show top 5 currencies by chaos equivalent
            top_currencies = sorted(currency_data, key=lambda x: x.chaos_equivalent, reverse=True)[:5]
            
            print("\nTop 5 currencies by chaos equivalent:")
            print(f"{'Currency':<30} {'Chaos Equiv':<12} {'Change %':<10} {'Confidence'}")
            print("-" * 70)
            
            for currency in top_currencies:
                change_str = f"{currency.total_change:+.1f}%" if currency.total_change else "N/A"
                print(f"{currency.currency_name:<30} {currency.chaos_equivalent:<12.2f} "
                      f"{change_str:<10} {currency.confidence_level}")
            
            # Test price change detection
            print("\n2. Testing price change detection...")
            significant_changes = client.get_price_changes(currency_data, min_change_threshold=5.0)
            
            if significant_changes:
                print(f"Found {len(significant_changes)} significant price changes:")
                for change in significant_changes[:3]:
                    print(f"  - {change['currency_name']}: {change['total_change']:+.1f}% "
                          f"({change['chaos_equivalent']:.2f} Chaos)")
            else:
                print("No significant price changes detected")
            
            # Test database storage
            print("\n3. Testing database storage...")
            success = await client.store_to_database(currency_data[:10])  # Store first 10 for testing
            print(f"Database storage: {'✓ Success' if success else '✗ Failed'}")
            
        else:
            print("✗ Failed to fetch data from poe.ninja")


async def test_live_data_integration():
    """Test integration of live data with prediction system."""
    print("\n" + "=" * 80)
    print("TESTING LIVE DATA INTEGRATION WITH PREDICTIONS")
    print("=" * 80)
    
    logger = MLLogger("TestLiveIntegration")
    
    # First, ensure we have some live data
    print("\n1. Fetching and storing live data...")
    async with PoENinjaClient(logger) as client:
        raw_data = await client.fetch_currency_overview("Standard", "Currency")
        if raw_data:
            currency_data = client.parse_currency_data(raw_data, "Standard")
            await client.store_to_database(currency_data)
            print(f"✓ Stored {len(currency_data)} live currency records")
        else:
            print("✗ Failed to fetch live data")
            return
    
    # Test prediction system with live data
    print("\n2. Testing predictions with live data...")
    try:
        # Find latest model directory
        models_base = Path("models")
        if models_base.exists():
            currency_dirs = [d for d in models_base.iterdir() 
                           if d.is_dir() and d.name.startswith('currency_')]
            if currency_dirs:
                latest_dir = max(currency_dirs, key=lambda d: d.stat().st_mtime)
                print(f"Using models from: {latest_dir}")
                
                # Initialize predictor
                predictor = ModelPredictor(latest_dir, logger=logger)
                available_models = predictor.load_available_models()
                
                if available_models:
                    print(f"✓ Loaded {len(available_models)} prediction models")
                    
                    # Test getting live data
                    print("\n3. Testing live data retrieval...")
                    live_data = predictor.get_current_league_data(use_live_data=True, days_back=1)
                    
                    if live_data is not None and not live_data.empty:
                        print(f"✓ Retrieved {len(live_data)} live data records")
                        print(f"  Available currency pairs: {live_data['currency_pair'].nunique()}")
                        print(f"  Date range: {live_data['date'].min()} to {live_data['date'].max()}")
                        
                        # Test prediction with live data
                        print("\n4. Testing predictions with live data...")
                        currency_pairs = live_data['currency_pair'].unique()[:3]  # Test first 3
                        
                        for currency_pair in currency_pairs:
                            result = predictor.predict_price(currency_pair)
                            if result:
                                print(f"✓ {currency_pair}:")
                                print(f"    Current: {result.current_price:.2f}")
                                print(f"    Predicted: {result.predicted_price:.2f}")
                                print(f"    Change: {result.price_change_percent:+.1f}%")
                                print(f"    Confidence: {result.confidence_score:.2f}")
                            else:
                                print(f"✗ Failed to predict {currency_pair}")
                    else:
                        print("✗ No live data available for predictions")
                        
                        # Test fallback to historical data
                        print("\n   Testing fallback to historical data...")
                        historical_data = predictor.get_current_league_data(use_live_data=False)
                        if historical_data is not None and not historical_data.empty:
                            print(f"✓ Fallback successful: {len(historical_data)} historical records")
                        else:
                            print("✗ No historical data available either")
                else:
                    print("✗ No prediction models available")
            else:
                print("✗ No trained models found")
        else:
            print("✗ Models directory not found")
            
    except Exception as e:
        print(f"✗ Error testing predictions: {str(e)}")


async def test_real_time_monitoring():
    """Test real-time monitoring capabilities."""
    print("\n" + "=" * 80)
    print("TESTING REAL-TIME MONITORING")
    print("=" * 80)
    
    logger = MLLogger("TestRealTimeMonitoring")
    
    print("\n1. Testing continuous data fetching (3 iterations)...")
    
    async with PoENinjaClient(logger) as client:
        for i in range(3):
            print(f"\nIteration {i+1}:")
            
            # Fetch data
            raw_data = await client.fetch_currency_overview("Standard", "Currency")
            if raw_data:
                currency_data = client.parse_currency_data(raw_data, "Standard")
                
                # Check for significant changes
                changes = client.get_price_changes(currency_data, min_change_threshold=1.0)
                
                print(f"  Fetched {len(currency_data)} currencies")
                print(f"  Found {len(changes)} price changes > 1%")
                
                if changes:
                    top_change = max(changes, key=lambda x: abs(x['total_change']))
                    print(f"  Biggest change: {top_change['currency_name']} "
                          f"({top_change['total_change']:+.1f}%)")
                
                # Store to database
                success = await client.store_to_database(currency_data)
                print(f"  Database storage: {'✓' if success else '✗'}")
            else:
                print("  ✗ Failed to fetch data")
            
            # Wait between iterations (except last one)
            if i < 2:
                print("  Waiting 10 seconds...")
                await asyncio.sleep(10)
    
    print("\n✓ Real-time monitoring test completed")


def test_data_analysis():
    """Test data analysis capabilities on stored live data."""
    print("\n" + "=" * 80)
    print("TESTING DATA ANALYSIS ON LIVE DATA")
    print("=" * 80)
    
    try:
        from utils.database import get_db_connection
        import pandas as pd
        
        conn = get_db_connection()
        
        # Check if we have live data
        check_query = """
        SELECT COUNT(*) as count, 
               MIN(sample_time) as earliest, 
               MAX(sample_time) as latest,
               COUNT(DISTINCT currency_name) as currencies
        FROM live_currency_prices
        """
        
        result = pd.read_sql(check_query, conn)
        
        if result.iloc[0]['count'] > 0:
            print(f"✓ Found {result.iloc[0]['count']} live data records")
            print(f"  Date range: {result.iloc[0]['earliest']} to {result.iloc[0]['latest']}")
            print(f"  Unique currencies: {result.iloc[0]['currencies']}")
            
            # Get top currencies by chaos equivalent
            top_currencies_query = """
            SELECT currency_name, 
                   AVG(chaos_equivalent) as avg_chaos_equiv,
                   COUNT(*) as data_points,
                   AVG(total_change) as avg_change
            FROM live_currency_prices 
            WHERE direction = 'receive' 
                AND chaos_equivalent > 0
            GROUP BY currency_name
            ORDER BY avg_chaos_equiv DESC
            LIMIT 10
            """
            
            top_currencies = pd.read_sql(top_currencies_query, conn)
            
            print("\nTop 10 currencies by average chaos equivalent:")
            print(f"{'Currency':<30} {'Avg Chaos':<12} {'Data Points':<12} {'Avg Change %'}")
            print("-" * 75)
            
            for _, row in top_currencies.iterrows():
                print(f"{row['currency_name']:<30} {row['avg_chaos_equiv']:<12.2f} "
                      f"{row['data_points']:<12} {row['avg_change']:<12.1f}")
            
            # Check for recent price alerts
            alerts_query = """
            SELECT COUNT(*) as alert_count
            FROM price_alerts
            WHERE alert_time >= NOW() - INTERVAL '1 hour'
            """
            
            try:
                alerts_result = pd.read_sql(alerts_query, conn)
                alert_count = alerts_result.iloc[0]['alert_count']
                print(f"\n✓ Recent price alerts (last hour): {alert_count}")
            except:
                print("\n- No price alerts table found (normal if service hasn't run)")
            
        else:
            print("✗ No live data found in database")
            print("  Run the live data ingestion service first:")
            print("  python ml/services/live_data_ingestion.py")
        
        conn.close()
        
    except Exception as e:
        print(f"✗ Error analyzing data: {str(e)}")


async def main():
    """Run all tests."""
    print("PoE.Ninja Live Data Integration Test Suite")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    
    try:
        # Test 1: Basic PoE Ninja client functionality
        await test_poe_ninja_client()
        
        # Test 2: Live data integration with predictions
        await test_live_data_integration()
        
        # Test 3: Real-time monitoring
        await test_real_time_monitoring()
        
        # Test 4: Data analysis
        test_data_analysis()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print(f"Test completed at: {datetime.now()}")
        
        print("\nNext steps:")
        print("1. Start the live data ingestion service:")
        print("   python ml/services/live_data_ingestion.py")
        print("\n2. Use the prediction API with live data:")
        print("   python ml/api/prediction_api.py")
        print("\n3. Make predictions with live data:")
        print("   python ml/scripts/predict_prices.py")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 