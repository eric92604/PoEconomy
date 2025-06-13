# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import warnings

# Suppress pandas SQLAlchemy warning for raw database connections
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

def identify_high_value_currencies(min_avg_value=50):
    """
    Identify currencies that have a minimum average value of min_avg_value chaos orbs.
    
    Args:
        min_avg_value: Minimum average chaos orb value threshold
    
    Returns:
        pd.DataFrame: High-value currencies with their average and max prices
    """
    conn = get_db_connection()
    
    query = """
    SELECT 
        gc.name as currency_name,
        MIN(cp.value) as min_price_chaos,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cp.value) as median_price_chaos,
        MAX(cp.value) as max_price_chaos,
        AVG(cp.value) as avg_price_chaos,
        STDDEV(cp.value) as price_volatility,
        COUNT(*) as total_records,
        MIN(cp.date) as first_seen,
        MAX(cp.date) as last_seen
    FROM currency_prices cp
    JOIN currency gc ON cp."getCurrencyId" = gc.id
    JOIN currency pc ON cp."payCurrencyId" = pc.id
    WHERE pc.name = 'Chaos Orb'  -- Only pairs priced in Chaos Orbs
        AND cp.value > 0  -- Remove invalid prices
    GROUP BY gc.name
    HAVING AVG(cp.value) >= %s  -- Filter by average value
        AND COUNT(*) >= 50  -- Ensure sufficient data points
    ORDER BY AVG(cp.value) DESC;
    """
    
    high_value_currencies = pd.read_sql(query, conn, params=[min_avg_value])
    conn.close()
    
    print(f"High-Value Currencies (Average >={min_avg_value} Chaos Orbs):")
    print("=" * 110)
    print(f"{'Currency':<25} {'Min':>6} {'Median':>7} {'Avg':>7} {'Max':>8} {'Vol':>6} {'Records':>7}")
    print("-" * 110)
    for _, row in high_value_currencies.iterrows():
        volatility = row['price_volatility'] if row['price_volatility'] is not None else 0
        median_val = row['median_price_chaos'] if row['median_price_chaos'] is not None else 0
        print(f"{row['currency_name']:<25} {row['min_price_chaos']:>6.1f} {median_val:>7.1f} "
              f"{row['avg_price_chaos']:>7.1f} {row['max_price_chaos']:>8.1f} {volatility:>6.1f} {row['total_records']:>7}")
    
    return high_value_currencies

def generate_target_currency_list():
    """
    Generate a prioritized list of currency pairs for ML model training.
    """
    print("\nGenerating Target Currency Pairs for ML Training...")
    print("=" * 60)
    
    # Get high-value currencies based on average value
    high_value = identify_high_value_currencies(min_avg_value=50)
    
    # Create target pairs list
    target_pairs = []
    
    # Priority 1: High-average-value currencies against Chaos Orb
    print(f"\nHigh-Average-Value Currencies (>=50 Chaos Average)")
    for _, currency in high_value.iterrows():
        pair = (currency['currency_name'], 'Chaos Orb')
        target_pairs.append({
            'get_currency': currency['currency_name'],
            'pay_currency': 'Chaos Orb',
            'priority': 1,
            'min_value': currency['min_price_chaos'],
            'median_value': currency['median_price_chaos'] if currency['median_price_chaos'] is not None else 0,
            'avg_value': currency['avg_price_chaos'],
            'max_value': currency['max_price_chaos'],
            'volatility': currency['price_volatility'] if currency['price_volatility'] is not None else 0,
            'records': currency['total_records']
        })
        median_val = currency['median_price_chaos'] if currency['median_price_chaos'] is not None else 0
        print(f"  {pair[0]} -> {pair[1]} (Min: {currency['min_price_chaos']:.1f}, "
              f"Med: {median_val:.1f}, Avg: {currency['avg_price_chaos']:.1f}, Max: {currency['max_price_chaos']:.1f})")
    
    # Remove duplicates and sort by priority
    unique_pairs = []
    seen_pairs = set()
    
    for pair in target_pairs:
        pair_tuple = (pair['get_currency'], pair['pay_currency'])
        if pair_tuple not in seen_pairs:
            unique_pairs.append(pair)
            seen_pairs.add(pair_tuple)
    
    # Sort by priority, then by average value (descending)
    unique_pairs.sort(key=lambda x: (x['priority'], -x['avg_value']))
    
    print(f"\nFinal Target Currency Pairs: {len(unique_pairs)} pairs")
    print("=" * 130)
    print(f"{'#':>2} {'Currency Pair':<35} {'Min':>6} {'Med':>6} {'Avg':>6} {'Max':>7} {'Vol':>5} {'Records':>7}")
    print("-" * 130)
    for i, pair in enumerate(unique_pairs, 1):
        pair_name = f"{pair['get_currency']} -> {pair['pay_currency']}"
        print(f"{i:2d}. {pair_name:<35} {pair['min_value']:>6.1f} {pair['median_value']:>6.1f} "
              f"{pair['avg_value']:>6.1f} {pair['max_value']:>7.1f} {pair['volatility']:>5.1f} {pair['records']:>7}")
    
    return unique_pairs

if __name__ == "__main__":
    print("CURRENCY ANALYSIS FOR EXPANDED ML DATASET")
    print("=" * 60)
    
    # Run the analysis
    target_pairs = generate_target_currency_list()
    
    print(f"*** Identified {len(target_pairs)} target currency pairs")