import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

def identify_high_value_currencies(min_value=50):
    """
    Identify currencies that have been worth more than min_value chaos orbs at any point.
    
    Args:
        min_value: Minimum chaos orb value threshold
    
    Returns:
        pd.DataFrame: High-value currencies with their max prices
    """
    conn = get_db_connection()
    
    query = """
    SELECT 
        gc.name as currency_name,
        MAX(cp.value) as max_price_chaos,
        AVG(cp.value) as avg_price_chaos,
        COUNT(*) as total_records,
        MIN(cp.date) as first_seen,
        MAX(cp.date) as last_seen
    FROM currency_prices cp
    JOIN currency gc ON cp."getCurrencyId" = gc.id
    JOIN currency pc ON cp."payCurrencyId" = pc.id
    WHERE pc.name = 'Chaos Orb'  -- Only pairs priced in Chaos Orbs
        AND cp.value > 0  -- Remove invalid prices
    GROUP BY gc.name
    HAVING MAX(cp.value) >= %s  -- High-value currencies
    ORDER BY MAX(cp.value) DESC;
    """
    
    high_value_currencies = pd.read_sql(query, conn, params=[min_value])
    conn.close()
    
    print(f"High-Value Currencies (>{min_value} Chaos Orbs):")
    print("=" * 60)
    for _, row in high_value_currencies.iterrows():
        print(f"{row['currency_name']:<25} Max: {row['max_price_chaos']:>8.1f} | "
              f"Avg: {row['avg_price_chaos']:>6.1f} | Records: {row['total_records']:>4}")
    
    return high_value_currencies

def generate_target_currency_list():
    """
    Generate a prioritized list of currency pairs for ML model training.
    """
    print("\nGenerating Target Currency Pairs for ML Training...")
    print("=" * 60)
    
    # Get high-value and high-volume currencies
    high_value = identify_high_value_currencies(min_value=50)
    
    # Create target pairs list
    target_pairs = []
    
    # Priority 1: High-value currencies against Chaos Orb
    print("\nPriority 1: High-Value Currencies (>50 Chaos)")
    for _, currency in high_value.iterrows():
        pair = (currency['currency_name'], 'Chaos Orb')
        target_pairs.append({
            'get_currency': currency['currency_name'],
            'pay_currency': 'Chaos Orb',
            'priority': 1,
            'max_value': currency['max_price_chaos'],
            'records': currency['total_records'],
            'reason': 'High-value currency'
        })
        print(f"  {pair[0]} -> {pair[1]}")
    
    # Remove duplicates and sort by priority
    unique_pairs = []
    seen_pairs = set()
    
    for pair in target_pairs:
        pair_tuple = (pair['get_currency'], pair['pay_currency'])
        if pair_tuple not in seen_pairs:
            unique_pairs.append(pair)
            seen_pairs.add(pair_tuple)
    
    # Sort by priority, then by records
    unique_pairs.sort(key=lambda x: (x['priority'], -x['records']))
    
    print(f"\nFinal Target Currency Pairs: {len(unique_pairs)} pairs")
    print("=" * 80)
    for i, pair in enumerate(unique_pairs, 1):
        print(f"{i:2d}. {pair['get_currency']:<20} -> {pair['pay_currency']:<15} | "
              f"P{pair['priority']} | Records: {pair['records']:>4} | {pair['reason']}")
    
    return unique_pairs

if __name__ == "__main__":
    print("CURRENCY ANALYSIS FOR EXPANDED ML DATASET")
    print("=" * 60)
    
    # Run the analysis
    target_pairs = generate_target_currency_list()
    
    print(f"\n*** Analysis completed!")
    print(f"*** Identified {len(target_pairs)} target currency pairs")
    print(f"*** Ready for expanded data collection") 