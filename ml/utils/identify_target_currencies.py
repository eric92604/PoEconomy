# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Suppress pandas SQLAlchemy warning for raw database connections
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection
from utils.currency_standardizer import CurrencyStandardizer

def check_availability_columns_exist() -> bool:
    """
    Check if the currency availability columns exist in the database.
    
    Returns:
        True if availability columns exist, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'currency' 
        AND column_name IN ('isAvailableInCurrentLeague', 'lastAvailabilityCheck')
        """
        
        cursor.execute(check_query)
        result = cursor.fetchall()
        conn.close()
        
        return len(result) >= 2  # Both columns should exist
        
    except Exception:
        return False

def identify_high_value_currencies(min_avg_value=50, filter_by_availability=True, availability_check_days=30):
    """
    Identify currencies that have a minimum average value of min_avg_value chaos orbs.
    
    Args:
        min_avg_value: Minimum average chaos orb value threshold
        filter_by_availability: Whether to filter by current league availability
        availability_check_days: Maximum days since last availability check
    
    Returns:
        pd.DataFrame: High-value currencies with their average and max prices
    """
    conn = get_db_connection()
    
    # Build availability filter clause
    availability_filter = ""
    if filter_by_availability and check_availability_columns_exist():
        cutoff_time = datetime.now() - timedelta(days=availability_check_days)
        availability_filter = f"""
        AND gc."isAvailableInCurrentLeague" = true
        AND gc."lastAvailabilityCheck" >= '{cutoff_time.isoformat()}'
        """
        print(f"Filtering currencies by availability (last checked within {availability_check_days} days)")
    else:
        if filter_by_availability:
            print("Availability filtering requested but columns don't exist - including all currencies")
        else:
            print("ℹAvailability filtering disabled - including all currencies")
    
    query = f"""
    SELECT 
        gc.name as currency_name,
        gc."isAvailableInCurrentLeague" as is_available,
        gc."lastAvailabilityCheck" as last_availability_check,
        gc."availabilitySource" as availability_source,
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
        {availability_filter}
    GROUP BY gc.name, gc."isAvailableInCurrentLeague", gc."lastAvailabilityCheck", gc."availabilitySource"
    HAVING AVG(cp.value) >= %s  -- Filter by average value
        AND COUNT(*) >= 50  -- Ensure sufficient data points
    ORDER BY AVG(cp.value) DESC;
    """
    
    high_value_currencies = pd.read_sql(query, conn, params=[min_avg_value])
    conn.close()
    
    # Display results with availability info
    print(f"High-Value Currencies (Average >={min_avg_value} Chaos Orbs):")
    print("=" * 125)
    print(f"{'Currency':<25} {'Available':>9} {'Min':>6} {'Median':>7} {'Avg':>7} {'Max':>8} {'Vol':>6} {'Records':>7}")
    print("-" * 125)
    
    for _, row in high_value_currencies.iterrows():
        volatility = row['price_volatility'] if row['price_volatility'] is not None else 0
        median_val = row['median_price_chaos'] if row['median_price_chaos'] is not None else 0
        is_available = "✓" if row.get('is_available', True) else "✗"
        
        print(f"{row['currency_name']:<25} {is_available:>9} {row['min_price_chaos']:>6.1f} {median_val:>7.1f} "
              f"{row['avg_price_chaos']:>7.1f} {row['max_price_chaos']:>8.1f} {volatility:>6.1f} {row['total_records']:>7}")
    
    # Show availability summary
    if filter_by_availability and check_availability_columns_exist():
        available_count = sum(1 for _, row in high_value_currencies.iterrows() if row.get('is_available', True))
        total_count = len(high_value_currencies)
        print(f"\nAvailability Summary: {available_count}/{total_count} currencies are available in current league")
    
    return high_value_currencies

def generate_all_currencies_list(
    min_avg_value: float = 1.0,
    min_records: int = 100,
    filter_by_availability: bool = True,
    only_available_currencies: bool = True,
    availability_check_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Generate a comprehensive list of ALL currency pairs that meet minimum criteria.
    
    Args:
        min_avg_value: Minimum average value (in Chaos Orbs) for inclusion
        min_records: Minimum number of historical records required
        filter_by_availability: Whether to apply availability filtering
        only_available_currencies: If True, only include available currencies
        availability_check_days: Maximum days since last availability check
    
    Returns:
        List of currency pair dictionaries for all qualifying currencies
    """
    print(f"\nGenerating ALL Currency Pairs for ML Training...")
    print(f"Minimum average value: {min_avg_value} Chaos Orbs")
    print(f"Minimum records: {min_records}")
    print("=" * 60)
    
    # Check if we should filter by availability
    apply_availability_filter = filter_by_availability and check_availability_columns_exist()
    
    if not apply_availability_filter and filter_by_availability:
        print("⚠️  Availability filtering requested but database columns not found")
        print("    Run 'python ml/scripts/add_currency_availability_column.py' first")
        print("    Proceeding without availability filtering...\n")
    
    # Get ALL currencies that meet minimum criteria
    all_qualifying = identify_high_value_currencies(
        min_avg_value=min_avg_value,
        filter_by_availability=apply_availability_filter,
        availability_check_days=availability_check_days
    )
    
    # Filter by minimum records
    all_qualifying = all_qualifying[all_qualifying['total_records'] >= min_records]
    
    # Create target pairs list
    target_pairs = []
    
    print(f"\nALL Qualifying Currency Pairs (>={min_avg_value} Chaos Average, >={min_records} records):")
    for _, currency in all_qualifying.iterrows():
        # Check availability if filtering is enabled
        if only_available_currencies and apply_availability_filter:
            if not currency.get('is_available', True):
                print(f"  SKIPPED: {currency['currency_name']} -> Chaos Orb (not available in current league)")
                continue
        
        pair = {
            'get_currency': currency['currency_name'],
            'pay_currency': 'Chaos Orb',
            'priority': 1,
            'min_value': currency['min_price_chaos'],
            'median_value': currency['median_price_chaos'] if currency['median_price_chaos'] is not None else 0,
            'avg_value': currency['avg_price_chaos'],
            'max_value': currency['max_price_chaos'],
            'volatility': currency['price_volatility'] if currency['price_volatility'] is not None else 0,
            'records': currency['total_records'],
            'is_available': currency.get('is_available', True),
            'availability_source': currency.get('availability_source', 'unknown'),
            'last_availability_check': currency.get('last_availability_check')
        }
        
        target_pairs.append(pair)
        
        median_val = currency['median_price_chaos'] if currency['median_price_chaos'] is not None else 0
        availability_indicator = "✓" if pair['is_available'] else "✗"
        print(f"  {availability_indicator} {pair['get_currency']} -> {pair['pay_currency']} "
              f"(Min: {currency['min_price_chaos']:.1f}, Med: {median_val:.1f}, "
              f"Avg: {currency['avg_price_chaos']:.1f}, Max: {currency['max_price_chaos']:.1f}, "
              f"Records: {currency['total_records']})")
    
    # Remove duplicates and sort by record count (descending), then by average value
    unique_pairs = []
    seen_pairs = set()
    
    for pair in target_pairs:
        pair_tuple = (pair['get_currency'], pair['pay_currency'])
        if pair_tuple not in seen_pairs:
            unique_pairs.append(pair)
            seen_pairs.add(pair_tuple)
    
    # Sort by record count (descending), then by average value (descending)
    unique_pairs.sort(key=lambda x: (-x['records'], -x['avg_value']))
    
    print(f"\nFinal ALL Currency Pairs: {len(unique_pairs)} pairs")
    if apply_availability_filter:
        available_pairs = sum(1 for p in unique_pairs if p['is_available'])
        print(f"Available pairs: {available_pairs}/{len(unique_pairs)}")
    
    print("=" * 140)
    print(f"{'#':>2} {'✓':>1} {'Currency Pair':<35} {'Min':>6} {'Med':>6} {'Avg':>6} {'Max':>7} {'Vol':>5} {'Records':>7} {'Source':>12}")
    print("-" * 140)
    
    for i, pair in enumerate(unique_pairs, 1):
        pair_name = f"{pair['get_currency']} -> {pair['pay_currency']}"
        availability_indicator = "✓" if pair['is_available'] else "✗"
        source = pair['availability_source'][:12] if pair['availability_source'] else 'unknown'
        
        print(f"{i:2d}. {availability_indicator} {pair_name:<35} {pair['min_value']:>6.1f} {pair['median_value']:>6.1f} "
              f"{pair['avg_value']:>6.1f} {pair['max_value']:>7.1f} {pair['volatility']:>5.1f} {pair['records']:>7} {source:>12}")
    
    return unique_pairs


def generate_target_currency_list(
    filter_by_availability: bool = True,
    only_available_currencies: bool = True,
    availability_check_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Generate a prioritized list of currency pairs for ML model training.
    
    Args:
        filter_by_availability: Whether to apply availability filtering
        only_available_currencies: If True, only include available currencies
        availability_check_days: Maximum days since last availability check
    
    Returns:
        List of currency pair dictionaries
    """
    print("\nGenerating Target Currency Pairs for ML Training...")
    print("=" * 60)
    
    # Check if we should filter by availability
    apply_availability_filter = filter_by_availability and check_availability_columns_exist()
    
    if not apply_availability_filter and filter_by_availability:
        print("⚠️  Availability filtering requested but database columns not found")
        print("    Run 'python ml/scripts/add_currency_availability_column.py' first")
        print("    Proceeding without availability filtering...\n")
    
    # Get high-value currencies based on average value
    high_value = identify_high_value_currencies(
        min_avg_value=5,
        filter_by_availability=apply_availability_filter,
        availability_check_days=availability_check_days
    )
    
    # Create target pairs list
    target_pairs = []
    
    # High-average-value currencies against Chaos Orb
    print(f"\nTarget Currency Pairs (>=10 Chaos Average):")
    for _, currency in high_value.iterrows():
        # Check availability if filtering is enabled
        if only_available_currencies and apply_availability_filter:
            if not currency.get('is_available', True):
                print(f"  SKIPPED: {currency['currency_name']} -> Chaos Orb (not available in current league)")
                continue
        
        pair = {
            'get_currency': currency['currency_name'],
            'pay_currency': 'Chaos Orb',
            'priority': 1,
            'min_value': currency['min_price_chaos'],
            'median_value': currency['median_price_chaos'] if currency['median_price_chaos'] is not None else 0,
            'avg_value': currency['avg_price_chaos'],
            'max_value': currency['max_price_chaos'],
            'volatility': currency['price_volatility'] if currency['price_volatility'] is not None else 0,
            'records': currency['total_records'],
            'is_available': currency.get('is_available', True),
            'availability_source': currency.get('availability_source', 'unknown'),
            'last_availability_check': currency.get('last_availability_check')
        }
        
        target_pairs.append(pair)
        
        median_val = currency['median_price_chaos'] if currency['median_price_chaos'] is not None else 0
        availability_indicator = "✓" if pair['is_available'] else "✗"
        print(f"  {availability_indicator} {pair['get_currency']} -> {pair['pay_currency']} "
              f"(Min: {currency['min_price_chaos']:.1f}, Med: {median_val:.1f}, "
              f"Avg: {currency['avg_price_chaos']:.1f}, Max: {currency['max_price_chaos']:.1f})")
    
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
    if apply_availability_filter:
        available_pairs = sum(1 for p in unique_pairs if p['is_available'])
        print(f"Available pairs: {available_pairs}/{len(unique_pairs)}")
    
    print("=" * 140)
    print(f"{'#':>2} {'✓':>1} {'Currency Pair':<35} {'Min':>6} {'Med':>6} {'Avg':>6} {'Max':>7} {'Vol':>5} {'Records':>7} {'Source':>12}")
    print("-" * 140)
    
    for i, pair in enumerate(unique_pairs, 1):
        pair_name = f"{pair['get_currency']} -> {pair['pay_currency']}"
        availability_indicator = "✓" if pair['is_available'] else "✗"
        source = pair['availability_source'][:12] if pair['availability_source'] else 'unknown'
        
        print(f"{i:2d}. {availability_indicator} {pair_name:<35} {pair['min_value']:>6.1f} {pair['median_value']:>6.1f} "
              f"{pair['avg_value']:>6.1f} {pair['max_value']:>7.1f} {pair['volatility']:>5.1f} {pair['records']:>7} {source:>12}")
    
    return unique_pairs

def get_available_currencies() -> List[str]:
    """
    Get list of currencies that are available in the current league.
    
    Returns:
        List of available currency names
    """
    if not check_availability_columns_exist():
        print("⚠️  Availability columns not found - returning all currencies")
        
        # Fallback: return all currencies
        conn = get_db_connection()
        query = "SELECT name FROM currency ORDER BY name"
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df['name'].tolist()
    
    try:
        conn = get_db_connection()
        
        query = """
        SELECT name 
        FROM currency 
        WHERE "isAvailableInCurrentLeague" = true
        ORDER BY name
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        available_currencies = df['name'].tolist()
        print(f"ℹ️  Found {len(available_currencies)} available currencies")
        
        return available_currencies
        
    except Exception as e:
        print(f"❌ Error getting available currencies: {str(e)}")
        return []

if __name__ == "__main__":
    print("CURRENCY ANALYSIS FOR EXPANDED ML DATASET")
    print("=" * 60)
    
    # Check if availability system is set up
    has_availability = check_availability_columns_exist()
    print(f"Currency availability system: {'✓ Active' if has_availability else '✗ Not configured'}")
    
    if not has_availability:
        print("\nTo enable availability filtering:")
        print("1. Run: python ml/scripts/add_currency_availability_column.py")
        print("2. Run: python ml/scripts/check_currency_availability.py")
        print("3. Re-run this script with availability filtering enabled\n")
    
    # Run the analysis with availability filtering if available
    target_pairs = generate_target_currency_list(
        filter_by_availability=has_availability,
        only_available_currencies=has_availability,
        availability_check_days=30
    )
    
    print(f"\n*** Identified {len(target_pairs)} target currency pairs")
    
    if has_availability:
        available_pairs = sum(1 for p in target_pairs if p['is_available'])
        print(f"*** Available pairs: {available_pairs}")
        print(f"*** Unavailable pairs: {len(target_pairs) - available_pairs}")
    
    print("\nNext steps:")
    print("1. Update training configuration if needed")
    print("2. Run feature engineering pipeline")
    print("3. Run model training pipeline")