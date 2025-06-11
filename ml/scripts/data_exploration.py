import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

def explore_database_structure():
    """Get overview of database structure and data volumes."""
    conn = get_db_connection()
    
    print("=== DATABASE STRUCTURE OVERVIEW ===")
    
    # Get table sizes
    query = """
    SELECT 
        schemaname,
        tablename,
        attname,
        n_distinct,
        correlation
    FROM pg_stats 
    WHERE schemaname = 'public'
    ORDER BY tablename, attname;
    """
    
    table_info = pd.read_sql(query, conn)
    print("\nTable Statistics:")
    print(table_info.groupby('tablename').size())
    
    # Get row counts for each table
    tables = ['leagues', 'currency', 'currency_prices', 'items', 'item_prices', 'predictions']
    
    print("\n=== ROW COUNTS ===")
    for table in tables:
        try:
            count_query = f"SELECT COUNT(*) as count FROM {table}"
            result = pd.read_sql(count_query, conn)
            print(f"{table}: {result['count'].iloc[0]:,} rows")
        except Exception as e:
            print(f"{table}: Error - {e}")
    
    conn.close()

def explore_currency_prices_data():
    """Deep dive into currency_prices table - our main ML data source."""
    conn = get_db_connection()
    
    print("\n=== CURRENCY PRICES DATA EXPLORATION ===")
    
    # Basic statistics
    query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT "leagueId") as unique_leagues,
        COUNT(DISTINCT "getCurrencyId") as unique_get_currencies,
        COUNT(DISTINCT "payCurrencyId") as unique_pay_currencies,
        MIN(date) as earliest_date,
        MAX(date) as latest_date,
        AVG(value) as avg_price,
        MIN(value) as min_price,
        MAX(value) as max_price
    FROM currency_prices;
    """
    
    basic_stats = pd.read_sql(query, conn)
    print("\nBasic Statistics:")
    for col in basic_stats.columns:
        print(f"{col}: {basic_stats[col].iloc[0]}")
    
    # Get recent leagues (last 6 months as example)
    recent_leagues_query = """
    SELECT 
        l.name as league_name,
        l."startDate",
        l."endDate", 
        l."isActive",
        COUNT(cp.id) as price_records
    FROM leagues l
    LEFT JOIN currency_prices cp ON l.id = cp."leagueId"
    WHERE l."startDate" >= NOW() - INTERVAL '6 months' OR l."isActive" = true
    GROUP BY l.id, l.name, l."startDate", l."endDate", l."isActive"
    ORDER BY l."startDate" DESC;
    """
    
    recent_leagues = pd.read_sql(recent_leagues_query, conn)
    print(f"\nRecent Leagues (last 6 months):")
    print(recent_leagues.to_string(index=False))
    
    # Currency pair analysis
    currency_pairs_query = """
    SELECT 
        gc.name as get_currency,
        pc.name as pay_currency,
        COUNT(*) as record_count,
        AVG(cp.value) as avg_price,
        MIN(cp.date) as first_date,
        MAX(cp.date) as last_date
    FROM currency_prices cp
    JOIN currency gc ON cp."getCurrencyId" = gc.id
    JOIN currency pc ON cp."payCurrencyId" = pc.id
    GROUP BY gc.name, pc.name
    ORDER BY record_count DESC
    LIMIT 20;
    """
    
    top_pairs = pd.read_sql(currency_pairs_query, conn)
    print(f"\nTop 20 Currency Pairs by Volume:")
    print(top_pairs.to_string(index=False))
    
    conn.close()
    return recent_leagues, top_pairs

def check_data_quality():
    """Check for data quality issues."""
    conn = get_db_connection()
    
    print("\n=== DATA QUALITY CHECKS ===")
    
    # Check for missing values and outliers
    quality_query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(*) - COUNT("leagueId") as missing_league_id,
        COUNT(*) - COUNT("getCurrencyId") as missing_get_currency,
        COUNT(*) - COUNT("payCurrencyId") as missing_pay_currency,
        COUNT(*) - COUNT(date) as missing_date,
        COUNT(*) - COUNT(value) as missing_value,
        COUNT(*) - COUNT(confidence) as missing_confidence,
        COUNT(CASE WHEN value <= 0 THEN 1 END) as negative_prices,
        COUNT(CASE WHEN value > 1000 THEN 1 END) as extreme_high_prices
    FROM currency_prices;
    """
    
    quality_stats = pd.read_sql(quality_query, conn)
    print("\nData Quality Issues:")
    for col in quality_stats.columns:
        value = quality_stats[col].iloc[0]
        if value > 0 and col != 'total_records':
            print(f"⚠️  {col}: {value}")
        elif col == 'total_records':
            print(f"✅ {col}: {value:,}")
    
    # Check date distribution
    date_dist_query = """
    SELECT 
        DATE_TRUNC('month', date) as month,
        COUNT(*) as records
    FROM currency_prices
    GROUP BY DATE_TRUNC('month', date)
    ORDER BY month DESC
    LIMIT 12;
    """
    
    date_dist = pd.read_sql(date_dist_query, conn)
    print(f"\nRecent Monthly Data Distribution:")
    print(date_dist.to_string(index=False))
    
    conn.close()
    return quality_stats, date_dist

def identify_target_currencies():
    """Identify the most important currencies for ML modeling."""
    conn = get_db_connection()
    
    print("\n=== TARGET CURRENCY IDENTIFICATION ===")
    
    # Most traded currencies (as both get and pay)
    currency_volume_query = """
    WITH currency_stats AS (
        SELECT 
            c.name,
            SUM(CASE WHEN cp."getCurrencyId" = c.id THEN 1 ELSE 0 END) as as_get_currency,
            SUM(CASE WHEN cp."payCurrencyId" = c.id THEN 1 ELSE 0 END) as as_pay_currency,
            COUNT(*) as total_trades
        FROM currency c
        LEFT JOIN currency_prices cp ON (c.id = cp."getCurrencyId" OR c.id = cp."payCurrencyId")
        GROUP BY c.id, c.name
    )
    SELECT 
        name,
        as_get_currency,
        as_pay_currency,
        total_trades,
        (as_get_currency + as_pay_currency) as total_volume
    FROM currency_stats
    WHERE total_trades > 0
    ORDER BY total_volume DESC
    LIMIT 15;
    """
    
    target_currencies = pd.read_sql(currency_volume_query, conn)
    print("Top Currencies by Trading Volume:")
    print(target_currencies.to_string(index=False))
    
    conn.close()
    return target_currencies

def main():
    """Main exploration function."""
    print("Starting Data Exploration for ML Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Database structure
        explore_database_structure()
        
        # Step 2: Currency prices deep dive
        recent_leagues, top_pairs = explore_currency_prices_data()
        
        # Step 3: Data quality assessment
        quality_stats, date_dist = check_data_quality()
        
        # Step 4: Identify target currencies
        target_currencies = identify_target_currencies()
        
        return {
            'recent_leagues': recent_leagues,
            'top_pairs': top_pairs,
            'quality_stats': quality_stats,
            'target_currencies': target_currencies
        }
        
    except Exception as e:
        print(f"❌ Error during exploration: {e}")
        return None

if __name__ == "__main__":
    results = main() 