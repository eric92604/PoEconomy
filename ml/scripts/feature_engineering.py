import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

def extract_base_features(target_currency_pair=None, months_back=12):
    """
    Extract base features for ML training, prioritizing recent leagues.
    
    Args:
        target_currency_pair: Tuple of (get_currency, pay_currency) or None for all
        months_back: How many months of data to include
    """
    conn = get_db_connection()
    
    # Build query with recency focus
    base_query = """
    SELECT 
        cp.id,
        cp."leagueId",
        cp."getCurrencyId", 
        cp."payCurrencyId",
        cp.date,
        cp.value as price,
        cp.confidence,
        l.name as league_name,
        l."startDate" as league_start,
        l."endDate" as league_end,
        l."isActive" as league_active,
        gc.name as get_currency,
        pc.name as pay_currency,
        -- League age in days
        EXTRACT(EPOCH FROM (cp.date - l."startDate")) / 86400 as league_age_days,
        -- Recency weight (higher for recent leagues)
        CASE 
            WHEN l."startDate" >= NOW() - INTERVAL '3 months' THEN 1.0
            WHEN l."startDate" >= NOW() - INTERVAL '6 months' THEN 0.8
            WHEN l."startDate" >= NOW() - INTERVAL '12 months' THEN 0.6
            ELSE 0.3
        END as recency_weight
    FROM currency_prices cp
    JOIN leagues l ON cp."leagueId" = l.id
    JOIN currency gc ON cp."getCurrencyId" = gc.id 
    JOIN currency pc ON cp."payCurrencyId" = pc.id
    WHERE cp.date >= NOW() - INTERVAL '{months_back} months'
        AND cp.value > 0  -- Remove negative prices
        AND cp.value < 10000  -- Remove extreme outliers
    """.format(months_back=months_back)
    
    # Add currency pair filter if specified
    if target_currency_pair:
        get_curr, pay_curr = target_currency_pair
        base_query += f" AND gc.name = '{get_curr}' AND pc.name = '{pay_curr}'"
    
    base_query += " ORDER BY cp.date DESC"
    
    print(f"Extracting base features for last {months_back} months...")
    df = pd.read_sql(base_query, conn)
    conn.close()
    
    print(f"Extracted {len(df):,} records")
    return df

def engineer_time_features(df):
    """Create time-based features."""
    print("Engineering time features...")
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # League phase (early/mid/late)
    df['league_phase'] = pd.cut(
        df['league_age_days'], 
        bins=[-1, 7, 30, 60, float('inf')], 
        labels=['very_early', 'early', 'mid', 'late']
    )
    
    return df

def engineer_price_features(df):
    """Create price-based features with rolling statistics."""
    print("Engineering price features...")
    
    # Sort by currency pair and date for rolling calculations
    df = df.sort_values(['get_currency', 'pay_currency', 'date'])
    
    # Rolling statistics (7, 14, 30 days)
    for window in [7, 14, 30]:
        df[f'price_mean_{window}d'] = (
            df.groupby(['get_currency', 'pay_currency'])['price']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[f'price_std_{window}d'] = (
            df.groupby(['get_currency', 'pay_currency'])['price']
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
    
    # Price momentum (current vs historical averages)
    df['momentum_7d'] = df['price'] / df['price_mean_7d'] - 1
    df['momentum_30d'] = df['price'] / df['price_mean_30d'] - 1
    
    # Volatility
    df['volatility_7d'] = df['price_std_7d'] / df['price_mean_7d']
    df['volatility_30d'] = df['price_std_30d'] / df['price_mean_30d']
    
    # Price change from previous day
    df['price_change_1d'] = (
        df.groupby(['get_currency', 'pay_currency'])['price']
        .pct_change(1)
        .fillna(0)
    )
    
    return df

def engineer_league_features(df):
    """Create league-specific features."""
    print("Engineering league features...")
    
    # League activity metrics
    league_stats = df.groupby('league_name').agg({
        'price': ['count', 'mean', 'std'],
        'date': ['min', 'max']
    }).round(4)
    
    league_stats.columns = ['_'.join(col).strip() for col in league_stats.columns]
    league_stats = league_stats.add_prefix('league_')
    league_stats = league_stats.reset_index()
    
    # Merge back to main dataframe
    df = df.merge(league_stats, on='league_name', how='left')
    
    return df

def engineer_target_variables(df, prediction_horizons=[1, 3, 7]):
    """Create target variables for different prediction horizons."""
    print(f"Engineering target variables for horizons: {prediction_horizons}")
    
    df = df.sort_values(['get_currency', 'pay_currency', 'date'])
    
    for horizon in prediction_horizons:
        # Future price
        df[f'target_price_{horizon}d'] = (
            df.groupby(['get_currency', 'pay_currency'])['price']
            .shift(-horizon)
        )
        
        # Price change percentage
        df[f'target_change_{horizon}d'] = (
            (df[f'target_price_{horizon}d'] / df['price'] - 1) * 100
        )
        
        # Direction (up/down/stable)
        df[f'target_direction_{horizon}d'] = pd.cut(
            df[f'target_change_{horizon}d'],
            bins=[-float('inf'), -2, 2, float('inf')],
            labels=['down', 'stable', 'up']
        )
    
    return df

def clean_and_validate_features(df):
    """Clean and validate the engineered features."""
    print("Cleaning and validating features...")
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with appropriate defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove rows where we don't have target variables (last few days)
    df = df.dropna(subset=[col for col in df.columns if col.startswith('target_')])
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Features: {len([col for col in df.columns if not col.startswith('target_')])}")
    print(f"Targets: {len([col for col in df.columns if col.startswith('target_')])}")
    
    return df

def create_ml_dataset(target_currency_pair=("Divine Orb", "Chaos Orb"), months_back=12):
    """
    Main function to create ML-ready dataset.
    
    Args:
        target_currency_pair: Tuple of (get_currency, pay_currency)
        months_back: How many months of historical data to include
    
    Returns:
        pd.DataFrame: ML-ready dataset with features and targets
    """
    print("=" * 60)
    print("CREATING ML DATASET")
    print("=" * 60)
    print(f"Target pair: {target_currency_pair}")
    print(f"Time window: {months_back} months")
    
    # Step 1: Extract base features
    df = extract_base_features(target_currency_pair, months_back)
    
    if len(df) < 100:
        print("⚠️  Insufficient data for this currency pair")
        return None
    
    # Step 2: Engineer features
    df = engineer_time_features(df)
    df = engineer_price_features(df)
    df = engineer_league_features(df)
    df = engineer_target_variables(df)
    
    # Step 3: Clean and validate
    df = clean_and_validate_features(df)
    
    print("✅ ML dataset creation completed!")
    return df

def save_dataset(df, filename="ml_dataset.parquet"):
    """Save the dataset for model training."""
    if df is not None:
        filepath = f"data/{filename}"
        os.makedirs("data", exist_ok=True)
        df.to_parquet(filepath, index=False)
        print(f"Dataset saved to: {filepath}")
        return filepath
    return None

if __name__ == "__main__":
    # Create dataset for Divine Orb pricing (most valuable currency)
    dataset = create_ml_dataset(
        target_currency_pair=("Divine Orb", "Chaos Orb"),
        months_back=12
    )
    
    if dataset is not None:
        save_dataset(dataset, "divine_orb_dataset.parquet")
        
        # Show sample of features
        print("\nDataset Preview:")
        print(dataset.head()[['date', 'price', 'league_name', 'recency_weight', 
                            'momentum_7d', 'target_change_1d']].to_string())
    else:
        print("❌ Dataset creation failed") 