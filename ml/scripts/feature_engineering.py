import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection
from identify_target_currencies import generate_target_currency_list

# Configure logging for console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline - focused on data preparation."""
    # League phase parameters
    max_league_days: int = 60  # Maximum days into each league to consider
    min_league_days: int = 0   # Minimum league duration to include
    
    # Data filtering
    min_records_per_pair: int = 50
    min_records_after_cleaning: int = 30
    
    # Feature engineering parameters
    rolling_windows: List[int] = None
    prediction_horizons: List[int] = None
    include_league_features: bool = True
    
    # Automation parameters
    experiment_id: str = None
    output_dir: str = "ml/training_data"
    log_level: str = "INFO"
    save_individual_datasets: bool = False
    create_combined_dataset: bool = True
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [1, 3, 7, 14]
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 3, 7]
        if self.experiment_id is None:
            self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_logging(config: FeatureEngineeringConfig) -> logging.Logger:
    """Setup logging for the feature engineering pipeline."""
    log_file = f"{config.output_dir}/logs/feature_engineering_{config.experiment_id}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_priority_pairs() -> List[Tuple[str, str]]:
    """
    Get priority currency pairs by directly calling identify_target_currencies functions.
    
    Returns:
        List of (from_currency, to_currency) tuples
    """
    logging.info("*** Identifying priority currency pairs...")
    
    try:
        # Directly call the function to get currency pairs
        target_pairs = generate_target_currency_list()
        
        if target_pairs:
            # Convert to simple tuple format expected by feature engineering
            priority_pairs = []
            for pair in target_pairs:
                priority_pairs.append((pair['get_currency'], pair['pay_currency']))
            
            logging.info(f"*** Identified {len(priority_pairs)} priority pairs from database analysis")
            return priority_pairs
        else:
            logging.warning("No target pairs returned from database analysis")
            
    except Exception as e:
        logging.warning(f"Failed to get target currencies from database: {str(e)}")

def get_league_phase_data(get_currency: str, pay_currency: str, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """
    Fetch currency price data based on league phases rather than linear time.
    
    Args:
        get_currency: Currency being received
        pay_currency: Currency being paid
        config: Feature engineering configuration
        
    Returns:
        pd.DataFrame: Currency price data from comparable league phases
    """
    conn = get_db_connection()
    
    query = """
    SELECT 
        cp.id,
        cp."leagueId",
        cp."getCurrencyId", 
        cp."payCurrencyId",
        cp.date AT TIME ZONE 'UTC' as date,
        cp.value as price,
        l.name as league_name,
        l."startDate" AT TIME ZONE 'UTC' as league_start,
        l."endDate" AT TIME ZONE 'UTC' as league_end,
        l."isActive" as league_active,
        gc.name as get_currency,
        pc.name as pay_currency,
        -- Calculate days into league
        EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) as league_day
    FROM currency_prices cp
    JOIN leagues l ON cp."leagueId" = l.id
    JOIN currency gc ON cp."getCurrencyId" = gc.id
    JOIN currency pc ON cp."payCurrencyId" = pc.id
    WHERE gc.name = %s 
        AND pc.name = %s
        AND cp.value > 0
        -- Only include data from the first X days of each league
        AND EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) <= %s
        AND EXTRACT(DAY FROM (cp.date AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) >= 0
        -- Only include leagues that lasted at least min_league_days
        AND (l."endDate" IS NULL OR 
             EXTRACT(DAY FROM (l."endDate" AT TIME ZONE 'UTC' - l."startDate" AT TIME ZONE 'UTC')) >= %s)
    ORDER BY l."startDate" DESC, cp.date ASC
    """
    
    df = pd.read_sql(query, conn, params=[
        get_currency, 
        pay_currency, 
        config.max_league_days,
        config.min_league_days
    ])
    conn.close()
    
    logging.debug(f"Fetched {len(df)} records for {get_currency} -> {pay_currency}")
    return df

def engineer_league_metadata_features(df: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """
    Create league metadata features for downstream recency weighting.
    
    This focuses on data preparation, NOT optimization.
    Recency weighting will be applied during model training.
    
    Args:
        df: DataFrame with raw currency price data from PostgreSQL
        config: Feature engineering configuration
        
    Returns:
        DataFrame with engineered league metadata features
    """
    if len(df) == 0:
        return df
    
    # Ensure timezone-naive datetime columns
    for col in ['date', 'league_start', 'league_end']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    
    df = df.sort_values(['league_start', 'date']).reset_index(drop=True)
    
    # League age and timing features
    df['league_age_days'] = df['league_day']
    df['days_into_league'] = df['league_age_days']
    
    # Calculate league recency metadata (for downstream weighting)
    latest_league_start = df['league_start'].max()
    df['league_days_old'] = (latest_league_start - df['league_start']).dt.days
    df['league_start_timestamp'] = df['league_start'].astype(int) // 10**9  # Unix timestamp
    
    # League phase indicators (discrete features)
    max_days = df['league_day'].max()
    if max_days > 0:
        early_threshold = max_days * 0.2
        late_threshold = max_days * 0.4
        
        df['league_phase_early'] = (df['league_day'] <= early_threshold).astype(int)
        df['league_phase_mid'] = ((df['league_day'] > early_threshold) & 
                                 (df['league_day'] <= late_threshold)).astype(int)
        df['league_phase_late'] = (df['league_day'] > late_threshold).astype(int)
    else:
        df['league_phase_early'] = 1
        df['league_phase_mid'] = 0
        df['league_phase_late'] = 0
    
    # Time-based features (relative to league start)
    df['week_in_league'] = (df['league_day'] / 7).astype(int)
    df['is_league_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # League activity and duration features
    if config.include_league_features:
        # League statistics (for context, not weighting)
        league_stats = df.groupby('league_name').agg({
            'price': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'league_day': 'max'
        }).reset_index()
        
        league_stats.columns = ['league_name', 'league_price_count', 'league_price_mean', 
                               'league_price_std', 'league_price_min', 'league_price_max', 
                               'league_price_median', 'league_duration_days']
        
        df = df.merge(league_stats, on='league_name', how='left')
        
        # League ranking by recency (for model to use)
        league_recency_rank = df.groupby('league_name')['league_start'].first().rank(method='dense', ascending=False)
        league_recency_mapping = league_recency_rank.to_dict()
        df['league_recency_rank'] = df['league_name'].map(league_recency_mapping)
    
    # Price-based features with league-aware rolling windows
    for window in config.rolling_windows:
        # Calculate rolling stats within each league (no cross-league contamination)
        df[f'price_mean_{window}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'price_std_{window}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        df[f'price_min_{window}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        df[f'price_max_{window}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        
        # Additional rolling features for better price modeling
        df[f'price_median_{window}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).median()
        )
        df[f'price_range_{window}d'] = df[f'price_max_{window}d'] - df[f'price_min_{window}d']
        
        # Robust z-score calculation with proper handling of zero standard deviation
        df[f'price_zscore_{window}d'] = df.groupby('league_name')['price'].transform(
            lambda x: (x - x.rolling(window=window, min_periods=1).mean()) / 
                     np.maximum(x.rolling(window=window, min_periods=1).std(), 1e-8)
        )
    
    # League-aware momentum and volatility
    for period in [3, 7]:
        df[f'momentum_{period}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.pct_change(periods=period)
        )
    
    for window in [7, 14]:
        rolling_mean = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        rolling_std = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=window).std()
        )
        df[f'volatility_{window}d'] = rolling_std / rolling_mean
    
    # Price change indicators within league
    df['price_change_1d'] = df.groupby('league_name')['price'].transform(lambda x: x.diff())
    df['price_change_pct_1d'] = df.groupby('league_name')['price'].transform(lambda x: x.pct_change())
    
    # Cross-league comparison features
    df['price_vs_league_mean'] = df['price'] / df['league_price_mean']
    df['price_percentile_in_league'] = df.groupby('league_name')['price'].transform(
        lambda x: x.rank(pct=True)
    )
    
    logging.info(f"  *** Created league metadata features for downstream recency weighting")
    logging.info(f"   *** League age range: {df['league_days_old'].min()}-{df['league_days_old'].max()} days")
    
    return df

def create_league_aware_targets(df: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """Create target variables that respect league boundaries."""
    if len(df) == 0:
        return df
    
    df = df.sort_values(['league_name', 'date']).reset_index(drop=True)
    
    for horizon in config.prediction_horizons:
        # Future price (within same league only)
        df[f'target_price_{horizon}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.shift(-horizon)
        )
        
        # Price change (absolute and percentage)
        df[f'target_change_{horizon}d'] = df[f'target_price_{horizon}d'] - df['price']
        df[f'target_change_pct_{horizon}d'] = (df[f'target_change_{horizon}d'] / df['price']) * 100
        
        # Direction (up/down/stable)
        df[f'target_direction_{horizon}d'] = pd.cut(
            df[f'target_change_pct_{horizon}d'],
            bins=[-np.inf, -2, 2, np.inf],
            labels=['down', 'stable', 'up']
        )
        
        # Volatility target
        df[f'target_volatility_{horizon}d'] = df.groupby('league_name')['price'].transform(
            lambda x: x.rolling(window=horizon).std().shift(-horizon)
        )
    
    return df

def clean_and_validate_league_features(df: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """Clean and validate engineered features with league-aware logic."""
    if len(df) == 0:
        return df
    
    logging.debug(f"Cleaning features: {df.shape[0]} records before cleaning")
    
    # Fill NaN values with league-aware forward fill
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df.groupby('league_name')[numeric_columns].ffill().fillna(0)
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows where ALL numeric features are NaN (but be more lenient with targets)
    feature_cols = [col for col in numeric_columns if not col.startswith('target_')]
    df = df.dropna(subset=feature_cols, how='all')
    
    # For targets, only require at least ONE valid target (not all targets)
    target_cols = [col for col in df.columns if col.startswith('target_') and not col.endswith('_direction_1d') and not col.endswith('_direction_3d') and not col.endswith('_direction_7d')]
    
    # Keep rows that have at least one valid target
    if target_cols:
        df = df.dropna(subset=target_cols, how='all')
    
    logging.debug(f"Cleaning complete: {df.shape[0]} records after cleaning")
    return df

def process_currency_pair_league_based(get_currency: str, pay_currency: str, config: FeatureEngineeringConfig) -> Optional[pd.DataFrame]:
    """Process a single currency pair through the league-based feature engineering pipeline."""
    logging.info(f"Processing {get_currency} -> {pay_currency}...")
    
    # Get league-phase data
    df = get_league_phase_data(get_currency, pay_currency, config)
    
    if len(df) < config.min_records_per_pair:
        logging.warning(f"  *** Insufficient data: {len(df)} records (minimum {config.min_records_per_pair} required)")
        return None
    
    leagues_count = df['league_name'].nunique()
    logging.info(f"  *** Raw data: {len(df)} records across {leagues_count} leagues")
    
    # Engineer features (data preparation only)
    df = engineer_league_metadata_features(df, config)
    
    # Create targets
    df = create_league_aware_targets(df, config)
    
    # Clean and validate
    df = clean_and_validate_league_features(df, config)
    
    if len(df) < config.min_records_after_cleaning:
        logging.warning(f"  *** Insufficient data after cleaning: {len(df)} records")
        return None
    
    # Add pair identifier
    df['currency_pair'] = f"{get_currency}_{pay_currency}"
    
    # Ensure league_name is preserved for model evaluation
    if 'league_name' not in df.columns:
        logging.warning("League name information missing - this may affect model evaluation")
    
    feature_count = len([c for c in df.columns if not c.startswith('target_')])
    target_count = len([c for c in df.columns if c.startswith('target_')])
    
    logging.info(f"  *** Processed: {len(df)} records, {feature_count} features, {target_count} targets")
    logging.info(f"  *** League coverage: {df['league_name'].nunique()} leagues" if 'league_name' in df.columns else "  *** No league information available")
    
    return df

def run_feature_engineering_experiment(config: FeatureEngineeringConfig) -> Dict[str, Any]:
    """
    Run a complete feature engineering experiment with the given configuration.
    
    Args:
        config: Feature engineering configuration
        
    Returns:
        Dictionary containing experiment results and metadata
    """
    
    logging.info("*** LEAGUE-BASED CURRENCY FEATURE ENGINEERING")
    logging.info("=" * 80)
    logging.info(f"Experiment ID: {config.experiment_id}")
    logging.info("=" * 80)
    
    # Get priority currency pairs
    priority_pairs = get_priority_pairs()
    
    # Log configuration
    logging.info(f"*** Feature engineering configured for league-based data preparation")
    logging.info(f"*** Ready to process {len(priority_pairs)} currency pairs")
    logging.info(f"*** League window: {config.max_league_days} days per league")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each currency pair
    logging.info("*** Starting data processing...")
    processed_datasets = []
    processing_stats = {
        "total_pairs": len(priority_pairs),
        "successful": 0,
        "failed": 0,
        "insufficient_data": 0
    }
    
    for i, (get_currency, pay_currency) in enumerate(priority_pairs, 1):
        logging.info(f"*** [{i}/{len(priority_pairs)}] Processing {get_currency} -> {pay_currency}")
        
        try:
            df = process_currency_pair_league_based(get_currency, pay_currency, config)
            
            if df is not None and len(df) > 0:
                processed_datasets.append(df)
                processing_stats["successful"] += 1
                
                # Save individual dataset if configured
                if config.save_individual_datasets:
                    try:
                        # Clean currency names for safe filenames
                        safe_get = get_currency.replace(" ", "_").replace("'", "")
                        safe_pay = pay_currency.replace(" ", "_").replace("'", "")
                        filename = f"{safe_get}_{safe_pay}_{config.experiment_id}.parquet"
                        filepath = output_dir / filename
                        
                        # Ensure parent directory exists
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        
                        df.to_parquet(filepath, index=False)
                        logging.info(f"  *** Individual dataset saved: {filepath}")
                        
                    except Exception as e:
                        logging.error(f"  *** Failed to save individual dataset for {get_currency}->{pay_currency}: {str(e)}")
                    
            else:
                processing_stats["insufficient_data"] += 1
                
        except Exception as e:
            logging.error(f"  *** Failed to process {get_currency} -> {pay_currency}: {str(e)}")
            processing_stats["failed"] += 1
    
    # Create combined dataset if configured and we have data
    combined_df = None
    logging.info(f"*** Combined dataset creation check:")
    logging.info(f"    - create_combined_dataset: {config.create_combined_dataset}")
    logging.info(f"    - processed_datasets length: {len(processed_datasets)}")
    
    if config.create_combined_dataset and processed_datasets:
        try:
            logging.info("*** Creating combined dataset...")
            combined_df = pd.concat(processed_datasets, ignore_index=True)
            
            # Save combined dataset
            combined_filename = f"combined_currency_features_{config.experiment_id}.parquet"
            combined_filepath = output_dir / combined_filename
            
            # Ensure parent directory exists
            combined_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            combined_df.to_parquet(combined_filepath, index=False)
            logging.info(f"*** Combined dataset saved: {combined_filepath}")
            logging.info(f"*** Combined dataset shape: {combined_df.shape}")
            
            # Log league distribution for model evaluation planning
            if 'league_name' in combined_df.columns:
                league_dist = combined_df['league_name'].value_counts()
                logging.info(f"*** League distribution in combined dataset:")
                for league, count in league_dist.items():
                    logging.info(f"    {league}: {count:,} records")
                
                if 'Settlers' in league_dist:
                    logging.info(f"*** Settlers league data available for model evaluation: {league_dist['Settlers']:,} records")
                else:
                    logging.warning("*** No Settlers league data found - models will use standard evaluation")
            else:
                logging.warning("*** No league information in combined dataset - models will use standard evaluation")
            
        except Exception as e:
            logging.error(f"Failed to save combined dataset: {str(e)}")
            combined_df = None
    else:
        if not config.create_combined_dataset:
            logging.info("*** Combined dataset creation disabled in config")
        if not processed_datasets:
            logging.info("*** No processed datasets available for combination")
    
    # Save experiment metadata
    metadata = {
        "experiment_id": config.experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "processing_stats": processing_stats,
        "output_files": {
            "combined_dataset": f"combined_currency_features_{config.experiment_id}.parquet" if combined_df is not None else None,
            "individual_datasets": config.save_individual_datasets
        }
    }
    
    try:
        metadata_file = output_dir / f"experiment_metadata_{config.experiment_id}.json"
        # Ensure parent directory exists
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"*** Experiment metadata saved: {metadata_file}")
        
    except Exception as e:
        logging.error(f"Failed to save experiment metadata: {str(e)}")
        metadata_file = output_dir / "metadata_save_failed.txt"
    
    logging.info("*** Processing complete!")
    logging.info(f"*** Successfully processed: {processing_stats['successful']}/{processing_stats['total_pairs']} pairs")
    logging.info(f"*** Insufficient data: {processing_stats['insufficient_data']} pairs")
    logging.info(f"*** Failed: {processing_stats['failed']} pairs")
    
    return {
        "experiment_id": config.experiment_id,
        "status": "completed",
        "processing_stats": processing_stats,
        "output_directory": str(output_dir),
        "combined_dataset_shape": combined_df.shape if combined_df is not None else None,
        "metadata_file": str(metadata_file)
    }

def main():
    """Main function to run feature engineering experiment."""
    
    # Create configuration
    config = FeatureEngineeringConfig()
    
    # Setup logging before processing
    setup_logging(config)
    
    # Ensure output directory exists and convert to absolute path
    output_path = Path(config.output_dir).resolve()
    config.output_dir = str(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting feature engineering experiment: {config.experiment_id}")
    logging.info(f"Output directory: {config.output_dir}")
    
    try:
        # Run experiment
        results = run_feature_engineering_experiment(config)
        
        print(f"\n*** Feature engineering completed!")
        print(f"Experiment ID: {results['experiment_id']}")
        print(f"Status: {results['status']}")
        print(f"Processed: {results['processing_stats']['successful']}/{results['processing_stats']['total_pairs']} currency pairs")
        
        if results['combined_dataset_shape']:
            print(f"Combined dataset shape: {results['combined_dataset_shape']}")
        
        print(f"Output directory: {results['output_directory']}")
        print(f"Metadata saved: {results['metadata_file']}")
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 