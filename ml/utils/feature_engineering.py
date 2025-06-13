"""
Feature engineering module for currency price prediction.

This module contains the core feature engineering logic, separated from the main data processing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings

from config.training_config import DataConfig, ProcessingConfig
from utils.logging_utils import MLLogger


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering process."""
    data: pd.DataFrame
    feature_names: List[str]
    transformations_applied: List[str]
    statistics: Dict[str, Any]


class FeatureEngineer:
    """Feature engineering for currency price prediction."""
    
    def __init__(self, config: DataConfig, processing_config: ProcessingConfig, logger: Optional[MLLogger] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Data configuration
            processing_config: Processing configuration
            logger: Optional logger instance
        """
        self.config = config
        self.processing_config = processing_config
        self.logger = logger or MLLogger("FeatureEngineer")
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
    
    def engineer_features(self, df: pd.DataFrame, currency_pair: str) -> FeatureEngineeringResult:
        """
        Comprehensive feature engineering pipeline.
        
        Args:
            df: Input dataframe
            currency_pair: Currency pair identifier
            
        Returns:
            FeatureEngineeringResult with engineered features
        """
        transformations_applied = []
        statistics = {}
        
        with self.logger.log_operation(f"Feature engineering for {currency_pair}"):
            # Start with a copy
            df_processed = df.copy()
            original_shape = df_processed.shape
            
            # 1. Basic preprocessing
            df_processed = self._preprocess_basic(df_processed)
            transformations_applied.append("basic_preprocessing")
            
            # 2. Time-based features
            df_processed = self._engineer_time_features(df_processed)
            transformations_applied.append("time_features")
            
            # 3. Price-based features
            df_processed = self._engineer_price_features(df_processed)
            transformations_applied.append("price_features")
            
            # 4. League-aware features
            if self.config.include_league_features:
                df_processed = self._engineer_league_features(df_processed)
                transformations_applied.append("league_features")
            
            # 5. Rolling window features
            df_processed = self._engineer_rolling_features(df_processed)
            transformations_applied.append("rolling_features")
            
            # 6. Target variables
            df_processed = self._create_targets(df_processed)
            transformations_applied.append("target_creation")
            
            # 7. Outlier removal
            if self.processing_config.outlier_removal:
                df_processed = self._remove_outliers(df_processed)
                transformations_applied.append("outlier_removal")
            
            # 8. Feature selection
            if self.processing_config.feature_selection:
                df_processed = self._select_features(df_processed, currency_pair)
                transformations_applied.append("feature_selection")
            
            # Statistics
            final_shape = df_processed.shape
            statistics = {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'features_added': final_shape[1] - original_shape[1],
                'records_retained': final_shape[0] / original_shape[0] if original_shape[0] > 0 else 0
            }
            
            feature_names = df_processed.columns.tolist()
            
            self.logger.info(
                f"Feature engineering completed for {currency_pair}",
                extra=statistics
            )
        
        return FeatureEngineeringResult(
            data=df_processed,
            feature_names=feature_names,
            transformations_applied=transformations_applied,
            statistics=statistics
        )
    
    def _preprocess_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing steps."""
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date within each league
        if 'league_name' in df.columns and 'date' in df.columns:
            df = df.sort_values(['league_name', 'date']).reset_index(drop=True)
        
        # Handle missing values in price
        if 'price' in df.columns:
            # Forward fill within each league
            if 'league_name' in df.columns:
                df['price'] = df.groupby('league_name')['price'].fillna(method='ffill')
            else:
                df['price'] = df['price'].fillna(method='ffill')
            
            # Drop remaining nulls
            df = df.dropna(subset=['price'])
        
        return df
    
    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based features."""
        if 'date' not in df.columns:
            return df
        
        # Basic time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # League-specific time features
        if 'league_name' in df.columns:
            # Calculate league start date and day within league
            league_starts = df.groupby('league_name')['date'].min()
            df['league_start'] = df['league_name'].map(league_starts)
            df['league_day'] = (df['date'] - df['league_start']).dt.days
            
            # League phase features
            max_league_day = df.groupby('league_name')['league_day'].max()
            df['league_max_day'] = df['league_name'].map(max_league_day)
            df['league_progress'] = df['league_day'] / (df['league_max_day'] + 1)
            
            # League age (how old is the league relative to others)
            latest_start = league_starts.max()
            league_age = (latest_start - league_starts).dt.days
            df['league_age_days'] = df['league_name'].map(league_age)
        
        return df
    
    def _engineer_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer price-based features."""
        if 'price' not in df.columns:
            return df
        
        # Log transformation if beneficial
        if self.processing_config.log_transform:
            price_range = df['price'].max() / df['price'].min()
            if price_range > self.processing_config.log_transform_ratio_threshold:
                df['price_log'] = np.log1p(df['price'])
        
        # Price changes
        if 'league_name' in df.columns:
            df['price_change_1d'] = df.groupby('league_name')['price'].diff()
            df['price_change_pct_1d'] = df.groupby('league_name')['price'].pct_change()
        else:
            df['price_change_1d'] = df['price'].diff()
            df['price_change_pct_1d'] = df['price'].pct_change()
        
        # Price momentum
        for period in self.config.momentum_periods:
            if period > 1:
                if 'league_name' in df.columns:
                    df[f'momentum_{period}d'] = df.groupby('league_name')['price'].transform(
                        lambda x: x.pct_change(periods=period)
                    )
                else:
                    df[f'momentum_{period}d'] = df['price'].pct_change(periods=period)
        
        return df
    
    def _engineer_league_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer league-specific features."""
        if 'league_name' not in df.columns:
            return df
        
        # League statistics
        league_stats = df.groupby('league_name')['price'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).add_prefix('league_price_')
        
        # Merge back to main dataframe
        df = df.merge(league_stats, left_on='league_name', right_index=True, how='left')
        
        # Price relative to league statistics
        df['price_vs_league_mean'] = df['price'] / df['league_price_mean']
        df['price_vs_league_median'] = df['price'] / df['league_price_median']
        
        # Price percentile within league
        df['price_percentile_in_league'] = df.groupby('league_name')['price'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # League recency ranking
        league_recency = df.groupby('league_name')['league_start'].first().rank(
            method='dense', ascending=False
        )
        df['league_recency_rank'] = df['league_name'].map(league_recency)
        
        return df
    
    def _engineer_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer rolling window features."""
        if 'price' not in df.columns:
            return df
        
        # Ensure data is properly sorted within each league
        if 'league_name' in df.columns:
            df = df.sort_values(['league_name', 'date'])
        else:
            df = df.sort_values('date')
        
        for window in self.config.rolling_windows:
            # Skip 1-day rolling std and zscore as they're always NaN
            if window == 1:
                # Only create mean, min, max for 1-day window (which are just the current values)
                if 'league_name' in df.columns:
                    df[f'price_mean_{window}d'] = df.groupby('league_name')['price'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df[f'price_min_{window}d'] = df.groupby('league_name')['price'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                    df[f'price_max_{window}d'] = df.groupby('league_name')['price'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                else:
                    df[f'price_mean_{window}d'] = df['price'].rolling(window=window, min_periods=1).mean()
                    df[f'price_min_{window}d'] = df['price'].rolling(window=window, min_periods=1).min()
                    df[f'price_max_{window}d'] = df['price'].rolling(window=window, min_periods=1).max()
                
                # Range is 0 for 1-day window
                df[f'price_range_{window}d'] = 0
                # Skip std and zscore for 1-day window
                continue
            
            # For windows > 1, use min_periods=2 for std calculation
            min_periods_std = max(2, min(window, 2))  # At least 2 points for std
            
            if 'league_name' in df.columns:
                # Rolling statistics within each league
                df[f'price_mean_{window}d'] = df.groupby('league_name')['price'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'price_std_{window}d'] = df.groupby('league_name')['price'].transform(
                    lambda x: x.rolling(window=window, min_periods=min_periods_std).std()
                )
                df[f'price_min_{window}d'] = df.groupby('league_name')['price'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[f'price_max_{window}d'] = df.groupby('league_name')['price'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
            else:
                # Simple rolling statistics
                df[f'price_mean_{window}d'] = df['price'].rolling(window=window, min_periods=1).mean()
                df[f'price_std_{window}d'] = df['price'].rolling(window=window, min_periods=min_periods_std).std()
                df[f'price_min_{window}d'] = df['price'].rolling(window=window, min_periods=1).min()
                df[f'price_max_{window}d'] = df['price'].rolling(window=window, min_periods=1).max()
            
            # Derived features
            df[f'price_range_{window}d'] = df[f'price_max_{window}d'] - df[f'price_min_{window}d']
            
            # Z-score (robust) - only calculate where std is available and > 0
            std_col = f'price_std_{window}d'
            mean_col = f'price_mean_{window}d'
            
            # Create z-score only where we have valid std values
            valid_std_mask = (df[std_col].notna()) & (df[std_col] > 1e-8)
            df[f'price_zscore_{window}d'] = np.nan
            df.loc[valid_std_mask, f'price_zscore_{window}d'] = (
                (df.loc[valid_std_mask, 'price'] - df.loc[valid_std_mask, mean_col]) / 
                df.loc[valid_std_mask, std_col]
            )
            
            # Debug: Log the rolling calculation results
            if hasattr(self, 'logger') and self.logger:
                nan_count = df[std_col].isna().sum()
                total_count = len(df)
                self.logger.info(f"Rolling {window}d std: {nan_count}/{total_count} NaN values")
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction."""
        if 'price' not in df.columns:
            return df
        
        original_len = len(df)
        
        for horizon in self.config.prediction_horizons:
            if 'league_name' in df.columns:
                # Future price within same league only
                df[f'target_price_{horizon}d'] = df.groupby('league_name')['price'].transform(
                    lambda x: x.shift(-horizon)
                )
            else:
                df[f'target_price_{horizon}d'] = df['price'].shift(-horizon)
            
            # Price change targets
            df[f'target_change_{horizon}d'] = df[f'target_price_{horizon}d'] - df['price']
            df[f'target_change_pct_{horizon}d'] = (
                df[f'target_change_{horizon}d'] / df['price']
            ) * 100
            
            # Direction targets
            df[f'target_direction_{horizon}d'] = pd.cut(
                df[f'target_change_pct_{horizon}d'],
                bins=[-np.inf, -2, 2, np.inf],
                labels=['down', 'stable', 'up']
            )
            
            # Log target creation statistics
            if hasattr(self, 'logger') and self.logger:
                target_col = f'target_price_{horizon}d'
                nan_count = df[target_col].isna().sum()
                valid_count = len(df) - nan_count
                self.logger.info(
                    f"Target creation for {horizon}d horizon:",
                    extra={
                        'original_samples': original_len,
                        'valid_targets': valid_count,
                        'nan_targets': nan_count,
                        'valid_ratio': valid_count / original_len if original_len > 0 else 0
                    }
                )
        
        # Get all target price columns
        target_cols = [col for col in df.columns if col.startswith('target_price_')]
        
        # Only remove rows where ALL targets are NaN
        df = df.dropna(subset=target_cols, how='all')
        
        if hasattr(self, 'logger') and self.logger:
            final_len = len(df)
            self.logger.info(
                "Final target filtering:",
                extra={
                    'original_samples': original_len,
                    'final_samples': final_len,
                    'removed_samples': original_len - final_len,
                    'retention_ratio': final_len / original_len if original_len > 0 else 0
                }
            )
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        if 'price' not in df.columns:
            return df
        
        original_len = len(df)
        
        # IQR-based outlier removal
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        
        multiplier = self.config.outlier_removal_iqr_multiplier
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        
        removed_count = original_len - len(df_clean)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outliers ({removed_count/original_len:.2%})")
        
        return df_clean
    
    def _select_features(self, df: pd.DataFrame, currency_pair: str) -> pd.DataFrame:
        """Select most important features."""
        # Identify feature columns (exclude metadata and targets)
        exclude_patterns = ['date', 'league_name', 'target_', 'league_start']
        feature_cols = [col for col in df.columns 
                       if not any(pattern in col for pattern in exclude_patterns)]
        
        if len(feature_cols) <= self.processing_config.max_features:
            return df
            
        # If we have too many features, just take the first max_features
        selected_features = feature_cols[:self.processing_config.max_features]
        
        # Keep all non-feature columns plus selected features
        non_feature_cols = [col for col in df.columns if col not in feature_cols]
        final_columns = non_feature_cols + selected_features
        
        return df[final_columns] 