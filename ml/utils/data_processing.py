"""
Comprehensive data processing utilities for ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
import logging

from config.training_config import DataConfig, ProcessingConfig
from utils.logging_utils import MLLogger


@dataclass
class DataValidationResult:
    """Result of data validation checks."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering process."""
    data: pd.DataFrame
    feature_names: List[str]
    transformations_applied: List[str]
    statistics: Dict[str, Any]


class DataValidator:
    """Comprehensive data validation for ML pipeline."""
    
    def __init__(self, config: DataConfig, logger: Optional[MLLogger] = None):
        """
        Initialize data validator.
        
        Args:
            config: Data configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or MLLogger("DataValidator")
    
    def validate_dataframe(self, df: pd.DataFrame, currency_pair: str) -> DataValidationResult:
        """
        Comprehensive validation of input dataframe.
        
        Args:
            df: Input dataframe
            currency_pair: Currency pair identifier
            
        Returns:
            DataValidationResult with validation status and details
        """
        errors = []
        warnings = []
        statistics = {}
        
        # Basic structure validation
        if df.empty:
            errors.append("DataFrame is empty")
            return DataValidationResult(False, errors, warnings, statistics)
        
        # Required columns check
        required_columns = ['price', 'date', 'league_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Data type validation
        if 'price' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['price']):
                errors.append("Price column must be numeric")
            else:
                # Price validation
                if (df['price'] <= 0).any():
                    warnings.append("Found non-positive prices")
                if df['price'].isnull().any():
                    errors.append("Found null values in price column")
        
        # Date validation
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except Exception:
                errors.append("Date column contains invalid dates")
        
        # Sample size validation
        if len(df) < self.config.min_records_per_pair:
            errors.append(f"Insufficient data: {len(df)} < {self.config.min_records_per_pair}")
        
        # Missing data analysis
        missing_ratio = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratio[missing_ratio > self.config.max_missing_ratio].index.tolist()
        if high_missing_cols:
            warnings.append(f"High missing data in columns: {high_missing_cols}")
        
        # Statistical validation
        if 'price' in df.columns and df['price'].dtype in ['int64', 'float64']:
            price_stats = {
                'mean': float(df['price'].mean()),
                'std': float(df['price'].std()),
                'min': float(df['price'].min()),
                'max': float(df['price'].max()),
                'median': float(df['price'].median()),
                'skewness': float(df['price'].skew()),
                'kurtosis': float(df['price'].kurtosis())
            }
            statistics['price_stats'] = price_stats
            
            # Outlier detection
            Q1 = df['price'].quantile(0.25)
            Q3 = df['price'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = 3.0
            outliers = df[(df['price'] < Q1 - outlier_threshold * IQR) | 
                         (df['price'] > Q3 + outlier_threshold * IQR)]
            
            if len(outliers) > 0:
                outlier_ratio = len(outliers) / len(df)
                statistics['outlier_ratio'] = outlier_ratio
                if outlier_ratio > 0.1:  # More than 10% outliers
                    warnings.append(f"High outlier ratio: {outlier_ratio:.2%}")
        
        # League distribution analysis
        if 'league_name' in df.columns:
            league_counts = df['league_name'].value_counts()
            statistics['league_distribution'] = league_counts.to_dict()
            
            # Check for league imbalance
            min_league_samples = league_counts.min()
            if min_league_samples < 10:
                warnings.append(f"Some leagues have very few samples: {min_league_samples}")
        
        # Time series validation
        if 'date' in df.columns:
            try:
                df_sorted = df.sort_values('date')
                date_diffs = pd.to_datetime(df_sorted['date']).diff().dt.days
                
                # Check for large gaps
                large_gaps = date_diffs[date_diffs > 7]  # More than 7 days
                if len(large_gaps) > 0:
                    warnings.append(f"Found {len(large_gaps)} large time gaps in data")
                
                statistics['date_range'] = {
                    'start': df['date'].min(),
                    'end': df['date'].max(),
                    'span_days': (pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days
                }
            except Exception as e:
                warnings.append(f"Could not validate time series: {str(e)}")
        
        # Overall statistics
        statistics['total_records'] = len(df)
        statistics['total_features'] = len(df.columns)
        statistics['missing_data_ratio'] = float(df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        is_valid = len(errors) == 0
        
        # Log validation results
        if is_valid:
            self.logger.info(f"Data validation passed for {currency_pair}", extra=statistics)
        else:
            self.logger.error(f"Data validation failed for {currency_pair}", extra={"errors": errors})
        
        if warnings:
            self.logger.warning(f"Data validation warnings for {currency_pair}", extra={"warnings": warnings})
        
        return DataValidationResult(is_valid, errors, warnings, statistics)


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
        for period in self.config.rolling_windows:
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
        
        for window in self.config.rolling_windows:
            if 'league_name' in df.columns:
                # Rolling statistics within each league
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
            else:
                # Simple rolling statistics
                df[f'price_mean_{window}d'] = df['price'].rolling(window=window, min_periods=1).mean()
                df[f'price_std_{window}d'] = df['price'].rolling(window=window, min_periods=1).std()
                df[f'price_min_{window}d'] = df['price'].rolling(window=window, min_periods=1).min()
                df[f'price_max_{window}d'] = df['price'].rolling(window=window, min_periods=1).max()
            
            # Derived features
            df[f'price_range_{window}d'] = df[f'price_max_{window}d'] - df[f'price_min_{window}d']
            
            # Z-score (robust)
            df[f'price_zscore_{window}d'] = (
                (df['price'] - df[f'price_mean_{window}d']) / 
                np.maximum(df[f'price_std_{window}d'], 1e-8)
            )
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction."""
        if 'price' not in df.columns:
            return df
        
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
        
        if len(feature_cols) <= self.config.max_features:
            return df
        
        # Use variance threshold first
        variance_selector = VarianceThreshold(threshold=self.config.min_variance_threshold)
        
        try:
            # Fit on feature columns only
            feature_data = df[feature_cols].fillna(0)  # Handle any remaining NaNs
            variance_mask = variance_selector.fit_transform(feature_data)
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) 
                               if variance_selector.get_support()[i]]
            
            # Keep all non-feature columns plus selected features
            non_feature_cols = [col for col in df.columns if col not in feature_cols]
            final_columns = non_feature_cols + selected_features
            
            df_selected = df[final_columns]
            
            self.logger.info(
                f"Feature selection for {currency_pair}: {len(feature_cols)} -> {len(selected_features)}"
            )
            
            return df_selected
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed for {currency_pair}: {str(e)}")
            return df


class DataProcessor:
    """Main data processing orchestrator."""
    
    def __init__(
        self,
        data_config: DataConfig,
        processing_config: ProcessingConfig,
        logger: Optional[MLLogger] = None
    ):
        """
        Initialize data processor.
        
        Args:
            data_config: Data configuration
            processing_config: Processing configuration
            logger: Optional logger instance
        """
        self.data_config = data_config
        self.processing_config = processing_config
        self.logger = logger or MLLogger("DataProcessor")
        
        self.validator = DataValidator(data_config, logger)
        self.feature_engineer = FeatureEngineer(data_config, processing_config, logger)
    
    def process_currency_data(
        self,
        df: pd.DataFrame,
        currency_pair: str
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Complete data processing pipeline for a currency pair.
        
        Args:
            df: Input dataframe
            currency_pair: Currency pair identifier
            
        Returns:
            Tuple of (processed_dataframe, processing_metadata)
        """
        metadata = {
            'currency_pair': currency_pair,
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'validation_result': None,
            'feature_engineering_result': None,
            'success': False
        }
        
        try:
            with self.logger.log_operation(f"Processing {currency_pair}"):
                # 1. Validation
                validation_result = self.validator.validate_dataframe(df, currency_pair)
                metadata['validation_result'] = validation_result
                
                if not validation_result.is_valid:
                    self.logger.error(f"Validation failed for {currency_pair}")
                    return None, metadata
                
                # 2. Feature Engineering
                feature_result = self.feature_engineer.engineer_features(df, currency_pair)
                metadata['feature_engineering_result'] = feature_result
                
                # 3. Final validation
                if len(feature_result.data) < self.data_config.min_records_after_cleaning:
                    self.logger.warning(
                        f"Insufficient data after processing for {currency_pair}: "
                        f"{len(feature_result.data)} < {self.data_config.min_records_after_cleaning}"
                    )
                    return None, metadata
                
                metadata['success'] = True
                self.logger.info(f"Successfully processed {currency_pair}")
                
                return feature_result.data, metadata
                
        except Exception as e:
            self.logger.error(f"Processing failed for {currency_pair}", exception=e)
            metadata['error'] = str(e)
            return None, metadata


def load_and_validate_data(
    file_path: Union[str, Path],
    logger: Optional[MLLogger] = None
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load and perform basic validation on data file.
    
    Args:
        file_path: Path to data file
        logger: Optional logger instance
        
    Returns:
        Tuple of (dataframe, metadata)
    """
    if logger is None:
        logger = MLLogger("DataLoader")
    
    metadata = {
        'file_path': str(file_path),
        'load_timestamp': pd.Timestamp.now().isoformat(),
        'success': False
    }
    
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            metadata['error'] = "File not found"
            return None, metadata
        
        # Load based on file extension
        if file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            metadata['error'] = f"Unsupported file format: {file_path.suffix}"
            return None, metadata
        
        # Basic validation
        if df.empty:
            logger.error("Loaded dataframe is empty")
            metadata['error'] = "Empty dataframe"
            return None, metadata
        
        metadata.update({
            'success': True,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        })
        
        logger.info(f"Successfully loaded data from {file_path}", extra=metadata)
        
        return df, metadata
        
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}", exception=e)
        metadata['error'] = str(e)
        return None, metadata 