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
        
        # Import feature engineer here to avoid circular imports
        from utils.feature_engineering import FeatureEngineer
        
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
            # 1. Validate data
            validation_result = self.validator.validate_dataframe(df, currency_pair)
            metadata['validation_result'] = validation_result.__dict__
            
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Data validation failed for {currency_pair}",
                    extra={'errors': validation_result.errors}
                )
                return None, metadata
            
            # 2. Engineer features
            feature_result = self.feature_engineer.engineer_features(df, currency_pair)
            metadata['feature_engineering_result'] = feature_result.__dict__
            
            if feature_result.data is None or feature_result.data.empty:
                self.logger.warning(f"Feature engineering failed for {currency_pair}")
                return None, metadata
            
            metadata['success'] = True
            return feature_result.data, metadata
            
        except Exception as e:
            self.logger.error(f"Processing failed for {currency_pair}", exception=e)
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