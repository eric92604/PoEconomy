"""
Comprehensive data processing utilities for ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

from ml.config.training_config import DataConfig, ProcessingConfig, MLConfig
from ml.utils.common_utils import MLLogger
from ml.utils.data_sources import create_data_source, DataSourceConfig, BaseDataSource




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
        from ml.utils.feature_engineering import FeatureEngineer
        
        self.feature_engineer = FeatureEngineer(data_config, processing_config, logger)
    
    def _clean_infinity_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean infinity and extreme values from dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Replace infinity values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Count infinity values before cleaning
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        if inf_count > 0:
            self.logger.warning(f"Found and replaced {inf_count} infinity values with NaN")
        
        # For numeric columns, cap extreme values to prevent overflow
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                # Cap values to reasonable range (e.g., 1e10 to -1e10)
                df[col] = df[col].clip(lower=-1e10, upper=1e10)
        
        # Log cleaning results
        if inf_count > 0:
            self.logger.info(f"Infinity value cleaning completed: {inf_count} values replaced")
        
        return df
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by removing or fixing invalid prices.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        if 'price' not in df.columns:
            return df
        
        original_len = len(df)
        
        # Remove rows with non-positive prices
        df_clean = df[df['price'] > 0].copy()
        
        removed_count = original_len - len(df_clean)
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} rows with non-positive prices ({removed_count/original_len:.2%})")
        
        return df_clean
    
    def process_currency_data(
        self,
        df: pd.DataFrame,
        currency: str
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Complete data processing pipeline for a currency.
        
        Args:
            df: Input dataframe
            currency: Currency identifier
            
        Returns:
            Tuple of (processed_dataframe, processing_metadata)
        """
        metadata = {
            'currency': currency,
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'validation_result': None,
            'feature_engineering_result': None,
            'success': False
        }
        
        try:
            # 1. Basic data validation
            if df.empty:
                self.logger.warning(f"Empty dataframe for {currency}")
                return None, metadata
            
            # 2. Clean price data
            cleaned_df = self._clean_price_data(df)
            
            # 3. Engineer features
            feature_result = self.feature_engineer.engineer_features(cleaned_df, currency)
            metadata['feature_engineering_result'] = feature_result.__dict__
            
            if feature_result.data is None or feature_result.data.empty:
                self.logger.warning(f"Feature engineering failed for {currency}")
                return None, metadata
            
            # 3. Clean infinity and extreme values
            cleaned_data = self._clean_infinity_values(feature_result.data)
            
            metadata['success'] = True
            return cleaned_data, metadata
            
        except Exception as e:
            self.logger.error(f"Processing failed for {currency}", exception=e)
            return None, metadata


# ------------------------------------------------------------------
# Currency Selection Utilities
# ------------------------------------------------------------------



def generate_all_currencies_list(
    data_source: Optional[BaseDataSource] = None,
    config: Optional[MLConfig] = None,
    min_avg_value: float = 1.0,
    min_records: int = 100,
    filter_by_availability: bool = True,
    only_available_currencies: bool = True,
    availability_check_days: int = 30,
) -> List[Dict[str, Any]]:
    """
    Return all currency pairs that satisfy the supplied filters.

    Args:
        data_source: Optional injected data source instance.
        config: MLConfig used to construct a data source when one is not provided.
        min_avg_value: Minimum average value (Chaos Orbs) required.
        min_records: Minimum number of price observations required.
        filter_by_availability: Whether to consider availability metadata.
        only_available_currencies: If True, exclude currencies that are currently unavailable.
        availability_check_days: Maximum age (in days) for availability information.
    """
    if data_source is None:
        if config is None:
            config = MLConfig()
        data_source_config = DataSourceConfig.from_dynamo_config(config.dynamo)
        data_source = create_data_source(data_source_config)
    source = data_source
    return source.select_currencies(
        min_avg_value=min_avg_value,
        min_records=min_records,
        filter_by_availability=filter_by_availability,
        only_available=only_available_currencies,
        availability_cutoff_days=availability_check_days,
    )


def generate_target_currency_list(
    data_source: Optional[BaseDataSource] = None,
    config: Optional[MLConfig] = None,
    filter_by_availability: bool = True,
    only_available_currencies: bool = True,
    availability_check_days: int = 30,
    min_avg_value: float = 5.0,
    min_records: int = 100,
) -> List[Dict[str, Any]]:
    """
    Return the prioritized list of high-value currencies for training.

    This function mirrors the previous behaviour by defaulting to a
    higher minimum average value (5 Chaos Orbs) while still allowing
    callers to override thresholds.
    """
    return generate_all_currencies_list(
        data_source=data_source,
        config=config,
        min_avg_value=min_avg_value,
        min_records=min_records,
        filter_by_availability=filter_by_availability,
        only_available_currencies=only_available_currencies,
        availability_check_days=availability_check_days,
    ) 