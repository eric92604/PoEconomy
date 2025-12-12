"""
Unified data source architecture for ML pipeline.

This module provides a clean, unified interface for accessing currency data
from multiple sources (DynamoDB, S3, local files) with a consistent API.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr, Key

from ml.config.training_config import DynamoConfig
from ml.utils.common_utils import MLLogger


@dataclass
class CurrencyStat:
    """Lightweight representation of currency metadata."""

    currency: str
    league: str
    avg_price: float
    median_price: float
    min_price: float
    max_price: float
    volatility: float
    total_records: int
    is_available: bool
    availability_source: Optional[str] = None
    last_availability_check: Optional[str] = None
    last_updated: Optional[str] = None

    @property
    def pay_currency(self) -> str:
        """Return the pay currency, defaulting to Chaos Orb."""
        return "Chaos Orb"


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    
    # Source type
    source_type: str  # 'dynamodb', 's3', 'local'
    
    # DynamoDB config
    region_name: str = "us-west-2"
    currency_metadata_table: Optional[str] = None
    currency_prices_table: Optional[str] = None
    league_metadata_table: Optional[str] = None
    
    # S3 config
    data_lake_bucket: Optional[str] = None
    historical_data_prefix: str = "historical-data/"
    processed_data_prefix: str = "processed-data/"
    
    # Local config
    local_data_path: Optional[str] = None
    
    @classmethod
    def from_dynamo_config(cls, dynamo_config: DynamoConfig) -> 'DataSourceConfig':
        """Create from existing DynamoConfig."""
        return cls(
            source_type='dynamodb',
            region_name=dynamo_config.region_name,
            currency_metadata_table=dynamo_config.currency_metadata_table,
            currency_prices_table=dynamo_config.currency_prices_table,
            league_metadata_table=dynamo_config.league_metadata_table
        )
    
    @classmethod
    def from_s3_config(cls, s3_config: Dict[str, Any]) -> 'DataSourceConfig':
        """Create from S3 configuration dictionary."""
        return cls(
            source_type='s3',
            region_name=s3_config.get('region_name', 'us-west-2'),
            data_lake_bucket=s3_config.get('data_lake_bucket'),
            historical_data_prefix=s3_config.get('historical_data_prefix', 'historical-data/'),
            processed_data_prefix=s3_config.get('processed_data_prefix', 'processed-data/')
        )
    
    @classmethod
    def from_local_config(cls, local_path: str) -> 'DataSourceConfig':
        """Create for local file access."""
        return cls(
            source_type='local',
            local_data_path=local_path
        )
    


class BaseDataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, config: DataSourceConfig, logger: Optional[MLLogger] = None):
        self.config = config
        self.logger = logger or MLLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def list_currency_stats(self) -> List[CurrencyStat]:
        """Get currency statistics."""
        pass
    
    def load_processed_parquet_data(
        self,
        data_lake_bucket: str,
        experiment_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load processed parquet data from S3 bucket.

        Default implementation returns None. Subclasses should override this method.

        Args:
            data_lake_bucket: S3 bucket containing processed data
            experiment_id: Optional experiment ID to load specific processed data

        Returns:
            DataFrame containing processed data, or None if not found
        """
        self.logger.warning("load_processed_parquet_data not implemented for this data source type")
        return None
    
    def load_processed_parquet_data_with_experiment_id(
        self,
        data_lake_bucket: str,
        experiment_id: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load processed parquet data from S3 bucket and return both data and experiment ID.

        Default implementation returns (None, None). Subclasses should override this method.

        Args:
            data_lake_bucket: S3 bucket containing processed data
            experiment_id: Optional experiment ID to load specific processed data

        Returns:
            Tuple of (DataFrame containing processed data, experiment_id) or (None, None) if not found
        """
        self.logger.warning("load_processed_parquet_data_with_experiment_id not implemented for this data source type")
        return None, None
    
    @abstractmethod
    def build_price_dataframe(
        self,
        currencies: Sequence[str],
        included_leagues: Optional[Sequence[str]] = None,
        max_league_days: int = 60,
        min_league_days: int = 0,
    ) -> Optional[pd.DataFrame]:
        """Build price dataframe for given currencies."""
        pass
    
    @abstractmethod
    def get_available_leagues(self) -> List[str]:
        """Get list of available leagues."""
        pass
    
    @abstractmethod
    def get_most_recent_league(self) -> Optional[str]:
        """Get the most recent league."""
        pass
    
    def select_currencies(
        self,
        min_avg_value: float,
        min_records: int,
        filter_by_availability: bool = True,
        only_available: bool = True,
        availability_cutoff_days: int = 7,
    ) -> List[Dict[str, Any]]:
        """Select currencies based on criteria."""
        stats = self.list_currency_stats()
        cutoff = datetime.now(timezone.utc) - timedelta(days=availability_cutoff_days)

        target_pairs: List[Dict[str, Any]] = []

        for stat in stats:
            if stat.total_records < min_records:
                continue
            if stat.avg_price < min_avg_value:
                continue

            availability_ok = True
            if filter_by_availability:
                availability_ok = stat.is_available
                if stat.last_availability_check:
                    last_check = self._coerce_datetime(stat.last_availability_check)
                    if last_check is not None and last_check < cutoff:
                        availability_ok = False
            if only_available and not availability_ok:
                continue

            target_pairs.append({
                "get_currency": stat.currency,
                "pay_currency": stat.pay_currency,
                "priority": 1,
                "min_value": stat.min_price,
                "median_value": stat.median_price,
                "avg_value": stat.avg_price,
                "max_value": stat.max_price,
                "volatility": stat.volatility,
                "records": stat.total_records,
                "is_available": stat.is_available if filter_by_availability else True,
                "availability_source": stat.availability_source,
                "last_availability_check": stat.last_availability_check,
            })

        target_pairs.sort(key=lambda x: (-x["avg_value"], x["get_currency"]))
        return target_pairs
    
    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        """Coerce various datetime formats to UTC datetime."""
        if value in (None, "", 0):
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if hasattr(value, '__float__'):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (ValueError, TypeError):
                return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    return None
        return None
    
    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Coerce value to float with default."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default




class DynamoDBDataSource(BaseDataSource):
    """DynamoDB-based data source."""
    
    def __init__(self, config: DataSourceConfig, logger: Optional[MLLogger] = None):
        super().__init__(config, logger)
        
        if config.source_type != 'dynamodb':
            raise ValueError("DynamoDBDataSource requires source_type='dynamodb'")
        
        session = boto3.session.Session(region_name=config.region_name)
        self._dynamodb = session.resource("dynamodb")
        
        self._prices_table = self._dynamodb.Table(config.currency_prices_table)
        self._metadata_table = self._dynamodb.Table(config.currency_metadata_table)
        self._league_table = (
            self._dynamodb.Table(config.league_metadata_table)
            if config.league_metadata_table
            else None
        )
        
        # Cache
        self._league_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._currency_name_cache: Optional[List[str]] = None
    
    def _scan_table(self, table: Any, filter_expression: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Scan a DynamoDB table with pagination."""
        items: List[Dict[str, Any]] = []
        kwargs = {}
        if filter_expression is not None:
            kwargs["FilterExpression"] = filter_expression

        while True:
            response = table.scan(**kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key

        return items
    
    def _load_league_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load league metadata from DynamoDB (cached)."""
        if self._league_cache is not None:
            return self._league_cache

        metadata: Dict[str, Dict[str, Any]] = {}

        if not self._league_table:
            self.logger.warning("League metadata table not configured; league features may be limited.")
            self._league_cache = metadata
            return metadata

        items = self._scan_table(self._league_table)
        for item in items:
            league_name = item.get("league") or item.get("name")
            if not league_name:
                continue

            start_raw = item.get("startDate") or item.get("start_date")
            end_raw = item.get("endDate") or item.get("end_date")

            metadata[league_name] = {
                "start": self._coerce_datetime(start_raw),
                "end": self._coerce_datetime(end_raw),
                "is_active": bool(item.get("isActive", False)),
            }

        self._league_cache = metadata
        return metadata
    
    def _count_price_records(self, currency: str, league: str) -> int:
        """Count price records for a currency in a league."""
        try:
            currency_league = f"{currency}#{league}"
            response = self._prices_table.query(
                KeyConditionExpression=Key("currency_league").eq(currency_league),
                Select="COUNT"
            )
            return int(response.get('Count', 0))
        except Exception as e:
            self.logger.warning(f"Failed to count price records for {currency} in {league}: {e}")
            return 0
    
    def list_currency_stats(self) -> List[CurrencyStat]:
        """Return aggregated currency statistics from the metadata table."""
        items = self._scan_table(self._metadata_table)
        stats: List[CurrencyStat] = []

        for item in items:
            currency = item.get("currency_name") or item.get("currency")
            league = item.get("league") or item.get("league_name") or "Unknown"
            if not currency:
                continue

            metadata_blob = item.get("metadata_json")
            metadata: Dict[str, Any] = {}
            if isinstance(metadata_blob, str):
                try:
                    metadata = json.loads(metadata_blob)
                except json.JSONDecodeError:
                    metadata = {}
            elif isinstance(metadata_blob, dict):
                metadata = metadata_blob

            avg_price = self._coerce_float(
                metadata.get("avg_price_chaos")
                or metadata.get("avg_price")
                or metadata.get("average_price")
                or item.get("avg_price")
            )
            median_price = self._coerce_float(
                metadata.get("median_price_chaos")
                or metadata.get("median_price")
                or item.get("median_price")
            )
            min_price = self._coerce_float(
                metadata.get("min_price_chaos")
                or metadata.get("min_price")
                or item.get("min_price")
            )
            max_price = self._coerce_float(
                metadata.get("max_price_chaos")
                or metadata.get("max_price")
                or item.get("max_price")
            )
            volatility = self._coerce_float(
                metadata.get("volatility")
                or metadata.get("price_volatility")
                or item.get("volatility")
            )
            
            total_records = self._count_price_records(currency, league)
            is_available = bool(
                metadata.get("is_available")
                if "is_available" in metadata
                else item.get("is_available", item.get("availability", True))
            )

            stats.append(
                CurrencyStat(
                    currency=currency,
                    league=league,
                    avg_price=avg_price,
                    median_price=median_price,
                    min_price=min_price,
                    max_price=max_price,
                    volatility=volatility,
                    total_records=total_records,
                    is_available=is_available,
                    availability_source=item.get("availability_source") or metadata.get("availability_source"),
                    last_availability_check=item.get("last_availability_check")
                    or metadata.get("last_availability_check"),
                    last_updated=item.get("last_updated") or metadata.get("last_updated"),
                )
            )

        return stats
    
    def get_available_leagues(self) -> List[str]:
        """Get list of available leagues."""
        league_metadata = self._load_league_metadata()
        return list(league_metadata.keys())
    
    def get_most_recent_league(self) -> Optional[str]:
        """Get the most recent active seasonal league."""
        league_metadata = self._load_league_metadata()
        if not league_metadata:
            return None
        
        # First, try to get the current active seasonal league from league metadata
        # Look for leagues with league_type = "seasonal" and is_active = True
        active_seasonal_leagues = []
        
        # We need to check the actual league metadata table for league_type
        if self._league_table:
            try:
                items = self._scan_table(self._league_table)
                for item in items:
                    league_name = item.get("league") or item.get("name")
                    league_type = item.get("league_type", "").lower()
                    is_active = item.get("isActive", False)
                    
                    # Filter out leagues with "Event" in the name
                    if league_name and "Event" in league_name:
                        continue
                    
                    if league_name and league_type == "seasonal" and is_active:
                        start_raw = item.get("startDate") or item.get("start_date")
                        start_date = self._coerce_datetime(start_raw)
                        if start_date:
                            active_seasonal_leagues.append((league_name, start_date))
                
                if active_seasonal_leagues:
                    # Return the seasonal league with the latest start date
                    latest_seasonal = max(active_seasonal_leagues, key=lambda x: x[1])
                    self.logger.info(f"Found current active seasonal league: {latest_seasonal[0]} (started: {latest_seasonal[1]})")
                    return latest_seasonal[0]
                    
            except Exception as e:
                self.logger.warning(f"Failed to get seasonal league from metadata table: {e}")
        
        # Fallback: try to find the most recent league from currency stats
        stats = self.list_currency_stats()
        if stats:
            best_league = None
            best_timestamp = None

            for stat in stats:
                # Filter out leagues with "Event" in the name
                if stat.league and "Event" in stat.league:
                    continue
                    
                timestamp = stat.last_updated or stat.last_availability_check
                dt = self._coerce_datetime(timestamp) if timestamp else None
                if dt is None:
                    continue
                if best_timestamp is None or dt > best_timestamp:
                    best_timestamp = dt
                    best_league = stat.league

            if best_league:
                self.logger.info(f"Using most recent league from currency stats: {best_league}")
                return best_league

        # Final fallback: use the league with the latest start date from metadata
        if league_metadata:
            # Filter out leagues with "Event" in the name
            filtered_metadata = {
                league: metadata for league, metadata in league_metadata.items()
                if league and "Event" not in league
            }
            if filtered_metadata:
                latest_league = max(
                    filtered_metadata.items(),
                    key=lambda item: item[1].get("start") or datetime.min,
                )[0]
                self.logger.info(f"Using fallback league with latest start date: {latest_league}")
                return latest_league
        
        return None
    
    def build_price_dataframe(
        self,
        currencies: Sequence[str],
        included_leagues: Optional[Sequence[str]] = None,
        max_league_days: int = 60,
        min_league_days: int = 0,
    ) -> Optional[pd.DataFrame]:
        """Fetch price history for the supplied currencies."""
        frames: List[pd.DataFrame] = []
        league_metadata = self._load_league_metadata()

        for currency in currencies:
            df = self._fetch_currency_prices(
                currency=currency,
                pay_currency="Chaos Orb",  # Always use Chaos Orb as pay currency
                included_leagues=included_leagues,
                max_league_days=max_league_days,
                min_league_days=min_league_days,
                league_metadata=league_metadata,
            )
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["league_start", "date"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined
    
    def _fetch_currency_prices(
        self,
        currency: str,
        pay_currency: str,
        included_leagues: Optional[Sequence[str]],
        max_league_days: int,
        min_league_days: int,
        league_metadata: Dict[str, Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        """Fetch price history for a single currency/pair."""
        items = []
        
        if included_leagues:
            for league in included_leagues:
                currency_league = f"{currency}#{league}"
                try:
                    response = self._prices_table.query(
                        KeyConditionExpression=Key("currency_league").eq(currency_league),
                        FilterExpression=Attr("pay_currency").eq(pay_currency)
                    )
                    items.extend(response.get('Items', []))
                except Exception as e:
                    self.logger.warning(f"Failed to query {currency_league}: {e}")
        else:
            filter_expression = Attr("currency").eq(currency) & Attr("pay_currency").eq(pay_currency)
            items = self._scan_table(self._prices_table, filter_expression=filter_expression)
        
        if not items:
            self.logger.warning(f"No DynamoDB price records found for {currency}")
            return None

        records: List[Dict[str, Any]] = []

        for item in items:
            league_name = item.get("league") or "Unknown"
            league_info = league_metadata.get(league_name, {})

            league_start = league_info.get("start")
            if league_start is None:
                continue

            league_end = league_info.get("end")
            league_is_active = league_info.get("is_active", False)

            timestamp_raw = item.get("timestamp")
            if timestamp_raw is None:
                continue

            trade_datetime = self._coerce_datetime(timestamp_raw)
            if trade_datetime is None:
                continue

            league_day = (trade_datetime - league_start).days
            self.logger.debug(f"League day calculation for {league_name}: trade_datetime={trade_datetime}, league_start={league_start}, league_day={league_day}, max_league_days={max_league_days}")
            
            if league_day < 0 or league_day > max_league_days:
                self.logger.debug(f"Skipping {league_name} record: league_day={league_day} outside range [0, {max_league_days}]")
                continue

            if league_end is not None:
                league_length = (league_end - league_start).days
                if league_length < min_league_days:
                    self.logger.debug(f"Skipping {league_name} record: league_length={league_length} < min_league_days={min_league_days}")
                    continue

            price = self._coerce_float(item.get("price"))
            records.append({
                "currency": currency,
                "get_currency": currency,
                "price": price,
                "date": trade_datetime,
                "league_name": league_name,
                "league_start": league_start,
                "league_end": league_end,
                "league_active": league_is_active,
                "league_day": league_day,
            })

        if not records:
            self.logger.warning(f"No qualifying records for {currency} after filtering.")
            return None

        return pd.DataFrame.from_records(records)


class S3DataSource(BaseDataSource):
    """S3-based data source for historical data."""
    
    def __init__(self, config: DataSourceConfig, logger: Optional[MLLogger] = None):
        super().__init__(config, logger)
        
        if config.source_type != 's3':
            raise ValueError("S3DataSource requires source_type='s3'")
        
        if not config.data_lake_bucket:
            raise ValueError("data_lake_bucket must be specified for S3DataSource")
        
        self.s3_client = boto3.client('s3', region_name=config.region_name)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._metadata_cache: Optional[List[CurrencyStat]] = None
    
    def list_currency_stats(self) -> List[CurrencyStat]:
        """Get currency statistics from S3 historical data."""
        if self._metadata_cache is not None:
            return self._metadata_cache
            
        self.logger.debug("Loading currency metadata from S3...")
        
        all_data = self._load_all_historical_data()
        
        if all_data.empty:
            self.logger.warning("No historical data found in S3")
            return []
        
        stats = []
        for currency in all_data['Get'].unique():
            currency_data = all_data[all_data['Get'] == currency]
            
            total_records = len(currency_data)
            avg_price = currency_data['Value'].mean() if 'Value' in currency_data.columns else 0.0
            median_price = currency_data['Value'].median() if 'Value' in currency_data.columns else 0.0
            min_price = currency_data['Value'].min() if 'Value' in currency_data.columns else 0.0
            max_price = currency_data['Value'].max() if 'Value' in currency_data.columns else 0.0
            volatility = currency_data['Value'].std() if 'Value' in currency_data.columns else 0.0
            
            latest_date = currency_data['Date'].max() if 'Date' in currency_data.columns else None
            if latest_date is not None:
                if isinstance(latest_date, str):
                    latest_date = pd.to_datetime(latest_date)
                if latest_date.tz is None:
                    latest_date = latest_date.tz_localize('UTC')
                is_available = True
            else:
                is_available = False
            
            stats.append(CurrencyStat(
                currency=currency,
                league="Historical",  # S3 data spans multiple leagues
                avg_price=float(avg_price),
                median_price=float(median_price),
                min_price=float(min_price),
                max_price=float(max_price),
                volatility=float(volatility),
                total_records=total_records,
                is_available=is_available,
                last_availability_check=latest_date.isoformat() if latest_date else None
            ))
        
        self._metadata_cache = stats
        self.logger.debug(f"Loaded statistics for {len(stats)} currencies")
        return stats
    
    def get_available_leagues(self) -> List[str]:
        """Get list of available leagues from S3."""
        all_data = self._load_all_historical_data()
        
        if all_data.empty:
            return []
        
        available_leagues = all_data['League'].unique().tolist()
        self.logger.debug(f"Discovered {len(available_leagues)} leagues in S3: {available_leagues}")
        return list(available_leagues)
    
    def get_most_recent_league(self) -> Optional[str]:
        """Get the most recent league from S3 data."""
        all_data = self._load_all_historical_data()
        
        if all_data.empty:
            return None
        
        if 'Date' in all_data.columns:
            latest_dates = all_data.groupby('League')['Date'].max()
            # Filter out leagues with "Event" in the name
            filtered_dates = latest_dates[
                ~latest_dates.index.str.contains("Event", case=False, na=False)
            ]
            if not filtered_dates.empty:
                most_recent_league = filtered_dates.idxmax()
                return str(most_recent_league)
        
        return None
    
    def build_price_dataframe(
        self,
        currencies: Sequence[str],
        included_leagues: Optional[Sequence[str]] = None,
        max_league_days: int = 60,
        min_league_days: int = 0,
    ) -> Optional[pd.DataFrame]:
        """Build price dataframe from S3 historical data."""
        self.logger.debug(f"Building price dataframe for {len(currencies)} currencies")
        
        all_data = self._load_all_historical_data()
        
        if all_data.empty:
            self.logger.warning("No historical data found in S3")
            return None
        
        # Filter by included leagues
        if included_leagues:
            all_data = all_data[all_data['League'].isin(included_leagues)]
            self.logger.debug(f"Filtered to leagues: {included_leagues}")
        else:
            available_leagues = all_data['League'].unique().tolist()
            self.logger.debug(f"Using all available leagues from S3: {available_leagues}")
        
        # Filter by currencies
        all_data = all_data[all_data['Get'].isin(currencies)]
        
        if all_data.empty:
            self.logger.warning("No data found for target currencies")
            return None
        
        # Standardize column names
        df = all_data.rename(columns={
            'Get': 'currency',
            'Pay': 'pay_currency',
            'Value': 'price',
            'Date': 'date',
            'League': 'league_name'
        })
        
        # Ensure required columns
        required_columns = ['currency', 'pay_currency', 'price', 'date', 'league_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values(['currency', 'date']).reset_index(drop=True)
        
        self.logger.debug(f"Built price dataframe: {df.shape}")
        return df
    
    def _load_all_historical_data(self) -> pd.DataFrame:
        """Load all historical data from S3."""
        cache_key = "all_historical_data"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        self.logger.debug("Loading all historical data from S3...")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.data_lake_bucket,
                Prefix=self.config.historical_data_prefix
            )
            
            csv_files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.currency.csv'):
                        csv_files.append(obj['Key'])
            
            if not csv_files:
                self.logger.warning("No currency CSV files found in historical-data/")
                return pd.DataFrame()
            
            self.logger.debug(f"Found {len(csv_files)} CSV files to process")
            
            all_dataframes = []
            
            for csv_key in csv_files:
                try:
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    self.s3_client.download_file(self.config.data_lake_bucket, csv_key, temp_path)
                    
                    df = pd.read_csv(temp_path, sep=';', low_memory=False, dtype={
                        'Get': 'string',
                        'Pay': 'string',
                        'Value': 'float64',
                        'Date': 'string',
                        'League': 'string'
                    })
                    
                    # Add league information from filename
                    league_name = Path(csv_key).stem.replace('.currency', '')
                    df['League'] = league_name
                    
                    all_dataframes.append(df)
                    os.unlink(temp_path)
                    
                    self.logger.debug(f"Loaded {len(df)} records from {csv_key}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load {csv_key}: {e}")
                    continue
            
            if not all_dataframes:
                self.logger.error("Failed to load any CSV files")
                return pd.DataFrame()
            
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            self._data_cache[cache_key] = combined_df
            
            self.logger.debug(f"Loaded {len(combined_df)} total records from {len(all_dataframes)} files")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data from S3: {e}")
            return pd.DataFrame()
    
    def load_processed_parquet_data(
        self,
        data_lake_bucket: str,
        experiment_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load processed parquet data from S3 data lake bucket.
        
        Args:
            data_lake_bucket: S3 bucket containing processed data
            experiment_id: Optional experiment ID to load specific processed data
            
        Returns:
            DataFrame containing processed data, or None if not found
        """
        try:
            import tempfile
            import os
            
            # Construct the S3 key for processed data
            if experiment_id:
                s3_key = f"processed_data/combined_currency_features_{experiment_id}.parquet"
            else:
                # Try to find the most recent processed data file
                response = self.s3_client.list_objects_v2(
                    Bucket=data_lake_bucket,
                    Prefix="processed_data/"
                )
                
                if 'Contents' not in response:
                    self.logger.info("No processed parquet files found in S3")
                    return None
                
                # Get the most recent parquet file
                parquet_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
                if not parquet_files:
                    return None
                
                # Sort by last modified and get the most recent
                parquet_files.sort(key=lambda x: response['Contents'][next(i for i, obj in enumerate(response['Contents']) if obj['Key'] == x)]['LastModified'], reverse=True)
                s3_key = parquet_files[0]
            
            self.logger.debug(f"Loading processed data from s3://{data_lake_bucket}/{s3_key}")
            
            # Download and load the parquet file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                temp_path = temp_file.name
            
            self.s3_client.download_file(data_lake_bucket, s3_key, temp_path)
            df = pd.read_parquet(temp_path)
            os.unlink(temp_path)
            
            self.logger.debug(f"Successfully loaded processed data: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to load processed parquet data from S3: {e}")
            return None
    
    def load_processed_parquet_data_with_experiment_id(
        self,
        data_lake_bucket: str,
        experiment_id: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load processed parquet data from S3 data lake bucket and return both data and experiment ID.

        Args:
            data_lake_bucket: S3 bucket containing processed data
            experiment_id: Optional experiment ID to load specific processed data

        Returns:
            Tuple of (DataFrame containing processed data, experiment_id) or (None, None) if not found
        """
        try:
            import tempfile
            import os
            
            # Construct the S3 key for processed data
            if experiment_id:
                s3_key = f"processed_data/combined_currency_features_{experiment_id}.parquet"
                found_experiment_id = experiment_id
            else:
                # Try to find the most recent processed data file
                response = self.s3_client.list_objects_v2(
                    Bucket=data_lake_bucket,
                    Prefix="processed_data/"
                )
                
                if 'Contents' not in response:
                    self.logger.info("No processed parquet files found in S3")
                    return None, None

                # Get the most recent parquet file
                parquet_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
                if not parquet_files:
                    return None, None

                # Sort by last modified and get the most recent
                parquet_files.sort(key=lambda x: x['LastModified'], reverse=True)
                s3_key = parquet_files[0]['Key']
                
                # Extract experiment ID from filename
                filename = os.path.basename(s3_key)
                if filename.startswith('combined_currency_features_') and filename.endswith('.parquet'):
                    # Prefix 'combined_currency_features_' is 27 characters
                    found_experiment_id = filename[27:-8]  # Remove prefix and suffix
                else:
                    found_experiment_id = "unknown"

            self.logger.debug(f"Loading processed data from s3://{data_lake_bucket}/{s3_key}")

            # Download and load the parquet file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                temp_path = temp_file.name

            self.s3_client.download_file(data_lake_bucket, s3_key, temp_path)
            df = pd.read_parquet(temp_path)
            os.unlink(temp_path)

            self.logger.debug(f"Successfully loaded processed data: {df.shape}")
            return df, found_experiment_id

        except Exception as e:
            self.logger.warning(f"Failed to load processed parquet data from S3: {e}")
            return None, None




def create_data_source(config: DataSourceConfig, logger: Optional[MLLogger] = None) -> BaseDataSource:
    """Factory function to create appropriate data source."""
    if config.source_type == 'dynamodb':
        return DynamoDBDataSource(config, logger)
    elif config.source_type == 's3':
        return S3DataSource(config, logger)
    elif config.source_type == 'local':
        return S3DataSource(config, logger)  # Use S3DataSource for local files
    else:
        raise ValueError(f"Unknown source type: {config.source_type}. Supported types: 'dynamodb', 's3', 'local'")


# Convenience functions for backward compatibility
def create_dynamo_data_source(dynamo_config: DynamoConfig, logger: Optional[MLLogger] = None) -> DynamoDBDataSource:
    """Create DynamoDB data source from DynamoConfig."""
    config = DataSourceConfig.from_dynamo_config(dynamo_config)
    return DynamoDBDataSource(config, logger)


def create_s3_data_source(s3_config: Dict[str, Any], logger: Optional[MLLogger] = None) -> S3DataSource:
    """Create S3 data source from config dictionary."""
    config = DataSourceConfig.from_s3_config(s3_config)
    return S3DataSource(config, logger)




__all__ = [
    "BaseDataSource",
    "DynamoDBDataSource", 
    "S3DataSource",
    "DataSourceConfig",
    "CurrencyStat",
    "create_data_source",
    "create_dynamo_data_source",
    "create_s3_data_source",
]
