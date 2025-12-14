#!/usr/bin/env python3
"""
Prediction refresh service.

Intended to run as an AWS Lambda handler (or CLI utility) that reads the latest
currency prices from DynamoDB, generates fresh predictions using the existing
trained models, and writes the results back to the predictions cache table.

Event structure examples:
{
  "currencies": ["Divine Orb", "Chaos Orb"],
  "horizons": ["1d", "3d"],
  "ttl_hours": 12
}
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import boto3
from boto3.dynamodb.conditions import Key

from ml.config.inference_config import get_inference_config_from_env, InferenceConfig
from ml.utils.data_sources import create_data_source, DataSourceConfig, BaseDataSource
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from ml.utils.common_utils import MLLogger
from ml.utils.data_processing import DataProcessor
from ml.utils.model_inference import (
    ModelPredictor,
    PredictionResult,
    _split_currency_label, 
    CurrencyModelBundle, 
    HORIZON_SUFFIXES
)


DEFAULT_HORIZONS = ("1d", "3d", "7d")


class DirectModelPredictor(ModelPredictor):
    """
    Direct version of ModelPredictor that loads models directly from the filesystem
    without zip files, using Lambda code storage for fast access.
    """
    
    def __init__(
        self,
        models_dir: Path | str,
        config: Optional[InferenceConfig] = None,
        logger: Optional[MLLogger] = None,
        data_source: Optional[BaseDataSource] = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.config = config or get_inference_config_from_env()
        self.logger = logger or MLLogger("DirectModelPredictor")
        if data_source is None:
            data_source_config = DataSourceConfig.from_dynamo_config(self.config.dynamo)
            self.data_source = create_data_source(data_source_config, self.logger)
        else:
            self.data_source = data_source

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        self.data_processor = DataProcessor(self.config.data, self.config.processing, self.logger)  # type: ignore[arg-type]
        
        # Find model directories for direct loading
        # Models are in /var/task/models/currency/ (from Lambda code storage)
        currency_dir = self.models_dir / "currency"
        if currency_dir.exists():
            self.model_dirs = [d for d in currency_dir.iterdir() if d.is_dir()]
            self.logger.info(f"Found {len(self.model_dirs)} model directories in currency folder")
        else:
            # Fallback: look directly in models_dir
            self.model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
            self.logger.info(f"Found {len(self.model_dirs)} model directories in models folder")
        
        if not self.model_dirs:
            raise FileNotFoundError(f"No model directories found in models directory: {self.models_dir}")
        
        self.model_registry = self._discover_models_from_directories()
        
        
        # Cache most recent league per invocation (doesn't change during Lambda execution)
        self._cached_most_recent_league: Optional[str] = None
    
    def _discover_models_from_directories(self) -> Dict[str, CurrencyModelBundle]:
        """Scan model directories and group artifacts per currency."""
        registry: Dict[str, CurrencyModelBundle] = {}
        
        for model_dir in self.model_dirs:
            try:
                # Find all metadata files in the directory
                metadata_files = list(model_dir.glob("**/model_metadata.json"))
                
                for metadata_file in metadata_files:
                    try:
                        # Read metadata from file
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        currency_label = metadata.get("currency")
                        if not currency_label:
                            continue
                        
                        # Check if corresponding model file exists
                        model_file = metadata_file.parent / "ensemble_model.pkl"
                        scaler_file = metadata_file.parent / "scaler.pkl"
                        
                        if not model_file.exists():
                            self.logger.warning(f"Model file missing for {currency_label}: {model_file}")
                            continue
                        
                        # Log successful model discovery for debugging
                        self.logger.debug(f"Discovered model: {currency_label} -> {model_file.parent}")
                        
                        # Create direct artifact
                        imputer_file = metadata_file.parent / "imputer.pkl"
                        artifact = DirectModelArtifact(
                            model_dir=metadata_file.parent,
                            scaler_path=scaler_file if scaler_file.exists() else None,
                            imputer_path=imputer_file if imputer_file.exists() else None,
                            metadata_path=metadata_file,
                            metadata=metadata,
                        )
                        
                        base_currency, suffix = _split_currency_label(currency_label)
                        bundle = registry.setdefault(base_currency, CurrencyModelBundle(primary=None, horizons={}))
                        if suffix and suffix in HORIZON_SUFFIXES:
                            bundle.horizons[suffix] = artifact
                        else:
                            bundle.primary = artifact
                            
                    except Exception as exc:
                        self.logger.warning(f"Failed to process metadata {metadata_file}: {exc}")
                        continue
                        
            except Exception as exc:
                self.logger.warning(f"Failed to read model directory {model_dir}: {exc}")
                continue
        
        self.logger.info(f"Discovered {len(registry)} currencies with models")
        
        # Handle case where no valid models were discovered
        if not registry:
            self.logger.error("No valid model artifacts discovered in any model directories")
            raise FileNotFoundError(
                f"No valid model artifacts found in models directory: {self.models_dir}. "
                f"Expected to find model_metadata.json files with corresponding ensemble_model.pkl files."
            )
        
        # Log details about discovered models for debugging
        for currency, bundle in registry.items():
            self.logger.info(f"Currency '{currency}': primary={bundle.primary is not None}, horizons={list(bundle.horizons.keys())}")
            # Log detailed information about each horizon model
            for horizon, artifact in bundle.horizons.items():
                self.logger.info(f"  Horizon '{horizon}': {artifact.model_dir}")
        
        return registry
    
    def _select_model(self, currency: str, horizon: str) -> Optional[DirectModelArtifact]:
        """Select the appropriate model artifact for the given currency and horizon."""
        self.logger.info(f"Selecting model for {currency} with horizon {horizon}")
        
        # Try exact currency name first
        bundle = self.model_registry.get(currency)
        self.logger.info(f"Bundle for exact currency '{currency}': {bundle is not None}")
        
        if bundle:
            self.logger.info(f"Found bundle for {currency}: primary={bundle.primary is not None}, horizons={list(bundle.horizons.keys())}")
        
        # If not found, try with horizon suffix (for models named like "Currency_1d")
        if not bundle:
            currency_with_horizon = f"{currency}_{horizon}"
            bundle = self.model_registry.get(currency_with_horizon)
            self.logger.info(f"Bundle for currency with horizon '{currency_with_horizon}': {bundle is not None}")
            if bundle:
                # If we found a bundle with horizon suffix, return the primary model
                self.logger.info(f"Found model for {currency} using key {currency_with_horizon}")
                return bundle.primary
        
        if not bundle:
            available_currencies = list(self.model_registry.keys())
            self.logger.warning(f"No models registered for {currency} or {currency}_{horizon}. Available currencies: {available_currencies[:10]}...")
            return None

        if horizon in bundle.horizons:
            self.logger.info(f"Found model for {currency} with horizon {horizon}")
            return bundle.horizons[horizon]

        if bundle.primary is not None:
            self.logger.info(f"Using primary model for {currency} (horizon {horizon} not available)")
            return bundle.primary

        # No model available for requested horizon
        self.logger.warning(f"No model available for {currency} horizon {horizon}")
        return None
    
    def _load_model_from_direct(self, artifact: DirectModelArtifact) -> Any:
        """Load a model directly from filesystem."""
        try:
            # Load model directly from file
            model = joblib.load(artifact.model_dir / "ensemble_model.pkl")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model from filesystem: {e}")
            raise
    
    def _load_scaler_from_direct(self, artifact: DirectModelArtifact) -> Optional[Any]:
        """Load a scaler directly from filesystem."""
        if not artifact.scaler_path:
            return None
        
        try:
            # Load scaler directly from file
            scaler = joblib.load(artifact.scaler_path)
            return scaler
        except Exception as e:
            self.logger.warning(f"Failed to load scaler from filesystem: {e}")
            return None
    
    def predict_currency(
        self,
        currency: str,
        horizon: str = "1d",
        pay_currency: str = "Chaos Orb",
        included_leagues: Optional[Sequence[str]] = None,
        days_back: Optional[int] = None,
    ) -> Optional[PredictionResult]:
        """
        Generate a prediction for a single currency using streaming model loading.
        """
        horizon = horizon.lower().replace(" ", "")
        
        # Try to use trained model first
        try:
            artifact = self._select_model(currency, horizon)
            if artifact is not None:
                self.logger.debug(f"Found artifact for {currency} ({horizon}): {artifact.model_dir.name}")
                
                # Verify model files exist before proceeding
                model_file = artifact.model_dir / "ensemble_model.pkl"
                scaler_file = artifact.model_dir / "scaler.pkl"
                metadata_file = artifact.model_dir / "model_metadata.json"
                
                if not model_file.exists():
                    self.logger.error(f"Model file missing: {model_file}")
                    return None
                
                if not metadata_file.exists():
                    self.logger.error(f"Metadata file missing: {metadata_file}")
                    return None
                
                self.logger.debug(f"Model files verified: model={model_file.exists()}, scaler={scaler_file.exists()}, metadata={metadata_file.exists()}")
                
                raw_df = self._load_price_history(
                    currency=currency,
                    pay_currency=pay_currency,
                    included_leagues=included_leagues,
                    days_back=days_back,
                )
            else:
                self.logger.warning(f"No artifact found for {currency} ({horizon})")
                return None
                
            if raw_df is not None and not raw_df.empty:
                # Pass model metadata to _prepare_features to use stored feature names if available
                processed_df, feature_columns = self._prepare_features(
                    raw_df, currency=currency, horizon=horizon, model_metadata=artifact.metadata
                )
                if processed_df is not None and not processed_df.empty:
                    X, feature_rows = self._extract_feature_matrix(processed_df, feature_columns)
                    if len(feature_rows) > 0:
                        # Load imputer, scaler, and model directly from filesystem
                        self.logger.debug(f"Loading model for {currency} ({horizon}) from filesystem")
                        
                        # Load imputer and apply transformation BEFORE scaling
                        imputer = None
                        if artifact.imputer_path and artifact.imputer_path.exists():
                            try:
                                imputer = joblib.load(artifact.imputer_path)
                                X = imputer.transform(X)
                                self.logger.debug(f"Applied saved imputer from training for {currency} ({horizon})")
                            except Exception as e:
                                self.logger.warning(f"Failed to load imputer for {currency} ({horizon}): {e}")
                                # If imputer fails to load, we cannot proceed safely
                                raise ValueError(f"Failed to load required imputer for {currency} ({horizon}). Model artifacts may be incomplete.")
                        else:
                            self.logger.warning(f"No imputer found for {currency} ({horizon}). This model may have been trained before imputation was saved.")
                            # Fallback: use median imputation (but log warning)
                            from sklearn.impute import SimpleImputer
                            fallback_imputer = SimpleImputer(strategy='median')
                            X = fallback_imputer.fit_transform(X)
                            self.logger.warning(f"Used fallback imputation for {currency} ({horizon}) - this may cause prediction inconsistencies")
                        
                        scaler = self._load_scaler_from_direct(artifact)
                        model = self._load_model_from_direct(artifact)
                        
                        if scaler is not None:
                            # Validate feature count before scaling
                            expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
                            self.logger.debug(f"Feature validation: expected {expected_features}, got {X.shape[1]}")
                            
                            if expected_features is not None and X.shape[1] != expected_features:
                                raise ValueError(
                                    f"Feature count mismatch: expected {expected_features} features, "
                                    f"but got {X.shape[1]} features. This indicates a mismatch between "
                                    f"training and inference feature engineering configurations.\n"
                                    f"NOTE: If you recently enabled horizon-based feature filtering to prevent "
                                    f"feature leakage, you need to retrain your models with the same filtering "
                                    f"logic applied during training. The model for {currency} ({horizon}) was "
                                    f"trained with {expected_features} features but inference is using {X.shape[1]} features."
                                )
                            X = scaler.transform(X)
                        latest_features = X[-1].reshape(1, -1)
                        
                        # Make prediction with uncertainty using ensemble spread
                        prediction_value, lower, upper = self._compute_interval_from_ensemble(
                            model, latest_features, confidence_level=0.95
                        )
                        
                        # Validate prediction value
                        if math.isnan(prediction_value) or math.isinf(prediction_value):
                            raise ValueError(f"Model prediction returned invalid value: {prediction_value}")

                        latest_row = feature_rows.iloc[-1]
                        current_price = float(latest_row["price"])
                        
                        # Validate current price
                        if math.isnan(current_price) or math.isinf(current_price) or current_price < 0:
                            raise ValueError(f"Invalid current price: {current_price}")
                        
                        # Calculate price change percentage with robust NaN handling
                        if (current_price and 
                            not math.isnan(current_price) and 
                            not math.isnan(prediction_value) and 
                            current_price > 1e-6):  # Avoid division by very small numbers
                            price_change_pct = ((prediction_value - current_price) / current_price) * 100
                        else:
                            # If current_price is invalid or too small, use a fallback calculation
                            if not math.isnan(prediction_value) and prediction_value > 0:
                                # Use a small positive value as baseline to calculate percentage
                                price_change_pct = ((prediction_value - 1.0) / 1.0) * 100
                            else:
                                price_change_pct = 0.0  # Default to no change

                        # Calculate confidence score based on relative interval width
                        interval_width = upper - lower
                        confidence = self._compute_confidence(prediction_value, interval_width)

                        result = PredictionResult(
                            currency=currency,
                            pay_currency=pay_currency,
                            horizon=horizon,
                            league=str(latest_row.get("league_name", "Unknown")),
                            current_price=current_price,
                            predicted_price=prediction_value,
                            price_change_percent=price_change_pct,
                            prediction_timestamp=pd.Timestamp.utcnow().isoformat(),
                            confidence_score=confidence,
                            prediction_lower=lower,
                            prediction_upper=upper,
                            features_used=len(feature_columns),
                            model_path=str(artifact.model_dir),
                            metadata=artifact.metadata,
                        )

                        self.logger.info(
                            f"Prediction for {currency} ({horizon}): {current_price:.2f} -> {prediction_value:.2f} "
                            f"({price_change_pct:+.1f}%, confidence: {confidence:.2f})"
                        )
                        
                        # Explicit cleanup of DataFrames and models to free memory
                        del raw_df, processed_df, feature_rows, X, latest_features, model, scaler
                        if 'imputer' in locals() and imputer is not None:
                            del imputer
                        return result

        except ValueError as exc:
            # Handle feature count mismatch specifically
            if "Feature count mismatch" in str(exc):
                self.logger.error(
                    f"Feature count mismatch for {currency} ({horizon}): {exc}",
                    extra={
                        "currency": currency,
                        "horizon": horizon,
                        "error_type": "feature_count_mismatch"
                    }
                )
            else:
                self.logger.warning(f"ValueError during prediction for {currency} ({horizon}): {exc}")
            return None
        except Exception as exc:
            # Enhanced error logging to identify specific failure points
            error_details = []
            if "artifact" in locals() and artifact is None:
                error_details.append("no model artifact found")
            elif "raw_df" in locals() and (raw_df is None or raw_df.empty):
                error_details.append("no price history data")
            elif "processed_df" in locals() and (processed_df is None or processed_df.empty):
                error_details.append("feature engineering failed")
            elif "X" in locals() and len(X) == 0:
                error_details.append("no feature matrix generated")
            elif "scaler" in locals() and scaler is None:
                error_details.append("scaler loading failed")
            elif "model" in locals() and model is None:
                error_details.append("model loading failed")
            elif "prediction_value" in locals() and (math.isnan(prediction_value) or math.isinf(prediction_value)):
                error_details.append("invalid prediction value")
            elif "current_price" in locals() and (math.isnan(current_price) or math.isinf(current_price) or current_price < 0):
                error_details.append("invalid current price")
            else:
                error_details.append(f"unexpected error: {type(exc).__name__}")
            
            self.logger.warning(f"Failed to generate prediction for {currency} ({horizon}): {exc} - {', '.join(error_details)}")
            return None


@dataclass
class DirectModelArtifact:
    """Model artifact stored directly in filesystem."""
    model_dir: Path
    scaler_path: Optional[Path]
    imputer_path: Optional[Path]  # Imputer fitted on training data (for RandomForest/ExtraTrees)
    metadata_path: Path
    metadata: Dict[str, Any]


@dataclass
class RefreshOptions:
    currencies: Optional[Sequence[str]]
    pay_currency: str
    horizons: Sequence[str]
    ttl_hours: int


def _get_current_price(currency: str, league: str, data_source: BaseDataSource, logger: MLLogger) -> Optional[float]:
    """Get the current price of a currency from the live prices table."""
    try:
        # Get the most recent price for this currency in this league
        currency_league = f"{currency}#{league}"
        
        # Query the live prices table for the most recent price
        response = data_source._prices_table.query(
            KeyConditionExpression=Key("currency_league").eq(currency_league),
            ScanIndexForward=False,  # Get most recent first
            Limit=1
        )
        
        items = response.get('Items', [])
        if not items:
            logger.debug(f"No current price found for {currency} in {league}")
            return None
            
        price = data_source._coerce_float(items[0].get('price'))
        if price is None or price <= 0:
            logger.debug(f"Invalid price for {currency} in {league}: {price}")
            return None
            
        return price
        
    except Exception as e:
        logger.warning(f"Failed to get current price for {currency} in {league}: {e}")
        return None


def _load_options(event: Optional[dict]) -> RefreshOptions:
    event = event or {}
    currencies = event.get("currencies")
    horizons = event.get("horizons", DEFAULT_HORIZONS)
    ttl_hours = int(event.get("ttl_hours", os.getenv("PREDICTION_CACHE_TTL_HOURS", 2)))
    return RefreshOptions(
        currencies=currencies,
        pay_currency="Chaos Orb",  # Always use Chaos Orb as pay currency
        horizons=horizons,
        ttl_hours=ttl_hours,
    )


def _select_currencies(
    options: RefreshOptions,
    data_source: BaseDataSource,
    predictor: ModelPredictor,
    logger: MLLogger,
    limit: Optional[int] = None,
) -> List[str]:
    if options.currencies:
        return list(options.currencies)

    logger.info("No currencies supplied; selecting defaults from metadata")
    stats = data_source.list_currency_stats()
    
    # Get most recent league once (cached per invocation in predictor, but this function is called separately)
    # Note: This is called once per Lambda invocation, so caching here is less critical but still good practice
    current_league = data_source.get_most_recent_league()
    if current_league:
        stats = [stat for stat in stats if stat.league == current_league]
        logger.info("Auto-selected current active seasonal league for currency selection", extra={"league": current_league})
    else:
        logger.info("No active seasonal league found, using all available leagues")
    
    # Get currencies that have both data availability AND trained models
    available_currencies_with_models = set(predictor.list_available_currencies())
    logger.info(f"Found {len(available_currencies_with_models)} currencies with trained models")
    
    # Filter stats to only include currencies with models
    stats_with_models = [stat for stat in stats if stat.currency in available_currencies_with_models]
    logger.info(f"Found {len(stats_with_models)} currencies with both data and models")
    
    stats_with_models = sorted(stats_with_models, key=lambda s: s.avg_price, reverse=True)
    
    # Filter to currencies worth 0.25 chaos orbs and above (based on current price)
    min_price_threshold = 0.25
    valuable_currencies = []
    
    for stat in stats_with_models:
        if not stat.is_available:
            continue
            
        # Get current price from live prices table
        current_price = _get_current_price(stat.currency, stat.league, data_source, logger)
        if current_price is not None and current_price >= min_price_threshold:
            valuable_currencies.append((stat.currency, current_price))
    
    available_currencies = [currency for currency, _ in valuable_currencies]
    
    if valuable_currencies:
        prices = [price for _, price in valuable_currencies]
        min_price = min(prices)
        max_price = max(prices)
        logger.info(f"Filtered to currencies worth {min_price_threshold}+ chaos orbs: {len(available_currencies)} currencies (current price range: {min_price:.2f} - {max_price:.2f} chaos)")
    else:
        logger.warning(f"No currencies found worth {min_price_threshold}+ chaos orbs")
    
    # Apply limit only if specified
    if limit is not None:
        available_currencies = available_currencies[:limit]
        logger.info(f"Selected {len(available_currencies)} currencies for prediction (limited to {limit})", extra={"currencies": available_currencies[:10]})
    else:
        logger.info(f"Selected {len(available_currencies)} currencies for prediction (no limit applied)", extra={"currencies": available_currencies[:10]})
    
    return available_currencies


def _write_predictions(
    table,
    predictions: Iterable[PredictionResult],
    ttl_hours: int,
    logger: MLLogger,
) -> None:
    if ttl_hours <= 0:
        ttl_epoch = None
    else:
        ttl_epoch = int(time.time()) + ttl_hours * 3600

    items_written = 0
    items_failed = 0
    key_names = [key["AttributeName"] for key in table.key_schema]
    
    try:
        with table.batch_writer(overwrite_by_pkeys=key_names) as batch:
            for result in predictions:
                try:
                    # Create the primary key for DynamoDB - include horizon to ensure unique records per horizon
                    currency_league_horizon = f"{result.currency}#{result.league}#{result.horizon}"
                    
                    # Get the base payload from the result
                    payload = result.to_dict()
                    
                    # Create the cache key for the prediction
                    cache_key = f"{result.currency}#{result.league}#{result.horizon}"
                    
                    # Prepare the DynamoDB item with proper structure
                    # Convert prediction_timestamp to epoch seconds for DynamoDB sort key
                    timestamp_epoch = int(result.prediction_timestamp.timestamp()) if hasattr(result.prediction_timestamp, 'timestamp') else int(time.time())
                    
                    # Helper function to safely convert float to Decimal with precision control
                    def safe_decimal(value, precision: int = 2):
                        """Convert float to Decimal with specified precision.
                        
                        Args:
                            value: Float value to convert
                            precision: Number of decimal places (2 for prices, 4 for percentages)
                        """
                        if value is None:
                            return None
                        if math.isnan(value) or math.isinf(value):
                            return None
                        try:
                            return Decimal(str(round(value, precision)))
                        except (ValueError, TypeError):
                            return None
                    
                    item = {
                        "currency_league_horizon": currency_league_horizon,  # Primary key (partition key) - includes horizon
                        "timestamp": timestamp_epoch,  # Sort key (epoch seconds)
                        "horizon": result.horizon,  # Additional field for filtering
                        "currency": result.currency,
                        "league": result.league,
                        "pay_currency": result.pay_currency,
                        "current_price": safe_decimal(result.current_price, precision=2),  # 2 decimal places for prices
                        "predicted_price": safe_decimal(result.predicted_price, precision=2),  # 2 decimal places for prices
                        "price_change_percent": safe_decimal(result.price_change_percent, precision=4),  # 4 decimal places for percentages
                        "prediction_timestamp": result.prediction_timestamp.isoformat() if hasattr(result.prediction_timestamp, 'isoformat') else str(result.prediction_timestamp),
                        "confidence_score": safe_decimal(result.confidence_score, precision=4),  # 4 decimal places for confidence scores
                        "prediction_lower": safe_decimal(result.prediction_lower, precision=2),  # 2 decimal places for prices
                        "prediction_upper": safe_decimal(result.prediction_upper, precision=2),  # 2 decimal places for prices
                        "features_used": result.features_used,
                    }
                    
                    # Add TTL if specified
                    if ttl_epoch:
                        item["ttl"] = ttl_epoch
                    
                    # Add metadata as JSON string (avoid DataFrame serialization issues)
                    if result.metadata:
                        try:
                            # Convert any non-serializable objects to strings
                            metadata_clean = {}
                            for key, value in result.metadata.items():
                                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                                    metadata_clean[key] = value
                                else:
                                    # Convert non-serializable objects to string representation
                                    metadata_clean[key] = str(value)
                            item["metadata"] = json.dumps(metadata_clean)
                        except Exception as e:
                            logger.warning(f"Failed to serialize metadata for {result.currency}: {e}")
                            item["metadata"] = "{}"
                    else:
                        item["metadata"] = "{}"
                    
                    # Filter out None values and put the item
                    clean_item = {k: v for k, v in item.items() if v is not None}
                    logger.debug(f"Writing item for {result.currency}: {clean_item}")
                    batch.put_item(Item=clean_item)
                    items_written += 1
                    
                except Exception as e:
                    logger.error(f"Failed to write prediction for {result.currency}: {e}")
                    items_failed += 1
                    continue
    except Exception as e:
        logger.error(f"Batch write operation failed: {e}")
        items_failed += len(list(predictions)) - items_written
    
    logger.info(f"Wrote {items_written} prediction records, {items_failed} failed", extra={"count": items_written, "failed": items_failed})


def refresh_predictions(event: Optional[dict] = None, context=None) -> dict:
    """Shared implementation for Lambda handler and CLI."""
    logger = MLLogger("PredictionRefresh", level=os.getenv("LOG_LEVEL", "INFO"))
    
    try:
        options = _load_options(event)
        logger.info(
            "Starting prediction refresh",
            extra={
                "currencies": options.currencies,
                "horizons": options.horizons,
                "ttl_hours": options.ttl_hours,
            },
        )

        # Use inference-specific configuration
        config = get_inference_config_from_env()
        models_dir = Path(os.getenv("MODELS_DIR", "ml/models/currency"))
        
        # Ensure models directory exists and contains bundled models
        actual_models_dir = _ensure_models_available(models_dir, logger)
        
        # Initialize predictor with error handling
        try:
            predictor = DirectModelPredictor(models_dir=actual_models_dir, config=config, logger=logger)
        except Exception as e:
            logger.error(f"Failed to initialize DirectModelPredictor: {e}")
            return {
                "status": "predictor_init_failed",
                "error": f"Failed to initialize predictor: {str(e)}"
            }
        
        # Initialize data source with error handling
        try:
            data_source_config = DataSourceConfig.from_dynamo_config(config.dynamo)
            data_source = create_data_source(data_source_config, logger)
        except Exception as e:
            logger.error(f"Failed to initialize data source: {e}")
            return {
                "status": "data_source_init_failed", 
                "error": f"Failed to initialize data source: {str(e)}"
            }

        target_currencies = _select_currencies(options, data_source, predictor, logger)
        if not target_currencies:
            logger.warning("No currencies found for prediction refresh")
            return {"status": "no_currencies", "error": "No currencies available"}

        # Check if we have sufficient data for predictions
        # This prevents the Lambda from running when there's insufficient historical data
        try:
            # Sample a few currencies to check data availability
            sample_currencies = target_currencies[:3]  # Check first 3 currencies
            insufficient_data_count = 0
            data_details = []
            
            for currency in sample_currencies:
                try:
                    raw_df = predictor._load_price_history(
                        currency=currency,
                        pay_currency=options.pay_currency,
                    )
                    record_count = len(raw_df) if raw_df is not None else 0
                    data_details.append(f"{currency}: {record_count} records")
                    
                    if raw_df is None or raw_df.empty or len(raw_df) < 1:
                        insufficient_data_count += 1
                        logger.debug(f"Insufficient data for {currency}: {record_count} records")
                    else:
                        logger.info(f"Found {record_count} records for {currency}")
                except Exception as e:
                    logger.debug(f"Could not check data for {currency}: {e}")
                    insufficient_data_count += 1
                    data_details.append(f"{currency}: error - {str(e)}")
            
            logger.info(f"Data availability check: {', '.join(data_details)}")
            
            # If most sample currencies have insufficient data, skip the refresh
            if insufficient_data_count >= len(sample_currencies) * 0.9:  # 90% threshold (more lenient)
                logger.warning(
                    f"Insufficient historical data for predictions. "
                    f"Sample check: {insufficient_data_count}/{len(sample_currencies)} currencies have insufficient data. "
                    f"Data details: {', '.join(data_details)}. "
                    f"Skipping prediction refresh."
                )
                return {
                    "status": "insufficient_data", 
                    "error": "Insufficient historical data for reliable predictions",
                    "sample_failures": insufficient_data_count,
                    "sample_size": len(sample_currencies),
                    "data_details": data_details
                }
                
        except Exception as e:
            logger.warning(f"Could not perform data sufficiency check: {e}")
            # Continue with prediction refresh if check fails

        predictions: List[PredictionResult] = []
        failed_predictions = 0
        
        # Track processed count for memory management
        processed_count = 0
        memory_cleanup_interval = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "10"))  # Cleanup every N predictions
        
        for currency in target_currencies:
            for horizon in options.horizons:
                logger.info(f"Processing {currency} for horizon {horizon}")
                try:
                    # Pre-validate model availability to provide better error messages
                    available_currencies = predictor.list_available_currencies()
                    if currency not in available_currencies:
                        logger.warning(f"Skipping {currency} ({horizon}) - no trained model available (available: {len(available_currencies)} currencies)")
                        failed_predictions += 1
                        continue
                    
                    logger.info(f"Attempting to generate prediction for {currency} ({horizon})")
                    prediction = predictor.predict_currency(
                        currency=currency,
                        horizon=horizon,
                        pay_currency=options.pay_currency,
                    )
                    
                    # Validate prediction result with more robust checks
                    if prediction is not None:
                        logger.info(f"Prediction generated for {currency} ({horizon}): {prediction.predicted_price if hasattr(prediction, 'predicted_price') else 'N/A'}")
                    
                    if (prediction and 
                        hasattr(prediction, 'currency') and 
                        hasattr(prediction, 'league') and
                        prediction.currency and 
                        prediction.league and
                        not math.isnan(prediction.current_price) and
                        not math.isnan(prediction.predicted_price) and
                        not math.isnan(prediction.confidence_score) and
                        not math.isnan(prediction.price_change_percent) and
                        prediction.current_price > 0 and  # Ensure positive current price
                        prediction.predicted_price > 0):  # Ensure positive predicted price
                        predictions.append(prediction)
                        logger.info(f"Successfully generated prediction for {currency} ({horizon})")
                        
                        processed_count += 1
                        
                        # Periodic memory cleanup to prevent OOM
                        if processed_count % memory_cleanup_interval == 0:
                            logger.info(f"Performing periodic memory cleanup (processed {processed_count} predictions)")
                            # Force garbage collection to free memory from models and DataFrames
                            gc.collect()
                            logger.debug("Memory cleanup completed")
                    else:
                        # Log more detailed information about why the prediction failed
                        failure_reasons = []
                        if not prediction:
                            failure_reasons.append("no prediction object")
                            # Add diagnostic information for None predictions
                            logger.debug(f"Prediction failed for {currency} ({horizon}) - checking model availability...")
                            available_currencies = predictor.list_available_currencies()
                            if currency not in available_currencies:
                                failure_reasons.append(f"no trained model (available: {len(available_currencies)} currencies)")
                            else:
                                failure_reasons.append("model exists but prediction generation failed")
                        elif not hasattr(prediction, 'currency') or not prediction.currency:
                            failure_reasons.append("missing currency")
                        elif not hasattr(prediction, 'league') or not prediction.league:
                            failure_reasons.append("missing league")
                        elif math.isnan(prediction.current_price):
                            failure_reasons.append("NaN current_price")
                        elif math.isnan(prediction.predicted_price):
                            failure_reasons.append("NaN predicted_price")
                        elif math.isnan(prediction.confidence_score):
                            failure_reasons.append("NaN confidence_score")
                        elif math.isnan(prediction.price_change_percent):
                            failure_reasons.append("NaN price_change_percent")
                        elif prediction.current_price <= 0:
                            failure_reasons.append(f"invalid current_price: {prediction.current_price}")
                        elif prediction.predicted_price <= 0:
                            failure_reasons.append(f"invalid predicted_price: {prediction.predicted_price}")
                        
                        logger.warning(f"Invalid prediction result for {currency} ({horizon}) - {', '.join(failure_reasons)}")
                        failed_predictions += 1
                        
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(
                        f"Failed to generate prediction for {currency} ({horizon}): {exc}",
                        exception=exc,
                        extra={"currency": currency, "horizon": horizon},
                    )
                    failed_predictions += 1
                    
                    # Cleanup on failure too to prevent memory accumulation
                    processed_count += 1
                    if processed_count % memory_cleanup_interval == 0:
                        gc.collect()

        # Final memory cleanup before writing to DynamoDB
        logger.info("Performing final memory cleanup before writing predictions")
        gc.collect()

        if not predictions:
            logger.warning("No valid predictions generated")
            return {
                "status": "no_predictions", 
                "error": "No valid predictions could be generated",
                "failed_count": failed_predictions
            }

        # Write predictions to DynamoDB
        try:
            dynamodb = boto3.resource("dynamodb", region_name=config.dynamo.region_name)
            predictions_table = dynamodb.Table(config.dynamo.predictions_table)
            _write_predictions(predictions_table, predictions, options.ttl_hours, logger)
            
            logger.info(
                "Prediction refresh completed successfully",
                extra={
                    "total_currencies": len(target_currencies),
                    "successful_predictions": len(predictions),
                    "failed_predictions": failed_predictions,
                    "horizons": list(options.horizons),
                }
            )
            
            return {
                "status": "ok",
                "currency_count": len(target_currencies),
                "prediction_count": len(predictions),
                "failed_count": failed_predictions,
                "horizons": list(options.horizons),
            }
            
        except Exception as e:
            logger.error(f"Failed to write predictions to DynamoDB: {e}")
            return {
                "status": "write_failed",
                "error": f"Failed to write predictions: {str(e)}",
                "prediction_count": len(predictions)
            }

    except Exception as e:
        logger.error(f"Prediction refresh failed with critical error: {e}")
        return {
            "status": "failed",
            "error": f"Critical error: {str(e)}"
        }


def lambda_handler(event, context):  # pragma: no cover
    return refresh_predictions(event, context)


def _parse_cli_args(argv: Sequence[str]) -> dict:
    parser = argparse.ArgumentParser(description="Refresh cached currency predictions")
    parser.add_argument("--currencies", nargs="*", help="Specific currencies to refresh")
    parser.add_argument("--pay-currency", default="Chaos Orb", help="Quote currency")
    parser.add_argument("--horizons", nargs="*", default=list(DEFAULT_HORIZONS), help="Prediction horizons (e.g. 1d 3d)")
    parser.add_argument("--ttl-hours", type=int, default=int(os.getenv("PREDICTION_CACHE_TTL_HOURS", 2)), help="Cache TTL in hours")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logger level")
    args = parser.parse_args(argv)
    os.environ.setdefault("LOG_LEVEL", args.log_level.upper())
    event = {
        "currencies": args.currencies or None,
        "horizons": args.horizons,
        "ttl_hours": args.ttl_hours,
    }
    return event


def _ensure_models_available(models_dir: Path, logger: MLLogger) -> Path:
    """Ensure models directory exists and contains models from Lambda code storage.
    
    Returns:
        Path: The models directory (for compatibility with existing code)
    """
    logger.info("Verifying models are available from Lambda code storage")
    
    # Models are available from Lambda code storage at /var/task/models
    # Just verify the directory exists
    if not models_dir.exists():
        raise ValueError(f"Models directory not found: {models_dir}")
    
    # Check for model directories (look in currency subdirectory first)
    currency_dir = models_dir / "currency"
    if currency_dir.exists():
        model_dirs = [d for d in currency_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(model_dirs)} model directories in currency folder from Lambda code storage")
        # Verify model directories are accessible
        for model_dir in model_dirs:
            try:
                model_files = list(model_dir.rglob("*.pkl")) + list(model_dir.rglob("*.json"))
                logger.info(f"Model directory {model_dir.name} contains {len(model_files)} model files")
            except Exception as e:
                logger.warning(f"Could not read model directory {model_dir.name}: {e}")
        return models_dir
    else:
        # Fallback: check directly in models_dir
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if model_dirs:
            logger.info(f"Found {len(model_dirs)} model directories from Lambda code storage")
            # Verify model directories are accessible
            for model_dir in model_dirs:
                try:
                    model_files = list(model_dir.rglob("*.pkl")) + list(model_dir.rglob("*.json"))
                    logger.info(f"Model directory {model_dir.name} contains {len(model_files)} model files")
                except Exception as e:
                    logger.warning(f"Could not read model directory {model_dir.name}: {e}")
            return models_dir
    
    # Fallback: check for any model files
    model_files = list(models_dir.rglob("*.pkl")) + list(models_dir.rglob("*.json"))
    logger.info(f"Found {len(model_files)} model files from Lambda code storage")
    
    if len(model_files) == 0:
        raise ValueError("No model files found in models directory - models must be available from Lambda code storage")
    
    return models_dir




if __name__ == "__main__":  # pragma: no cover
    cli_event = _parse_cli_args(sys.argv[1:])
    result = refresh_predictions(cli_event)
    print(json.dumps(result, indent=2))
