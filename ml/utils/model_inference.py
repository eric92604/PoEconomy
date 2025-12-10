"""
Inference utilities for DynamoDB-backed currency price predictions.

This module loads trained model artifacts, assembles the latest feature vectors
using the shared DataProcessor/FeatureEngineer pipeline, and generates
predictions for requested currencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math

import joblib
import numpy as np
import pandas as pd

from ml.config.inference_config import InferenceConfig, get_inference_config_from_env
from ml.utils.data_processing import DataProcessor
from ml.utils.data_sources import create_data_source, DataSourceConfig, BaseDataSource
from ml.utils.common_utils import MLLogger

HORIZON_SUFFIXES = {"1d", "3d", "7d", "14d", "30d"}




@dataclass
class ModelArtifact:
    """Storage for a single model artifact (model + optional scaler)."""

    model_dir: Path
    scaler_path: Optional[Path]
    metadata_path: Path
    metadata: Dict


@dataclass
class CurrencyModelBundle:
    """Aggregated artifacts for a currency across horizons."""

    primary: Optional[ModelArtifact]
    horizons: Dict[str, ModelArtifact]


@dataclass
class PredictionResult:
    """Structured prediction payload."""

    currency: str
    pay_currency: str
    horizon: str
    league: str
    current_price: float
    predicted_price: float
    price_change_percent: float
    prediction_timestamp: str
    confidence_score: float
    prediction_lower: float
    prediction_upper: float
    features_used: int
    model_path: str
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "currency": self.currency,
            "horizon": self.horizon,
            "league": self.league,
            "current_price": self.current_price,
            "predicted_price": self.predicted_price,
            "price_change_percent": self.price_change_percent,
            "prediction_timestamp": self.prediction_timestamp,
            "confidence_score": self.confidence_score,
            "prediction_lower": self.prediction_lower,
            "prediction_upper": self.prediction_upper,
            "features_used": self.features_used,
            "model_path": self.model_path,
            "metadata": self.metadata,
        }


class ModelPredictor:
    """
    Generates price predictions using trained models and DynamoDB data.
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
        self.logger = logger or MLLogger("ModelPredictor")
        if data_source is None:
            data_source_config = DataSourceConfig.from_dynamo_config(self.config.dynamo)
            self.data_source = create_data_source(data_source_config, self.logger)
        else:
            self.data_source = data_source

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        self.data_processor = DataProcessor(self.config.data, self.config.processing, self.logger)  # type: ignore[arg-type]
        self.model_registry = self._discover_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_available_currencies(self) -> List[str]:
        """Return currencies for which model artifacts exist."""
        return sorted(self.model_registry.keys())

    def predict_currency(
        self,
        currency: str,
        horizon: str = "1d",
        pay_currency: str = "Chaos Orb",
        included_leagues: Optional[Sequence[str]] = None,
        days_back: Optional[int] = None,
    ) -> PredictionResult:
        """
        Generate a prediction for a single currency.
        """
        horizon = horizon.lower().replace(" ", "")
        
        # Try to use trained model first
        try:
            artifact = self._select_model(currency, horizon)
            if artifact is not None:
                raw_df = self._load_price_history(
                    currency=currency,
                    pay_currency=pay_currency,
                    included_leagues=included_leagues,
                    days_back=days_back,
                )
                if raw_df is not None and not raw_df.empty:
                    # Pass model metadata to _prepare_features to use stored feature names if available
                    processed_df, feature_columns = self._prepare_features(
                        raw_df, currency=currency, horizon=horizon, model_metadata=artifact.metadata
                    )
                    if processed_df is not None and not processed_df.empty:
                        X, feature_rows = self._extract_feature_matrix(processed_df, feature_columns)
                        if len(feature_rows) > 0:
                            # Load scaler
                            scaler = joblib.load(artifact.scaler_path) if artifact.scaler_path and artifact.scaler_path.exists() else None
                            if scaler is not None:
                                X = scaler.transform(X)

                            # Load model
                            model = joblib.load(artifact.model_dir / "ensemble_model.pkl")
                            
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
                                model_dir=str(artifact.model_dir),
                                metadata=artifact.metadata,
                            )

                            self.logger.info(
                                "Generated prediction",
                                extra={
                                    "currency": currency,
                                    "horizon": horizon,
                                    "prediction": prediction_value,
                                    "current_price": current_price,
                                    "confidence": confidence,
                                },
                            )
                            return result
        except Exception as e:
            self.logger.warning(f"Trained model prediction failed for {currency}: {e}")

        # No trained model available - return error gracefully
        self.logger.error(f"No trained model available for {currency} {horizon}")
        raise ValueError(f"No trained model available for currency '{currency}' with horizon '{horizon}'. Please train models first.")



    def _get_current_league(self) -> str:
        """Get the current active league."""
        try:
            # Try to get from data source
            latest_league = self.data_source.get_most_recent_league()
            if latest_league:
                return latest_league
        except Exception:
            pass
        
        # No fallback - this should not happen in production
        raise RuntimeError("Could not determine current league from data source. This indicates a data source configuration issue.")

    def _calculate_league_day(self, league: str) -> int:
        """Calculate the current league day for the given league."""
        try:
            # For now, return a default league day since we don't have league metadata
            # This could be enhanced to use actual league start dates
            from datetime import datetime, timezone
            current_time = datetime.now(timezone.utc)
            # Assume league started 30 days ago as a fallback
            league_day = 30
            return max(1, league_day)  # Ensure at least day 1
        except Exception as e:
            self.logger.warning(f"Could not calculate league day for {league}: {e}")
        
        # No fallback - this should not happen in production
        raise RuntimeError(f"Could not calculate league day for {league}. This indicates a league metadata configuration issue.")

    def _get_current_price(self, currency: str, league: str) -> Optional[float]:
        """Get the most recent price for a currency using primary key query."""
        try:
            # Query the prices table using primary key for efficiency
            from boto3.dynamodb.conditions import Key
            # For now, return a default price since we don't have direct table access
            # This could be enhanced to use the data source's query methods
            # No fallback - this should not happen in production
            raise RuntimeError(f"Could not get current price for {currency} in {league}. This indicates a data source configuration issue.")
        except Exception as e:
            self.logger.warning(f"Failed to get current price for {currency}: {e}")
        return None



    def predict_many(
        self,
        currencies: Sequence[str],
        horizon: str = "1d",
        pay_currency: str = "Chaos Orb",
        included_leagues: Optional[Sequence[str]] = None,
        days_back: Optional[int] = None,
    ) -> List[PredictionResult]:
        """Batch prediction helper."""
        results = []
        for currency in currencies:
            try:
                results.append(
                    self.predict_currency(
                        currency=currency,
                        horizon=horizon,
                        pay_currency=pay_currency,
                        included_leagues=included_leagues,
                        days_back=days_back,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error(f"Prediction failed for {currency}: {exc}")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_models(self) -> Dict[str, CurrencyModelBundle]:
        """Scan the models directory and group artifacts per currency."""
        registry: Dict[str, CurrencyModelBundle] = {}

        for metadata_path in self.models_dir.rglob("model_metadata.json"):
            try:
                with open(metadata_path, "r", encoding="utf-8") as fp:
                    metadata = json.load(fp)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to read metadata {metadata_path}: {exc}")
                continue
            currency_label = metadata.get("currency")
            if not currency_label:
                continue

            model_dir = metadata_path.parent
            model_path = model_dir / "ensemble_model.pkl"
            if not model_path.exists():
                self.logger.warning(f"Model file missing for {currency_label}: {model_path}")
                continue

            scaler_path = model_dir / "scaler.pkl"
            artifact = ModelArtifact(
                model_dir=model_dir,
                scaler_path=scaler_path if scaler_path.exists() else None,
                metadata_path=metadata_path,
                metadata=metadata,
            )

            base_currency, suffix = _split_currency_label(currency_label)
            bundle = registry.setdefault(base_currency, CurrencyModelBundle(primary=None, horizons={}))
            if suffix and suffix in HORIZON_SUFFIXES:
                bundle.horizons[suffix] = artifact
            else:
                bundle.primary = artifact

        return registry

    def _select_model(self, currency: str, horizon: str) -> Optional[ModelArtifact]:
        bundle = self.model_registry.get(currency)
        if not bundle:
            self.logger.warning(f"No models registered for {currency}")
            return None

        if horizon in bundle.horizons:
            return bundle.horizons[horizon]

        if bundle.primary is not None:
            return bundle.primary

        # No model available for requested horizon
        return None

    def _load_price_history(
        self,
        currency: str,
        pay_currency: str,
        included_leagues: Optional[Sequence[str]] = None,
        days_back: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        pairs = [{"get_currency": currency}]

        if included_leagues is None:
            if self.config.data.included_leagues:
                leagues = list(self.config.data.included_leagues)
            else:
                latest_league = self.data_source.get_most_recent_league()
                if latest_league:
                    leagues = [latest_league]
                    self.logger.info(f"Auto-selected most recent league: {latest_league}")
                else:
                    leagues = None
        else:
            leagues = list(included_leagues)

        df = self.data_source.build_price_dataframe(
            currencies=[pair['get_currency'] for pair in pairs],
            included_leagues=leagues,
            max_league_days=self.config.data.max_league_days,
            min_league_days=self.config.data.min_league_days,
        )

        if df is None or df.empty:
            return df

        if days_back is not None and "date" in df.columns:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)
            df = df[df["date"] >= cutoff]

        return df.reset_index(drop=True)

    def _prepare_features(
        self,
        df: pd.DataFrame,
        currency: str,
        horizon: Optional[str] = None,
        model_metadata: Optional[Dict] = None,
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        processed_data, metadata = self.data_processor.process_currency_data(df, currency)
        if processed_data is None or processed_data.empty:
            self.logger.warning(f"No processed data for {currency}", extra=metadata)
            return None, []

        processed_data = processed_data.sort_values("date").reset_index(drop=True)
        
        # Require stored feature names in model metadata for exact feature matching
        if not model_metadata or 'feature_names' not in model_metadata:
            error_msg = (
                f"Model metadata missing 'feature_names' for {currency} ({horizon}). "
                f"This model was trained before feature names were stored in metadata. "
                f"Please retrain the model to include feature names in metadata."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        stored_feature_names = model_metadata['feature_names']
        self.logger.info(
            f"Using stored feature names from model metadata ({len(stored_feature_names)} features)",
            extra={"currency": currency, "horizon": horizon}
        )
        
        # Verify all stored features exist in processed data
        available_features = []
        missing_features = []
        for feat_name in stored_feature_names:
            if feat_name in processed_data.columns:
                available_features.append(feat_name)
            else:
                missing_features.append(feat_name)
        
        if missing_features:
            error_msg = (
                f"Missing {len(missing_features)} stored features in processed data for {currency} ({horizon}). "
                f"This indicates a mismatch between training and inference feature engineering. "
                f"Missing features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
            )
            self.logger.error(error_msg, extra={"missing_features": missing_features})
            raise ValueError(error_msg)
        
        if not available_features:
            error_msg = (
                f"None of the stored feature names are available in processed data for {currency} ({horizon}). "
                f"This indicates a critical mismatch between training and inference feature engineering."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Use stored feature names exactly as they were during training
        feature_columns = available_features
        self.logger.debug(f"Using {len(feature_columns)} features from stored metadata (expected {len(stored_feature_names)})")
        
        return processed_data, feature_columns

    def _extract_feature_matrix(
        self,
        processed_df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        if not feature_columns:
            return np.empty((0, 0)), processed_df.iloc[0:0]

        feature_frame = processed_df[feature_columns].copy()
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
        feature_matrix = feature_frame.to_numpy(dtype=float, na_value=np.nan)

        # More robust handling of NaN values for small datasets
        # Instead of requiring ALL features to be finite, we'll:
        # 1. Impute NaN values with column means/medians
        # 2. Use a more lenient filtering approach
        
        # Count NaN values per row
        nan_counts = np.isnan(feature_matrix).sum(axis=1)
        total_features = feature_matrix.shape[1]
        
        # For small datasets, be more lenient with NaN tolerance
        # Allow rows with up to 50% NaN values (for 5 data points, this is reasonable)
        max_nan_ratio = 0.5
        max_nan_count = int(total_features * max_nan_ratio)
        
        # Filter rows with too many NaN values
        valid_mask = nan_counts <= max_nan_count
        
        if not np.any(valid_mask):
            # If no rows pass the NaN filter, use the row with the fewest NaN values
            min_nan_idx = np.argmin(nan_counts)
            valid_mask = np.zeros(len(feature_matrix), dtype=bool)
            valid_mask[min_nan_idx] = True
            self.logger.warning(f"No rows passed NaN filter, using row with fewest NaN values ({nan_counts[min_nan_idx]}/{total_features})")
        
        filtered_matrix = feature_matrix[valid_mask]
        filtered_rows = processed_df[valid_mask].reset_index(drop=True)
        
        # Impute remaining NaN values with column means
        for col_idx in range(filtered_matrix.shape[1]):
            col_data = filtered_matrix[:, col_idx]
            if np.any(np.isnan(col_data)):
                # Use median for imputation (more robust than mean)
                median_val = np.nanmedian(col_data)
                if np.isnan(median_val):
                    # If all values are NaN, use 0
                    median_val = 0.0
                filtered_matrix[np.isnan(col_data), col_idx] = median_val
                self.logger.debug(f"Imputed {np.isnan(col_data).sum()} NaN values in column {col_idx} with {median_val}")
        
        self.logger.debug(f"Feature matrix extraction: {len(filtered_matrix)} rows, {filtered_matrix.shape[1]} features")
        return filtered_matrix, filtered_rows

    def _compute_interval_from_ensemble(
        self, 
        model: Any, 
        X: np.ndarray, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute prediction interval using ensemble spread method.
        
        Uses the variance across ensemble model predictions as uncertainty measure.
        This method adapts to prediction difficulty and handles heteroscedasticity naturally.
        
        Args:
            model: Trained ensemble model
            X: Input features (single sample)
            confidence_level: Confidence level for intervals
        
        Returns:
            Tuple of (prediction, lower_bound, upper_bound)
        """
        if hasattr(model, 'predict_with_uncertainty'):
            # Use ensemble spread method
            mean_pred, lower, upper = model.predict_with_uncertainty(X, confidence_level)
            prediction = float(np.asarray(mean_pred).ravel()[0])
            lower = float(np.asarray(lower).ravel()[0])
            upper = float(np.asarray(upper).ravel()[0])
            return prediction, lower, upper
        else:
            # Fallback: if model doesn't support uncertainty, use simple prediction
            # This should not happen with EnsembleModel, but handle gracefully
            prediction = model.predict(X)
            prediction_value = float(np.asarray(prediction).ravel()[0])
            # Use a conservative default interval (20% of prediction)
            margin = prediction_value * 0.2
            lower = max(0.0, prediction_value - margin)
            upper = prediction_value + margin
            return prediction_value, lower, upper

    def _compute_confidence(self, prediction: float, interval_width: float) -> float:
        """
        Compute confidence score based on relative interval width.
        
        Uses a square root transformation to make confidence less sensitive to range width.
        This means small changes in range width have less impact on confidence scores.
        """
        if prediction == 0:
            return 0.5  # Default confidence for zero predictions
        
        relative_width = interval_width / abs(prediction)
        # Use square root to reduce sensitivity - small range changes have less impact
        # Scale by 0.5 to further dampen the effect
        scaled_relative_width = 0.5 * (relative_width ** 0.5)
        confidence = 1.0 / (1.0 + scaled_relative_width)
        return float(max(0.0, min(1.0, confidence)))


def _split_currency_label(label: str) -> Tuple[str, Optional[str]]:
    for suffix in HORIZON_SUFFIXES:
        token = f"_{suffix}"
        if label.endswith(token):
            return label[: -len(token)], suffix
    return label, None
