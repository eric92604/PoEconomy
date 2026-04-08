"""
Feature engineering module for currency price prediction.

This module implements the complete feature engineering pipeline used for both
training and inference.  All features are designed to be:

  - **Causal**: only information available at prediction time is used.
  - **Consistent**: identical computation for training and inference paths.
  - **Non-redundant**: duplicate columns from earlier implementations removed.

Feature Catalogue
-----------------
Time:
    month, league_day, league_progress, league_age_days,
    supply_shock_decay  (exp(-league_day/5), replaces binary is_supply_shock)

Price lags (causal look-back windows):
    price_lag_{1,2,3,5,7,14}d
    Note: price_lag_14d is NaN for the first 14 league days and is imputed
    (median) at inference time by the existing SimpleImputer.

Price changes:
    price_change_1d, momentum_{3,5,7}d

Price log (always computed; comparable across currencies):
    price_log  (log1p(price))

League expanding stats (causal — from league start to current day only):
    price_exp_mean, price_exp_std, price_exp_min, price_exp_max,
    price_vs_exp_mean, price_exp_zscore, league_recency_rank,
    price_pct_of_league_high  (price / expanding max),
    price_pct_of_league_low   (price / expanding min),
    price_recovery_strength   (Williams %R in league range: 0=low, 1=high),
    days_since_league_high, days_since_league_low

Rolling windows (default: 3, 5, 7, 14 days):
    price_mean_{w}d, price_std_{w}d, price_min_{w}d, price_max_{w}d,
    price_zscore_{w}d
    Note: 14d rolling stats are NaN before day 2 and are imputed at inference.

Exponential moving averages:
    price_ema_{3,7,14}d, price_to_ema7

MACD (fast=5, slow=14, signal=3):
    macd, macd_signal, macd_hist
    Note: MACD is excluded for the 7d horizon model (short-term oscillator).

Volatility (windows 3, 5, 7):
    price_volatility_{w}d  (CoV = std / mean)
    volatility_garch_{w}d  (EWM realised vol of percentage returns)
    volatility_clustering_7d  (mean squared successive difference, 7d only)

Trend (windows >= 7):
    trend_strength_{w}d  (OLS slope over rolling w-day window)
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ml.config.training_config import DataConfig, ProcessingConfig
from ml.utils.common_utils import MLLogger


def compute_price_log(price: pd.Series) -> pd.Series:
    """log1p(price) with non-negative clip; used in FE and when backfilling missing columns."""
    return np.log1p(price.clip(lower=0.0))


@dataclass
class FeatureEngineeringResult:
    """Result of the feature engineering process."""

    data: pd.DataFrame
    feature_names: List[str]
    transformations_applied: List[str]
    statistics: Dict[str, Any]


class FeatureEngineer:
    """
    Feature engineering for currency price prediction.

    Implements a complete, leak-free feature engineering pipeline.  All
    features are computed using only information available at the time of
    prediction (causal / no look-ahead bias).
    """

    def __init__(
        self,
        config: DataConfig,
        processing_config: ProcessingConfig,
        logger: Optional[MLLogger] = None,
    ) -> None:
        self.config = config
        self.processing_config = processing_config
        self.logger = logger or MLLogger("FeatureEngineer")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer_features(
        self, df: pd.DataFrame, currency: str, is_inference: bool = False
    ) -> FeatureEngineeringResult:
        """
        Run the complete feature engineering pipeline.

        Args:
            df: Raw currency price dataframe (will be sorted internally).
            currency: Currency pair identifier used for logging.
            is_inference: When True, skip target creation and outlier removal.
                Target columns require future prices that are unavailable at
                prediction time, and the dropna they trigger would remove the
                most-recent observation — exactly the row the model predicts
                from.  Outlier removal is also skipped because it is already
                disabled via InferenceProcessingConfig.outlier_removal=False,
                and the current league may have too few rows for a stable IQR.

        Returns:
            FeatureEngineeringResult with the enriched dataframe.
        """
        transformations_applied: List[str] = []

        with self.logger.log_operation(f"Feature engineering for {currency}"):
            df_out = df.copy()
            original_shape = df_out.shape

            df_out = self._preprocess_basic(df_out)
            transformations_applied.append("basic_preprocessing")

            df_out = self._engineer_time_features(df_out)
            transformations_applied.append("time_features")

            df_out = self._engineer_price_features(df_out)
            transformations_applied.append("price_features")

            if getattr(self.config, "include_league_features", True):
                df_out = self._engineer_league_features(df_out)
                transformations_applied.append("league_features")

            df_out = self._engineer_rolling_features(df_out)
            transformations_applied.append("rolling_features")

            if not is_inference:
                df_out = self._create_targets(df_out)
                transformations_applied.append("target_creation")

                if self.processing_config.outlier_removal:
                    df_out = self._remove_outliers(df_out)
                    transformations_applied.append("outlier_removal")

            final_shape = df_out.shape
            statistics: Dict[str, Any] = {
                "original_shape": original_shape,
                "final_shape": final_shape,
                "features_added": final_shape[1] - original_shape[1],
                "records_retained": (
                    final_shape[0] / original_shape[0] if original_shape[0] > 0 else 0.0
                ),
            }

            self.logger.debug(
                f"Feature engineering completed for {currency}", extra=statistics
            )

        return FeatureEngineeringResult(
            data=df_out,
            feature_names=df_out.columns.tolist(),
            transformations_applied=transformations_applied,
            statistics=statistics,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _preprocess_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort by league/date, parse dates, forward-fill missing prices, and
        normalise sub-daily data to one row per (league, calendar day).

        Training data is already daily (one Parquet row per day).  Inference
        data comes from the live-prices DynamoDB table which stores one row per
        hour.  All subsequent feature computations (lags, rolling windows,
        EMAs, etc.) are designed for daily granularity, so we must collapse
        hourly rows to a single daily observation before computing features.
        The last record of each calendar day is kept (closest to end-of-day).
        Numeric columns are aggregated as the last value; all other columns
        retain their last value too so that metadata fields are preserved.
        """
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if "league_name" in df.columns and "date" in df.columns:
            df = df.sort_values(["league_name", "date"]).reset_index(drop=True)
        elif "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        # Resample sub-daily data to daily granularity.
        if "date" in df.columns:
            date_floor = df["date"].dt.normalize()
            group_cols = (
                ["league_name", date_floor]
                if "league_name" in df.columns
                else [date_floor]
            )
            # Check whether any calendar-day bucket holds more than one row.
            if df.groupby(group_cols).size().max() > 1:
                self.logger.debug(
                    "Sub-daily data detected — resampling to daily (last price per day)."
                )
                df["_date_day"] = date_floor
                agg_cols = ["_date_day"] + (
                    ["league_name"] if "league_name" in df.columns else []
                )
                # Keep the last record for every (league, day) bucket.
                df = (
                    df.sort_values("date")
                    .groupby(agg_cols, sort=True)
                    .last()
                    .reset_index()
                )
                df["date"] = df["_date_day"]
                df = df.drop(columns=["_date_day"])
                if "league_name" in df.columns:
                    df = df.sort_values(["league_name", "date"]).reset_index(drop=True)
                else:
                    df = df.sort_values("date").reset_index(drop=True)

        if "price" in df.columns:
            if "league_name" in df.columns:
                df["price"] = df.groupby("league_name")["price"].ffill()
            else:
                df["price"] = df["price"].ffill()
            df = df.dropna(subset=["price"])

        return df

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar and league-cycle time features.

        Calendar features are limited to ``month`` — day-of-week and
        day-of-month carry negligible importance in PoE markets since there
        is no market close or weekly settlement cycle.

        ``supply_shock_decay`` replaces the old binary ``is_supply_shock``
        flag with a continuous exponential decay (1.0 on league day 0,
        ~0.14 by day 10) that captures the well-known early-league supply
        constraint without an artificial threshold.
        """
        if "date" not in df.columns:
            return df

        df["month"] = df["date"].dt.month

        if "league_name" in df.columns:
            league_starts = df.groupby("league_name")["date"].min()
            df["league_start"] = df["league_name"].map(league_starts)
            df["league_day"] = (df["date"] - df["league_start"]).dt.days

            assumed_max_days: int = getattr(self.config, "max_league_days", 60)
            df["league_progress"] = df["league_day"] / (assumed_max_days + 1)

            latest_start = league_starts.max()
            df["league_age_days"] = df["league_name"].map(
                (latest_start - league_starts).dt.days
            )

            # Continuous supply-shock decay: prices spike on days 0-3 before
            # content is farmable, then normalise. exp(-day/5) gives a smooth
            # decay to ~0.14 by day 10 with no arbitrary threshold.
            df["supply_shock_decay"] = np.exp(-df["league_day"] / 5.0)

        return df

    def _engineer_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-derived features: lags, changes, log returns, momentum.

        Explicit lag features (the raw price at t-k) are the highest-impact
        inputs for gradient-boosted tree models on time-series data because
        the model cannot infer temporal ordering from feature names alone.
        """
        if "price" not in df.columns:
            return df

        df["price_log"] = compute_price_log(df["price"])

        df = self._add_lag_features(df)

        # 1-day change
        if "league_name" in df.columns:
            df["price_change_1d"] = df.groupby("league_name")["price"].diff()
        else:
            df["price_change_1d"] = df["price"].diff()

        # Multi-day momentum (percentage change over N days, within each league)
        momentum_periods: List[int] = getattr(self.config, "momentum_periods", [3, 5, 7])
        for period in momentum_periods:
            if period > 1:
                if "league_name" in df.columns:
                    df[f"momentum_{period}d"] = df.groupby("league_name")[
                        "price"
                    ].transform(lambda x, p=period: x.pct_change(periods=p))
                else:
                    df[f"momentum_{period}d"] = df["price"].pct_change(periods=period)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add explicit price lag features (price value at t-k).

        These are typically the most predictive features for gradient-boosted
        trees on time-series tasks because they directly encode the temporal
        ordering of observations that the model cannot infer on its own.
        """
        if "price" not in df.columns:
            return df

        lag_periods: List[int] = getattr(self.config, "lag_periods", [1, 2, 3, 5, 7])
        for lag in lag_periods:
            if "league_name" in df.columns:
                df[f"price_lag_{lag}d"] = df.groupby("league_name")["price"].transform(
                    lambda x, k=lag: x.shift(k)
                )
            else:
                df[f"price_lag_{lag}d"] = df["price"].shift(lag)

        return df

    def _engineer_league_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create causal league-context features using expanding windows.

        Only expanding (cumulative-from-start) statistics are used.  Global
        per-league aggregates were removed because they incorporate future
        prices at the time of early-league observations (data leakage).
        """
        if "league_name" not in df.columns or "price" not in df.columns:
            return df

        grp = df.groupby("league_name")["price"]

        df["price_exp_mean"] = grp.transform(lambda x: x.expanding().mean())
        df["price_exp_std"] = grp.transform(lambda x: x.expanding().std())
        df["price_exp_min"] = grp.transform(lambda x: x.expanding().min())
        df["price_exp_max"] = grp.transform(lambda x: x.expanding().max())

        # Price relative to expanding mean (mean-reversion signal)
        df["price_vs_exp_mean"] = df["price"] / df["price_exp_mean"].replace(0, np.nan)

        # Expanding z-score: where are we in the league's experienced price range?
        valid_std = df["price_exp_std"] > 1e-8
        df["price_exp_zscore"] = np.nan
        df.loc[valid_std, "price_exp_zscore"] = (
            (df.loc[valid_std, "price"] - df.loc[valid_std, "price_exp_mean"])
            / df.loc[valid_std, "price_exp_std"]
        )

        # Price position relative to league high/low (causal expanding bounds)
        df["price_pct_of_league_high"] = df["price"] / df["price_exp_max"].replace(0, np.nan)
        df["price_pct_of_league_low"] = df["price"] / df["price_exp_min"].replace(0, np.nan)

        # Williams %R in league context: 0 = at league low, 1 = at league high.
        # More directly interpretable than z-score for tree splits.
        league_range = (df["price_exp_max"] - df["price_exp_min"]).replace(0, np.nan)
        df["price_recovery_strength"] = (df["price"] - df["price_exp_min"]) / league_range

        # Days since the expanding high/low was last set.
        # If the league high was yesterday → likely momentum continuation.
        # If 20 days ago → likely in a sustained downtrend.
        def _days_since_extreme(prices: pd.Series, mode: str) -> pd.Series:
            result = np.zeros(len(prices), dtype=float)
            extreme = prices.iloc[0]
            last_extreme_pos = 0
            for i, val in enumerate(prices):
                if mode == "high" and val >= extreme:
                    extreme = val
                    last_extreme_pos = i
                elif mode == "low" and val <= extreme:
                    extreme = val
                    last_extreme_pos = i
                result[i] = i - last_extreme_pos
            return pd.Series(result, index=prices.index)

        df["days_since_league_high"] = df.groupby("league_name")["price"].transform(
            lambda x: _days_since_extreme(x, "high")
        )
        df["days_since_league_low"] = df.groupby("league_name")["price"].transform(
            lambda x: _days_since_extreme(x, "low")
        )

        # League recency rank: newer leagues get a lower rank number.
        # This is static metadata — no future price information.
        if "date" in df.columns:
            league_starts = df.groupby("league_name")["date"].min()
        elif "league_start" in df.columns:
            league_starts = df.groupby("league_name")["league_start"].first()
        else:
            return df

        recency_rank = league_starts.rank(method="dense", ascending=False)
        df["league_recency_rank"] = df["league_name"].map(recency_rank)

        return df

    def _engineer_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling-window statistics, Bollinger Band %B, EMA, and MACD.

        Window=1 is intentionally skipped: a 1-period rolling mean/min/max
        is identical to the raw price, and a 1-period range is always 0 —
        both carry zero additional information.
        """
        if "price" not in df.columns:
            return df

        if "league_name" in df.columns and "date" in df.columns:
            df = df.sort_values(["league_name", "date"])
        elif "date" in df.columns:
            df = df.sort_values("date")

        rolling_windows: List[int] = getattr(self.config, "rolling_windows", [3, 5, 7])

        for window in rolling_windows:
            if window < 2:
                continue  # window=1 stats equal the raw price — no value added

            min_periods_std = max(2, min(window, 2))

            if "league_name" in df.columns:
                df[f"price_mean_{window}d"] = df.groupby("league_name")[
                    "price"
                ].transform(lambda x, w=window: x.rolling(w, min_periods=1).mean())

                df[f"price_std_{window}d"] = df.groupby("league_name")[
                    "price"
                ].transform(
                    lambda x, w=window, mp=min_periods_std: x.rolling(
                        w, min_periods=mp
                    ).std()
                )

                df[f"price_min_{window}d"] = df.groupby("league_name")[
                    "price"
                ].transform(lambda x, w=window: x.rolling(w, min_periods=1).min())

                df[f"price_max_{window}d"] = df.groupby("league_name")[
                    "price"
                ].transform(lambda x, w=window: x.rolling(w, min_periods=1).max())
            else:
                df[f"price_mean_{window}d"] = df["price"].rolling(
                    window, min_periods=1
                ).mean()
                df[f"price_std_{window}d"] = df["price"].rolling(
                    window, min_periods=min_periods_std
                ).std()
                df[f"price_min_{window}d"] = df["price"].rolling(
                    window, min_periods=1
                ).min()
                df[f"price_max_{window}d"] = df["price"].rolling(
                    window, min_periods=1
                ).max()

            # Z-score (only where std is well-defined and positive)
            std_col = f"price_std_{window}d"
            mean_col = f"price_mean_{window}d"
            valid_std = df[std_col].notna() & (df[std_col] > 1e-8)
            df[f"price_zscore_{window}d"] = np.nan
            df.loc[valid_std, f"price_zscore_{window}d"] = (
                (df.loc[valid_std, "price"] - df.loc[valid_std, mean_col])
                / df.loc[valid_std, std_col]
            )

            self.logger.debug(
                f"Rolling {window}d: {df[std_col].isna().sum()}/{len(df)} NaN in std"
            )

        self._add_ema_macd_features(df)
        self._add_volatility_features(df)
        self._add_trend_strength_features(df)

        return df

    def _add_ema_macd_features(self, df: pd.DataFrame) -> None:
        """
        Add exponential moving average features and the MACD oscillator.

        EMA gives greater weight to recent observations than a simple moving
        average, making it more responsive to short-term price moves.

        MACD spans (5/14/3) are scaled for league lengths of ≤60 days.
        The standard 12/26/9 configuration would use more than a quarter of
        the available league history just to warm up.
        """
        if "price" not in df.columns:
            return

        ema_spans: List[int] = getattr(self.config, "ema_spans", [3, 7, 14])
        for span in ema_spans:
            if "league_name" in df.columns:
                df[f"price_ema_{span}d"] = df.groupby("league_name")[
                    "price"
                ].transform(lambda x, s=span: x.ewm(span=s, adjust=False).mean())
            else:
                df[f"price_ema_{span}d"] = df["price"].ewm(
                    span=span, adjust=False
                ).mean()

        # Price-to-EMA7 ratio: compact mean-reversion signal
        if "price_ema_7d" in df.columns:
            df["price_to_ema7"] = df["price"] / df["price_ema_7d"].replace(0, np.nan)

        fast_span: int = getattr(self.config, "macd_fast_span", 5)
        slow_span: int = getattr(self.config, "macd_slow_span", 14)
        signal_span: int = getattr(self.config, "macd_signal_span", 3)

        if "league_name" in df.columns:
            ema_fast = df.groupby("league_name")["price"].transform(
                lambda x, s=fast_span: x.ewm(span=s, adjust=False).mean()
            )
            ema_slow = df.groupby("league_name")["price"].transform(
                lambda x, s=slow_span: x.ewm(span=s, adjust=False).mean()
            )
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df.groupby("league_name")["macd"].transform(
                lambda x, s=signal_span: x.ewm(span=s, adjust=False).mean()
            )
        else:
            ema_fast = df["price"].ewm(span=fast_span, adjust=False).mean()
            ema_slow = df["price"].ewm(span=slow_span, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=signal_span, adjust=False).mean()

        df["macd_hist"] = df["macd"] - df["macd_signal"]

    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        """
        Add volatility features derived from rolling statistics.

        Two measures are computed per window:

        - ``price_volatility_{w}d``: coefficient of variation (std / mean).
          Normalised volatility, comparable across different price levels.
        - ``volatility_garch_{w}d``: EWM realised volatility of percentage returns.
          Captures volatility clustering — large moves tend to follow large moves.

        ``volatility_clustering_7d`` (mean squared successive difference) is
        computed for the 7-day window only — the 3d/5d variants had negligible
        importance and are redundant with the garch measure.

        ``price_std_{w}d`` and ``price_mean_{w}d`` (computed in
        ``_engineer_rolling_features``) are reused here to avoid redundant
        computation.
        """
        if "price" not in df.columns:
            return

        volatility_windows: List[int] = getattr(
            self.config, "volatility_windows", [3, 5, 7]
        )
        ewm_alpha: float = 0.1

        for window in volatility_windows:
            std_col = f"price_std_{window}d"
            mean_col = f"price_mean_{window}d"

            # Coefficient of Variation (normalised volatility)
            if std_col in df.columns and mean_col in df.columns:
                valid_mean = df[mean_col] > 1e-8
                df[f"price_volatility_{window}d"] = 0.0
                df.loc[valid_mean, f"price_volatility_{window}d"] = (
                    df.loc[valid_mean, std_col] / df.loc[valid_mean, mean_col]
                )
            else:
                df[f"price_volatility_{window}d"] = 0.0

            # EWM realised volatility of log returns (GARCH-like)
            if "league_name" in df.columns:
                df[f"volatility_garch_{window}d"] = df.groupby("league_name")[
                    "price"
                ].transform(
                    lambda x, a=ewm_alpha, w=window: x.pct_change()
                    .ewm(alpha=a, min_periods=max(w // 2, 1))
                    .std()
                )
            else:
                df[f"volatility_garch_{window}d"] = (
                    df["price"]
                    .pct_change()
                    .ewm(alpha=ewm_alpha, min_periods=max(window // 2, 1))
                    .std()
                )
            df[f"volatility_garch_{window}d"] = (
                df[f"volatility_garch_{window}d"].fillna(0.0)
            )

            # Volatility clustering: mean squared successive price difference.
            # Only computed for the 7d window — 3d/5d variants are redundant
            # with volatility_garch and had negligible importance.
            if window == 7:
                if "league_name" in df.columns:
                    df["volatility_clustering_7d"] = df.groupby("league_name")[
                        "price"
                    ].transform(
                        lambda x: x.rolling(7, min_periods=2).apply(
                            lambda y: float(
                                np.sum(np.diff(y) ** 2) / max(len(y) - 1, 1)
                            ),
                            raw=True,
                        )
                    )
                else:
                    df["volatility_clustering_7d"] = df["price"].rolling(
                        7, min_periods=2
                    ).apply(
                        lambda y: float(np.sum(np.diff(y) ** 2) / max(len(y) - 1, 1)),
                        raw=True,
                    )
                df["volatility_clustering_7d"] = df["volatility_clustering_7d"].fillna(0.0)

    def _add_trend_strength_features(self, df: pd.DataFrame) -> None:
        """
        Add linear trend slope features over rolling windows.

        The OLS slope of the last *w* prices indicates whether price has been
        trending up (positive) or down (negative).  Only computed for windows
        >= 7 — shorter windows (3d, 5d) produced near-zero importance and OLS
        over 3 data points is statistically unreliable.
        """
        if "price" not in df.columns:
            return

        def _polyfit_slope(prices: np.ndarray, window: int) -> float:
            if len(prices) < window:
                return 0.0
            x = np.arange(window, dtype=float)
            try:
                return float(np.polyfit(x, prices[-window:], 1)[0])
            except Exception:
                return 0.0

        rolling_windows: List[int] = getattr(self.config, "rolling_windows", [3, 5, 7])
        trend_windows = [w for w in rolling_windows if w >= 7]

        for window in trend_windows:
            if "league_name" in df.columns:
                df[f"trend_strength_{window}d"] = df.groupby("league_name")[
                    "price"
                ].transform(
                    lambda x, w=window: x.rolling(w, min_periods=w).apply(
                        lambda y: _polyfit_slope(y, w), raw=True
                    )
                )
            else:
                df[f"trend_strength_{window}d"] = df["price"].rolling(
                    window, min_periods=window
                ).apply(lambda y, w=window: _polyfit_slope(y, w), raw=True)


    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create forward-shifted price targets for each prediction horizon."""
        if "price" not in df.columns:
            return df

        original_len = len(df)
        prediction_horizons: List[int] = getattr(
            self.config, "prediction_horizons", [1, 3, 7]
        )

        for horizon in prediction_horizons:
            if "league_name" in df.columns:
                df[f"target_price_{horizon}d"] = df.groupby("league_name")[
                    "price"
                ].transform(lambda x, h=horizon: x.shift(-h))
            else:
                df[f"target_price_{horizon}d"] = df["price"].shift(-horizon)

            df[f"target_change_{horizon}d"] = (
                df[f"target_price_{horizon}d"] - df["price"]
            )
            df[f"target_change_pct_{horizon}d"] = (
                df[f"target_change_{horizon}d"] / df["price"]
            ) * 100.0

            df[f"target_direction_{horizon}d"] = pd.cut(
                df[f"target_change_pct_{horizon}d"],
                bins=[-np.inf, -2.0, 2.0, np.inf],
                labels=["down", "stable", "up"],
            )

        if len(prediction_horizons) > 1:
            target_price_cols = [f"target_price_{h}d" for h in prediction_horizons]
            df["_multi_output_targets"] = ",".join(target_price_cols)

        target_price_cols = [c for c in df.columns if c.startswith("target_price_")]
        df = df.dropna(subset=target_price_cols, how="all")

        self.logger.debug(
            "Target creation complete",
            extra={
                "original_samples": original_len,
                "final_samples": len(df),
                "removed_samples": original_len - len(df),
            },
        )

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove price outliers using a per-league IQR filter.

        Computing IQR within each league avoids incorrectly flagging prices
        that are valid for one league but extreme relative to another (e.g.
        a currency priced at 50c in an old high-inflation league and 5c in a
        new economy-reset league).
        """
        if "price" not in df.columns:
            return df

        original_len = len(df)
        multiplier: float = getattr(
            self.config, "outlier_removal_iqr_multiplier", 3.0
        )

        if "league_name" in df.columns:
            bounds = (
                df.groupby("league_name")["price"]
                .quantile([0.25, 0.75])
                .unstack()
                .rename(columns={0.25: "q1", 0.75: "q3"})
            )
            bounds["iqr"] = bounds["q3"] - bounds["q1"]
            bounds["_outlier_lower"] = bounds["q1"] - multiplier * bounds["iqr"]
            bounds["_outlier_upper"] = bounds["q3"] + multiplier * bounds["iqr"]

            df = df.join(
                bounds[["_outlier_lower", "_outlier_upper"]], on="league_name"
            )
            mask = (df["price"] >= df["_outlier_lower"]) & (
                df["price"] <= df["_outlier_upper"]
            )
            df = df.loc[mask].drop(columns=["_outlier_lower", "_outlier_upper"])
        else:
            q1 = df["price"].quantile(0.25)
            q3 = df["price"].quantile(0.75)
            iqr = q3 - q1
            df = df[
                (df["price"] >= q1 - multiplier * iqr)
                & (df["price"] <= q3 + multiplier * iqr)
            ]

        removed = original_len - len(df)
        if removed > 0:
            self.logger.debug(
                f"Removed {removed} outliers ({removed / original_len:.2%})"
            )

        return df.reset_index(drop=True)
