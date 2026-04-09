"""Shared configuration helpers for AWS Lambda handlers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required for this Lambda function.")
    return value


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class AppEnvironment:
    """Resolved environment variables injected into the Lambda functions."""

    region_name: str
    data_lake_bucket: Optional[str]
    models_bucket: Optional[str]
    currency_metadata_table: str
    live_prices_table: str
    predictions_table: str
    league_metadata_table: Optional[str]
    historical_league_prices_table: Optional[str]
    default_leagues: List[str]
    models_s3_prefix: Optional[str]
    models_local_dir: str


def load_environment(default_leagues: Optional[List[str]] = None) -> AppEnvironment:
    """Build an :class:`AppEnvironment` from Lambda environment variables."""

    leagues_from_env = _split_csv(os.getenv("DEFAULT_LEAGUES"))
    if not leagues_from_env and default_leagues:
        leagues_from_env = list(default_leagues)

    return AppEnvironment(
        region_name=os.getenv("AWS_REGION", "us-west-2"),
        data_lake_bucket=os.getenv("DATA_LAKE_BUCKET"),
        models_bucket=os.getenv("MODELS_BUCKET"),
        currency_metadata_table=_require_env("DYNAMO_CURRENCY_METADATA_TABLE"),
        live_prices_table=_require_env("DYNAMO_CURRENCY_PRICES_TABLE"),
        predictions_table=os.getenv("DYNAMO_PREDICTIONS_TABLE"),
        league_metadata_table=os.getenv("DYNAMO_LEAGUE_METADATA_TABLE"),
        historical_league_prices_table=os.getenv("HISTORICAL_LEAGUE_PRICES_TABLE"),
        default_leagues=leagues_from_env,
        models_s3_prefix=os.getenv("MODELS_S3_PREFIX"),
        models_local_dir=os.getenv("MODELS_DIR", "/var/task/models"),
    )


__all__ = ["AppEnvironment", "load_environment"]

