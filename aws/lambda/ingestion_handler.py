"""AWS Lambda handler that ingests currency data from POE Watch."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

import boto3

from ml.config.training_config import MLConfig
from ml.services.poe_watch_ingestion import (
    build_metadata_items,
    build_price_items,
    fetch_poe_watch_data,
)
from ml.utils.common_utils import MLLogger, setup_standard_logging

from .config import AppEnvironment, load_environment

# Set up standardized logging
LOGGER = setup_standard_logging(
    name="IngestionHandler",
    level=os.getenv("LOG_LEVEL", "INFO"),
    console_output=True,
    suppress_external=True
)


def _resolve_leagues(app_env: AppEnvironment, event: Optional[dict]) -> List[str]:
    if event and isinstance(event, dict) and event.get("leagues"):
        leagues = event["leagues"]
        if isinstance(leagues, str):
            return [leagues]
        if isinstance(leagues, list):
            return [league for league in leagues if isinstance(league, str)]

    leagues_env = os.getenv("INGEST_LEAGUES")
    if leagues_env:
        return [entry.strip() for entry in leagues_env.split(",") if entry.strip()]

    if app_env.default_leagues:
        return app_env.default_leagues

    # Try to get current active seasonal league from POE Watch API
    try:
        from ml.utils.data_sources import create_data_source, DataSourceConfig
        from ml.config.training_config import DynamoConfig
        
        # Create data source to get current active seasonal league
        dynamo_config = DynamoConfig(
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
            currency_metadata_table=os.getenv("DYNAMO_CURRENCY_METADATA_TABLE"),
            currency_prices_table=os.getenv("DYNAMO_CURRENCY_PRICES_TABLE"),
            league_metadata_table=os.getenv("DYNAMO_LEAGUE_METADATA_TABLE")
        )
        data_source_config = DataSourceConfig.from_dynamo_config(dynamo_config)
        data_source = create_data_source(data_source_config)
        current_league = data_source.get_most_recent_league()
        if current_league:
            LOGGER.info("Using current active seasonal league: %s", current_league)
            return [current_league]
        else:
            LOGGER.warning("No active seasonal league found")
                
    except Exception as e:
        LOGGER.warning("Failed to get current active seasonal league: %s", e)
    
    # Fallback to available historical leagues if no configuration is available
    try:
        ml_config = MLConfig()
        if ml_config.data.included_leagues:
            return list(ml_config.data.included_leagues)
    except Exception as e:
        LOGGER.warning("Failed to load ML config: %s", e)
    
    # Final fallback - use Standard league which is always available
    LOGGER.warning("Using Standard league as final fallback")
    return ["Standard"]


def _resolve_ttl_days(event: Optional[dict]) -> int:
    if event and isinstance(event, dict):
        ttl_override = event.get("ttl_days")
        if isinstance(ttl_override, int) and ttl_override > 0:
            return ttl_override
    ttl_env = os.getenv("INGEST_TTL_DAYS")
    if ttl_env and ttl_env.isdigit():
        return int(ttl_env)
    return 14


def _resolve_timeout(event: Optional[dict]) -> int:
    if event and isinstance(event, dict):
        timeout_override = event.get("timeout_seconds")
        if isinstance(timeout_override, int) and timeout_override > 0:
            return timeout_override
    timeout_env = os.getenv("POE_WATCH_TIMEOUT")
    if timeout_env and timeout_env.isdigit():
        return int(timeout_env)
    return 30


def _resolve_categories(event: Optional[dict]) -> List[str]:
    """Resolve categories to ingest from event or environment."""
    if event and isinstance(event, dict) and event.get("categories"):
        categories = event["categories"]
        if isinstance(categories, str):
            return [categories]
        if isinstance(categories, list):
            return [cat for cat in categories if isinstance(cat, str)]
    
    categories_env = os.getenv("INGEST_CATEGORIES")
    if categories_env:
        return [entry.strip() for entry in categories_env.split(",") if entry.strip()]
    
    # Default to both currency and fragment
    return ["currency", "fragment"]


def lambda_handler(event: Optional[dict], _context) -> Dict[str, object]:
    """Entry point for the POE Watch ingestion Lambda."""
    LOGGER.debug("Received event: %s", json.dumps(event or {}))

    app_env = load_environment()
    leagues = _resolve_leagues(app_env, event)
    if not leagues:
        raise ValueError("No leagues configured for ingestion.")

    ttl_days = _resolve_ttl_days(event)
    timeout = _resolve_timeout(event)
    categories = _resolve_categories(event)

    session = boto3.session.Session(region_name=app_env.region_name)
    dynamodb = session.resource("dynamodb")
    prices_table = dynamodb.Table(app_env.live_prices_table)
    metadata_table = dynamodb.Table(app_env.currency_metadata_table)

    ttl_epoch = int(time.time()) + ttl_days * 86400
    iso_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger = MLLogger("POEWatchIngestionLambda", level=os.getenv("LOG_LEVEL", "INFO"))

    results: Dict[str, object] = {
        "processed_leagues": [],
        "total_price_items": 0,
        "total_metadata_items": 0,
        "ttl_days": ttl_days,
        "timeout_seconds": timeout,
        "categories": categories,
    }

    for league in leagues:
        logger.info(f"Ingesting data for league {league} (categories: {categories})")
        payload = fetch_poe_watch_data(league, timeout=timeout, categories=categories)
        price_items = build_price_items(payload, league, ttl_epoch)
        metadata_items = build_metadata_items(payload, league, iso_now)

        _write_batch(prices_table, price_items, logger, "price")
        _write_batch(metadata_table, metadata_items, logger, "metadata")

        results["processed_leagues"].append(league)
        results["total_price_items"] += len(price_items)
        results["total_metadata_items"] += len(metadata_items)

    logger.info("POE Watch ingestion complete.")
    
    # Trigger prediction refresh after successful ingestion
    _trigger_prediction_refresh(app_env, logger)
    
    return results


def _trigger_prediction_refresh(app_env: AppEnvironment, logger: MLLogger) -> None:
    """Trigger prediction refresh after successful ingestion."""
    try:
        lambda_client = boto3.client("lambda", region_name=app_env.region_name)
        function_name = os.getenv("PREDICTION_REFRESH_FUNCTION_NAME")
        
        if not function_name:
            logger.warning("PREDICTION_REFRESH_FUNCTION_NAME not set, skipping prediction refresh trigger")
            return
        
        # Invoke prediction refresh asynchronously
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="Event",  # Asynchronous invocation
            Payload=json.dumps({
                "currencies": None,  # Let it select default currencies
                "horizons": ["1d"],  # Default horizon
                "ttl_hours": int(os.getenv("PREDICTION_CACHE_TTL_HOURS", "2")),  # Use env var for TTL
            })
        )
        
        logger.info(f"Triggered prediction refresh: {function_name}")
        logger.debug(f"Lambda response: {response}")
        
    except Exception as exc:
        logger.warning(f"Failed to trigger prediction refresh: {exc}")
        # Don't fail the ingestion if prediction refresh fails


def _write_batch(table, items: List[dict], logger: MLLogger, item_type: str) -> None:
    if not items:
        logger.warning(f"No {item_type} items to write.")
        return

    key_names = [key["AttributeName"] for key in table.key_schema]
    with table.batch_writer(overwrite_by_pkeys=key_names) as batch:
        for item in items:
            sanitized = {k: v for k, v in item.items() if v is not None}
            batch.put_item(Item=sanitized)
    logger.info(f"Wrote {len(items)} {item_type} records to {table.name}")


__all__ = ["lambda_handler"]

