#!/usr/bin/env python3
"""One-shot POE Watch ingestion pipeline that writes to DynamoDB."""

from __future__ import annotations

import argparse
import json
import time
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Optional

import boto3
import requests

from ml.config.training_config import MLConfig
from ml.utils.common_utils import MLLogger


POE_WATCH_URL = "https://api.poe.watch/get"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync POE Watch currency and fragment data into DynamoDB tables.",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="Leagues to ingest (defaults to MLConfig.included_leagues).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["currency", "fragment"],
        help="Categories to ingest (defaults to currency and fragment).",
    )
    parser.add_argument("--ttl-days", type=int, default=14, help="TTL in days for price entries.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout (seconds).")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def fetch_poe_watch_data(league: str, timeout: int, categories: List[str] = None) -> List[dict]:
    """Fetch data from POE Watch API for specified categories.
    
    Args:
        league: League name to fetch data for
        timeout: HTTP timeout in seconds
        categories: List of categories to fetch (defaults to ["currency", "fragment"])
        
    Returns:
        List of items from POE Watch API
        
    Raises:
        requests.RequestException: If API request fails
        ValueError: If response format is unexpected
    """
    if categories is None:
        categories = ["currency", "fragment"]
    
    all_data = []
    for category in categories:
        params = {"category": category, "league": league}
        response = requests.get(POE_WATCH_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected response for league {league}, category {category}: {payload!r}")
        all_data.extend(payload)
    
    return all_data


def to_decimal(value: Optional[float]) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(round(float(value), 6)))
    except (InvalidOperation, TypeError, ValueError):
        return None


def build_price_items(data: Iterable[dict], league: str, ttl_epoch: int) -> List[dict]:
    """Build price items for DynamoDB storage.
    
    Args:
        data: POE Watch API response data
        league: League name
        ttl_epoch: TTL timestamp for item expiration
        
    Returns:
        List of price items for DynamoDB batch write
    """
    timestamp = int(time.time())
    items: List[dict] = []
    for entry in data:
        name = entry.get("name")
        category = entry.get("category")
        mean_price = entry.get("mean")
        
        # Process items with category = "currency" or "fragment" and valid price data
        if not name or category not in ["currency", "fragment"] or mean_price is None:
            continue
            
        confidence = 0.5 if entry.get("lowConfidence") else 0.85
        item = {
            "currency_league": f"{name}#{league}",
            "timestamp": timestamp,  # Keep as int for DynamoDB sort key
            "currency": name,
            "league": league,  # Separate field for GSI
            "pay_currency": "Chaos Orb",  # Required for prediction system
            "price": to_decimal(mean_price) or Decimal("0"),  # Keep as Decimal for precision
            "confidence": to_decimal(confidence) or Decimal("0.5"),  # Keep as Decimal for precision
            "ttl": ttl_epoch,  # Keep as int for DynamoDB TTL
        }
        items.append(item)
    return items


def build_metadata_items(data: Iterable[dict], league: str, iso_now: str) -> List[dict]:
    """Build currency and fragment metadata items for DynamoDB storage.
    
    Args:
        data: POE Watch API response data
        league: League name
        iso_now: Current timestamp in ISO format
        
    Returns:
        List of metadata items for DynamoDB batch write
    """
    items: List[dict] = []
    for entry in data:
        name = entry.get("name")
        category = entry.get("category")
        
        # Process items with category = "currency" or "fragment"
        if not name or category not in ["currency", "fragment"]:
            continue

        # Store currency/fragment metadata as individual fields (excluding price data)
        item = {
            "currency": name,  # Primary key for DynamoDB table
            "league": league,
            "currency_id": int(entry.get("id", 0)) if entry.get("id") is not None else None,
            "category": category,  # "currency" or "fragment"
            "group": entry.get("group"),
            "frame": int(entry.get("frame", 0)) if entry.get("frame") is not None else None,
            "influences": entry.get("influences"),
            "icon_url": entry.get("icon"),
            "implicits": entry.get("implicits"),
            "explicits": entry.get("explicits"),
            "item_level": int(entry.get("itemLevel", 0)) if entry.get("itemLevel") is not None else None,
            "width": int(entry.get("width", 0)) if entry.get("width") is not None else None,
            "height": int(entry.get("height", 0)) if entry.get("height") is not None else None,
            "is_available": not entry.get("lowConfidence", False),
            "last_updated": iso_now,
        }
        
        # Remove None values to avoid DynamoDB complaints
        sanitized_item = {k: v for k, v in item.items() if v is not None}
        items.append(sanitized_item)
    
    return items


def write_batch(table, items: List[dict], logger: MLLogger, item_type: str) -> None:
    if not items:
        logger.warning(f"No {item_type} items to write.")
        return

    key_names = [key["AttributeName"] for key in table.key_schema]
    with table.batch_writer(overwrite_by_pkeys=key_names) as batch:
        for item in items:
            # Remove None values to avoid Dynamo complaints
            sanitized = {k: v for k, v in item.items() if v is not None}
            batch.put_item(Item=sanitized)
    logger.info(f"Wrote {len(items)} {item_type} records to {table.name}")


def main() -> None:
    args = parse_args()
    logger = MLLogger("POEWatchIngestion", level="DEBUG" if args.debug else "INFO")
    config = MLConfig()

    leagues = args.leagues or list(config.data.included_leagues)
    if not leagues:
        raise ValueError("No leagues specified and MLConfig.included_leagues is empty.")

    session = boto3.session.Session(region_name=config.dynamo.region_name)
    dynamodb = session.resource("dynamodb")
    prices_table = dynamodb.Table(config.dynamo.currency_prices_table)
    metadata_table = dynamodb.Table(config.dynamo.currency_metadata_table)

    ttl_epoch = int(time.time()) + args.ttl_days * 86400
    iso_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for league in leagues:
        logger.info(f"Ingesting POE Watch data for league {league} (categories: {args.categories})")
        data = fetch_poe_watch_data(league, timeout=args.timeout, categories=args.categories)
        price_items = build_price_items(data, league, ttl_epoch)
        metadata_items = build_metadata_items(data, league, iso_now)

        write_batch(prices_table, price_items, logger, "price")
        write_batch(metadata_table, metadata_items, logger, "metadata")

    logger.info("POE Watch ingestion complete.")


if __name__ == "__main__":
    main()
