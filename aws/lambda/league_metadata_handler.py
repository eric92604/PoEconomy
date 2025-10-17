"""AWS Lambda handler that ingests league metadata from POE Watch."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

import boto3

from ml.services.league_metadata_ingestion import (
    build_league_metadata_items,
    fetch_poe_watch_leagues,
)
from ml.utils.common_utils import MLLogger, setup_standard_logging

try:
    from .config import AppEnvironment, load_environment
except ImportError:
    # Handle case when running as standalone module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import AppEnvironment, load_environment

# Set up standardized logging
LOGGER = setup_standard_logging(
    name="LeagueMetadataHandler",
    level=os.getenv("LOG_LEVEL", "INFO"),
    console_output=True,
    suppress_external=True
)


def _resolve_timeout(event: Optional[dict]) -> int:
    """Resolve timeout from event or environment variable."""
    if event and isinstance(event, dict):
        timeout_override = event.get("timeout_seconds")
        if isinstance(timeout_override, int) and timeout_override > 0:
            return timeout_override
    timeout_env = os.getenv("POE_WATCH_TIMEOUT")
    if timeout_env and timeout_env.isdigit():
        return int(timeout_env)
    return 30


def _resolve_force_update(event: Optional[dict]) -> bool:
    """Resolve force update flag from event or environment variable."""
    if event and isinstance(event, dict):
        force_override = event.get("force_update")
        if isinstance(force_override, bool):
            return force_override
    force_env = os.getenv("FORCE_LEAGUE_UPDATE", "false").lower()
    return force_env in ("true", "1", "yes")


def lambda_handler(event: Optional[dict], _context) -> Dict[str, object]:
    """Entry point for the POE Watch league metadata ingestion Lambda."""
    LOGGER.debug("Received event: %s", json.dumps(event or {}))

    app_env = load_environment()
    timeout = _resolve_timeout(event)
    force_update = _resolve_force_update(event)

    session = boto3.session.Session(region_name=app_env.region_name)
    dynamodb = session.resource("dynamodb")
    league_metadata_table = dynamodb.Table(app_env.league_metadata_table)

    iso_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger = MLLogger("POEWatchLeagueMetadataLambda", level=os.getenv("LOG_LEVEL", "INFO"))

    results: Dict[str, object] = {
        "processed_leagues": [],
        "total_league_items": 0,
        "updated_leagues": [],
        "new_leagues": [],
        "timeout_seconds": timeout,
        "force_update": force_update,
    }

    try:
        logger.info("Fetching league data from POE Watch API")
        leagues_data = fetch_poe_watch_leagues(timeout=timeout)
        
        if not leagues_data:
            logger.warning("No league data received from POE Watch API")
            return results

        logger.info(f"Received {len(leagues_data)} leagues from POE Watch API")
        
        # Build league metadata items
        league_items = build_league_metadata_items(leagues_data, iso_now, force_update)
        
        if not league_items:
            logger.warning("No league metadata items to process")
            return results

        # Write to DynamoDB
        _write_batch(league_metadata_table, league_items, logger, "league metadata")

        # Update results
        results["total_league_items"] = len(league_items)
        results["processed_leagues"] = [item["league"] for item in league_items]
        
        # Categorize leagues by update type
        for item in league_items:
            if item.get("is_new_league", False):
                results["new_leagues"].append(item["league"])
            else:
                results["updated_leagues"].append(item["league"])

        logger.info("POE Watch league metadata ingestion complete.")
        logger.info(f"Processed {len(league_items)} leagues: {len(results['new_leagues'])} new, {len(results['updated_leagues'])} updated")

    except Exception as e:
        logger.error(f"Failed to ingest league metadata: {e}")
        raise

    return results


def _write_batch(table, items: List[dict], logger: MLLogger, item_type: str) -> None:
    """Write items to DynamoDB table using batch writer."""
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
