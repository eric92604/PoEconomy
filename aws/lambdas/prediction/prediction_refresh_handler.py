"""Lambda handler that refreshes cached predictions."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Iterable, Optional, Set, Tuple

from ml.services.prediction_refresh import refresh_predictions
from ml.utils.common_utils import setup_standard_logging

# Set up standardized logging
LOGGER = setup_standard_logging(
    name="PredictionRefreshHandler",
    level=os.getenv("LOG_LEVEL", "INFO"),
    console_output=True,
    suppress_external=True
)


def lambda_handler(event: Optional[dict], _context) -> Dict[str, object]:
    """
    Refresh cached predictions.

    Supports three invocation patterns:
    1. Direct invocation with an event compatible with prediction_refresh.refresh_predictions.
    2. EventBridge schedule (event is ``None`` or empty) to refresh the default top currencies.
    3. DynamoDB stream records from the LivePrices table to refresh only the affected currencies.
    """
    if event and event.get("Records") and _is_dynamodb_stream(event["Records"]):
        LOGGER.debug("Handling DynamoDB stream event.")
        dynamic_event = _build_event_from_stream(event["Records"])
        if not dynamic_event:
            LOGGER.info("No qualifying currencies found in stream batch.")
            return {"status": "no_currencies"}
        LOGGER.debug("Generated refresh payload from stream: %s", json.dumps(dynamic_event))
        return refresh_predictions(dynamic_event)

    LOGGER.debug("Invoking refresh_predictions with event: %s", json.dumps(event or {}))
    return refresh_predictions(event or {})


def _is_dynamodb_stream(records: Iterable[dict]) -> bool:
    for record in records:
        if record.get("eventSource") == "aws:dynamodb":
            return True
    return False


def _build_event_from_stream(records: Iterable[dict]) -> Optional[dict]:
    """
    Convert DynamoDB stream records into the expected event payload for refresh_predictions.
    """
    currencies: Set[Tuple[str, str]] = set()

    for record in records:
        dynamodb = record.get("dynamodb") or {}
        new_image = dynamodb.get("NewImage") or {}
        old_image = dynamodb.get("OldImage") or {}

        for image in (new_image, old_image):
            currency = _read_string(image, "currency")
            league = _read_string(image, "league")
            if currency and league:
                currencies.add((currency, league))

    if not currencies:
        return None

    payload = {
        "currencies": sorted({currency for currency, _ in currencies}),
        "leagues": sorted({league for _, league in currencies}),
    }

    if os.getenv("STREAM_REFRESH_TTL_HOURS"):
        try:
            payload["ttl_hours"] = int(os.getenv("STREAM_REFRESH_TTL_HOURS"))
        except ValueError:
            LOGGER.warning("Invalid STREAM_REFRESH_TTL_HOURS value; ignoring.")

    return payload


def _read_string(image: dict, field: str) -> Optional[str]:
    value = image.get(field)
    if isinstance(value, dict):
        return value.get("S") or value.get("s")
    if isinstance(value, str):
        return value
    return None


__all__ = ["lambda_handler"]
