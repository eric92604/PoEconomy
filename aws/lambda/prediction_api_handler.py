"""Lambda handler that exposes the prediction API over API Gateway."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Key

from .config import AppEnvironment, load_environment

# Set up standardized logging
from ml.utils.common_utils import setup_standard_logging
LOGGER = setup_standard_logging(
    name="PredictionAPIHandler",
    level=os.getenv("LOG_LEVEL", "INFO"),
    console_output=True,
    suppress_external=True
)

_APP_ENV: Optional[AppEnvironment] = None
_DYNAMO_RESOURCE = None
_METADATA_TABLE = None
_PREDICTIONS_TABLE = None
_PRICES_TABLE = None

# In-memory cache for live prices (since data only changes every hour)
_PRICES_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
_CACHE_TTL: int = 55 * 60  # 55 minutes (5 minutes before next ingestion)

def lambda_handler(event: dict, _context) -> dict:
    """Handle API Gateway proxy events."""
    global _APP_ENV, _DYNAMO_RESOURCE, _METADATA_TABLE, _PREDICTIONS_TABLE, _PRICES_TABLE

    if _APP_ENV is None:
        _APP_ENV = load_environment()

    if _DYNAMO_RESOURCE is None:
        _DYNAMO_RESOURCE = boto3.resource("dynamodb", region_name=_APP_ENV.region_name)
        _METADATA_TABLE = _DYNAMO_RESOURCE.Table(_APP_ENV.currency_metadata_table)
        _PREDICTIONS_TABLE = _DYNAMO_RESOURCE.Table(_APP_ENV.predictions_table)
        _PRICES_TABLE = _DYNAMO_RESOURCE.Table(_APP_ENV.live_prices_table)

    request = ApiRequest.from_event(event)
    LOGGER.debug("Handling %s %s", request.http_method, request.path)

    try:
        if request.path in ("/health", "") and request.http_method == "GET":
            response_body = {"status": "ok"}
            status = HTTPStatus.OK
        elif request.path == "/predict/currencies" and request.http_method == "GET":
            response_body = _list_currencies()
            status = HTTPStatus.OK
        elif request.path == "/predict/leagues" and request.http_method == "GET":
            response_body = _list_leagues()
            status = HTTPStatus.OK
        elif request.path == "/predict/single" and request.http_method == "POST":
            response_body, status = _handle_single_prediction(request.json_body)
        elif request.path == "/predict/batch" and request.http_method == "POST":
            response_body, status = _handle_batch_prediction(request.json_body)
        elif request.path == "/predict/currency" and request.http_method == "GET":
            response_body, status = _handle_currency_predictions(request.query_params)
        elif request.path == "/predict/latest" and request.http_method == "GET":
            response_body, status = _handle_latest_predictions(request.query_params)
        elif request.path == "/prices/live" and request.http_method == "GET":
            response_body, status = _handle_live_prices(request.query_params)
        else:
            response_body = {"message": f"Unsupported route {request.http_method} {request.path}"}
            status = HTTPStatus.NOT_FOUND
    except ClientFacingError as exc:
        LOGGER.warning("Client error: %s", exc)
        response_body = {"message": str(exc)}
        status = exc.status
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Unhandled error during API invocation")
        response_body = {"message": "Internal server error", "detail": str(exc)}
        status = HTTPStatus.INTERNAL_SERVER_ERROR

    # Add cache headers for specific endpoints
    headers = _cors_headers()
    if request.path == "/prices/live" and status == HTTPStatus.OK:
        headers.update({
            "Cache-Control": "public, max-age=300",  # 5 minutes browser cache
            "ETag": f'"{hash(str(response_body))}"',  # Simple ETag for conditional requests
        })
    elif request.path == "/predict/latest" and status == HTTPStatus.OK:
        headers.update({
            "Cache-Control": "public, max-age=600",  # 10 minutes browser cache (optimal for fresh data)
            "ETag": f'"{hash(str(response_body))}"',  # Simple ETag for conditional requests
        })
    
    return {
        "statusCode": status.value,
        "headers": headers,
        "body": json.dumps(response_body, default=_dynamodb_encoder),
    }


@dataclass
class ApiRequest:
    http_method: str
    raw_path: str
    path: str
    json_body: Dict[str, Any]
    query_params: Dict[str, str]

    @classmethod
    def from_event(cls, event: dict) -> "ApiRequest":
        http_method = (event.get("httpMethod") or "GET").upper()
        path = event.get("path") or event.get("resource") or "/"

        path_params = event.get("pathParameters") or {}
        if isinstance(path_params, dict) and path_params.get("proxy"):
            path = "/" + path_params["proxy"]

        body = event.get("body")
        if body and isinstance(body, str):
            try:
                json_body = json.loads(body)
            except json.JSONDecodeError:
                json_body = {}
        elif isinstance(body, dict):
            json_body = body
        else:
            json_body = {}

        # Parse query parameters
        query_params = event.get("queryStringParameters") or {}
        if not isinstance(query_params, dict):
            query_params = {}

        normalised_path = "/" + path.strip("/")
        if normalised_path == "//":
            normalised_path = "/"

        return cls(
            http_method=http_method,
            raw_path=path,
            path=normalised_path,
            json_body=json_body,
            query_params=query_params,
        )


class ClientFacingError(Exception):
    """Raised for validation errors that should be returned to the client."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def _list_currencies() -> Dict[str, Any]:
    assert _METADATA_TABLE is not None
    response = _METADATA_TABLE.scan(ProjectionExpression="currency_name, league, metadata_json, last_updated")
    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = _METADATA_TABLE.scan(
            ProjectionExpression="currency_name, league, metadata_json, last_updated",
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        items.extend(response.get("Items", []))

    currencies: Dict[str, Dict[str, Any]] = {}
    for item in items:
        currency = item.get("currency_name")
        league = item.get("league")
        if not currency or not league:
            continue
        metadata = {}
        if item.get("metadata_json"):
            try:
                metadata = json.loads(item["metadata_json"])
            except json.JSONDecodeError:
                metadata = {}
        currencies.setdefault(currency, {})[league] = metadata

    return {"currencies": currencies}


def _list_leagues() -> Dict[str, Any]:
    assert _METADATA_TABLE is not None
    response = _METADATA_TABLE.scan(ProjectionExpression="league, last_updated")
    leagues: Dict[str, Any] = {}
    for item in response.get("Items", []):
        league = item.get("league")
        if not league:
            continue
        leagues.setdefault(league, {"currency_count": 0})
        leagues[league]["currency_count"] += 1
    while "LastEvaluatedKey" in response:
        response = _METADATA_TABLE.scan(
            ProjectionExpression="league, last_updated",
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        for item in response.get("Items", []):
            league = item.get("league")
            if not league:
                continue
            leagues.setdefault(league, {"currency_count": 0})
            leagues[league]["currency_count"] += 1
    return {"leagues": leagues}


def _handle_single_prediction(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], HTTPStatus]:
    currency = _require_string(payload, "currency")
    league = payload.get("league")
    horizon = payload.get("horizon", "1d")

    stored = _fetch_cached_prediction(currency, league, horizon)
    if stored:
        return stored, HTTPStatus.OK

    return (
        {"message": f"No cached prediction for {currency} (league={league!r}, horizon={horizon})"},
        HTTPStatus.NOT_FOUND,
    )


def _handle_batch_prediction(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], HTTPStatus]:
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ClientFacingError("requests must be a non-empty list.")

    responses = []
    for entry in requests:
        if not isinstance(entry, dict):
            raise ClientFacingError("Each request must be an object.")
        currency = _require_string(entry, "currency")
        league = entry.get("league")
        horizon = entry.get("horizon", "1d")
        result = _fetch_cached_prediction(currency, league, horizon)
        if result:
            responses.append(result)

    return {"results": responses}, HTTPStatus.OK


def _handle_currency_predictions(query_params: Dict[str, str]) -> Tuple[Dict[str, Any], HTTPStatus]:
    """Handle currency predictions endpoint using the new GSI for efficient queries.
    
    GET /predict/currency?currency=Divine Orb&league=Mercenaries&horizons=1d,3d,7d
    
    Args:
        query_params: Query parameters from the request
        
    Returns:
        Tuple of (response_body, status_code)
    """
    currency = query_params.get("currency")
    if not currency:
        raise ClientFacingError("currency parameter is required")
    
    league = query_params.get("league")
    horizons_param = query_params.get("horizons", "1d,3d,7d")
    horizons = [h.strip() for h in horizons_param.split(",") if h.strip()]
    
    if not horizons:
        raise ClientFacingError("at least one horizon must be specified")
    
    # Use the new GSI to efficiently get all predictions for this currency
    predictions = []
    for horizon in horizons:
        prediction = _fetch_cached_prediction(currency, league, horizon)
        if prediction:
            predictions.append(prediction)
    
    return {
        "currency": currency,
        "league": league or "auto-detected",
        "predictions": predictions,
        "metadata": {
            "total_horizons_requested": len(horizons),
            "predictions_found": len(predictions),
            "source": "cache",
            "query_efficiency": "GSI-optimized"
        }
    }, HTTPStatus.OK


def _handle_latest_predictions(query_params: Dict[str, str]) -> Tuple[Dict[str, Any], HTTPStatus]:
    """Handle latest predictions endpoint for frontend dashboard.
    
    This endpoint is optimized for frontend applications that need to load
    all currency predictions across all horizons for the latest prediction time.
    
    GET /predict/latest?league=Mercenaries&horizons=1d,3d,7d&limit=50
    
    Args:
        query_params: Query parameters from the request
        
    Returns:
        Tuple of (response_body, status_code)
    """
    league = query_params.get("league")
    horizons_param = query_params.get("horizons", "1d,3d,7d")
    horizons = [h.strip() for h in horizons_param.split(",") if h.strip()]
    limit = int(query_params.get("limit", "100"))
    
    # Validate parameters
    if limit <= 0 or limit > 500:
        raise ClientFacingError("limit must be between 1 and 500")
    if not horizons:
        raise ClientFacingError("at least one horizon must be specified")
    
    # Fetch fresh data from DynamoDB using efficient GSI queries
    # Note: Caching is handled by Cloudflare Worker for better performance
    predictions_data = _fetch_latest_predictions_from_db(league, horizons, limit)
    
    return predictions_data, HTTPStatus.OK


def _fetch_latest_predictions_from_db(
    league: Optional[str], 
    horizons: List[str], 
    limit: int
) -> Dict[str, Any]:
    """Fetch latest predictions for all currencies across specified horizons.
    
    This function uses the currency-horizon-index GSI for optimal performance
    and aggregates results by currency and horizon.
    
    Args:
        league: League name (optional, will auto-detect if not provided)
        horizons: List of prediction horizons to fetch
        limit: Maximum number of currencies to return per horizon
        
    Returns:
        Dictionary containing organized prediction data
    """
    assert _PREDICTIONS_TABLE is not None
    
    # Auto-detect league if not provided
    if not league:
        league = _infer_latest_league("Divine Orb")  # Use a common currency for league detection
        if not league:
            raise ClientFacingError("Unable to determine current league")
    
    # Get list of available currencies from metadata table
    available_currencies = _get_available_currencies(league)
    if not available_currencies:
        raise ClientFacingError(f"No currencies found for league: {league}")
    
    # Limit currencies to most popular ones for performance
    top_currencies = available_currencies[:limit]
    
    # Fetch predictions for each currency across all horizons
    all_predictions = {}
    prediction_timestamps = set()
    
    for currency in top_currencies:
        currency_predictions = {}
        
        for horizon in horizons:
            try:
                # Use GSI for efficient querying
                response = _PREDICTIONS_TABLE.query(
                    IndexName='currency-horizon-index',
                    KeyConditionExpression=Key('currency').eq(currency) & Key('horizon').eq(horizon),
                    FilterExpression=Key('league').eq(league),
                    ScanIndexForward=False,  # Most recent first
                    Limit=1
                )
                
                items = response.get("Items", [])
                if items:
                    item = items[0]
                    prediction_data = _format_prediction_item(item, currency, league, horizon)
                    currency_predictions[horizon] = prediction_data
                    prediction_timestamps.add(item.get("timestamp", 0))
                    
            except Exception as e:
                LOGGER.warning(f"Failed to fetch prediction for {currency} {horizon}: {e}")
                continue
        
        if currency_predictions:
            all_predictions[currency] = currency_predictions
    
    # Calculate latest prediction time
    latest_timestamp = max(prediction_timestamps) if prediction_timestamps else 0
    latest_time_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(latest_timestamp)) if latest_timestamp else "unknown"
    
    # Organize data for frontend consumption
    organized_data = {
        "league": league,
        "latest_prediction_time": latest_time_str,
        "latest_timestamp": latest_timestamp,
        "horizons": horizons,
        "currencies": all_predictions,
        "metadata": {
            "total_currencies": len(all_predictions),
            "total_currencies_available": len(available_currencies),
            "horizons_requested": len(horizons),
            "source": "cache",
            "query_efficiency": "GSI-optimized",
            "cache_handled_by": "Cloudflare Worker"
        }
    }
    
    return organized_data


def _get_available_currencies(league: str) -> List[str]:
    """Get list of available currencies for a specific league.
    
    Args:
        league: League name
        
    Returns:
        List of currency names, sorted by popularity/importance
    """
    assert _METADATA_TABLE is not None
    
    try:
        # Get currencies from metadata table
        response = _METADATA_TABLE.scan(
            FilterExpression=Key('league').eq(league) if 'league' in _METADATA_TABLE.attribute_definitions else None,
            ProjectionExpression='currency'
        )
        
        currencies = [item.get('currency') for item in response.get('Items', []) if item.get('currency')]
        
        # Sort by importance (common currencies first)
        priority_currencies = [
            'Divine Orb', 'Chaos Orb', 'Exalted Orb', 'Mirror of Kalandra',
            'Orb of Fusing', 'Chromatic Orb', 'Jeweller\'s Orb', 'Orb of Alchemy',
            'Vaal Orb', 'Orb of Scouring', 'Regal Orb', 'Blessed Orb'
        ]
        
        # Prioritize common currencies, then sort alphabetically
        sorted_currencies = []
        for priority_currency in priority_currencies:
            if priority_currency in currencies:
                sorted_currencies.append(priority_currency)
                currencies.remove(priority_currency)
        
        sorted_currencies.extend(sorted(currencies))
        return sorted_currencies
        
    except Exception as e:
        LOGGER.warning(f"Failed to get available currencies: {e}")
        # Fallback to common currencies
        return [
            'Divine Orb', 'Chaos Orb', 'Exalted Orb', 'Orb of Fusing',
            'Chromatic Orb', 'Jeweller\'s Orb', 'Orb of Alchemy', 'Vaal Orb'
        ]


def _format_prediction_item(item: Dict[str, Any], currency: str, league: str, horizon: str) -> Dict[str, Any]:
    """Format a prediction item for consistent API response.
    
    Args:
        item: Raw prediction item from DynamoDB
        currency: Currency name
        league: League name
        horizon: Prediction horizon
        
    Returns:
        Formatted prediction data
    """
    # Handle both old and new data formats
    if "prediction_data" in item:
        # Old format: prediction_data field contains JSON
        payload = item.get("prediction_data")
        if isinstance(payload, str):
            try:
                prediction_dict = json.loads(payload)
            except json.JSONDecodeError:
                prediction_dict = {}
        elif isinstance(payload, dict):
            prediction_dict = payload
        else:
            prediction_dict = {}
    else:
        # New format: prediction data is stored directly in item fields
        prediction_dict = {
            "predicted_price": float(item.get("predicted_price", 0)) if item.get("predicted_price") else None,
            "current_price": float(item.get("current_price", 0)) if item.get("current_price") else None,
            "price_change_percent": float(item.get("price_change_percent", 0)) if item.get("price_change_percent") else None,
            "confidence_score": float(item.get("confidence_score", 0)) if item.get("confidence_score") else None,
            "prediction_lower": float(item.get("prediction_lower", 0)) if item.get("prediction_lower") else None,
            "prediction_upper": float(item.get("prediction_upper", 0)) if item.get("prediction_upper") else None,
            "features_used": item.get("features_used", []),
            "model_path": item.get("model_path", ""),
        }
    
    # Set common fields
    prediction_dict.update({
        "currency": currency,
        "league": league,
        "horizon": horizon,
        "source": "cache",
        "prediction_timestamp": item.get("prediction_timestamp", item.get("created_at", "")),
        "timestamp": item.get("timestamp", 0)
    })
    
    return prediction_dict


def _fetch_cached_prediction(currency: str, league: Optional[str], horizon: str) -> Optional[Dict[str, Any]]:
    assert _PREDICTIONS_TABLE is not None
    league_value = league or _infer_latest_league(currency)
    if not league_value:
        return None
    
    # Use the new currency-horizon-index GSI for much more efficient queries
    # This avoids the need for FilterExpression and provides direct access by currency + horizon
    try:
        response = _PREDICTIONS_TABLE.query(
            IndexName='currency-horizon-index',
            KeyConditionExpression=Key('currency').eq(currency) & Key('horizon').eq(horizon),
            FilterExpression=Key('league').eq(league_value) if league_value else None,
            ScanIndexForward=False,  # Most recent first
            Limit=1
        )
    except Exception as e:
        LOGGER.warning(f"GSI query failed, falling back to main table query: {e}")
        # Fallback to original query method if GSI is not available yet
        currency_league = f"{currency}#{league_value}"
        response = _PREDICTIONS_TABLE.query(
            KeyConditionExpression=Key("currency_league").eq(currency_league),
            FilterExpression=Key("horizon").eq(horizon),
            ScanIndexForward=False,
            Limit=1
        )
    
    items = response.get("Items", [])
    if not items:
        return None
    
    item = items[0]  # Get the most recent prediction
    
    # Handle both old and new data formats
    if "prediction_data" in item:
        # Old format: prediction_data field contains JSON
        payload = item.get("prediction_data")
        if isinstance(payload, str):
            try:
                payload_dict = json.loads(payload)
            except json.JSONDecodeError:
                payload_dict = {}
        elif isinstance(payload, dict):
            payload_dict = payload
        else:
            payload_dict = {}
    else:
        # New format: prediction data is stored directly in item fields
        payload_dict = {
            "predicted_price": float(item.get("predicted_price", 0)) if item.get("predicted_price") else None,
            "current_price": float(item.get("current_price", 0)) if item.get("current_price") else None,
            "price_change_percent": float(item.get("price_change_percent", 0)) if item.get("price_change_percent") else None,
            "confidence_score": float(item.get("confidence_score", 0)) if item.get("confidence_score") else None,
            "prediction_lower": float(item.get("prediction_lower", 0)) if item.get("prediction_lower") else None,
            "prediction_upper": float(item.get("prediction_upper", 0)) if item.get("prediction_upper") else None,
            "features_used": item.get("features_used", []),
            "model_path": item.get("model_path", ""),
        }
    
    # Set common fields
    payload_dict.setdefault("currency", currency)
    payload_dict.setdefault("league", league_value)
    payload_dict.setdefault("horizon", horizon)
    payload_dict["source"] = "cache"
    payload_dict["prediction_timestamp"] = item.get("prediction_timestamp", item.get("created_at", ""))
    
    return payload_dict


def _infer_latest_league(currency: str) -> Optional[str]:
    # First try to get current active seasonal league from POE Watch API
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
            return current_league
    except Exception as e:
        LOGGER.warning("Failed to get current active seasonal league: %s", e)
    
    # Fallback: try to get league from metadata table
    try:
        from ml.config.training_config import DynamoConfig
        
        # Create DynamoDB configuration
        dynamo_config = DynamoConfig(
            currency_metadata_table=_METADATA_TABLE.table_name if _METADATA_TABLE else None,
            currency_prices_table=_PRICES_TABLE.table_name if _PRICES_TABLE else None,
            league_metadata_table=_METADATA_TABLE.table_name if _METADATA_TABLE else None,  # Assuming same table for now
            region_name=os.getenv("AWS_REGION", "us-west-2")
        )
        
        data_source_config = DataSourceConfig.from_dynamo_config(dynamo_config)
        data_source = create_data_source(data_source_config)
        most_recent_league = data_source.get_most_recent_league()
        if most_recent_league:
            return most_recent_league
            
    except Exception as e:
        LOGGER.warning("Failed to get league from metadata table: %s", e)
    
    # Final fallback: check cached predictions
    assert _PREDICTIONS_TABLE is not None
    response = _PREDICTIONS_TABLE.query(
        IndexName="currency-timestamp-index",
        KeyConditionExpression=Key("currency").eq(currency),
        ScanIndexForward=False,
        Limit=1,
    )
    items = response.get("Items")
    if not items:
        return None
    prediction_key = items[0].get("prediction_key", "")
    if "#" in prediction_key:
        try:
            _, league, _ = prediction_key.split("#", 2)
            return league
        except ValueError:
            return None
    return items[0].get("league")










def _require_string(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ClientFacingError(f"{key} must be a non-empty string.")
    return value.strip()


def _cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }


def _handle_live_prices(query_params: Dict[str, str]) -> Tuple[Dict[str, Any], HTTPStatus]:
    """Handle live currency prices endpoint with caching.
    
    Args:
        query_params: Query parameters from the request
        
    Returns:
        Tuple of (response_body, status_code)
    """
    currency = query_params.get("currency")
    league = query_params.get("league")
    limit = int(query_params.get("limit", "100"))
    hours = int(query_params.get("hours", "24"))
    
    # Validate parameters
    if limit <= 0 or limit > 1000:
        raise ClientFacingError("limit must be between 1 and 1000")
    if hours <= 0 or hours > 168:  # Max 1 week
        raise ClientFacingError("hours must be between 1 and 168")
    
    # Create cache key
    cache_key = f"prices:{currency or 'all'}:{league or 'all'}:{limit}:{hours}"
    
    # Check in-memory cache first
    current_time = time.time()
    if cache_key in _PRICES_CACHE:
        cached_data, cache_time = _PRICES_CACHE[cache_key]
        if current_time - cache_time < _CACHE_TTL:
            LOGGER.debug("Returning cached live prices data")
            return cached_data, HTTPStatus.OK
    
    # Fetch from DynamoDB
    prices_data = _fetch_live_prices_from_db(currency, league, limit, hours)
    
    # Cache the result
    _PRICES_CACHE[cache_key] = (prices_data, current_time)
    
    return prices_data, HTTPStatus.OK


def _fetch_live_prices_from_db(
    currency: Optional[str], 
    league: Optional[str], 
    limit: int, 
    hours: int
) -> Dict[str, Any]:
    """Fetch live prices from DynamoDB with efficient queries.
    
    Args:
        currency: Optional currency filter
        league: Optional league filter  
        limit: Maximum number of results
        hours: Hours back to look for data
        
    Returns:
        Dictionary containing prices and metadata
    """
    assert _PRICES_TABLE is not None
    
    # Calculate timestamp threshold (hours ago)
    current_time = int(time.time())
    hours_ago = current_time - (hours * 3600)
    
    prices = []
    total_count = 0
    
    try:
        if currency and league:
            # Query specific currency in specific league using GSI
            response = _PRICES_TABLE.query(
                IndexName="currency-timestamp-index",
                KeyConditionExpression=Key("currency").eq(currency) & Key("timestamp").gte(hours_ago),
                FilterExpression=Key("league").eq(league),
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )
            prices = response.get("Items", [])
            total_count = response.get("Count", 0)
            
        elif currency:
            # Query specific currency across all leagues using GSI
            response = _PRICES_TABLE.query(
                IndexName="currency-timestamp-index", 
                KeyConditionExpression=Key("currency").eq(currency) & Key("timestamp").gte(hours_ago),
                ScanIndexForward=False,
                Limit=limit
            )
            prices = response.get("Items", [])
            total_count = response.get("Count", 0)
            
        elif league:
            # Query specific league using GSI
            response = _PRICES_TABLE.query(
                IndexName="league-timestamp-index",
                KeyConditionExpression=Key("league").eq(league) & Key("timestamp").gte(hours_ago),
                ScanIndexForward=False,
                Limit=limit
            )
            prices = response.get("Items", [])
            total_count = response.get("Count", 0)
            
        else:
            # Scan for all recent prices (less efficient, but works for small datasets)
            response = _PRICES_TABLE.scan(
                FilterExpression=Key("timestamp").gte(hours_ago),
                Limit=limit
            )
            prices = response.get("Items", [])
            total_count = response.get("Count", 0)
            
    except Exception as e:
        LOGGER.error(f"Error fetching live prices: {e}")
        raise ClientFacingError("Failed to fetch live prices data")
    
    # Format response
    formatted_prices = []
    for item in prices:
        formatted_prices.append({
            "currency": item.get("currency"),
            "league": item.get("league"),
            "price": float(item.get("price", 0)),
            "confidence": float(item.get("confidence", 0)),
            "timestamp": item.get("timestamp")
        })
    
    # Get latest update time from metadata
    latest_update = _get_latest_price_update_time()
    
    return {
        "prices": formatted_prices,
        "metadata": {
            "total_count": total_count,
            "returned_count": len(formatted_prices),
            "time_range_hours": hours,
            "filters": {
                "currency": currency,
                "league": league
            },
            "last_updated": latest_update,
            "cache_info": {
                "cached": True,
                "cache_ttl_minutes": _CACHE_TTL // 60
            }
        }
    }


def _get_latest_price_update_time() -> str:
    """Get the timestamp of the most recent price update."""
    try:
        assert _PRICES_TABLE is not None
        # Get the most recent item to determine last update time
        response = _PRICES_TABLE.scan(
            Limit=1,
            ScanIndexForward=False
        )
        items = response.get("Items", [])
        if items:
            timestamp = items[0].get("timestamp")
            if timestamp:
                return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(timestamp))
    except Exception as e:
        LOGGER.warning(f"Could not determine latest update time: {e}")
    
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _dynamodb_encoder(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


__all__ = ["lambda_handler"]
