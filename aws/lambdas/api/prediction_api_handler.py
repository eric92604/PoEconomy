"""Lambda handler that exposes the prediction API over API Gateway.

- Read from DynamoDB tables
- Return JSON responses
- Handle HTTP requests

"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Key

from ml.utils.common_utils import setup_standard_logging
from ..config import AppEnvironment, load_environment

# Set up standardized logging
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
_PRICING_TABLE = None


def lambda_handler(event: dict, _context) -> dict:
    """Handle API Gateway proxy events."""
    global _APP_ENV, _DYNAMO_RESOURCE, _METADATA_TABLE, _PREDICTIONS_TABLE, _PRICING_TABLE

    if _APP_ENV is None:
        _APP_ENV = load_environment()

    if _DYNAMO_RESOURCE is None:
        _DYNAMO_RESOURCE = boto3.resource("dynamodb", region_name=_APP_ENV.region_name)
        _METADATA_TABLE = _DYNAMO_RESOURCE.Table(_APP_ENV.currency_metadata_table)
        if _APP_ENV.predictions_table:
            _PREDICTIONS_TABLE = _DYNAMO_RESOURCE.Table(_APP_ENV.predictions_table)
        else:
            raise RuntimeError("DYNAMO_PREDICTIONS_TABLE environment variable is required for the API handler")
        _PRICING_TABLE = _DYNAMO_RESOURCE.Table(_APP_ENV.live_prices_table)

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
        elif request.path == "/prices/historical" and request.http_method == "GET":
            response_body, status = _handle_historical_prices(request.query_params)
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

    headers = _cors_headers()
    
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
    """List all available currencies from metadata table."""
    assert _METADATA_TABLE is not None
    response = _METADATA_TABLE.scan(ProjectionExpression="currency, league, last_updated, icon_url, is_available, category")
    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = _METADATA_TABLE.scan(
            ProjectionExpression="currency, league, last_updated, icon_url, is_available, category",
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        items.extend(response.get("Items", []))

    currencies: Dict[str, Dict[str, Any]] = {}
    for item in items:
        currency = item.get("currency")
        league = item.get("league")
        if not currency or not league:
            continue
        
        # Create metadata from available fields
        metadata = {
            "last_updated": item.get("last_updated"),
            "icon_url": item.get("icon_url"),
            "is_available": item.get("is_available"),
            "category": item.get("category")
        }
        currencies.setdefault(currency, {})[league] = metadata

    return {"currencies": currencies}


def _list_leagues() -> Dict[str, Any]:
    """List all available leagues from metadata table."""
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
    """Handle single prediction request."""
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
    """Handle batch prediction request."""
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
    """Handle currency predictions endpoint using the new GSI for efficient queries."""
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
    """Handle latest predictions endpoint for frontend dashboard."""
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
    predictions_data = _fetch_latest_predictions_from_db(league, horizons, limit)
    
    return predictions_data, HTTPStatus.OK


def _fetch_latest_predictions_from_db(
    league: Optional[str], 
    horizons: List[str], 
    limit: int
) -> Dict[str, Any]:
    """Fetch latest predictions for all currencies across specified horizons."""
    assert _PREDICTIONS_TABLE is not None
    
    # Auto-detect league if not provided
    if not league:
        LOGGER.info("No league specified, attempting to auto-detect...")
        league = _infer_latest_league("Divine Orb")  # Use a common currency for league detection
        if not league:
            LOGGER.info("Divine Orb failed, trying Chaos Orb...")
            # Try with a different common currency
            league = _infer_latest_league("Chaos Orb")
            if not league:
                LOGGER.warning("All league inference attempts failed")
                # Get available leagues for better error message
                try:
                    leagues_data = _list_leagues()
                    available_leagues = list(leagues_data.get("leagues", {}).keys())
                    if available_leagues:
                        raise ClientFacingError(f"Unable to determine current league. Available leagues: {', '.join(available_leagues)}. Please specify a league parameter.")
                    else:
                        raise ClientFacingError("Unable to determine current league. No leagues found in metadata. Please ensure data is properly loaded.")
                except Exception as e:
                    raise ClientFacingError("Unable to determine current league. Please specify a league parameter or ensure prediction data is available.")
        else:
            LOGGER.info(f"Auto-detected league: {league}")
    
    # Get list of available currencies from metadata table
    available_currencies = _get_available_currencies(league)
    if not available_currencies:
        raise ClientFacingError(f"No currencies found for league: {league}")
    
    LOGGER.info(f"Found {len(available_currencies)} available currencies for league: {league}")
    
    # Limit currencies to most popular ones for performance
    top_currencies = available_currencies[:limit]
    LOGGER.info(f"Processing {len(top_currencies)} currencies with limit: {limit}")
    
    # Fetch predictions for each currency across all horizons
    all_predictions = {}
    prediction_timestamps = set()
    currencies_with_predictions = 0
    
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
        
        # Include currency even if no prediction data (for complete statistics)
        all_predictions[currency] = currency_predictions
        if currency_predictions:
            currencies_with_predictions += 1
    
    LOGGER.info(f"Returning {len(all_predictions)} currencies ({currencies_with_predictions} with prediction data)")
    
    # Calculate latest prediction time
    latest_timestamp = max(prediction_timestamps) if prediction_timestamps else 0
    # Convert Decimal to int for time.gmtime()
    latest_timestamp_int = int(latest_timestamp) if latest_timestamp else 0
    latest_time_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(latest_timestamp_int)) if latest_timestamp_int else "unknown"
    
    # Organize data for frontend consumption
    organized_data = {
        "predictions": all_predictions,
        "metadata": {
            "league": league,
            "total_currencies": len(all_predictions),
            "horizons_requested": horizons,
            "latest_prediction_time": latest_time_str,
            "query_efficiency": "GSI-optimized"
        }
    }
    
    return organized_data


def _get_available_currencies(league: str) -> List[str]:
    """Get list of available currencies for a specific league."""
    assert _METADATA_TABLE is not None
    
    try:
        # Get currencies from metadata table
        # Check if league attribute exists in table schema
        has_league_attribute = any(
            attr.get('AttributeName') == 'league' 
            for attr in _METADATA_TABLE.attribute_definitions
        )
        
        scan_params = {
            'ProjectionExpression': 'currency'
        }
        
        if has_league_attribute:
            scan_params['FilterExpression'] = Key('league').eq(league)
        
        response = _METADATA_TABLE.scan(**scan_params)
        
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


def _parse_prediction_data_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract prediction data from DynamoDB item, handling both old and new formats."""
    # Handle both old and new data formats
    if "prediction_data" in item:
        # Old format: prediction_data field contains JSON (legacy support)
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
        }
    return prediction_dict


def _normalize_prediction_dict(
    prediction_dict: Dict[str, Any],
    item: Dict[str, Any],
    currency: str,
    league: str,
    horizon: str
) -> Dict[str, Any]:
    """Normalize prediction dictionary with common fields, timestamps, and validation."""
    # Set common fields and ensure frontend compatibility
    prediction_dict.update({
        "currency": currency,
        "league": league,
        "horizon": horizon,
        "source": "cache",
    })
    
    # Convert timestamp to string format expected by frontend
    timestamp = item.get("prediction_timestamp", item.get("created_at", item.get("timestamp", "")))
    if isinstance(timestamp, (int, float, Decimal)):
        # Convert Unix timestamp to ISO string (handle Decimal from DynamoDB)
        import datetime
        timestamp_value = float(timestamp) if isinstance(timestamp, Decimal) else timestamp
        prediction_dict["timestamp"] = datetime.datetime.fromtimestamp(timestamp_value, tz=datetime.timezone.utc).isoformat()
    else:
        prediction_dict["timestamp"] = str(timestamp) if timestamp else ""
    
    # Calculate price_change if not present
    if "price_change" not in prediction_dict:
        current_price = prediction_dict.get("current_price", 0)
        predicted_price = prediction_dict.get("predicted_price", 0)
        if current_price and predicted_price:
            prediction_dict["price_change"] = predicted_price - current_price
        else:
            prediction_dict["price_change"] = 0
    
    # Ensure confidence field name matches frontend expectation
    if "confidence_score" in prediction_dict and "confidence" not in prediction_dict:
        prediction_dict["confidence"] = prediction_dict["confidence_score"]
    
    # Validate and clamp prediction bounds to ensure they're reasonable
    # Clamp negative values to 0 (shouldn't happen with ensemble range, but handle old data)
    if prediction_dict.get("prediction_lower") is not None:
        prediction_dict["prediction_lower"] = max(0.0, float(prediction_dict["prediction_lower"]))
    if prediction_dict.get("prediction_upper") is not None:
        prediction_dict["prediction_upper"] = max(0.0, float(prediction_dict["prediction_upper"]))
    
    # Ensure upper >= lower (fix any data inconsistencies)
    if (prediction_dict.get("prediction_lower") is not None and 
        prediction_dict.get("prediction_upper") is not None):
        if prediction_dict["prediction_upper"] < prediction_dict["prediction_lower"]:
            # If upper < lower, swap them or set upper = lower
            prediction_dict["prediction_upper"] = max(
                prediction_dict["prediction_lower"], 
                prediction_dict["prediction_upper"]
            )
    
    return prediction_dict


def _format_prediction_item(item: Dict[str, Any], currency: str, league: str, horizon: str) -> Dict[str, Any]:
    """Format a prediction item for consistent API response."""
    prediction_dict = _parse_prediction_data_from_item(item)
    return _normalize_prediction_dict(prediction_dict, item, currency, league, horizon)


def _fetch_cached_prediction(currency: str, league: Optional[str], horizon: str) -> Optional[Dict[str, Any]]:
    """Fetch a cached prediction from DynamoDB."""
    assert _PREDICTIONS_TABLE is not None
    league_value = league or _infer_latest_league(currency)
    if not league_value:
        return None
    
    # Use the currency-horizon-index GSI for efficient queries
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
        # Use new key structure that includes horizon
        currency_league_horizon = f"{currency}#{league_value}#{horizon}"
        response = _PREDICTIONS_TABLE.query(
            KeyConditionExpression=Key("currency_league_horizon").eq(currency_league_horizon),
            ScanIndexForward=False,
            Limit=1
        )
    
    items = response.get("Items", [])
    if not items:
        return None
    
    item = items[0]  # Get the most recent prediction
    
    # Use shared formatting logic
    prediction_dict = _parse_prediction_data_from_item(item)
    return _normalize_prediction_dict(prediction_dict, item, currency, league_value, horizon)


def _infer_latest_league(currency: str) -> Optional[str]:
    """Simple league inference without ML dependencies."""
    # First try to get league from cached predictions
    assert _PREDICTIONS_TABLE is not None
    try:
        # Try GSI first - use currency-horizon-index with any horizon
        # We need to query with a specific horizon, so try common ones
        items = None
        for horizon in ["1d", "3d", "7d"]:
            response = _PREDICTIONS_TABLE.query(
                IndexName="currency-horizon-index",
                KeyConditionExpression=Key("currency").eq(currency) & Key("horizon").eq(horizon),
                ScanIndexForward=False,
                Limit=1,
            )
            items = response.get("Items")
            if items:
                break
        
        if items:
            prediction_key = items[0].get("prediction_key", "")
            if "#" in prediction_key:
                try:
                    _, league, _ = prediction_key.split("#", 2)
                    return league
                except ValueError:
                    pass
            league = items[0].get("league")
            if league:
                return league
    except Exception as e:
        LOGGER.warning(f"GSI query failed for league inference: {e}")
    
    # Fallback: try main table query
    try:
        response = _PREDICTIONS_TABLE.scan(
            FilterExpression=Key("currency").eq(currency),
            Limit=1
        )
        items = response.get("Items")
        if items:
            prediction_key = items[0].get("prediction_key", "")
            if "#" in prediction_key:
                try:
                    _, league, _ = prediction_key.split("#", 2)
                    return league
                except ValueError:
                    pass
            league = items[0].get("league")
            if league:
                return league
    except Exception as e:
        LOGGER.warning(f"Main table query failed for league inference: {e}")
    
    # Final fallback: get any available league from metadata
    try:
        assert _METADATA_TABLE is not None
        response = _METADATA_TABLE.scan(
            ProjectionExpression="league",
            Limit=1
        )
        items = response.get("Items")
        if items:
            league = items[0].get("league")
            if league:
                LOGGER.info(f"Using fallback league from metadata: {league}")
                return league
    except Exception as e:
        LOGGER.warning(f"Metadata table query failed for league inference: {e}")
    
    return None


def _require_string(payload: Dict[str, Any], key: str) -> str:
    """Require a non-empty string from payload."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ClientFacingError(f"{key} must be a non-empty string.")
    return value.strip()


def _cors_headers() -> Dict[str, str]:
    """Return CORS headers."""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }


def _handle_live_prices(query_params: Dict[str, str]) -> Tuple[Dict[str, Any], HTTPStatus]:
    """Handle live currency prices endpoint."""
    currency = query_params.get("currency")
    league = query_params.get("league")
    limit = int(query_params.get("limit", "100"))
    hours = int(query_params.get("hours", "24"))
    
    # Validate parameters
    if limit <= 0 or limit > 1000:
        raise ClientFacingError("limit must be between 1 and 1000")
    if hours <= 0 or hours > 168:  # Max 1 week
        raise ClientFacingError("hours must be between 1 and 168")
    
    prices_data = _fetch_live_prices_from_db(currency, league, limit, hours)
    
    return prices_data, HTTPStatus.OK


def _handle_historical_prices(query_params: Dict[str, str]) -> Tuple[Dict[str, Any], HTTPStatus]:
    """Handle historical daily price data endpoint."""
    currency = query_params.get("currency")
    league = query_params.get("league")
    start_date = query_params.get("start_date")  # Optional
    end_date = query_params.get("end_date")  # Optional
    limit = int(query_params.get("limit", "1000"))  # Default to 1000 to get more data
    
    # Validate required parameters
    if not currency:
        raise ClientFacingError("currency parameter is required")
    if not league:
        raise ClientFacingError("league parameter is required")
    
    # Validate date format if provided
    if start_date:
        try:
            from datetime import datetime
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ClientFacingError("start_date must be in YYYY-MM-DD format")
    
    if end_date:
        try:
            from datetime import datetime
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ClientFacingError("end_date must be in YYYY-MM-DD format")
    
    # Validate limit
    if limit <= 0 or limit > 1000:
        raise ClientFacingError("limit must be between 1 and 1000")
    
    # Fetch historical prices
    historical_data = _fetch_historical_prices_from_db(
        currency=currency,
        league=league,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    return historical_data, HTTPStatus.OK


def _fetch_historical_prices_from_db(
    currency: str,
    league: str,
    start_date: str = None,
    end_date: str = None,
    limit: int = 1000
) -> Dict[str, Any]:
    """Fetch historical daily prices from DynamoDB."""
    # Get the daily prices table name from environment (not in shared AppEnvironment)
    daily_prices_table_name = os.getenv("DAILY_PRICES_TABLE", "poeconomy-daily-prices")
    
    try:
        # Initialize DynamoDB resource
        dynamodb = boto3.resource("dynamodb", region_name=_APP_ENV.region_name)
        daily_prices_table = dynamodb.Table(daily_prices_table_name)
        
        currency_league = f"{currency}#{league}"
        
        # Query the daily prices table
        if start_date and end_date:
            # Both start and end date provided
            response = daily_prices_table.query(
                KeyConditionExpression=Key("currency_league").eq(currency_league) & 
                                     Key("date").between(start_date, end_date),
                ScanIndexForward=True,  # Oldest first
                Limit=limit
            )
        elif start_date:
            # Only start date provided
            response = daily_prices_table.query(
                KeyConditionExpression=Key("currency_league").eq(currency_league) & 
                                     Key("date").gte(start_date),
                ScanIndexForward=True,  # Oldest first
                Limit=limit
            )
        elif end_date:
            # Only end date provided
            response = daily_prices_table.query(
                KeyConditionExpression=Key("currency_league").eq(currency_league) & 
                                     Key("date").lte(end_date),
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )
        else:
            # No date filters - get all available data
            response = daily_prices_table.query(
                KeyConditionExpression=Key("currency_league").eq(currency_league),
                ScanIndexForward=True,  # Oldest first
                Limit=limit
            )
        
        items = response.get("Items", [])
        
        # Format the response
        historical_prices = []
        for item in items:
            price_data = {
                "date": item.get("date"),
                "avg_price": float(item.get("avg_price", 0))
            }
            historical_prices.append(price_data)
        
        return {
            "currency": currency,
            "league": league,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(historical_prices),
            "prices": historical_prices,
            "source": "daily_aggregation",
            "last_updated": _get_latest_price_update_time()
        }
        
    except Exception as e:
        LOGGER.error(f"Error fetching historical prices: {e}")
        return {
            "currency": currency,
            "league": league,
            "start_date": start_date,
            "end_date": end_date,
            "count": 0,
            "prices": [],
            "error": str(e),
            "source": "daily_aggregation"
        }


def _fetch_live_prices_from_db(
    currency: Optional[str], 
    league: Optional[str], 
    limit: int, 
    hours: int
) -> Dict[str, Any]:
    """Fetch live prices from DynamoDB with efficient queries."""
    assert _PRICING_TABLE is not None
    
    # Calculate timestamp threshold (hours ago)
    current_time = int(time.time())
    hours_ago = current_time - (hours * 3600)
    
    prices = []
    total_count = 0
    
    try:
        if currency and league:
            # Query specific currency in specific league using GSI
            response = _PRICING_TABLE.query(
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
            response = _PRICING_TABLE.query(
                IndexName="currency-timestamp-index", 
                KeyConditionExpression=Key("currency").eq(currency) & Key("timestamp").gte(hours_ago),
                ScanIndexForward=False,
                Limit=limit
            )
            prices = response.get("Items", [])
            total_count = response.get("Count", 0)
            
        elif league:
            # Query specific league using GSI - get all prices
            response = _PRICING_TABLE.query(
                IndexName="league-timestamp-index",
                KeyConditionExpression=Key("league").eq(league),
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )
            prices = response.get("Items", [])
            total_count = response.get("Count", 0)
            
        else:
            # Scan for all recent prices (less efficient, but works for small datasets)
            response = _PRICING_TABLE.scan(
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
                "cached": False,
                "cache_handled_by": "Cloudflare Worker"
            }
        }
    }


def _get_latest_price_update_time() -> str:
    """Get the timestamp of the most recent price update."""
    try:
        assert _PRICING_TABLE is not None
        # Get the most recent item to determine last update time
        response = _PRICING_TABLE.scan(
            Limit=1
        )
        items = response.get("Items", [])
        if items:
            # Find the item with the highest timestamp
            latest_item = max(items, key=lambda x: int(x.get("timestamp", 0)))
            timestamp = latest_item.get("timestamp")
            if timestamp:
                # Convert Decimal to int for time.gmtime()
                timestamp_int = int(timestamp) if timestamp else 0
                return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(timestamp_int))
    except Exception as e:
        LOGGER.warning(f"Could not determine latest update time: {e}")
    
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _dynamodb_encoder(value: Any) -> Any:
    """Encode DynamoDB types for JSON serialization."""
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
