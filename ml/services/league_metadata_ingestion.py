"""League metadata ingestion service for POE Watch API."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

from ml.utils.common_utils import MLLogger

POE_WATCH_LEAGUES_URL = "https://api.poe.watch/leagues"


def fetch_poe_watch_leagues(timeout: int = 30) -> List[dict]:
    """
    Fetch league data from POE Watch API.
    
    Args:
        timeout: HTTP timeout in seconds
        
    Returns:
        List of league data from POE Watch API
        
    Raises:
        requests.RequestException: If API request fails
        ValueError: If response format is unexpected
    """
    try:
        response = requests.get(POE_WATCH_LEAGUES_URL, timeout=timeout)
        response.raise_for_status()
        leagues = response.json()
        
        if not isinstance(leagues, list):
            raise ValueError(f"Unexpected response format: {leagues!r}")
            
        return leagues
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch leagues from POE Watch API: {e}")


def build_league_metadata_items(
    leagues_data: List[dict], 
    iso_now: str, 
    force_update: bool = False
) -> List[dict]:
    """
    Build league metadata items for DynamoDB storage.
    
    Args:
        leagues_data: League data from POE Watch API
        iso_now: Current timestamp in ISO format
        force_update: Whether to force update existing leagues
        
    Returns:
        List of league metadata items for DynamoDB batch write
    """
    items: List[dict] = []
    
    for league in leagues_data:
        league_name = league.get("name")
        if not league_name:
            continue
            
        # Parse start and end dates
        start_date = _parse_league_date(league.get("start_date"))
        end_date = _parse_league_date(league.get("end_date"))
        
        # Determine if league is active
        is_active = _determine_league_activity(league, start_date, end_date)
        
        # Determine league type
        league_type = _determine_league_type(league_name)
        
        ttl = int((datetime.now(timezone.utc) + timedelta(days=3)).timestamp())

        item = {
            "league_name": league_name,
            "league": league_name,  # Keep both for backward compatibility
            "startDate": start_date.isoformat() if start_date else None,
            "endDate": end_date.isoformat() if end_date else None,
            "isActive": is_active,
            "league_type": league_type,
            "display_name": league.get("display", league_name),
            "description": league.get("description"),
            "url": league.get("url"),
            "last_updated": iso_now,
            "force_update": force_update,
            "ttl": ttl,
        }
        
        # Add metadata about whether this is a new league
        # This will be used by the lambda to categorize results
        item["is_new_league"] = True  # We'll determine this based on existing data
        
        # Remove None values to avoid DynamoDB complaints
        sanitized_item = {k: v for k, v in item.items() if v is not None}
        items.append(sanitized_item)
    
    return items


def _parse_league_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse league date string to datetime object.
    
    Args:
        date_str: Date string from POE Watch API
        
    Returns:
        Parsed datetime object or None if parsing fails
    """
    if not date_str or date_str in ('0001-01-01T00:00:00+00:00', '0001-01-01T00:00:00Z'):
        return None
        
    try:
        # POE Watch API typically returns dates in ISO format
        if date_str.endswith('Z'):
            return datetime.fromisoformat(date_str[:-1]).replace(tzinfo=timezone.utc)
        else:
            return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _determine_league_activity(
    league_data: dict, 
    start_date: Optional[datetime], 
    end_date: Optional[datetime]
) -> bool:
    """
    Determine if a league is currently active.
    
    Args:
        league_data: Raw league data from API
        start_date: Parsed start date
        end_date: Parsed end date
        
    Returns:
        True if league is active, False otherwise
    """
    now = datetime.now(timezone.utc)
    
    # Check if league has started
    if start_date and now < start_date:
        return False
        
    # Check if league has ended
    if end_date and now > end_date:
        return False
        
    # Check for explicit active flag in API response
    if "active" in league_data:
        return bool(league_data["active"])
        
    # Default to active if no end date and has started
    return start_date is None or now >= start_date


def _determine_league_type(league_name: str) -> str:
    """
    Determine the type of league based on its name.
    
    Args:
        league_name: Name of the league
        
    Returns:
        League type: 'permanent', 'seasonal', 'hardcore', 'ruthless', or 'ssf'
    """
    name_lower = league_name.lower()
    
    if name_lower == "standard":
        return "permanent"
    elif "hardcore" in name_lower:
        return "hardcore"
    elif "ruthless" in name_lower:
        return "ruthless"
    elif "ssf" in name_lower or "solo self-found" in name_lower:
        return "ssf"
    else:
        return "seasonal"


def main() -> None:
    """Main function for testing the league metadata ingestion."""
    logger = MLLogger("LeagueMetadataIngestion", level="INFO")
    
    try:
        logger.info("Fetching league data from POE Watch API")
        leagues_data = fetch_poe_watch_leagues()
        logger.info(f"Received {len(leagues_data)} leagues")
        
        iso_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        league_items = build_league_metadata_items(leagues_data, iso_now)
        
        logger.info(f"Built {len(league_items)} league metadata items")
        
        # Print sample items
        for i, item in enumerate(league_items[:3]):
            logger.info(f"Sample league {i+1}: {item}")
            
    except Exception as e:
        logger.error(f"Failed to process league metadata: {e}")
        raise


if __name__ == "__main__":
    main()
