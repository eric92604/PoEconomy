import re
from typing import List, Optional, Tuple


def parse_horizon_to_days(horizon: str) -> Optional[int]:
    if not horizon:
        return None
    
    horizon = str(horizon).lower().strip()
    if horizon.endswith('d'):
        try:
            return int(horizon[:-1])
        except ValueError:
            pass
    return None


def extract_window_size_from_feature(feature_name: str) -> Optional[int]:
    if not feature_name:
        return None
    
    window_pattern = re.compile(r'_(\d+)d(?:_|$)')
    
    match = window_pattern.search(feature_name)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            pass
    
    return None


def filter_features_by_horizon(
    feature_columns: List[str],
    horizon: str,
    strict_mode: bool = True
) -> Tuple[List[str], List[str]]:
    horizon_days = parse_horizon_to_days(horizon)
    if horizon_days is None:
        return feature_columns, []
    
    filtered = []
    excluded = []
    
    for col in feature_columns:
        window_days = extract_window_size_from_feature(col)
        
        if window_days is not None:
            if strict_mode:
                should_exclude = window_days >= horizon_days
            else:
                should_exclude = window_days == horizon_days
            
            if should_exclude:
                excluded.append(col)
                continue
        
        filtered.append(col)
    
    return filtered, excluded


def get_feature_filtering_stats(
    feature_columns: List[str],
    horizon: str,
    strict_mode: bool = True
) -> dict:
    filtered, excluded = filter_features_by_horizon(
        feature_columns, horizon, strict_mode
    )
    
    horizon_days = parse_horizon_to_days(horizon)
    
    excluded_by_window = {}
    for feat in excluded:
        window = extract_window_size_from_feature(feat)
        if window:
            excluded_by_window[window] = excluded_by_window.get(window, 0) + 1
    
    return {
        'total_features': len(feature_columns),
        'filtered_features': len(filtered),
        'excluded_features': len(excluded),
        'exclusion_rate': len(excluded) / len(feature_columns) if feature_columns else 0.0,
        'horizon_days': horizon_days,
        'strict_mode': strict_mode,
        'excluded_by_window': excluded_by_window,
        'excluded_feature_names': excluded[:20]
    }

