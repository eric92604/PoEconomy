#!/usr/bin/env python3
"""Display currency availability and pricing metadata from DynamoDB."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import List, Optional

from ml.config.training_config import MLConfig
from ml.utils.data_sources import create_data_source, DataSourceConfig, CurrencyStat
from ml.utils.common_utils import MLLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect currency availability using DynamoDB metadata",
    )
    parser.add_argument("--league", help="Filter to a specific league")
    parser.add_argument("--min-avg-value", type=float, default=0.0, help="Minimum average Chaos value")
    parser.add_argument("--min-records", type=int, default=0, help="Minimum number of price records")
    parser.add_argument(
        "--show-unavailable",
        action="store_true",
        help="Include currencies marked as unavailable",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximum rows to display")
    parser.add_argument(
        "--sort-by",
        choices=["avg", "volatility", "records"],
        default="avg",
        help="Sort column",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def filter_stats(
    stats: List[CurrencyStat],
    league: Optional[str],
    min_avg_value: float,
    min_records: int,
    show_unavailable: bool,
) -> List[CurrencyStat]:
    filtered = []
    for stat in stats:
        if league and stat.league != league:
            continue
        if stat.avg_price < min_avg_value:
            continue
        if stat.total_records < min_records:
            continue
        if not show_unavailable and not stat.is_available:
            continue
        filtered.append(stat)
    return filtered


def sort_stats(stats: List[CurrencyStat], sort_by: str) -> List[CurrencyStat]:
    key_map = {
        "avg": lambda s: s.avg_price,
        "volatility": lambda s: s.volatility,
        "records": lambda s: s.total_records,
    }
    key_func = key_map.get(sort_by, key_map["avg"])
    return sorted(stats, key=key_func, reverse=True)


def print_table(stats: List[CurrencyStat], limit: int) -> None:
    headers = [
        "Currency",
        "League",
        "Avg",
        "Median",
        "Min",
        "Max",
        "Volatility",
        "Records",
        "Available",
        "Last Check",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for stat in stats[:limit]:
        print(
            "| {currency:<25} | {league:<15} | {avg:>8.2f} | {median:>8.2f} | {min_:>8.2f} | {max_:>8.2f} | {vol:>9.2f} | {records:>7} | {available:^9} | {last_check:<20} |".format(
                currency=stat.currency[:25],
                league=stat.league[:15],
                avg=stat.avg_price,
                median=stat.median_price,
                min_=stat.min_price,
                max_=stat.max_price,
                vol=stat.volatility,
                records=stat.total_records,
                available="Y" if stat.is_available else "N",
                last_check=(stat.last_availability_check or "n/a")[:20],
            )
        )


def print_json(stats: List[CurrencyStat], limit: int) -> None:
    payload = [asdict(stat) for stat in stats[:limit]]
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    logger = MLLogger("Availability", level="DEBUG" if args.debug else "INFO")
    config = MLConfig()
    data_source_config = DataSourceConfig.from_dynamo_config(config.dynamo)
    data_source = create_data_source(data_source_config, logger)

    stats = data_source.list_currency_stats()
    selected_league = args.league
    if selected_league is None:
        # Use consistent interface method to get most recent league
        if hasattr(data_source, 'get_most_recent_league'):
            selected_league = data_source.get_most_recent_league()
        if selected_league:
            logger.info("Auto-selected most recent league", extra={"league": selected_league})

    filtered = filter_stats(
        stats,
        league=selected_league,
        min_avg_value=args.min_avg_value,
        min_records=args.min_records,
        show_unavailable=args.show_unavailable,
    )
    sorted_stats = sort_stats(filtered, args.sort_by)

    if args.format == "json":
        print_json(sorted_stats, args.limit)
    else:
        print_table(sorted_stats, args.limit)

    logger.info(
        "Displayed %s currencies (requested limit=%s, total available=%s)",
        min(args.limit, len(sorted_stats)),
        args.limit,
        len(sorted_stats),
    )


if __name__ == "__main__":
    main()
