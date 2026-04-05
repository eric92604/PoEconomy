#!/usr/bin/env python3
"""Generate a DynamoDB-backed investment opportunity report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ml.config.training_config import MLConfig
from ml.utils.data_sources import create_data_source, DataSourceConfig, CurrencyStat
from ml.utils.common_utils import MLLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a currency investment report from DynamoDB metadata",
    )
    parser.add_argument("--league", help="Limit report to a specific league")
    parser.add_argument("--top-n", type=int, default=25, help="Number of currencies to include")
    parser.add_argument(
        "--min-records",
        type=int,
        default=200,
        help="Minimum number of records required for inclusion",
    )
    parser.add_argument(
        "--min-avg-value",
        type=float,
        default=5.0,
        help="Minimum average Chaos value required",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write the report to",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def score_currency(stat: CurrencyStat) -> float:
    """Simple scoring heuristic balancing price and stability."""
    stability = max(1.0, 1.0 / (1.0 + stat.volatility))
    return stat.avg_price * stability * (1.0 if stat.is_available else 0.6)


def stats_to_dataframe(stats: List[CurrencyStat]) -> pd.DataFrame:
    records = []
    for stat in stats:
        records.append(
            {
                "currency": stat.currency,
                "league": stat.league,
                "avg_price": stat.avg_price,
                "median_price": stat.median_price,
                "min_price": stat.min_price,
                "max_price": stat.max_price,
                "volatility": stat.volatility,
                "total_records": stat.total_records,
                "is_available": stat.is_available,
                "score": score_currency(stat),
                "last_check": stat.last_availability_check,
                "availability_source": stat.availability_source,
            }
        )
    return pd.DataFrame.from_records(records)


def filter_dataframe(df: pd.DataFrame, league: Optional[str], min_records: int, min_avg_value: float) -> pd.DataFrame:
    if league:
        df = df[df["league"] == league]
    df = df[df["total_records"] >= min_records]
    df = df[df["avg_price"] >= min_avg_value]
    return df


def render_report(df: pd.DataFrame, top_n: int) -> None:
    if df.empty:
        print("No currencies matched the criteria.")
        return

    display_cols = [
        "currency",
        "league",
        "avg_price",
        "median_price",
        "volatility",
        "total_records",
        "is_available",
        "score",
    ]
    print(df[display_cols].head(top_n).to_string(index=False, justify="left"))


def export_report(df: pd.DataFrame, output: Path, top_n: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.head(top_n).to_csv(output, index=False)
    print(f"Report written to {output}")


def main() -> None:
    args = parse_args()
    logger = MLLogger("InvestmentReport", level="DEBUG" if args.debug else "INFO")
    config = MLConfig()
    data_source_config = DataSourceConfig.from_dynamo_config(config.dynamo)
    data_source = create_data_source(data_source_config, logger)

    stats = data_source.list_currency_stats()
    df = stats_to_dataframe(stats)
    df = filter_dataframe(df, args.league, args.min_records, args.min_avg_value)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    render_report(df, args.top_n)

    if args.output:
        export_report(df, args.output, args.top_n)

    logger.info(
        "Generated investment report",
        extra={
            "rows_considered": len(stats),
            "rows_after_filter": len(df),
            "top_n": args.top_n,
            "league": args.league,
        },
    )


if __name__ == "__main__":
    main()
