"""Migrate historical league CSV data into the historical_league_prices DynamoDB table.

CSV format (semicolon-delimited):
  League;Date;Get;Pay;Value;Confidence

Rules:
  - Keep only rows where Pay == "Chaos Orb" — Value is the price of Get currency in chaos.
  - Group by (League, Get) and aggregate per date: avg/high/low of Value.
  - Derive league_start_date as the earliest date seen for that league.
  - Compute price_change_percent relative to the previous day for each (League, Get) pair.
  - Write to DynamoDB table using batch_writer().

Usage:
  AWS_PROFILE=<profile> python migrate_csv_to_historical_league_prices.py \\
      --env production \\
      [--csv-dir <path>]   # default: training_data/ relative to repo root
      [--dry-run]          # print items without writing
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_csv_dir(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            sys.exit(f"Error: --csv-dir '{explicit}' is not a directory.")
        return path
    # Default: repo-root/training_data
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    path = repo_root / "training_data"
    if not path.is_dir():
        sys.exit(f"Error: expected training_data/ at {path}. Pass --csv-dir explicitly.")
    return path


def _parse_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def _quantize(value: Decimal, places: int) -> Decimal:
    """Round a Decimal using fixed places for DynamoDB-safe numeric values."""
    return value.quantize(Decimal(f"1.{'0' * places}"), rounding=ROUND_HALF_UP)


def _aggregate(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Decimal]]:
    """Return {(league, currency, date): [values...]} for Pay==Chaos Orb rows."""
    data: Dict[Tuple[str, str, str], List[Decimal]] = defaultdict(list)
    for row in rows:
        if row.get("Pay", "").strip() != "Chaos Orb":
            continue
        league = row.get("League", "").strip()
        currency = row.get("Get", "").strip()
        date = row.get("Date", "").strip()
        value_str = row.get("Value", "").strip()
        if not (league and currency and date and value_str):
            continue
        try:
            value = Decimal(value_str)
        except InvalidOperation:
            continue
        if value <= 0:
            continue
        data[(league, currency, date)].append(value)
    return data


def _build_items(
    aggregated: Dict[Tuple[str, str, str], List[Decimal]],
) -> List[Dict[str, Any]]:
    """Convert aggregated price data into DynamoDB item dicts."""
    # Group by (league, currency) to compute league_start_date and price_change_percent
    series: Dict[Tuple[str, str], List[Tuple[str, Decimal, Decimal, Decimal]]] = defaultdict(list)
    for (league, currency, date), values in aggregated.items():
        avg = sum(values) / Decimal(len(values))
        high = max(values)
        low = min(values)
        series[(league, currency)].append((date, avg, high, low))

    items = []
    for (league, currency), day_list in series.items():
        day_list.sort(key=lambda t: t[0])  # sort by date ascending
        league_start_date = day_list[0][0]

        prev_avg: Optional[Decimal] = None
        for date, avg, high, low in day_list:
            if prev_avg is not None and prev_avg > 0:
                price_change_percent = _quantize(((avg - prev_avg) / prev_avg) * Decimal("100"), 4)
            else:
                price_change_percent = Decimal("0")
            prev_avg = avg

            items.append({
                "currency_league": f"{currency}#{league}",
                "date": date,
                "currency": currency,
                "league": league,
                "league_start_date": league_start_date,
                "avg_price": _quantize(avg, 6),
                "high_price": _quantize(high, 6),
                "low_price": _quantize(low, 6),
                "price_change_percent": price_change_percent,
            })

    return items


def _write_to_dynamo(items: List[Dict[str, Any]], table_name: str, region: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would write {len(items)} items to {table_name}")
        for item in items[:3]:
            print(f"  {item}")
        if len(items) > 3:
            print(f"  ... ({len(items) - 3} more)")
        return

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    written = 0
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)
            written += 1
            if written % 500 == 0:
                print(f"  Written {written}/{len(items)}...")

    print(f"Done. Wrote {written} items to {table_name}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="production", help="Environment name (default: production)")
    parser.add_argument("--region", default="us-west-2", help="AWS region (default: us-west-2)")
    parser.add_argument("--csv-dir", default=None, help="Path to directory containing *.csv files")
    parser.add_argument("--dry-run", action="store_true", help="Print items without writing to DynamoDB")
    args = parser.parse_args()

    csv_dir = _find_csv_dir(args.csv_dir)
    table_name = f"poeconomy-{args.env}-historical-league-prices"

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No *.csv files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV file(s) in {csv_dir}")
    print(f"Target table: {table_name} ({args.region})")

    all_rows: List[Dict[str, str]] = []
    for csv_path in csv_files:
        rows = _parse_csv(csv_path)
        print(f"  {csv_path.name}: {len(rows)} rows")
        all_rows.extend(rows)

    print(f"\nTotal rows: {len(all_rows)}")

    aggregated = _aggregate(all_rows)
    print(f"Unique (league, currency, date) combinations after filtering: {len(aggregated)}")

    items = _build_items(aggregated)
    print(f"DynamoDB items to write: {len(items)}")

    # Report leagues found
    leagues = sorted({item["league"] for item in items})
    print(f"Leagues: {', '.join(leagues)}")

    _write_to_dynamo(items, table_name, args.region, args.dry_run)


if __name__ == "__main__":
    main()
