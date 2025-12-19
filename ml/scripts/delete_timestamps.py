#!/usr/bin/env python3
"""Delete rows from DynamoDB live-prices table matching specific timestamps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import boto3
from boto3.dynamodb.conditions import Attr

from ml.utils.common_utils import MLLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete rows from DynamoDB table matching specific timestamps",
    )
    parser.add_argument(
        "--table-name",
        default="poeconomy-production-live-prices",
        help="DynamoDB table name (default: poeconomy-production-live-prices)",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )
    parser.add_argument(
        "--timestamps",
        nargs="+",
        type=int,
        help="List of timestamps to delete (space-separated)",
    )
    parser.add_argument(
        "--file",
        help="File containing timestamps (one per line)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def load_timestamps(args: argparse.Namespace) -> List[int]:
    """Load timestamps from arguments or file."""
    timestamps: List[int] = []
    
    if args.timestamps:
        timestamps.extend(args.timestamps)
    
    if args.file:
        try:
            with open(args.file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        timestamps.append(int(line))
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
            sys.exit(1)
    
    if not timestamps:
        print("Error: No timestamps provided. Use --timestamps or --file.", file=sys.stderr)
        sys.exit(1)
    
    # Remove duplicates and sort
    unique_timestamps = sorted(set(timestamps))
    return unique_timestamps


def find_items_by_timestamps(
    table, timestamps: List[int], logger: MLLogger
) -> List[dict]:
    """Find all items in the table matching the given timestamps."""
    logger.info(f"Scanning table for {len(timestamps)} timestamps...")
    
    # Build filter expression: timestamp IN (ts1, ts2, ...)
    # DynamoDB supports up to 100 values in an IN expression
    # We'll process in batches if needed
    all_items: List[dict] = []
    batch_size = 100
    
    for i in range(0, len(timestamps), batch_size):
        batch = timestamps[i : i + batch_size]
        filter_expr = Attr("timestamp").is_in(batch)
        
        logger.debug(f"Scanning batch {i // batch_size + 1} ({len(batch)} timestamps)...")
        
        # Scan with pagination
        response = table.scan(FilterExpression=filter_expr)
        all_items.extend(response.get("Items", []))
        
        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = table.scan(
                FilterExpression=filter_expr,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            all_items.extend(response.get("Items", []))
    
    logger.info(f"Found {len(all_items)} items matching the timestamps")
    return all_items


def delete_items(
    table, items: List[dict], logger: MLLogger, dry_run: bool = False
) -> int:
    """Delete items from the table using batch writer."""
    if not items:
        logger.info("No items to delete.")
        return 0
    
    if dry_run:
        logger.info(f"[DRY RUN] Would delete {len(items)} items")
        # Show sample items
        for i, item in enumerate(items[:5]):
            logger.info(
                f"  Sample {i+1}: currency_league={item.get('currency_league')}, "
                f"timestamp={item.get('timestamp')}, currency={item.get('currency')}, "
                f"league={item.get('league')}"
            )
        if len(items) > 5:
            logger.info(f"  ... and {len(items) - 5} more items")
        return len(items)
    
    deleted_count = 0
    
    # Use batch_writer for efficient deletion
    key_names = [key["AttributeName"] for key in table.key_schema]
    with table.batch_writer() as batch:
        for item in items:
            # Extract primary key
            key = {key_name: item[key_name] for key_name in key_names}
            batch.delete_item(Key=key)
            deleted_count += 1
            
            if deleted_count % 100 == 0:
                logger.info(f"Deleted {deleted_count}/{len(items)} items...")
    
    logger.info(f"Successfully deleted {deleted_count} items")
    return deleted_count


def main() -> None:
    args = parse_args()
    logger = MLLogger("DeleteTimestamps", level="DEBUG" if args.debug else "INFO")
    
    # Load timestamps
    timestamps = load_timestamps(args)
    logger.info(f"Loaded {len(timestamps)} unique timestamps to delete")
    logger.debug(f"Timestamps: {timestamps}")
    
    # Initialize DynamoDB
    dynamodb = boto3.resource("dynamodb", region_name=args.region)
    table = dynamodb.Table(args.table_name)
    
    # Verify table exists
    try:
        table.load()
        logger.info(f"Connected to table: {table.table_name}")
    except Exception as e:
        logger.error(f"Failed to connect to table '{args.table_name}': {e}")
        sys.exit(1)
    
    # Find items matching timestamps
    items = find_items_by_timestamps(table, timestamps, logger)
    
    if not items:
        logger.info("No items found matching the specified timestamps.")
        return
    
    # Show summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Table: {args.table_name}")
    print(f"  Timestamps: {len(timestamps)}")
    print(f"  Items found: {len(items)}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'DELETE'}")
    print(f"{'='*60}\n")
    
    # Group items by timestamp for display
    items_by_timestamp: dict[int, int] = {}
    for item in items:
        ts = item.get("timestamp")
        if ts:
            items_by_timestamp[ts] = items_by_timestamp.get(ts, 0) + 1
    
    print("Items per timestamp:")
    for ts in sorted(timestamps):
        count = items_by_timestamp.get(ts, 0)
        print(f"  {ts}: {count} item(s)")
    print()
    
    # Confirmation
    if not args.dry_run and not args.confirm:
        response = input(
            f"Are you sure you want to delete {len(items)} items? (yes/no): "
        )
        if response.lower() not in ["yes", "y"]:
            print("Deletion cancelled.")
            return
    
    # Delete items
    deleted_count = delete_items(table, items, logger, dry_run=args.dry_run)
    
    if not args.dry_run:
        logger.info(f"Deletion complete. {deleted_count} items deleted.")


if __name__ == "__main__":
    main()

