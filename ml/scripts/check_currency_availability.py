#!/usr/bin/env python3
"""
Currency availability checker script.

This script determines which currencies are available in the current league
by checking multiple data sources:
1. Recent price data in the database
2. Manual overrides

Updates the currency table with availability status.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.database import get_db_connection
from utils.logging_utils import MLLogger
from utils.currency_standardizer import CurrencyStandardizer
import pandas as pd


class CurrencyAvailabilityChecker:
    """
    Checks currency availability across multiple data sources.
    """
    
    def __init__(self, logger: Optional[MLLogger] = None):
        """Initialize the availability checker."""
        self.logger = logger or MLLogger("CurrencyAvailabilityChecker")
        self.current_league = None
        self.availability_results = {}
        self.standardizer = CurrencyStandardizer()
        
    def get_current_league(self) -> Optional[str]:
        """Get the current active league from database."""
        try:
            conn = get_db_connection()
            
            # Get the current active league
            query = """
            SELECT name, "startDate", "endDate", "isActive"
            FROM leagues 
            WHERE "isActive" = true 
            ORDER BY "startDate" DESC 
            LIMIT 1
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if not df.empty:
                league_name = df['name'].iloc[0]
                self.logger.info(f"Current active league: {league_name}")
                return league_name
            else:
                # Fallback to most recent league
                conn = get_db_connection()
                fallback_query = """
                SELECT name FROM leagues 
                ORDER BY "startDate" DESC 
                LIMIT 1
                """
                df = pd.read_sql(fallback_query, conn)
                conn.close()
                
                if not df.empty:
                    league_name = df['name'].iloc[0]
                    self.logger.warning(f"No active league found, using most recent: {league_name}")
                    return league_name
                
                self.logger.error("No leagues found in database")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get current league: {str(e)}")
            return None
    
    def check_availability_from_price_data(self, league_name: str, days_back: int = 7) -> Set[str]:
        """
        Check currency availability based on recent price data.
        
        Args:
            league_name: Name of the league to check
            days_back: Number of days to look back for recent data
            
        Returns:
            Set of available currency names
        """
        try:
            conn = get_db_connection()
            
            # Check for recent price data
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = """
            SELECT DISTINCT c.name
            FROM currency c
            JOIN currency_prices cp ON (c.id = cp."getCurrencyId" OR c.id = cp."payCurrencyId")
            JOIN leagues l ON cp."leagueId" = l.id
            WHERE l.name = %s
              AND cp.date >= %s
              AND cp.value > 0
            """
            
            df = pd.read_sql(query, conn, params=[league_name, cutoff_date])
            conn.close()
            
            available_currencies = set(df['name'].tolist())
            
            self.logger.info(f"Found {len(available_currencies)} currencies with recent price data in {league_name}")
            
            return available_currencies
            
        except Exception as e:
            self.logger.error(f"Failed to check availability from price data: {str(e)}")
            return set()
    
    def check_availability_from_poe_watch(self, days_back: int = 7) -> Set[str]:
        """
        Check currency availability based on recent POE Watch data.
        Uses currency standardization to match POE Watch names to database names.
        
        Args:
            days_back: Number of days to look back for recent data
            
        Returns:
            Set of available currency names (standardized to database names)
        """
        try:
            conn = get_db_connection()
            
            # Check for recent POE Watch data
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = """
            SELECT DISTINCT currency_name
            FROM live_poe_watch
            WHERE fetch_time >= %s
              AND mean_price > 0
              AND daily_volume > 0
            """
            
            df = pd.read_sql(query, conn, params=[cutoff_date])
            conn.close()
            
            poe_watch_currencies = set(df['currency_name'].tolist())
            self.logger.info(f"Found {len(poe_watch_currencies)} currencies with recent POE Watch data")
            
            # Standardize POE Watch currency names to match database names
            standardized_currencies = set()
            successful_matches = 0
            
            for poe_watch_name in poe_watch_currencies:
                # Try direct match first
                standardized_name = self.standardizer.standardize_currency_name(poe_watch_name)
                if standardized_name:
                    standardized_currencies.add(standardized_name)
                    successful_matches += 1
                elif poe_watch_name in self.standardizer.get_all_currency_names():
                    # Direct match without standardization needed
                    standardized_currencies.add(poe_watch_name)
                    successful_matches += 1
            
            self.logger.info(f"Successfully standardized {successful_matches}/{len(poe_watch_currencies)} POE Watch currencies to database names")
            
            return standardized_currencies
            
        except Exception as e:
            self.logger.error(f"Failed to check availability from POE Watch: {str(e)}")
            return set()
    

    
    def get_manual_overrides(self) -> Tuple[Set[str], Set[str]]:
        """
        Get manual availability overrides.
        
        Returns:
            Tuple of (force_enabled, force_disabled) currency names
        """
        # These are currencies that should always be considered available
        # or always disabled, regardless of data availability
        force_enabled = {
            'Chaos Orb',  # Base currency - always available
            'Divine Orb',  # Primary high-value currency
            'Exalted Orb',  # Common high-value currency
            'Chromatic Orb',  # Common crafting currency
            'Orb of Fusing',  # Common crafting currency
            'Orb of Alchemy',  # Common crafting currency
        }
        
        # Currencies to disable (e.g., legacy items, test currencies)
        force_disabled = {
            'Eternal Orb',  # Legacy currency
        }
        
        self.logger.info(f"Manual overrides: {len(force_enabled)} force enabled, {len(force_disabled)} force disabled")
        
        return force_enabled, force_disabled
    
    async def run_availability_check(self) -> Dict[str, bool]:
        """
        Run comprehensive availability check.
        
        Returns:
            Dictionary mapping currency names to availability status
        """
        self.logger.info("Starting comprehensive currency availability check...")
        
        # Get current league
        self.current_league = self.get_current_league()
        if not self.current_league:
            self.logger.error("Cannot determine current league")
            return {}
        
        # Check availability from multiple sources
        price_data_currencies = self.check_availability_from_price_data(self.current_league, days_back=14)
        poe_watch_currencies = self.check_availability_from_poe_watch(days_back=7)
        
        # Get manual overrides
        force_enabled, force_disabled = self.get_manual_overrides()
        
        # Get all currencies from database
        try:
            conn = get_db_connection()
            all_currencies_df = pd.read_sql("SELECT name FROM currency ORDER BY name", conn)
            conn.close()
            all_currencies = set(all_currencies_df['name'].tolist())
        except Exception as e:
            self.logger.error(f"Failed to get currency list: {str(e)}")
            return {}
        
        # Determine availability for each currency
        availability_results = {}
        
        for currency_name in all_currencies:
            # Start with manual overrides
            if currency_name in force_disabled:
                availability_results[currency_name] = False
                continue
            
            if currency_name in force_enabled:
                availability_results[currency_name] = True
                continue
            
            # Check if currency appears in data sources
            has_price_data = currency_name in price_data_currencies
            has_poe_watch_data = currency_name in poe_watch_currencies
            
            # Currency is available if it appears in ANY data source
            is_available = has_price_data or has_poe_watch_data
            availability_results[currency_name] = is_available
            
            # Log decision reasoning
            sources = []
            if has_price_data:
                sources.append("price_data")
            if has_poe_watch_data:
                sources.append("poe_watch")
            
            source_str = ", ".join(sources) if sources else "none"
            self.logger.debug(f"{currency_name}: {'available' if is_available else 'unavailable'} (sources: {source_str})")
        
        # Summary statistics
        available_count = sum(availability_results.values())
        total_count = len(availability_results)
        
        # Debug: Show breakdown of availability sources
        poe_watch_enabled = sum(1 for name, avail in availability_results.items() 
                               if avail and name in poe_watch_currencies and name not in force_enabled)
        price_data_enabled = sum(1 for name, avail in availability_results.items() 
                                if avail and name in price_data_currencies and name not in force_enabled and name not in poe_watch_currencies)
        manual_enabled = sum(1 for name, avail in availability_results.items() 
                           if avail and name in force_enabled)
        
        self.logger.info(f"Availability check complete: {available_count}/{total_count} currencies available")
        self.logger.info(f"  - Manual enabled: {manual_enabled}")
        self.logger.info(f"  - POE Watch enabled: {poe_watch_enabled}")
        self.logger.info(f"  - Price data only enabled: {price_data_enabled}")
        self.logger.info(f"Price data sources: {len(price_data_currencies)} currencies")
        self.logger.info(f"POE Watch sources: {len(poe_watch_currencies)} currencies")
        self.logger.info(f"Manual overrides: {len(force_enabled)} enabled, {len(force_disabled)} disabled")
        
        self.availability_results = availability_results
        return availability_results
    
    def update_database(self, availability_results: Dict[str, bool]) -> bool:
        """
        Update the database with availability results.
        
        Args:
            availability_results: Dictionary mapping currency names to availability
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get manual overrides for source determination
            force_enabled, force_disabled = self.get_manual_overrides()
            
            # Prepare batch update
            update_query = """
            UPDATE currency 
            SET "isAvailableInCurrentLeague" = %s,
                "lastAvailabilityCheck" = %s,
                "availabilitySource" = %s
            WHERE name = %s
            """
            
            current_time = datetime.now()
            update_data = []
            
            for currency_name, is_available in availability_results.items():
                # Determine source
                source = "combined_check"
                if currency_name in force_enabled:
                    source = "manual_enabled"
                elif currency_name in force_disabled:
                    source = "manual_disabled"
                
                update_data.append((is_available, current_time, source, currency_name))
            
            # Execute batch update
            cursor.executemany(update_query, update_data)
            updated_rows = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Successfully updated {updated_rows} currency availability records")
            
            # Log summary
            enabled_count = sum(availability_results.values())
            disabled_count = len(availability_results) - enabled_count
            
            self.logger.info(f"Availability summary: {enabled_count} enabled, {disabled_count} disabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update database: {str(e)}")
            return False
    
    def generate_report(self) -> str:
        """Generate a detailed availability report."""
        if not self.availability_results:
            return "No availability results to report"
        
        available = [name for name, avail in self.availability_results.items() if avail]
        unavailable = [name for name, avail in self.availability_results.items() if not avail]
        
        report = f"""
Currency Availability Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Current League: {self.current_league or 'Unknown'}

SUMMARY:
- Total currencies: {len(self.availability_results)}
- Available: {len(available)}
- Unavailable: {len(unavailable)}

AVAILABLE CURRENCIES ({len(available)}):
""" + "\n".join(f"  ✓ {name}" for name in sorted(available))
        
        if unavailable:
            report += f"""

UNAVAILABLE CURRENCIES ({len(unavailable)}):
""" + "\n".join(f"  ✗ {name}" for name in sorted(unavailable))
        
        return report


async def main():
    """Main function to run currency availability check."""
    logger = MLLogger("CurrencyAvailabilityMain")
    
    try:
        checker = CurrencyAvailabilityChecker(logger)
        
        print("🔍 Checking currency availability in current league...")
        
        # Run availability check
        results = await checker.run_availability_check()
        
        if not results:
            print("❌ Failed to check currency availability")
            return
        
        # Update database
        print("💾 Updating database with availability results...")
        success = checker.update_database(results)
        
        if success:
            print("✅ Currency availability updated successfully!")
            
            # Generate and display report
            report = checker.generate_report()
            print("\n" + "="*80)
            print(report)
            print("="*80)
            
        else:
            print("❌ Failed to update database")
            
    except Exception as e:
        logger.error(f"Availability check failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 