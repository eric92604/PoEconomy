#!/usr/bin/env python3
"""
Database migration script to add currency availability tracking.

This script adds a new column to the currency table to track whether
each currency is available in the current league based on live data.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.database import get_db_connection
from utils.logging_utils import MLLogger


def add_currency_availability_column():
    """
    Add isAvailableInCurrentLeague column to currency table.
    
    This column will track whether each currency is currently available
    based on recent price data and live API checks.
    """
    logger = MLLogger("CurrencyMigration")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column already exists
        check_column_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'currency' 
        AND column_name = 'isAvailableInCurrentLeague'
        """
        
        cursor.execute(check_column_query)
        result = cursor.fetchone()
        
        if result:
            logger.info("Column 'isAvailableInCurrentLeague' already exists")
            conn.close()
            return True
        
        # Add the new column
        alter_query = """
        ALTER TABLE currency 
        ADD COLUMN "isAvailableInCurrentLeague" BOOLEAN DEFAULT true,
        ADD COLUMN "lastAvailabilityCheck" TIMESTAMP WITH TIME ZONE DEFAULT now(),
        ADD COLUMN "availabilitySource" VARCHAR(50) DEFAULT 'manual'
        """
        
        logger.info("Adding currency availability columns...")
        cursor.execute(alter_query)
        
        # Create index for better query performance
        index_query = """
        CREATE INDEX IF NOT EXISTS idx_currency_availability 
        ON currency("isAvailableInCurrentLeague")
        """
        
        cursor.execute(index_query)
        
        # Add comment to explain the columns
        comment_queries = [
            """
            COMMENT ON COLUMN currency."isAvailableInCurrentLeague" IS 
            'Indicates if the currency is available in the current active league based on recent price data'
            """,
            """
            COMMENT ON COLUMN currency."lastAvailabilityCheck" IS 
            'Timestamp of the last availability check against live data'
            """,
            """
            COMMENT ON COLUMN currency."availabilitySource" IS 
            'Source of availability data: manual, poe_ninja, price_data, or api_check'
            """
        ]
        
        for comment_query in comment_queries:
            cursor.execute(comment_query)
        
        conn.commit()
        logger.info("Successfully added currency availability columns")
        
        # Initialize availability for existing currencies
        initialize_query = """
        UPDATE currency 
        SET "isAvailableInCurrentLeague" = true,
            "lastAvailabilityCheck" = now(),
            "availabilitySource" = 'default'
        WHERE "isAvailableInCurrentLeague" IS NULL
        """
        
        cursor.execute(initialize_query)
        affected_rows = cursor.rowcount
        conn.commit()
        
        logger.info(f"Initialized availability for {affected_rows} currencies")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to add currency availability column: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False


def verify_migration():
    """Verify that the migration was successful."""
    logger = MLLogger("CurrencyMigrationVerify")
    
    try:
        conn = get_db_connection()
        
        # Check table structure
        structure_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = 'currency'
        AND column_name IN ('isAvailableInCurrentLeague', 'lastAvailabilityCheck', 'availabilitySource')
        ORDER BY column_name
        """
        
        cursor = conn.cursor()
        cursor.execute(structure_query)
        columns = cursor.fetchall()
        
        if len(columns) == 3:
            logger.info("✓ All currency availability columns added successfully")
            for col in columns:
                logger.info(f"  - {col[0]}: {col[1]} (nullable: {col[2]}, default: {col[3]})")
        else:
            logger.error(f"✗ Expected 3 columns, found {len(columns)}")
            return False
        
        # Check sample data
        sample_query = """
        SELECT 
            name,
            "isAvailableInCurrentLeague",
            "lastAvailabilityCheck",
            "availabilitySource"
        FROM currency 
        LIMIT 5
        """
        
        cursor.execute(sample_query)
        samples = cursor.fetchall()
        
        logger.info("Sample currency availability data:")
        for sample in samples:
            logger.info(f"  {sample[0]}: available={sample[1]}, last_check={sample[2]}, source={sample[3]}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Migration verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Adding currency availability tracking to database...")
    
    success = add_currency_availability_column()
    if success:
        print("✓ Migration completed successfully")
        
        print("\nVerifying migration...")
        if verify_migration():
            print("✓ Migration verification passed")
        else:
            print("✗ Migration verification failed")
            sys.exit(1)
    else:
        print("✗ Migration failed")
        sys.exit(1)
    
    print("\nCurrency availability tracking is now enabled!")
    print("Next steps:")
    print("1. Run the currency availability checker script")
    print("2. Update training configuration to use availability filtering")
    print("3. Re-run feature engineering and model training") 