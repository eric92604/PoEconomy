"""
Database utilities for ML scripts.
Provides reusable database connection and common query functions.
"""

import os
import psycopg2
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from server/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "server", ".env"))

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using environment variables.
    
    Returns:
        psycopg2.connection: Database connection object
        
    Raises:
        ValueError: If required environment variables are missing
        psycopg2.Error: If database connection fails
    """
    # Get database credentials from environment
    db_config = {
        'dbname': os.getenv("DB_NAME"),
        'user': os.getenv("DB_USER"),
        'password': os.getenv("DB_PASSWORD"),
        'host': os.getenv("DB_HOST", "localhost"),
        'port': os.getenv("DB_PORT", "5432")
    }
    
    # Validate all required credentials are present
    missing_vars = [key for key, value in db_config.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except psycopg2.Error as e:
        raise psycopg2.Error(f"Database connection failed: {e}")

def get_table_row_count(table_name: str) -> Optional[int]:
    """
    Get the number of rows in a table.
    
    Args:
        table_name: Name of the table to count
        
    Returns:
        int: Number of rows, or None if error
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"Error counting rows in {table_name}: {e}")
        return None

def test_connection() -> bool:
    """
    Test the database connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[0] == 1
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the connection when run directly
    print("Testing database connection...")
    if test_connection():
        print("*** Database connection successful!")
        
        # Show table counts
        tables = ['leagues', 'currency', 'currency_prices', 'items', 'item_prices', 'predictions']
        print("\nTable row counts:")
        for table in tables:
            count = get_table_row_count(table)
            if count is not None:
                print(f"  {table}: {count:,}")
    else:
        print("❌ Database connection failed!") 