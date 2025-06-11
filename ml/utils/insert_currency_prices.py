import os
import csv
import sys
from psycopg2 import sql

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

def get_id(cursor, table, name_column, value):
    """Fetches the id from a table where name_column = value."""
    cursor.execute(
        sql.SQL("SELECT id FROM {} WHERE {} = %s").format(
            sql.Identifier(table),
            sql.Identifier(name_column)
        ),
        [value]
    )
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"{table}.{name_column} '{value}' not found in database.")
    return result[0]

def get_or_create_currency_id(cursor, currency_name):
    """Fetches the id from currency table by name, or inserts it if not found."""
    cursor.execute(
        sql.SQL("SELECT id FROM currency WHERE name = %s"),
        [currency_name]
    )
    result = cursor.fetchone()
    if result:
        return result[0]
    # Insert new currency with only the name set
    cursor.execute(
        sql.SQL("INSERT INTO currency (name) VALUES (%s) RETURNING id"),
        [currency_name]
    )
    new_id = cursor.fetchone()[0]
    print(f"Inserted new currency: {currency_name} (id={new_id})")
    return new_id

def insert_currency_prices(csv_path):
    """Reads a semicolon-delimited CSV and inserts rows into public.currency_prices."""
    conn = get_db_connection()
    cursor = conn.cursor()
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            league_id = get_id(cursor, "leagues", "name", row["League"])
            get_currency_id = get_or_create_currency_id(cursor, row["Get"])
            pay_currency_id = get_or_create_currency_id(cursor, row["Pay"])
            cursor.execute(
                """
                INSERT INTO public.currency_prices
                ("leagueId", "getCurrencyId", "payCurrencyId", "date", "value", "confidence")
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT ("leagueId", "getCurrencyId", "payCurrencyId", "date") DO NOTHING
                """,
                (
                    league_id,
                    get_currency_id,
                    pay_currency_id,
                    row["Date"],
                    row["Value"],
                    row["Confidence"]
                )
            )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted data from {csv_path} into public.currency_prices.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python insert_currency_prices.py <csv_file_path>")
        sys.exit(1)
    csv_file = sys.argv[1]
    if not os.path.isfile(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)
    insert_currency_prices(csv_file) 