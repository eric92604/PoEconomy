import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path="server/.env")

BACKUP_DIR = "pg_backup"
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
    raise ValueError("Database credentials are not fully set in the environment.")

os.makedirs(BACKUP_DIR, exist_ok=True)

def get_backup_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{DB_NAME}_backup_{timestamp}.sql"

def backup_postgres():
    backup_file = os.path.join(BACKUP_DIR, get_backup_filename())
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD
    cmd = [
        "pg_dump",
        "-h", DB_HOST,
        "-p", str(DB_PORT),
        "-U", DB_USER,
        "-F", "c",  # custom format
        "-b",        # include blobs
        "-v",        # verbose
        "-f", backup_file,
        DB_NAME
    ]
    try:
        print(f"Starting backup: {backup_file}")
        subprocess.run(cmd, env=env, check=True)
        print(f"Backup completed: {backup_file}")
    except FileNotFoundError:
        print("Error: pg_dump is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Backup failed: {e}")

if __name__ == "__main__":
    backup_postgres() 