"""Reset the local test database and recreate the schema."""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.storage.postgres.client import postgres_connection
from app.storage.postgres.schema import init_postgres_schema

def main() -> None:
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute('DROP SCHEMA public CASCADE')
        cursor.execute('CREATE SCHEMA public')
    init_postgres_schema()
    print('PostgreSQL test database reset and initialized.')

if __name__ == '__main__':
    main()
