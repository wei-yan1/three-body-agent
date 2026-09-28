"""Initialize PostgreSQL database and auth tables."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.storage.postgres.schema import init_postgres_schema


def main() -> None:
    init_postgres_schema()
    print("PostgreSQL schema initialized.")


if __name__ == "__main__":
    main()

