"""Initialize MySQL database and auth tables."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from app.storage.mysql.schema import init_mysql_schema


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    init_mysql_schema()
    print("MySQL schema initialized.")


if __name__ == "__main__":
    main()
