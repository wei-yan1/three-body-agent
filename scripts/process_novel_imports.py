"""Run queued/interrupted StoryRole novel imports.

This provides a small durable-worker mode for local deployments. Production can
replace it with Celery, RQ or another queue without changing the import API.
"""

from __future__ import annotations

import argparse

from app.services.novel_import_service import NovelImportService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    service = NovelImportService()
    recovered = service.recover_pending_jobs(limit=args.limit)
    print({"recovered": recovered, "count": len(recovered)})


if __name__ == "__main__":
    main()
