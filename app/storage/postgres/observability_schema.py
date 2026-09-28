"""Compatibility entry point for the unified schema."""
from app.storage.postgres.schema import init_postgres_schema

def init_observability_schema() -> None:
    init_postgres_schema()
