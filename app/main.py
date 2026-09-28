"""FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.a2a import router as a2a_router
from app.api.v1.mcp import router as mcp_router
from app.api.v1.auth import router as auth_router
from app.api.v1.novels import router as novels_router
from app.api.v1.storyrole import router as storyrole_router
from app.api.v1.models import router as models_router
from app.api.v1.settings import router as settings_router
from app.storage.postgres.schema import init_postgres_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="StoryRole Immersive Novel Character Platform")


@app.on_event("startup")
def initialize_database() -> None:
    """Initialize the schema once before serving requests."""
    init_postgres_schema()


app.include_router(a2a_router)
app.include_router(mcp_router)
app.include_router(auth_router)
app.include_router(novels_router)
app.include_router(storyrole_router)
app.include_router(models_router)
app.include_router(settings_router)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "frontend/static"), name="static")


@app.get("/")
def login_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/login.html")


@app.get("/agents")
def agents_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/storyrole.html")


@app.get("/storyrole")
def storyrole_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/storyrole.html")


@app.get("/settings")
def settings_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/settings.html")


@app.get("/models")
def models_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/models.html")
