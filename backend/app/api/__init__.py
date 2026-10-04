"""HTTP API. Every route lives under `/api`; see `build_router`."""
from __future__ import annotations

from fastapi import APIRouter

API_VERSION = "2.0.0"


def build_router() -> APIRouter:
    """Assemble all API routers (imported here so `app.api` itself stays import-light)."""
    from app.api import (
        routes_analyze,
        routes_data,
        routes_export,
        routes_lab,
        routes_meta,
        routes_watch,
    )

    router = APIRouter(prefix="/api")
    for module in (routes_meta, routes_analyze, routes_data, routes_lab, routes_watch, routes_export):
        router.include_router(module.router)
    return router
