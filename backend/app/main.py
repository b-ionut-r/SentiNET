"""SentiNET FastAPI application: the `/api` routes plus the built single-page app.

Run: `uvicorn app.main:app` (or `python -m app serve`). Lifespan opens the
SQLite store, starts the watchlist monitor (if enabled) and, on shutdown,
stops background work and closes the shared HTTP client.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api import API_VERSION, build_router
from app.api.spa import mount_frontend
from app.api.web_settings import web_settings
from app.config import settings
from app.core.http import close_client
from app.services import alerts, analyzer
from app.services.errors import ServiceError
from app.services.monitor import Monitor
from app.storage import db

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sentinet")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(db.init_db)
    monitor = Monitor()
    app.state.monitor = monitor
    if settings.monitor_enabled:
        monitor.start()
        logger.info("watchlist monitor on (every %s min)", monitor.status.interval_minutes)
    try:
        yield
    finally:
        await monitor.stop()
        await alerts.drain()
        await analyzer.shutdown()
        await close_client()
        db.close_db()


async def _service_error(_: Request, exc: Exception) -> JSONResponse:
    status = exc.status_code if isinstance(exc, ServiceError) else 500
    return JSONResponse({"detail": str(exc)}, status_code=status)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    """422 with the same `{"detail": str}` shape as every other error (raw list under `errors`)."""
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    parts = []
    for err in errors:
        loc = [str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path")]
        parts.append(f"{'.'.join(loc) or 'request'}: {err.get('msg', 'invalid')}")
    return JSONResponse({"detail": "; ".join(parts) or "Invalid request.", "errors": jsonable_encoder(errors)},
                        status_code=422)


async def _unhandled_error(_: Request, __: Exception) -> JSONResponse:
    # Starlette re-raises after responding, so the server logs the traceback once.
    return JSONResponse({"detail": "Internal server error. Please retry; if it persists, check the logs."},
                        status_code=500)


def create_app() -> FastAPI:
    app = FastAPI(
        title="SentiNET",
        version=API_VERSION,
        summary="Market-intelligence terminal: evidence-backed sentiment from news, crowds, analysts, "
                "insiders, filings and global news tone.",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)  # SSE (text/event-stream) is never compressed
    app.add_middleware(CORSMiddleware, allow_origins=web_settings.cors_origin_list,
                       allow_methods=["*"], allow_headers=["*"])
    app.add_exception_handler(ServiceError, _service_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)
    app.include_router(build_router())
    mount_frontend(app)  # last: the SPA catch-all must not shadow /api
    return app


app = create_app()
