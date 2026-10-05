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
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api import API_VERSION, build_router
from app.api.limits import DEFAULT_MAX_BODY, BodySizeLimitMiddleware
from app.api.spa import mount_frontend
from app.api.web_settings import web_settings
from app.config import settings
from app.core.http import close_client
from app.services import alerts, analyzer, lab
from app.services.errors import ServiceError
from app.services.monitor import Monitor
from app.storage import db

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sentinet")

# The lab's JSON body: its whole character budget even fully escaped (an astral
# character as a `\uXXXX\uXXXX` pair is 12 bytes), plus quoting for 500 texts.
LAB_MAX_BODY = lab.MAX_TOTAL_CHARS * 12 + 1024 * 1024
# Pydantic error fields that may carry (echo) client input; only type/loc/msg go back.
_ECHO_FIELDS = ("input", "ctx", "url")


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
    retry = exc.retry_after if isinstance(exc, ServiceError) else None
    headers = {"Retry-After": str(retry)} if retry else None
    return JSONResponse({"detail": str(exc)}, status_code=status, headers=headers)


MAX_LISTED_ERRORS = 10


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    """422 with the same `{"detail": str}` shape as every other error (`errors`: type/loc/msg only).

    The rejected input is never echoed back: a 5 MB invalid body must not
    produce a 5 MB error response.
    """
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    errors = [{k: v for k, v in err.items() if k not in _ECHO_FIELDS} for err in errors]
    parts = []
    for err in errors[:MAX_LISTED_ERRORS]:
        loc = [str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path")]
        parts.append(f"{'.'.join(loc) or 'request'}: {err.get('msg', 'invalid')}")
    if len(errors) > MAX_LISTED_ERRORS:
        parts.append(f"… and {len(errors) - MAX_LISTED_ERRORS} more")
    return JSONResponse({"detail": "; ".join(parts) or "Invalid request.",
                         "errors": jsonable_encoder(errors[:MAX_LISTED_ERRORS])}, status_code=422)


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
    # Retry-After is exposed so a cross-origin UI can wait out a busy lab (503).
    app.add_middleware(CORSMiddleware, allow_origins=web_settings.cors_origin_list,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["Retry-After"])
    app.add_middleware(BodySizeLimitMiddleware, default=DEFAULT_MAX_BODY,
                       overrides={"/api/lab/score": LAB_MAX_BODY})
    # Outermost: requests for other Host names (DNS rebinding, LAN scans) never reach the API.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=web_settings.allowed_host_list, www_redirect=False)
    app.add_exception_handler(ServiceError, _service_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)
    app.include_router(build_router())
    mount_frontend(app)  # last: the SPA catch-all must not shadow /api
    return app


app = create_app()
