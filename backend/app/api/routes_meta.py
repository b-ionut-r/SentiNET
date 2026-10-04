"""Service metadata: health and the source catalog."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.api import API_VERSION
from app.api.spa import frontend_available
from app.config import settings
from app.schemas import HealthResponse, SourceInfo
from app.services.tasks import run_bounded
from app.storage import db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["meta"])


def source_infos() -> list[SourceInfo]:
    """Every registered source with its enabled/configured state (empty if the registry fails)."""
    try:
        from app.sources.registry import all_sources

        sources = all_sources()
    except Exception:
        logger.exception("source registry unavailable")
        return []
    disabled = settings.disabled_source_set
    infos = []
    for s in sources:
        try:
            configured = bool(s.configured())
        except Exception:  # noqa: BLE001 - a broken source shows as unconfigured, not a 500
            configured = False
        infos.append(SourceInfo(
            key=s.key, label=s.label, kind=s.kind, enabled=s.key not in disabled,
            requires_key=s.requires_key, configured=configured, description=s.description,
            docs_url=s.docs_url,
        ))
    return infos


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + capabilities. Cheap: no upstream calls."""
    infos = source_infos()
    db_ok = (await run_bounded(db.watch_tickers, 3.0, name="db-health")).ok
    monitor = getattr(request.app.state, "monitor", None)
    features = {
        "database": db_ok,
        "frontend": frontend_available(),
        "monitor": bool(monitor is not None and monitor.running),
        "alert_webhook": bool(settings.alert_webhook_url),
        "transformer_engine": settings.sentiment_engine.startswith("finbert"),
    }
    return HealthResponse(
        status="ok" if db_ok and infos else "degraded",
        version=API_VERSION,
        engine=settings.sentiment_engine,
        sources=infos,
        features=features,
    )


@router.get("/sources", response_model=list[SourceInfo])
async def list_sources() -> list[SourceInfo]:
    """The source catalog, including keyed sources you can unlock with a free API key."""
    return source_infos()
