"""Meta endpoints: health + source listing."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse, SourceInfo
from app.sources.registry import ALL_SOURCES

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    disabled = settings.disabled_source_set
    return HealthResponse(
        status="ok",
        sentiment_engine=settings.sentiment_engine,
        sources=[
            SourceInfo(
                name=s.name,
                label=s.label,
                kind=s.kind,
                enabled=s.name not in disabled,
            )
            for s in ALL_SOURCES
        ],
    )


@router.get("/sources", response_model=list[SourceInfo])
async def sources() -> list[SourceInfo]:
    disabled = settings.disabled_source_set
    return [
        SourceInfo(name=s.name, label=s.label, kind=s.kind, enabled=s.name not in disabled)
        for s in ALL_SOURCES
    ]
