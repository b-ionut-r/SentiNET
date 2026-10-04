"""The core endpoints: full analysis, as JSON or as a live Server-Sent Events stream."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.deps import Symbol
from app.schemas import Analysis, ProgressEvent
from app.services import analyzer
from app.services.errors import ServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])

RefreshParam = Query(False, description="Bypass the analysis cache and recompute now")


@router.get("/analyze/{ticker}", response_model=Analysis)
async def get_analysis(symbol: Symbol, refresh: bool = RefreshParam) -> Analysis:
    """Synthesized intel for one ticker (cached for `ANALYZE_CACHE_TTL`; `cached=true` when served from cache)."""
    return await analyzer.analyze(symbol, refresh=refresh)


def _error_event(message: str, status: int) -> ServerSentEvent:
    return ServerSentEvent(event="error", raw_data=json.dumps({"detail": message, "status": status}))


@router.get("/analyze/{ticker}/stream", response_class=EventSourceResponse)
async def stream_analysis(symbol: Symbol, refresh: bool = RefreshParam) -> AsyncIterator[ServerSentEvent]:
    """Live analysis as Server-Sent Events.

    Emits `event: progress` (a `ProgressEvent`) as each source/intel task starts
    and finishes, then exactly one terminal event: `event: result` (the
    `Analysis`) or `event: error` (`{"detail", "status"}`). Clients should close
    the EventSource on the terminal event. Disconnecting does not abort the
    shared analysis; it still completes and is cached.
    """
    queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()

    async def on_progress(event: ProgressEvent) -> None:
        queue.put_nowait(ServerSentEvent(event="progress", raw_data=event.model_dump_json()))

    async def run() -> None:
        try:
            result = await analyzer.analyze(symbol, refresh=refresh, progress=on_progress)
            queue.put_nowait(ServerSentEvent(event="result", raw_data=result.model_dump_json()))
        except ServiceError as exc:
            queue.put_nowait(_error_event(str(exc), exc.status_code))
        except Exception:
            logger.exception("stream analysis of %s failed", symbol)
            queue.put_nowait(_error_event("Internal error while analyzing; please retry.", 500))
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(run(), name=f"sse:{symbol}")
    try:
        while (item := await queue.get()) is not None:
            yield item
    finally:
        if not task.done():
            task.cancel()  # only detaches this listener; the shared run keeps going
