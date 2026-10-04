"""Watchlist, stored snapshots and alert rules/events."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.deps import Symbol
from app.schemas import (
    AlertEvent,
    AlertRule,
    AlertRuleIn,
    Snapshot,
    WatchAdd,
    WatchItem,
)
from app.services import alerts, analyzer
from app.services.errors import InvalidInput, NotFound
from app.storage import db

router = APIRouter(tags=["watchlist & alerts"])

MAX_WATCHLIST = 50
MAX_RULES = 200


async def _display_name(symbol: str) -> str | None:
    """Company name for a new watchlist row: from a recent analysis, else a quick resolve."""
    recent = analyzer.latest_analysis(symbol)
    if recent is not None and recent.profile is not None:
        return recent.profile.name
    company = await analyzer.resolve_or_bare(symbol, timeout=6.0)
    return company.name if company.name and company.name != symbol else None


# ---- watchlist ---------------------------------------------------------------- #
@router.get("/watchlist", response_model=list[WatchItem])
async def list_watchlist() -> list[WatchItem]:
    """Watched tickers with their latest snapshot, a ~daily Δ baseline and a score sparkline."""
    return await db.watch_items()


@router.post("/watchlist", response_model=list[WatchItem])
async def add_to_watchlist(body: WatchAdd) -> list[WatchItem]:
    """Add a ticker (idempotent); the monitor keeps it fresh. Returns the full watchlist."""
    symbol = analyzer.normalize(body.ticker)
    current = await db.watch_tickers()
    if symbol not in current:
        if len(current) >= MAX_WATCHLIST:
            raise InvalidInput(f"Watchlist is full ({MAX_WATCHLIST} tickers). Remove one first.")
        await db.add_watch(symbol, await _display_name(symbol))
    return await db.watch_items()


@router.delete("/watchlist/{ticker}", response_model=list[WatchItem])
async def remove_from_watchlist(symbol: Symbol) -> list[WatchItem]:
    """Remove a ticker (idempotent; its snapshots and alert rules are kept). Returns the watchlist."""
    await db.remove_watch(symbol)
    return await db.watch_items()


@router.get("/snapshots/{ticker}", response_model=list[Snapshot])
async def list_snapshots(
    symbol: Symbol, limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> list[Snapshot]:
    """Stored analysis snapshots, newest first."""
    return await db.list_snapshots(symbol, limit)


# ---- alerts -------------------------------------------------------------------- #
@router.get("/alerts", response_model=list[AlertRule])
async def list_alert_rules() -> list[AlertRule]:
    return await db.list_rules()


@router.post("/alerts", response_model=AlertRule, status_code=status.HTTP_201_CREATED)
async def create_alert_rule(body: AlertRuleIn) -> AlertRule:
    """Create a rule. `threshold` per kind: score_above/score_below → SentiNET level (default 70/30);
    score_change → points vs ~24h ago (10); attention_spike → heat (75); new_narrative → min items (3);
    analyst_action → ignored. The ticker is refreshed by the monitor even if not on the watchlist."""
    symbol = analyzer.normalize(body.ticker)
    try:
        rule = alerts.normalize_rule(body.model_copy(update={"ticker": symbol}))
    except ValueError as exc:
        raise InvalidInput(str(exc)) from exc
    if len(await db.list_rules()) >= MAX_RULES:
        raise InvalidInput(f"Too many alert rules (max {MAX_RULES}). Delete some first.")
    return await db.create_rule(rule)


@router.get("/alerts/events", response_model=list[AlertEvent])
async def list_alert_events(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ticker: Annotated[str | None, Query(description="Only events for this ticker")] = None,
) -> list[AlertEvent]:
    """Triggered alerts, newest first."""
    symbol = analyzer.normalize(ticker) if ticker else None
    return await db.list_alert_events(limit, symbol)


@router.delete("/alerts/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert_rule(rule_id: int) -> Response:
    if not await db.delete_rule(rule_id):
        raise NotFound(f"Alert rule {rule_id} does not exist.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
