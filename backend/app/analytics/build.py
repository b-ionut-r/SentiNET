"""`build_analysis(inputs) -> Analysis`: the analytics entry point.

Pure, synchronous and deterministic given its inputs (no network, no clock:
`inputs.now` is the only notion of time). Steps:

    sanitize structured intel → prepare items → tone summaries → narratives → themes/keywords/timeline →
    crowd & attention → pending-deal check → six components → composite verdict → catalysts →
    delta vs previous snapshot → insights → brief → ranked signals

One bad provider value must never cost the user the whole analysis: structured
intel is sanitized first, and each component (and the catalyst list) is built
in isolation — one that still fails is marked unavailable and reported as a
data-quality insight.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from datetime import timedelta
from typing import TypeVar

from app.analytics.aggregate import keyword_list, source_reports, summarize, theme_stats, timeline
from app.analytics.brief import build_brief
from app.analytics.catalysts import build_catalysts
from app.analytics.composite import (
    DEGRADED_MAX_DISTANCE,
    LABELS,
    ComponentKey,
    Part,
    analysts_part,
    compose,
    deal_anchored,
    insiders_part,
    momentum_part,
    news_part,
    social_part,
    technicals_part,
)
from app.analytics.crowd import as_float, as_int, attention_view, crowd_view, merged_metrics, stocktwits_tally
from app.analytics.deals import pending_deal
from app.analytics.delta import build_delta
from app.analytics.facts import Facts
from app.analytics.inputs import AnalysisInputs
from app.analytics.insights import build_insights
from app.analytics.narratives import Story, build_narratives
from app.analytics.prepare import Item, prepare
from app.analytics.sanitize import sanitize_inputs
from app.analytics.verdict import build_verdict
from app.schemas import Analysis, Signal

log = logging.getLogger(__name__)
T = TypeVar("T")

MAX_SIGNALS = 200
RECENT_WINDOW = timedelta(hours=48)
OLDER_WINDOW = timedelta(days=7)


def build_analysis(inputs: AnalysisInputs) -> Analysis:
    """The full analysis of one ticker; `generated_at` is `inputs.now` (the caller sets cache/timing fields)."""
    inputs = sanitize_inputs(inputs)  # NaN/inf numbers and naive datetimes from providers
    now = inputs.now
    company = inputs.company
    prepared = prepare(company, inputs.source_runs, now)
    items = prepared.items
    news_items = [it for it in items if it.group == "news"]
    social_items = [it for it in items if it.group == "social"]
    overall, news, social = summarize(items), summarize(news_items), summarize(social_items)
    recent = summarize(it for it in news_items if it.timestamp and now - it.timestamp <= RECENT_WINDOW)
    older = summarize(it for it in news_items
                      if it.timestamp and RECENT_WINDOW < now - it.timestamp <= OLDER_WINDOW)

    stories = build_narratives(items, company, now, inputs.previous, previous_ids=_previous_story_ids(inputs),
                               actions=inputs.analysts.actions if inputs.analysts is not None else ())
    metrics = merged_metrics(inputs.source_runs)
    crowd = crowd_view(metrics)
    tally = stocktwits_tally(metrics)
    attention = attention_view(inputs.tone, inputs.wiki_views, crowd, items, now)
    market_cap = inputs.quote.market_cap if inputs.quote is not None else None
    currency = (inputs.quote.currency if inputs.quote is not None else None) or "USD"
    reporting = "USD" if currency == "USD" else None  # see Facts.reporting_currency
    asset = company.quote_type
    failed: list[str] = []

    deal = _guard("Deal detection", lambda: pending_deal(inputs.filings, company, now.date()), lambda: None, failed)

    def part(key: ComponentKey, make: Callable[[], Part]) -> Part:
        built = _guard(LABELS[key], make, lambda: Part(key, detail="could not be computed"), failed)
        # A pending acquisition pins the price to the deal terms (see deals.py).
        return deal_anchored(built) if deal is not None and key in ("analysts", "technicals") else built

    composite = compose([
        part("news", lambda: news_part(news, as_float(metrics.get("av_sentiment")),
                                       as_int(metrics.get("av_articles")))),
        part("social", lambda: social_part(social, crowd, tally)),
        part("analysts", lambda: analysts_part(inputs.analysts, now, asset, currency)),
        part("insiders", lambda: insiders_part(inputs.insiders, market_cap, now, asset, reporting)),
        part("momentum", lambda: momentum_part(inputs.tone, recent, older)),
        part("technicals", lambda: technicals_part(inputs.technicals)),
    ], max_distance=DEGRADED_MAX_DISTANCE if prepared.engine_error else None)
    themes = theme_stats(items)
    facts = Facts(
        inputs=inputs, prepared=prepared, overall=overall, news=news, social=social, news_recent=recent,
        news_older=older, stories=stories, themes=themes, metrics=metrics, crowd=crowd, stocktwits=tally,
        attention=attention, composite=composite, failed=failed, deal=deal,
    )
    verdict = build_verdict(facts)
    facts.catalysts = _guard("Catalysts", lambda: build_catalysts(facts), list, failed)
    sentiment = overall.stat()
    delta = build_delta(inputs.previous, verdict, sentiment, inputs.quote, stories, inputs.previous_components)
    insights = build_insights(facts, verdict, delta)
    brief = build_brief(facts, verdict, insights)

    return Analysis(
        ticker=company.ticker, generated_at=now, engine=inputs.engine_name,
        profile=inputs.profile, quote=inputs.quote, technicals=inputs.technicals,
        verdict=verdict, brief=brief, delta=delta, insights=insights,
        sentiment=sentiment, news=news.stat(), social=social.stat(),
        narratives=[s.narrative for s in stories], themes=themes, keywords=keyword_list(items, company),
        timeline=timeline(items, now), tone=inputs.tone,
        analysts=inputs.analysts, insiders=inputs.insiders, earnings=inputs.earnings, filings=list(inputs.filings),
        crowd=crowd, attention=attention, catalysts=facts.catalysts,
        sources=source_reports(inputs.source_runs, prepared), signals=select_signals(items, stories),
    )


def _guard(what: str, make: Callable[[], T], fallback: Callable[[], T], failed: list[str]) -> T:
    """`make()`, or `fallback()` when it raises (logged and recorded in `failed`)."""
    try:
        return make()
    except Exception:  # noqa: BLE001 - one malformed provider value must not sink the analysis
        log.exception("analytics: %s failed; continuing without it", what)
        failed.append(what)
        return fallback()


def _previous_story_ids(inputs: AnalysisInputs) -> Collection[Collection[str]] | None:
    """Member signal ids of the previous snapshot's stories, when the caller provides them.

    Read defensively: `previous_story_ids` is a requested (optional) addition to
    `AnalysisInputs`; until it exists NEW flags rely on timing and headlines."""
    ids = getattr(inputs, "previous_story_ids", None)
    return ids if isinstance(ids, (list, tuple)) else None


def select_signals(items: list[Item], stories: list[Story], limit: int = MAX_SIGNALS) -> list[Signal]:
    """Heaviest items first (≤ `limit`): every narrative member, then published media up to half
    the list when there is that much of it, then the heaviest of the rest — so a busy StockTwits
    stream cannot crowd the reporting out of the explorer."""
    members = {it.id for s in stories for it in s.members}
    chosen = [it for it in items if it.id in members]
    rest = [it for it in items if it.id not in members]  # `items` is heaviest first
    media = [it for it in rest if it.group == "news"]
    reserved = media[: max(0, min(limit // 2 - sum(it.group == "news" for it in chosen), limit - len(chosen)))]
    taken = {it.id for it in reserved}
    chosen += reserved
    chosen += [it for it in rest if it.id not in taken][: max(0, limit - len(chosen))]
    chosen.sort(key=lambda it: (-it.weight, it.id))
    return [it.to_signal() for it in chosen[:limit]]
