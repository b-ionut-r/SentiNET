"""`build_analysis(inputs) -> Analysis`: the analytics entry point.

Pure, synchronous and deterministic given its inputs (no network, no clock:
`inputs.now` is the only notion of time). Steps:

    sanitize structured intel → prepare items → tone summaries → narratives → themes/keywords/timeline →
    crowd & attention → six components → composite verdict → catalysts →
    delta vs previous snapshot → insights → brief → ranked signals
"""
from __future__ import annotations

from datetime import timedelta

from app.analytics.aggregate import keyword_list, source_reports, summarize, theme_stats, timeline
from app.analytics.brief import build_brief
from app.analytics.catalysts import build_catalysts
from app.analytics.composite import (
    DEGRADED_MAX_DISTANCE,
    analysts_part,
    compose,
    insiders_part,
    momentum_part,
    news_part,
    social_part,
    technicals_part,
)
from app.analytics.crowd import as_float, as_int, attention_view, crowd_view, merged_metrics, stocktwits_tally
from app.analytics.delta import build_delta
from app.analytics.facts import Facts
from app.analytics.inputs import AnalysisInputs
from app.analytics.insights import build_insights
from app.analytics.narratives import Story, build_narratives
from app.analytics.prepare import Item, prepare
from app.analytics.sanitize import sanitize_inputs
from app.analytics.verdict import build_verdict
from app.schemas import Analysis, Signal

MAX_SIGNALS = 200
RECENT_WINDOW = timedelta(hours=48)
OLDER_WINDOW = timedelta(days=7)


def build_analysis(inputs: AnalysisInputs) -> Analysis:
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

    stories = build_narratives(items, company, now, inputs.previous)
    metrics = merged_metrics(inputs.source_runs)
    crowd = crowd_view(metrics)
    tally = stocktwits_tally(metrics)
    attention = attention_view(inputs.tone, inputs.wiki_views, crowd, items, now)
    market_cap = inputs.quote.market_cap if inputs.quote is not None else None
    composite = compose([
        news_part(news, as_float(metrics.get("av_sentiment")), as_int(metrics.get("av_articles"))),
        social_part(social, crowd, tally),
        analysts_part(inputs.analysts, now),
        insiders_part(inputs.insiders, market_cap, now),
        momentum_part(inputs.tone, recent, older),
        technicals_part(inputs.technicals),
    ], max_distance=DEGRADED_MAX_DISTANCE if prepared.engine_error else None)
    themes = theme_stats(items)
    facts = Facts(
        inputs=inputs, prepared=prepared, overall=overall, news=news, social=social, news_recent=recent,
        news_older=older, stories=stories, themes=themes, metrics=metrics, crowd=crowd, stocktwits=tally,
        attention=attention, composite=composite,
    )
    verdict = build_verdict(facts)
    facts.catalysts = build_catalysts(facts)
    sentiment = overall.stat()
    delta = build_delta(inputs.previous, verdict, sentiment, inputs.quote, stories)
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


def select_signals(items: list[Item], stories: list[Story], limit: int = MAX_SIGNALS) -> list[Signal]:
    """Heaviest items first, always including every narrative member (≤ `limit`)."""
    members = {it.id for s in stories for it in s.members}
    chosen = [it for it in items if it.id in members]
    chosen += [it for it in items if it.id not in members][: max(0, limit - len(chosen))]
    chosen.sort(key=lambda it: (-it.weight, it.id))
    return [it.to_signal() for it in chosen[:limit]]
