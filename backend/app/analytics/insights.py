"""Insights: noteworthy, actionable observations — each emitted only when its
evidence clears a threshold, and always with the numbers behind it.

Checks (thresholds):
* divergence   news vs crowd leaning opposite ways vs their norms (news tone ±0.08 off
               typical on >= 6 articles; crowd: StockTwits tags >= 1.64 SE off 62%, social
               text ±0.10, WSB ±0.15); price vs news (the 1M / 5D move >= 1σ / 1.5σ against
               the news lean)
* attention    GDELT volume z >= 2, Reddit mentions >= +100% (>= 10 mentions),
               Wikipedia views z >= 2, Reddit mentions collapsing <= -60% (>= 15 before)
* crowding     StockTwits bull share >= 85% or <= 35% with >= 15 tagged; top-5 WSB ticker
* reversal     GDELT 7d vs 30d tone sign flip (|Δ| >= 0.5); SentiNET Δ vs previous >= 12
* momentum     GDELT tone at a 90d high/low (pct >= 0.9 / <= 0.1); 48h headline
               tone shift >= 0.2 (>= 8 items each side)
* smart_money  >= 2 upgrades/downgrades or >= 3 PT raises/cuts in 30d; >= 2 insider
               buyers in 90d or an officer buy >= $500K; insider sales >= 0.5% of market cap
* catalyst     earnings <= 14 days; ex-dividend <= 7 days
* risk         lawsuit/probe/regulatory-setback events (>= 3 articles — or 2 incl. a major
               outlet — from >= 2 outlets, tone <= -0.1); red-flag 8-Ks; bankruptcy/going
               concern, delisting, short reports (corroborated); dilution
* quality      < 8 relevant items; >= 3 sources failed; engine failure; slow/failed feeds
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from app.analytics.composite import NEWS_BASELINE, SOCIAL_BASELINE
from app.analytics.composite import STOCKTWITS_BASELINE as STOCKTWITS_NORM
from app.analytics import textkit
from app.analytics.crowd import reddit_change_pct
from app.analytics.facts import Facts
from app.analytics.prepare import Item
from app.analytics.util import (
    count,
    join_and,
    money,
    ordinal,
    pct,
    quote,
    short_date,
    signed,
    tone_polarity,
    weighted_mean,
)
from app.schemas import DeltaView, Insight, Polarity, Verdict

MAX_INSIGHTS = 8
_SEVERITY_RANK = {"alert": 0, "watch": 1, "info": 2}
_OFFICER = ("chief", "ceo", "cfo", "president", "chair", "founder", "coo")
MIN_NEWS_FOR_DIVERGENCE = 6
LEGAL_EVENTS = frozenset({"lawsuit", "investigation", "regulatory_setback", "data_breach"})
MAJOR_TRUST = 1.1  # wires and majors (Reuters, Bloomberg, WSJ …)
RED_FLAG_EVENTS = {
    "bankruptcy": "Bankruptcy / going-concern risk",
    "delisting": "Delisting risk",
    "short_report": "Short-seller report",
}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass
class _Cand:
    insight: Insight
    priority: float


InsightKind = Literal["divergence", "attention", "reversal", "crowding", "catalyst", "smart_money", "risk",
                      "momentum", "quality"]
Severity = Literal["info", "watch", "alert"]


def _make(kind: InsightKind, severity: Severity, polarity: Polarity, title: str, detail: str,
          priority: float) -> _Cand:
    return _Cand(Insight(kind=kind, severity=severity, polarity=polarity, title=title, detail=detail), priority)


def build_insights(f: Facts, verdict: Verdict, delta: DeltaView) -> list[Insight]:
    cands: list[_Cand] = []
    checks: list[Callable[[], Iterator[_Cand]]] = [
        lambda: _divergences(f), lambda: _attention(f), lambda: _crowding(f),
        lambda: _reversals(f, verdict, delta), lambda: _momentum(f), lambda: _smart_money(f),
        lambda: _catalysts(f), lambda: _risks(f), lambda: _quality(f),
    ]
    for check in checks:
        cands.extend(check())
    cands.sort(key=lambda c: (_SEVERITY_RANK[c.insight.severity], -c.priority))
    seen: set[str] = set()
    out: list[Insight] = []
    for c in cands:
        if c.insight.title in seen:
            continue
        seen.add(c.insight.title)
        out.append(c.insight)
    return out[:MAX_INSIGHTS]


# --------------------------------------------------------------------------- #
# Divergences
# --------------------------------------------------------------------------- #
def crowd_lean(f: Facts) -> tuple[int, str] | None:
    """(+1 bullish / -1 bearish, evidence) when retail clearly leans one way vs its norms.

    StockTwits tags must differ from the 62% norm by >= 1.64 standard errors
    (one-sided 95%), social text by >= 0.10 from its typical tone on >= 15
    posts, WSB by >= 0.15 on >= 50 comments; conflicting votes cancel."""
    votes: list[tuple[int, str]] = []
    c = f.crowd
    if c is not None and c.stocktwits_bull_ratio is not None:
        ratio = c.stocktwits_bull_ratio
        tagged = (c.stocktwits_bullish or 0) + (c.stocktwits_bearish or 0)
        if tagged >= 10:
            z = (ratio - STOCKTWITS_NORM) / math.sqrt(STOCKTWITS_NORM * (1 - STOCKTWITS_NORM) / tagged)
            if abs(z) >= 1.64:
                votes.append((1 if z > 0 else -1, f"{ratio:.0%} of {tagged} tagged StockTwits posts are bullish "
                                                  f"(typical {STOCKTWITS_NORM:.0%})"))
    if f.social.n >= 15 and f.social.mean is not None and abs(f.social.mean - SOCIAL_BASELINE) >= 0.10:
        votes.append((1 if f.social.mean > SOCIAL_BASELINE else -1,
                      f"social posts average {signed(f.social.shrunk)} across {f.social.n}"))
    if c is not None and c.wsb_sentiment is not None and (c.wsb_comments or 0) >= 50 and abs(c.wsb_sentiment) >= 0.15:
        votes.append((1 if c.wsb_sentiment > 0 else -1, f"WallStreetBets sentiment is {signed(c.wsb_sentiment)}"))
    if not votes or len({v for v, _ in votes}) != 1:
        return None
    return votes[0][0], join_and([text for _, text in votes])


def news_lean(f: Facts) -> int:
    """+1 / -1 when news tone is clearly off its typical level (>= 6 articles), else 0."""
    if f.news.n < MIN_NEWS_FOR_DIVERGENCE or f.news.mean is None:
        return 0
    rel = f.news.mean - NEWS_BASELINE
    return 1 if rel >= 0.08 else -1 if rel <= -0.08 else 0


def _divergences(f: Facts) -> Iterator[_Cand]:
    parts = f.composite.parts
    crowd, news = crowd_lean(f), news_lean(f)
    if crowd is not None and news and crowd[0] != news:
        crowd_bull = crowd[0] > 0
        title = "Crowd bullish, news bearish" if crowd_bull else "News bullish, crowd bearish"
        tail = ("retail is positioned against the news flow" if crowd_bull
                else "retail is not buying the positive coverage")
        text = crowd[1]
        yield _make("divergence", "watch", "bear" if crowd_bull else "bull", title,
                    f"{text[0].upper() + text[1:]} while news tone is {signed(f.news.shrunk)} across "
                    f"{count(f.news.coverage or f.news.n, 'article')} — {tail}.", abs(parts["news"].x - parts["social"].x) + 0.5)

    t = f.inputs.technicals
    if t is not None and news:
        tone = f.news.shrunk
        sigma = parts["technicals"].facts.get("sigma_month") or (
            max(t.volatility_30d / math.sqrt(12), 1.5) if t.volatility_30d else 8.0)
        moves = []
        if t.return_1m is not None:
            moves.append((t.return_1m / sigma, 1.0, f"{pct(t.return_1m)} over 1 month"))
        if t.return_5d is not None:
            moves.append((t.return_5d / (sigma * math.sqrt(5 / 21)), 1.5, f"{pct(t.return_5d)} over 5 days"))
        for z, need, text in moves:
            if news > 0 and z <= -need:
                yield _make("divergence", "watch", "bear", "Price falling despite upbeat news",
                            f"{f.name} is {text} ({abs(z):.1f}σ) while news tone is {signed(tone)} across "
                            f"{count(f.news.coverage or f.news.n, 'article')} — the tape is not confirming the headlines.", abs(z))
                break
            if news < 0 and z >= need:
                yield _make("divergence", "watch", "bull", "Price rising despite negative news",
                            f"{f.name} is {text} ({z:.1f}σ) while news tone is {signed(tone)} across "
                            f"{count(f.news.coverage or f.news.n, 'article')} — buyers are looking through the headlines.", abs(z))
                break


# --------------------------------------------------------------------------- #
# Attention & crowding
# --------------------------------------------------------------------------- #
def _attention(f: Facts) -> Iterator[_Cand]:
    att, crowd = f.attention, f.crowd
    recent_pol = tone_polarity(f.news_recent.mean, 0.1) if f.news_recent.n >= 5 else "neutral"
    if att is not None and att.news_volume_z is not None and att.news_volume_z >= 2 and f.inputs.tone:
        vols = [p.volume for p in sorted(f.inputs.tone.series, key=lambda p: p.date) if p.volume is not None]
        recent = sum(vols[-2:]) / 2
        base = sorted(vols[-30:-2])
        typical = base[len(base) // 2] if base else 0
        sev: Severity = "alert" if att.news_volume_z >= 3.5 else "watch"
        multiple = f"{recent / typical:.1f}× its" if typical else "far above its"
        sigma = "≥ 10σ" if att.news_volume_z >= 10 else f"{att.news_volume_z:.1f}σ"
        yield _make("attention", sev, recent_pol, "Global news volume spiking",
                    f"GDELT article count is {multiple} 4-week norm ({recent:,.0f}/day over the last 2 days vs "
                    f"{typical:,.0f} typical; {sigma}).", min(att.news_volume_z, 10.0))
    change = reddit_change_pct(crowd)
    if crowd is not None and change is not None and crowd.reddit_mentions is not None:
        rank = ""
        if crowd.reddit_rank is not None:
            rank = f"; rank #{crowd.reddit_rank}" + (
                f" (from #{crowd.reddit_rank_prev})" if crowd.reddit_rank_prev is not None else "")
        tracked = f.metrics.get("reddit_tracked")
        rank += f" of {tracked} tracked tickers" if rank and isinstance(tracked, int) else ""
        if change >= 100 and crowd.reddit_mentions >= 10:
            yield _make("attention", "watch", "neutral", f"Reddit mentions {pct(change)} in 24h",
                        f"{crowd.reddit_mentions} mentions vs {crowd.reddit_mentions_prev} a day earlier{rank}.",
                        math.log2(1 + change / 100) + 1)
        elif change <= -60 and (crowd.reddit_mentions_prev or 0) >= 15:
            yield _make("attention", "info", "neutral", "Retail attention fading",
                        f"Reddit mentions fell {pct(abs(change), sign=False)} in 24h ({crowd.reddit_mentions_prev} → "
                        f"{crowd.reddit_mentions}){rank}.", abs(change) / 100)
    if att is not None and att.wiki_views_z is not None and att.wiki_views_z >= 2 and att.wiki_views_7d:
        yield _make("attention", "watch", "neutral", "Wikipedia attention spiking",
                    f"Pageviews are {att.wiki_views_z:.1f}σ above their 4-week norm "
                    f"({att.wiki_views_7d:,.0f}/day on average over the last 7 days) — the story is reaching "
                    f"a general audience.", att.wiki_views_z)


def _crowding(f: Facts) -> Iterator[_Cand]:
    crowd = f.crowd
    if crowd is None:
        return
    ratio = crowd.stocktwits_bull_ratio
    tagged = (crowd.stocktwits_bullish or 0) + (crowd.stocktwits_bearish or 0)
    if ratio is not None and tagged >= 15:
        if ratio >= 0.85:
            yield _make("crowding", "watch", "bear", "Crowded long on StockTwits",
                        f"{ratio:.0%} of {tagged} tagged posts are bullish vs a {STOCKTWITS_NORM:.0%} norm — "
                        f"one-sided retail positioning is vulnerable to bad news.", (ratio - STOCKTWITS_NORM) * 4)
        elif ratio <= 0.35:
            yield _make("crowding", "watch", "bull", "Retail capitulation on StockTwits",
                        f"Only {ratio:.0%} of {tagged} tagged posts are bullish vs a {STOCKTWITS_NORM:.0%} norm — "
                        f"washed-out retail sentiment is a classic contrarian setup.", (STOCKTWITS_NORM - ratio) * 4)
    wsb_rank = f.metrics.get("wsb_rank")
    if isinstance(wsb_rank, int) and wsb_rank <= 5:
        mood = f", sentiment {crowd.wsb_label} ({signed(crowd.wsb_sentiment)})" if (
            crowd.wsb_label and crowd.wsb_sentiment is not None) else ""
        comments = f" with {count(crowd.wsb_comments, 'comment')}" if crowd.wsb_comments else ""
        yield _make("crowding", "info", "neutral", f"#{wsb_rank} on WallStreetBets",
                    f"One of the most discussed tickers on r/wallstreetbets today{comments}{mood}.",
                    (6 - wsb_rank) / 5)


# --------------------------------------------------------------------------- #
# Reversals & momentum
# --------------------------------------------------------------------------- #
def _reversals(f: Facts, verdict: Verdict, delta: DeltaView) -> Iterator[_Cand]:
    tone = f.inputs.tone
    if tone is not None and tone.tone_7d is not None and tone.tone_30d is not None:
        t7, t30 = tone.tone_7d, tone.tone_30d
        if t7 * t30 < 0 and abs(t7 - t30) >= 0.5:
            up = t7 > 0
            yield _make("reversal", "watch", "bull" if up else "bear",
                        f"Global news tone flipped {'positive' if up else 'negative'}",
                        f"GDELT 7-day tone is {signed(t7)} vs {signed(t30)} over 30 days, across thousands "
                        f"of outlets worldwide.", abs(t7 - t30))
    if delta.sentinel_change is not None and abs(delta.sentinel_change) >= 12 and delta.previous_at:
        d = delta.sentinel_change
        prev = verdict.score - d
        yield _make("reversal", "alert" if abs(d) >= 20 else "watch", "bull" if d > 0 else "bear",
                    f"SentiNET {'jumped' if d > 0 else 'dropped'} {signed(d, 'd')} since {short_date(delta.previous_at)}",
                    f"Score moved {prev} → {verdict.score} ({verdict.label})"
                    + (f"; price {pct(delta.price_change_pct)} over the same period" if delta.price_change_pct is not None
                       else "") + ".", abs(d) / 10)


def _momentum(f: Facts) -> Iterator[_Cand]:
    tone = f.inputs.tone
    if tone is not None and tone.percentile_7d is not None and tone.tone_7d is not None:
        p = tone.percentile_7d
        if p >= 0.9 or p <= 0.1:
            high = p >= 0.9
            yield _make("momentum", "watch" if not high else "info", "bull" if high else "bear",
                        f"News tone at a 90-day {'high' if high else 'low'}",
                        f"GDELT 7-day tone of {signed(tone.tone_7d)} ranks in the {ordinal(round(p * 100))} "
                        f"percentile of the last 90 days.", abs(p - 0.5) * 2)
    r, o = f.news_recent, f.news_older
    if r.n >= 8 and o.n >= 8 and r.mean is not None and o.mean is not None and abs(r.mean - o.mean) >= 0.2:
        up = r.mean > o.mean
        yield _make("momentum", "watch", "bull" if up else "bear",
                    f"Headline tone turned {'up' if up else 'down'} in the last 48h",
                    f"Last 48h: {signed(r.mean)} across {count(r.n, 'headline')} vs {signed(o.mean)} across "
                    f"{o.n} in the prior days.", abs(r.mean - o.mean) * 3)


# --------------------------------------------------------------------------- #
# Smart money
# --------------------------------------------------------------------------- #
def _firms(names: list[str], limit: int = 3) -> str:
    uniq = list(dict.fromkeys(names))
    shown = join_and(uniq[:limit])
    return shown + (f" +{len(uniq) - limit} more" if len(uniq) > limit else "")


def _smart_money(f: Facts) -> Iterator[_Cand]:
    analysts = f.composite.parts["analysts"]
    rev = analysts.facts.get("revisions")
    if rev is not None:
        if rev.upgrades_30d >= 2 or rev.raises_30d >= 3:
            bits = []
            if rev.raises_30d:
                bits.append(f"{count(rev.raises_30d, 'price-target raise')}"
                            + (f" ({_firms(rev.raise_firms)})" if rev.raise_firms else ""))
            if rev.upgrades_30d:
                bits.append(f"{count(rev.upgrades_30d, 'upgrade')} ({_firms(rev.upgrade_firms)})")
            if rev.cuts_30d or rev.downgrades_30d:
                bits.append(f"vs {rev.cuts_30d} cuts / {rev.downgrades_30d} downgrades")
            yield _make("smart_money", "watch", "bull", "Analysts turning more bullish",
                        f"{'; '.join(bits)} in the last 30 days.", rev.upgrades_30d + 0.5 * rev.raises_30d)
        if rev.downgrades_30d >= 2 or rev.cuts_30d >= 3:
            bits = []
            if rev.cuts_30d:
                bits.append(f"{count(rev.cuts_30d, 'price-target cut')}"
                            + (f" ({_firms(rev.cut_firms)})" if rev.cut_firms else ""))
            if rev.downgrades_30d:
                bits.append(f"{count(rev.downgrades_30d, 'downgrade')} ({_firms(rev.downgrade_firms)})")
            yield _make("smart_money", "watch", "bear", "Analysts turning cautious",
                        f"{'; '.join(bits)} in the last 30 days.", rev.downgrades_30d + 0.5 * rev.cuts_30d)

    view = f.inputs.insiders
    if view is None:
        return
    since = f.now.date() - timedelta(days=90)
    buys = [t for t in view.transactions if t.kind == "buy" and t.date >= since]
    buyers = list(dict.fromkeys(t.insider for t in buys))
    total = sum(t.value or 0 for t in buys)
    if len(buyers) >= 2:
        top = max(buys, key=lambda t: t.value or 0)
        lead = f"; largest: {top.insider}" + (f" ({top.position})" if top.position else "") + (
            f" {money(top.value)} on {short_date(top.date)}" if top.value else "")
        yield _make("smart_money", "alert" if len(buyers) >= 3 else "watch", "bull", "Insider cluster buying",
                    f"{count(len(buyers), 'insider')} bought {money(total)} on the open market in the last 90 days"
                    f"{lead}.", 2 + len(buyers))
    elif buys:
        top = max(buys, key=lambda t: t.value or 0)
        role = (top.position or "").lower()
        if (top.value or 0) >= 500_000 and any(k in role for k in _OFFICER):
            yield _make("smart_money", "watch", "bull", f"{top.position} bought {money(top.value or 0)}",
                        f"{top.insider} bought {money(top.value or 0)} of stock on the open market on "
                        f"{short_date(top.date)} — officers rarely buy without conviction.", 2)
    bps = f.composite.parts["insiders"].facts.get("sell_bps")
    if bps is not None and bps >= 50 and not buys:
        yield _make("smart_money", "watch", "bear", "Heavy insider selling",
                    f"Insiders sold {money(view.sell_value)} in {view.window_days} days — {bps / 100:.2f}% of market "
                    f"cap, well above routine levels.", bps / 50)


# --------------------------------------------------------------------------- #
# Catalysts
# --------------------------------------------------------------------------- #
def _catalysts(f: Facts) -> Iterator[_Cand]:
    e = f.inputs.earnings
    today = f.now.date()
    if e is not None and e.next_date is not None and 0 <= (e.next_date - today).days <= 14:
        days = (e.next_date - today).days
        when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
        bits = [f"Reports {short_date(e.next_date)}"]
        reported = [h for h in e.history if h.eps_actual is not None and h.eps_estimate is not None]
        if e.beat_rate is not None and reported:
            bits.append(f"beat EPS estimates in {round(e.beat_rate * len(reported))} of the last {len(reported)} quarters")
        last = next((h for h in e.history if h.surprise_pct is not None), None)
        if last is not None and last.surprise_pct is not None:
            bits.append(f"last surprise {pct(last.surprise_pct)}")
        if e.eps_estimate is not None:
            bits.append(f"consensus EPS {money(e.eps_estimate, price=True)}")
        yield _make("catalyst", "watch" if days <= 7 else "info", "neutral", f"Earnings {when}",
                    "; ".join(bits) + ".", 3 - days / 7)
    for c in f.inputs.calendar_catalysts:
        if c.kind != "dividend" or not c.upcoming or not c.title.lower().startswith("ex-dividend"):
            continue
        days = (c.date.date() - today).days
        if 0 <= days <= 7:
            when = "today" if days == 0 else f"in {count(days, 'day')}"
            detail = f"Ex-dividend {short_date(c.date)}" + (f" · {c.detail}" if c.detail else "") + "."
            yield _make("catalyst", "info", "neutral", f"Ex-dividend {when}", detail, 1 - days / 7)


# --------------------------------------------------------------------------- #
# Risks
# --------------------------------------------------------------------------- #
def _corroborated(hits: list[Item]) -> bool:
    """>= 2 outlets, or one major outlet clearly about the company."""
    outlets = {o for it in hits for o in it.outlets()}
    return len(outlets) >= 2 or any(it.trust >= MAJOR_TRUST and it.relevance >= 0.8 for it in hits)


def _risks(f: Facts) -> Iterator[_Cand]:
    news = [it for it in f.prepared.items if it.group == "news" and it.scored and it.relevance >= 0.5]
    legal = [it for it in news if LEGAL_EVENTS & set(it.event_keys) and not it.press_release]
    solicitations = [it for it in news if it.press_release and "legal" in it.themes]
    outlets = {o for it in legal for o in it.outlets()}
    articles = sum(it.coverage for it in legal)
    tone, _ = weighted_mean((it.score, it.weight) for it in legal)
    enough = len(legal) >= 3 or (len(legal) >= 2 and any(it.trust >= MAJOR_TRUST for it in legal))
    if enough and len(outlets) >= 2 and tone is not None and tone <= -0.1:
        top = max(legal, key=lambda it: it.weight)
        kinds = Counter(k for it in legal for k in it.event_keys if k in LEGAL_EVENTS)
        about = join_and([textkit.event_label(k).lower() for k, _ in kinds.most_common(2)])
        story = next((s for s in f.stories[:3] if s.narrative.id == top.narrative_id), None)
        severe = story is not None and story.narrative.impact >= 0.5
        extra = (f" Plus {count(len(solicitations), 'law-firm solicitation')} — usually a sign a securities "
                 f"class action is being assembled." if len(solicitations) >= 2 else "")
        yield _make("risk", "alert" if severe else "watch", "bear", "Legal/regulatory overhang",
                    f"{count(articles, 'article')} from {count(len(outlets), 'outlet')} on {about} "
                    f"(tone {signed(tone)}); top: {quote(top.title, 80)}.{extra}", len(legal) / 3 + abs(tone))
    elif len(solicitations) >= 3:
        yield _make("risk", "watch", "bear", "Class-action solicitations",
                    f"{count(len(solicitations), 'law-firm press release')} soliciting shareholders — usually filed "
                    f"after a sharp drop; check for an underlying lawsuit.", len(solicitations) / 3)

    for key, title in RED_FLAG_EVENTS.items():
        hits = [it for it in news if key in it.event_keys and it.relevance >= 0.6 and not it.press_release]
        if hits and _corroborated(hits):
            top = max(hits, key=lambda it: it.weight)
            yield _make("risk", "alert", "bear", title,
                        f"{count(sum(it.coverage for it in hits), 'article')} — top: {quote(top.title, 90)}"
                        + (f" ({top.publisher})" if top.publisher else "") + ".", 5 + len(hits))
    offering = [it for it in news if "offering" in it.event_keys]
    if len(offering) >= 2 and _corroborated(offering):
        top = max(offering, key=lambda it: it.weight)
        yield _make("risk", "watch", "bear", "Dilution risk: share offering",
                    f"{count(sum(it.coverage for it in offering), 'article')} on a share offering — top: "
                    f"{quote(top.title, 90)}.",
                    len(offering))

    for filing in f.inputs.filings:
        if not filing.form.upper().startswith("8-K"):
            continue
        age = (f.now.date() - filing.date).days
        items = f" (item {', '.join(filing.items)})" if filing.items else ""
        if filing.importance == "high" and filing.polarity == "bear" and 0 <= age <= 120:
            yield _make("risk", "alert", "bear", f"Red-flag filing: {filing.title.split(':')[0][:60]}",
                        f"Form {filing.form}{items} filed {short_date(filing.date)}: {filing.title}.", 6 - age / 30)
        elif "3.02" in filing.items and 0 <= age <= 60:
            yield _make("risk", "watch", "bear", "Dilution: unregistered equity sale",
                        f"Form {filing.form}{items} filed {short_date(filing.date)}: {filing.title}.", 2)


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def _quality(f: Facts) -> Iterator[_Cand]:
    if f.prepared.engine_error:
        yield _make("quality", "alert", "neutral", "Sentiment engine failed",
                    f"{count(len(f.prepared.items), 'text')} could not be scored ({f.prepared.engine_error}); the "
                    f"verdict rests on structured data only.", 10)
    n = f.overall.n
    sources = sum(1 for r in f.inputs.source_runs if r.status == "ok")
    if not f.prepared.engine_error and n < 8 and sources:
        yield _make("quality", "watch" if n < 3 else "info", "neutral", "Thin coverage",
                    f"Only {count(n, 'relevant item')} from {count(sources, 'source')} — text scores are shrunk "
                    f"toward neutral; weigh the structured data more.", 8 - n)
    down = f.sources_down
    if down and not sources:
        yield _make("quality", "alert", "neutral", "No news or social data this run",
                    f"All {count(len(down), 'text source')} failed ({join_and(down)}); retry shortly.", 9)
    elif len(down) >= 3:
        yield _make("quality", "watch", "neutral", f"{len(down)} sources failed",
                    f"{join_and(down)} returned errors; the read uses the remaining {count(sources, 'source')}.",
                    len(down))
    pending = f.intel_pending()
    if "tone" in pending and f.inputs.tone is None:
        yield _make("quality", "info", "neutral", "Global news tone still loading",
                    "GDELT history is still being fetched; momentum uses headline flow only until the next "
                    "refresh.", 0.5)
    failed = f.intel_failed()
    if failed:
        names = {"analysts": "analyst ratings", "insiders": "insider trades", "earnings": "earnings",
                 "technicals": "technicals", "quote": "the quote", "filings": "SEC filings", "tone": "GDELT tone",
                 "wiki": "Wikipedia pageviews", "profile": "the profile", "calendar": "the dividend calendar"}
        shown = [names.get(k, k) for k in failed]
        yield _make("quality", "info", "neutral", "Some market data unavailable",
                    f"{_cap(join_and(shown))} could not be loaded this run; affected components are "
                    f"marked n/a.", len(shown) / 3)

