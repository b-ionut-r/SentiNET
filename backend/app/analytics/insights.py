"""Insights: noteworthy, actionable observations — each emitted only when its
evidence clears a threshold, and always with the numbers behind it.

Checks (thresholds):
* divergence   news vs crowd leaning opposite ways vs their norms (news tone ±0.08 off
               typical on >= 6 articles; crowd: StockTwits tags (per account when
               available) >= 1.64 SE off 62% on >= 10 tags, social
               text ±0.10, WSB ±0.15); price vs news (the 1M / 5D move >= 1σ / 1.5σ against
               the news lean)
* attention    GDELT volume z >= 2, Reddit mentions >= +100% (>= 10 mentions), a Reddit
               rank breakout (top 25 from outside the top 100, >= 10 mentions),
               Wikipedia views z >= 2, Reddit mentions collapsing <= -60% (>= 15 before)
* crowding     StockTwits bull share >= 85% or <= 35% with >= 15 tags (per account when
               available); top-5 WSB ticker
* reversal     GDELT 7d vs 30d tone sign flip (|Δ| >= 0.5); SentiNET Δ vs previous >= 12
* momentum     GDELT tone at a 90d high/low (shown pct >= 90th / <= 10th); 48h headline
               tone shift >= 0.2 after discounting decay toward the typical tone (>= 8 items
               each side; a mere return toward typical is never a "turn"); otherwise the
               momentum component itself when it reads <= 35 or >= 65 and moves the score
               >= 1 point (not when a GDELT sign flip already tells the story)
* smart_money  >= 2 upgrades/downgrades or >= 3 PT raises/cuts in 30d; >= 2 insiders each
               buying >= $25K in 90d, together >= $100K (or 0.5 bp of the USD market cap), and
               not dwarfed (> 10×) by discretionary insider sales — or an officer buy >= $500K;
               insider sales >= 0.5% of the USD market cap
* catalyst     earnings <= 14 days; ex-dividend <= 7 days
* deal         a pending acquisition of the company (signed merger agreement in its 8-Ks,
               see deals.py) — always an alert, ranked first; else a deal in play: fresh,
               corroborated M&A coverage involving the company (deals.py) — an alert when the
               quoted value is >= 10% of the USD market cap, none under 1%, else watch
* risk         lawsuit/probe/regulatory-setback events (>= 3 articles — or 2 incl. a major
               outlet — from >= 2 outlets, tone <= -0.1); red-flag 8-Ks; bankruptcy/going
               concern, delisting, short reports (corroborated); dilution (incl. new shares
               reported inside the deal-in-play coverage). A listing-rule
               notice is not raised once a later filing reports regained compliance or
               while the company is being acquired; a compliance notice is not a red flag
* quality      no relevant text at all (whatever the source statuses); < 8 relevant items;
               >= 3 sources failed; engine failure; a component that failed on bad data;
               slow/failed feeds
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from app.analytics import textkit
from app.analytics.composite import (
    LABELS,
    MATERIAL_BUY,
    NEWS_BASELINE,
    SOCIAL_BASELINE,
    ComponentKey,
    buyers,
    crowded,
    discretionary_sales,
    headline_shift,
    token_factor,
)
from app.analytics.composite import STOCKTWITS_BASELINE as STOCKTWITS_NORM
from app.analytics.crowd import BREAKOUT_RANK, reddit_breakout, reddit_change_pct
from app.analytics.facts import COMPONENT_FEEDS, FEED_NAMES, Facts
from app.analytics.prepare import Item
from app.analytics.util import (
    count,
    filing_parts,
    join_and,
    money,
    ordinal,
    pct,
    quote,
    short_date,
    signed,
    tone_polarity,
    trim,
    weighted_mean,
)
from app.schemas import DeltaView, Insight, Polarity, Verdict

MAX_INSIGHTS = 8
CLUSTER_DAYS = 90  # window of the insider cluster-buying check
CLUSTER_MIN_VALUE = 100_000.0  # USD the material buyers must put in together (or CLUSTER_MIN_BPS of the cap)
CLUSTER_MIN_BPS = 0.5
_SEVERITY_RANK = {"alert": 0, "watch": 1, "info": 2}
_OFFICER = ("chief", "ceo", "cfo", "president", "chair", "founder", "coo")
MIN_NEWS_FOR_DIVERGENCE = 6
HEADLINE_TURN = 0.2  # decay-discounted 48h-vs-prior headline tone change worth an insight
MOMENTUM_CLEAR = 0.3  # |x| of the momentum component (score <= 35 or >= 65) worth an insight
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
                      "momentum", "quality", "deal"]
Severity = Literal["info", "watch", "alert"]


def _make(kind: InsightKind, severity: Severity, polarity: Polarity, title: str, detail: str,
          priority: float) -> _Cand:
    return _Cand(Insight(kind=kind, severity=severity, polarity=polarity, title=title, detail=detail), priority)


def build_insights(f: Facts, verdict: Verdict, delta: DeltaView) -> list[Insight]:
    cands: list[_Cand] = []
    checks: list[Callable[[], Iterator[_Cand]]] = [
        lambda: _divergences(f), lambda: _attention(f), lambda: _crowding(f),
        lambda: _reversals(f, verdict, delta), lambda: _momentum(f), lambda: _smart_money(f),
        lambda: _catalysts(f), lambda: _deals(f), lambda: _risks(f), lambda: _quality(f),
    ]
    for check in checks:
        cands.extend(check())
    if all(c.insight.kind == "quality" for c in cands):
        cands.extend(_main_drag(f, verdict))  # the headline names it, so the rail must not say "nothing"
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
    c, tally = f.crowd, f.stocktwits
    if tally is not None and tally.ratio is not None and tally.n >= 10:
        ratio, tagged = tally.ratio, tally.n
        z = (ratio - STOCKTWITS_NORM) / math.sqrt(STOCKTWITS_NORM * (1 - STOCKTWITS_NORM) / tagged)
        if abs(z) >= 1.64:
            votes.append((1 if z > 0 else -1, f"{ratio:.0%} of {tally.described} are bullish "
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
    breakout = reddit_breakout(crowd)
    if breakout and crowd is not None and crowd.reddit_rank is not None:
        tracked = f.metrics.get("reddit_tracked")
        of = f" of {tracked} tracked tickers" if isinstance(tracked, int) else ""
        since = f"from #{crowd.reddit_rank_prev}" if crowd.reddit_rank_prev is not None else "from unranked"
        before = (f" vs {crowd.reddit_mentions_prev} a day earlier" if crowd.reddit_mentions_prev is not None else "")
        yield _make("attention", "watch", "neutral", f"Reddit breakout: #{crowd.reddit_rank} {since}",
                    f"{crowd.reddit_mentions} mentions in 24h{before} — {f.name} jumped into Reddit's top "
                    f"{BREAKOUT_RANK}{of}.", 3.0)
    if crowd is not None and change is not None and crowd.reddit_mentions is not None and not breakout:
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
    crowd, tally = f.crowd, f.stocktwits
    if crowd is None:
        return
    side = crowded(tally)
    if tally is not None and tally.ratio is not None and side:
        ratio = tally.ratio
        if side > 0:
            yield _make("crowding", "watch", "bear", "Crowded long on StockTwits",
                        f"{ratio:.0%} of {tally.described} are bullish vs a {STOCKTWITS_NORM:.0%} norm — "
                        f"one-sided retail positioning is vulnerable to bad news.", (ratio - STOCKTWITS_NORM) * 4)
        else:
            yield _make("crowding", "watch", "bull", "Retail capitulation on StockTwits",
                        f"Only {ratio:.0%} of {tally.described} are bullish vs a {STOCKTWITS_NORM:.0%} norm — "
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
        if _gdelt_flip(f):
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


def _main_drag(f: Facts, verdict: Verdict) -> Iterator[_Cand]:
    """The component the headline names as the main drag (or offset), as a watch item.

    Only used when no other check fired: every clause of the headline then
    still has its evidence on the rail."""
    if verdict.stance == "neutral":
        return
    sign = 1 if verdict.stance == "bullish" else -1
    comp = f.composite
    against = [(k, c) for k, c in comp.contributions.items()
               if c * sign <= -1.0 and comp.parts[k].phrase and comp.parts[k].reason]
    if not against:
        return
    key, points = min(against, key=lambda kc: kc[1] * sign)
    part = comp.parts[key]
    lead, _, evidence = (part.reason or "").partition(": ")
    role = "drag on" if sign > 0 else "offset to"
    yield _make(part_kind(key), "watch", "bear" if sign > 0 else "bull", lead,
                f"{_cap(evidence)} — the main {role} the {verdict.label} read ({signed(points, '.1f')} points).",
                abs(points))


def part_kind(key: str) -> InsightKind:
    """Insight kind for a component's evidence."""
    return {"momentum": "momentum", "analysts": "smart_money", "insiders": "smart_money",
            "social": "crowding"}.get(key, "divergence")  # type: ignore[return-value]


def restated_component(f: Facts, insight: Insight) -> ComponentKey | None:
    """The component whose evidence line an insight restates (its title is that line's lead:
    the main-drag and momentum-component insights), so the brief states that fact once."""
    for key, part in f.composite.parts.items():
        if part.reason and part.reason.partition(": ")[0] == insight.title:
            return key
    return None


def _gdelt_flip(f: Facts) -> bool:
    """GDELT 7-day tone on the other side of zero from its 30-day level (by >= 0.5)."""
    tone = f.inputs.tone
    if tone is None or tone.tone_7d is None or tone.tone_30d is None:
        return False
    return tone.tone_7d * tone.tone_30d < 0 and abs(tone.tone_7d - tone.tone_30d) >= 0.5


def _momentum(f: Facts) -> Iterator[_Cand]:
    fired = _gdelt_flip(f)  # the reversal insight already tells the tone story
    tone = f.inputs.tone
    if tone is not None and tone.percentile_7d is not None and tone.tone_7d is not None:
        p = tone.percentile_7d
        rank = round(p * 100)  # judged as shown ("10th pct" is a 90-day low)
        if rank >= 90 or rank <= 10:
            high = rank >= 90
            fired = True
            yield _make("momentum", "watch" if not high else "info", "bull" if high else "bear",
                        f"News tone at a 90-day {'high' if high else 'low'}",
                        f"GDELT 7-day tone of {signed(tone.tone_7d)} ranks in the {ordinal(rank)} "
                        f"percentile of the last 90 days.", abs(p - 0.5) * 2)
    r, o = f.news_recent, f.news_older
    shift = headline_shift(r, o)
    # Tone drifting back to normal after an event day is not a turn (the component may still
    # report it below, worded "Headline tone normalizing", when it clearly moves the score).
    if shift is not None and r.n >= 8 and o.n >= 8 and not shift.normalizing \
            and abs(shift.effective) >= HEADLINE_TURN:
        up = shift.change > 0
        fired = True
        yield _make("momentum", "watch", "bull" if up else "bear",
                    f"Headline tone turned {'up' if up else 'down'} in the last 48h",
                    f"Last 48h: {signed(shift.recent)} across {count(r.n, 'headline')} vs {signed(shift.older)} "
                    f"across {o.n} in the prior days.", abs(shift.effective) * 3)
    # The momentum component moving the verdict clearly is noteworthy even below those bars.
    part = f.composite.parts["momentum"]
    points = f.composite.contributions.get("momentum", 0.0)
    if not fired and part.reason and abs(part.x) >= MOMENTUM_CLEAR and abs(points) >= 1.0:
        lead, _, evidence = part.reason.partition(": ")
        yield _make("momentum", "watch", "bull" if part.x > 0 else "bear", lead,
                    f"{_cap(evidence)} — the momentum component reads {part.score:.0f}/100 and moves the score "
                    f"{signed(points, '.1f')} points.", abs(part.x) * 2)


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
                bits.append(f"vs {count(rev.cuts_30d, 'cut')} / {count(rev.downgrades_30d, 'downgrade')}")
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
    usd = f.insider_currency
    since = f.now.date() - timedelta(days=CLUSTER_DAYS)
    buys = [t for t in view.transactions if t.kind == "buy" and t.date >= since]
    material = [b for b in buyers(view, f.now.date(), since) if b.material]
    total = sum(b.value or 0.0 for b in material)
    sold = sum(t.value or 0.0 for t in view.transactions if t.kind == "sell" and t.date >= since)
    freely_sold = discretionary_sales(view, since)
    floor = min(CLUSTER_MIN_VALUE, CLUSTER_MIN_BPS * 1e-4 * f.market_cap_usd) if f.market_cap_usd else CLUSTER_MIN_VALUE
    # Several insiders each putting real money in, and not dwarfed by their colleagues' selling.
    if len(material) >= 2 and total >= floor and token_factor(total, freely_sold) >= 1:
        top = material[0]
        lead = f"; largest: {top.name}" + (f" ({top.position})" if top.position else "") + (
            f" {money(top.value, currency=usd)}" if top.value else "")
        smaller = len({t.insider for t in buys}) - len(material)
        token = f" (+{count(smaller, 'smaller buyer')} under {money(MATERIAL_BUY, currency=usd)})" if smaller else ""
        against = f"; {money(sold, currency=usd)} sold over the same period" if sold else "; no sales over the same period"
        yield _make("smart_money", "alert" if len(material) >= 3 else "watch", "bull", "Insider cluster buying",
                    f"{count(len(material), 'insider')} bought {money(total, currency=usd)} on the open market in the last "
                    f"{CLUSTER_DAYS} days{token}{lead}{against}.", 2 + len(material))
    elif buys and not material[1:]:
        top = max(buys, key=lambda t: t.value or 0)
        role = (top.position or "").lower()
        if (top.value or 0) >= 500_000 and any(k in role for k in _OFFICER) and token_factor(top.value or 0, freely_sold) >= 1:
            yield _make("smart_money", "watch", "bull", f"{top.position} bought {money(top.value or 0, currency=usd)}",
                        f"{top.insider} bought {money(top.value or 0, currency=usd)} of stock on the open market on "
                        f"{short_date(top.date)} — officers rarely buy without conviction.", 2)
    bps = f.composite.parts["insiders"].facts.get("sell_bps")
    if bps is not None and bps >= 50 and not material:
        yield _make("smart_money", "watch", "bear", "Heavy insider selling",
                    f"Insiders sold {money(view.sell_value, currency=usd)} in {view.window_days} days — {bps / 100:.2f}% of "
                    f"market cap, well above routine levels.", bps / 50)


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
            bits.append(f"consensus EPS {money(e.eps_estimate, price=True, currency=f.reporting_currency)}")
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
# Pending acquisition
# --------------------------------------------------------------------------- #
def _deals(f: Facts) -> Iterator[_Cand]:
    d = f.deal
    if d is not None:
        items = f" (item {', '.join(d.items)})" if d.items else ""
        said = d.excerpt.rstrip(".") + ("" if d.excerpt.endswith("…") else ".")
        yield _make("deal", "alert", "neutral", f"Pending acquisition: merger agreement ({d.when})",
                    f"Form {d.form}{items}: {said} {f.name} is the company being acquired{d.by}, so its share price "
                    f"now tracks the deal terms and the odds of closing; analyst targets and the price trend are "
                    f"discounted in the score.", 20)
        return
    play = f.deal_in_play
    if play is None:
        return
    size = play.size
    title = "Deal in play" + (f": {size}" if size else "")
    stakes = (" — a deal this size would transform the company" if play.ratio is not None and play.ratio >= 1
              else " — material for the company" if play.material else "")
    value = (f" Quoted deal value {size}{stakes}." if size
             else " No deal value is quoted yet.")
    when = f", latest {short_date(play.latest)}" if play.latest else ""
    yield _make("deal", "alert" if play.material else "watch", "neutral", title,
                f"{count(play.articles, 'article')} from {count(play.outlets, 'outlet')} on M&A involving {f.name}"
                f"{when}: {quote(play.lead.title, 100)}.{value}",
                10 + min(play.ratio or 0.0, 10.0) if play.material else 4)


# --------------------------------------------------------------------------- #
# Risks
# --------------------------------------------------------------------------- #
_COMPLIANCE_RE = re.compile(r"\bregained compliance\b|\bback in compliance\b|\bcompliance (?:has been|was) regained\b",
                            re.IGNORECASE)
_LISTING_RE = re.compile(r"\bdelist|\blisting[- ]rule|\bminimum bid price\b|\bcontinued listing\b", re.IGNORECASE)


def _listing_cleared(f: Facts) -> date | None:
    """Date of the latest filing reporting regained listing compliance (None when there is none)."""
    dates = [x.date for x in f.inputs.filings if _COMPLIANCE_RE.search(x.title)]
    return max(dates) if dates else None


def _stale_listing_flag(f: Facts, when: date, cleared: date | None) -> bool:
    """A listing-rule problem dated `when` that a later compliance notice or a pending deal supersedes."""
    return f.deal is not None or (cleared is not None and cleared >= when)


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

    cleared = _listing_cleared(f)
    for key, title in RED_FLAG_EVENTS.items():
        hits = [it for it in news if key in it.event_keys and it.relevance >= 0.6 and not it.press_release]
        if key == "delisting" and hits:
            latest = max((it.timestamp.date() for it in hits if it.timestamp), default=None)
            if latest is not None and _stale_listing_flag(f, latest, cleared):
                continue
        if hits and _corroborated(hits):
            top = max(hits, key=lambda it: it.weight)
            yield _make("risk", "alert", "bear", title,
                        f"{count(sum(it.coverage for it in hits), 'article')} — top: {quote(top.title, 90)}"
                        + (f" ({top.publisher})" if top.publisher else "") + ".", 5 + len(hits))
    play = f.deal_in_play
    funding = [m for m in play.dilution if not m.press_release] if play is not None else []
    if funding:  # new shares to pay for the deal ("Shareholders Back Bigger Share Count To Support … Acquisition")
        top = max(funding, key=lambda it: it.weight)
        source = ", ".join(x for x in (top.publisher, short_date(top.timestamp) if top.timestamp else None) if x)
        yield _make("risk", "watch", "bear", "Dilution risk: new shares to fund the deal",
                    f"{quote(top.title, 110)}" + (f" ({source})" if source else "")
                    + " — reported inside the deal coverage: paying with new stock would dilute holders.",
                    len(funding) + 1)
    used = {it.id for it in funding}
    offering = [it for it in news if "offering" in it.event_keys and it.id not in used]
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
        label, desc = filing_parts(filing.title)
        said = trim(desc or label, 160).rstrip(".")
        said += "" if said.endswith("…") else "."
        if _COMPLIANCE_RE.search(filing.title):
            continue  # "regained compliance" resolves a listing problem; it is not a red flag
        listing = "3.01" in filing.items or bool(_LISTING_RE.search(filing.title))
        if listing and _stale_listing_flag(f, filing.date, cleared):
            continue
        if filing.importance == "high" and filing.polarity == "bear" and 0 <= age <= 120:
            yield _make("risk", "alert", "bear", f"Red-flag filing: {trim(label, 60)}",
                        f"Form {filing.form}{items} filed {short_date(filing.date)}: {said}", 6 - age / 30)
        elif "3.02" in filing.items and 0 <= age <= 60:
            yield _make("risk", "watch", "bear", "Dilution: unregistered equity sale",
                        f"Form {filing.form}{items} filed {short_date(filing.date)}: {said}", 2)


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
    down = f.sources_down
    all_down = bool(down) and not sources
    if not f.prepared.engine_error and n == 0 and not all_down:
        fetched = sum(f.prepared.fetched.values())
        off_topic = f.prepared.dropped.get("irrelevant", 0)
        answered = sum(1 for r in f.inputs.source_runs if r.status in ("ok", "empty"))
        why = (f"{count(fetched, 'item')} fetched, {off_topic} off-topic" if fetched
               else f"{count(answered, 'source')} answered with nothing" if answered else "no text source ran")
        yield _make("quality", "watch", "neutral", "No relevant news or social items",
                    f"Nothing about {f.name} this run ({why}) — the read rests on structured data only.", 8.5)
    elif not f.prepared.engine_error and n < 8 and sources:
        yield _make("quality", "watch" if n < 3 else "info", "neutral", "Thin coverage",
                    f"Only {count(n, 'relevant item')} from {count(sources, 'source')} — text scores are shrunk "
                    f"toward neutral; weigh the structured data more.", 8 - n)
    if all_down:
        yield _make("quality", "alert", "neutral", "No news or social data this run",
                    f"All {count(len(down), 'text source')} failed ({join_and(down)}); retry shortly.", 9)
    elif len(down) >= 3:
        yield _make("quality", "watch", "neutral", f"{len(down)} sources failed",
                    f"{join_and(down)} returned errors; the read uses the remaining {count(sources, 'source')}.",
                    len(down))
    if f.failed:
        yield _make("quality", "watch", "neutral", "Part of the analysis could not be computed",
                    f"{_cap(join_and(f.failed))} failed on this run's data and {'was' if len(f.failed) == 1 else 'were'}"
                    f" left out (logged for repair); the rest of the analysis is unaffected.", 7)
    pending = f.intel_pending()
    if "tone" in pending and f.inputs.tone is None:
        # Said as what this run used: the caller may already have waited for it (the CLI waits up to 30 s).
        momentum = f.composite.parts.get("momentum")
        flow = (f"momentum uses the last 48 h of headlines ({f.news_recent.n}) vs the prior days "
                f"({f.news_older.n}) instead" if momentum is not None and momentum.available
                else "the momentum component is n/a")
        yield _make("quality", "info", "neutral", "Global news tone not loaded this run",
                    f"GDELT history did not arrive in time for this read; {flow}.", 0.5)
    # Every other feed that timed out (still loading) or failed: a run missing analysts, insiders and
    # price history must say so, not read as complete (live NVDA with a 0.2 s intel budget).
    slow = [k for k in pending if k != "tone"]  # slow GDELT has its own note above
    for keys, title, how in ((slow, "Market data not loaded in time", "did not arrive in time for this read"),
                             (list(f.intel_failed()), "Some market data unavailable", "could not be loaded this run")):
        if not keys:
            continue
        shown = [FEED_NAMES.get(k, k) for k in keys]
        gaps = [LABELS[p.key] for p in f.unloaded_parts() if COMPONENT_FEEDS[p.key] in keys]
        rest = [LABELS[p.key].lower() for p in f.composite.available()]
        n_a = (f"; the {join_and(gaps)} component{'s are' if len(gaps) > 1 else ' is'} n/a"
               + (f", so the score rests on {join_and(rest)} alone" if rest else "") if gaps else "")
        yield _make("quality", "watch" if gaps else "info", "neutral", title,
                    f"{_cap(join_and(shown))} {how}{n_a}.", (6.5 + len(gaps)) if gaps else len(shown) / 3)

