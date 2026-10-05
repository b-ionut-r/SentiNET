"""The analyst brief: a short numbers-backed summary, the bull and bear case, and what to watch.

Deterministic templates fed only by evidence already computed for the
verdict, narratives, insights and catalysts — nothing generic is ever added.
"""
from __future__ import annotations

from datetime import timedelta

from app.analytics.composite import consensus_name
from app.analytics.crowd import reddit_move
from app.analytics.facts import Facts
from app.analytics.util import cap_share, count, join_and, money, pct, polarity_of, quote, short_date, signed
from app.analytics.narratives import Story
from app.analytics.verdict import story_points
from app.schemas import Brief, Insight, Verdict

MAX_POINTS = 5
MAX_STORIES_PER_SIDE = 2


def build_brief(f: Facts, verdict: Verdict, insights: list[Insight]) -> Brief:
    bull, bear = _cases(f, insights)
    return Brief(summary=_summary(f, verdict), bull_points=bull, bear_points=bear,
                 watch=_watch(f, insights, said=set(bull) | set(bear)))


# --------------------------------------------------------------------------- #
# Summary paragraph
# --------------------------------------------------------------------------- #
def _evidence_base(f: Facts) -> str:
    n = f.overall.n
    sources = len({it.source for it in f.prepared.items})
    extra = []
    a = f.inputs.analysts
    if f.composite.parts["analysts"].available and a is not None:
        extra.append(f"{count(a.total, 'analyst')}" if a.total else "analyst targets")
    if f.composite.parts["insiders"].available:
        extra.append("insider trades")
    if f.inputs.tone is not None and f.inputs.tone.series:
        extra.append("GDELT global tone")
    if f.composite.parts["technicals"].available:
        extra.append("price action")
    if n:
        texts = f"{count(n, 'relevant news and social item')} from {count(sources, 'source')}"
        return texts + (f", plus {join_and(extra)}" if extra else "")
    return f"{join_and(extra)} only (no relevant news or social items)" if extra else "no data"


def _summary(f: Facts, verdict: Verdict) -> str:
    if not f.composite.available():
        return (f"No read on {f.name}: every news, social and market-data feed failed or came back empty this run "
                f"— retry shortly.")
    sentences = [f"{f.name} reads {verdict.label} at {verdict.score}/100 with {verdict.confidence} confidence, "
                 f"based on {_evidence_base(f)}."]
    if f.stories:
        n = f.stories[0].narrative
        new = " (new since the last look)" if n.is_new else ""
        sentences.append(f"The dominant story is {quote(n.headline)}{new}: {count(n.count, 'article')} from "
                         f"{count(len(n.publishers), 'outlet')}, tone {signed(n.score)}.")
    smart_crowd = _smart_vs_crowd(f)
    if smart_crowd:
        sentences.append(smart_crowd)
    nxt = _next_catalyst(f)
    if nxt and len(sentences) < 4:
        sentences.append(nxt)
    return " ".join(sentences[:4])


def _smart_vs_crowd(f: Facts) -> str | None:
    smart, crowd = [], []
    a = f.inputs.analysts
    if a is not None and f.composite.parts["analysts"].available:
        name = consensus_name(a)
        if name:
            smart.append(f"analysts rate it {name}" + (
                f" with {pct(a.upside_pct)} to the mean target" if a.upside_pct is not None else ""))
        elif a.upside_pct is not None:
            smart.append(f"the mean analyst target implies {pct(a.upside_pct)}")
    ins = f.inputs.insiders
    if ins is not None and f.composite.parts["insiders"].available:
        if ins.buys:
            smart.append(f"insiders bought {ins.buys}× vs sold {ins.sells}× in {ins.window_days}d")
        else:
            smart.append(f"insiders only sold ({ins.sells}×) in {ins.window_days}d")
    c, tally = f.crowd, f.stocktwits
    if c is not None:
        if tally is not None and tally.ratio is not None and tally.n >= 5:
            crowd.append(f"StockTwits is {tally.ratio:.0%} bullish ({tally.sample})")
        move = reddit_move(c)
        if move is not None:
            crowd.append(f"Reddit mentions {move}")
        if c.wsb_label and c.wsb_sentiment is not None:
            crowd.append(f"WallStreetBets leans {c.wsb_label}")
    if not smart and not crowd:
        return None
    if smart and crowd:
        text = f"{join_and(smart)}, while {join_and(crowd)}"
    else:
        text = join_and(smart or crowd)
    return text[0].upper() + text[1:] + "."


def _next_catalyst(f: Facts) -> str | None:
    upcoming = [c for c in f.catalysts if c.upcoming]
    if not upcoming:
        return None
    c = upcoming[0]
    detail = f" — {c.detail}" if c.detail else ""
    return f"Next catalyst: {c.title[0].lower() + c.title[1:]} ({short_date(c.date)}){detail}."


# --------------------------------------------------------------------------- #
# Bull / bear case, watch list
# --------------------------------------------------------------------------- #
# Insight kinds that argue a side (bull/bear case) vs. ones that say "keep an eye on it".
CASE_KINDS = frozenset({"smart_money", "risk", "momentum", "reversal"})
WATCH_KINDS = frozenset({"divergence", "crowding", "attention", "quality", "catalyst"})

# Secondary evidence: facts that argue a side even when they barely move the score.
# They rank below every point that does move it (weights < 0.5).
INSIDER_SELL_MIN = 5_000_000.0  # $ sold (with no meaningful buying) worth stating in the bear case
ROUTINE_SELL_BPS = 5.0  # below this share of market cap, large-cap selling is usually scheduled
STRETCHED_200DMA = 25.0  # % above the 200-day average that reads as extended
DOWNGRADES_MIN = 2  # downgrades in 90 days (or 3 PT cuts in 30 days) worth stating in the bear case
NOTABLE_STORY_TONE = 0.25  # a story the news average washes out still argues its side above these bars
NOTABLE_STORY_IMPACT = 0.35


def _cases(f: Facts, insights: list[Insight]) -> tuple[list[str], list[str]]:
    """Bull and bear case: score-moving evidence first, then threshold-clearing counterpoints.

    Each side lists the components, stories and insights that move the score
    its way (ranked by points), then — so a side is never empty while material
    facts argue it — insider flow, analyst targets/revisions, stretched or
    washed-out technicals and crowding extremes that clear their thresholds."""
    scored: dict[str, list[tuple[float, str, str]]] = {"bull": [], "bear": []}  # (weight, text, topic)
    for key, points in f.composite.contributions.items():
        part = f.composite.parts[key]
        side = polarity_of(part.score)
        if part.reason and abs(points) >= 0.5 and side != "neutral":
            scored[side].append((abs(points), part.reason, key))
    stories = {"bull": 0, "bear": 0}
    for story in f.stories:
        n = story.narrative
        points = story_points(f, story)
        if points <= 0 and _notable(story):
            points = 0.3  # a clear, corroborated story that the news average happens to wash out
        side = "bull" if n.score > 0 else "bear"
        if points <= 0 or stories[side] >= MAX_STORIES_PER_SIDE:
            continue
        stories[side] += 1
        outlets = f" from {count(len(n.publishers), 'outlet')}" if n.publishers else ""
        scored[side].append((points, f"Story: {quote(n.headline)} — {count(n.count, 'article')}{outlets}, "
                                     f"tone {signed(n.score)}", f"story:{n.id}"))
    momentum_reason = bool(f.composite.parts["momentum"].reason) and \
        abs(f.composite.contributions.get("momentum", 0.0)) >= 0.5
    for ins in insights:
        if ins.kind not in CASE_KINDS or ins.polarity == "neutral":
            continue
        if ins.kind == "momentum" and momentum_reason and "48h" in ins.title:
            continue  # already stated by the momentum component's evidence line
        weight = {"alert": 6.0, "watch": 4.0, "info": 2.0}[ins.severity]
        # Insights restating a component's evidence share its topic, so only the stronger line is kept.
        if ins.kind == "smart_money" and ins.title.lower().startswith("insider"):
            topic = "insiders"
        elif ins.kind == "momentum" and "GDELT" in ins.detail:
            topic = "momentum"
        else:
            topic = f"insight:{ins.title}"
        scored[ins.polarity].append((weight, f"{ins.title}: {ins.detail}", topic))
    for lean, weight, text, topic in _counterpoints(f, insights):
        if not any(t == topic for _, _, t in scored[lean]):
            scored[lean].append((weight, text, topic))

    def top(side: str) -> list[str]:
        """Heaviest first, one point per topic (an insider insight and the insider component say one thing)."""
        out: list[str] = []
        topics: set[str] = set()
        for _, text, topic in sorted(scored[side], key=lambda x: -x[0]):
            if text not in out and topic not in topics:
                out.append(text)
                topics.add(topic)
        return out[:MAX_POINTS]

    return top("bull"), top("bear")


def _notable(story: Story) -> bool:
    """A story worth a case point on its own: clear tone, corroborated, reasonably prominent."""
    n = story.narrative
    return (story.directional and abs(n.score) >= NOTABLE_STORY_TONE and n.impact >= NOTABLE_STORY_IMPACT
            and len(n.publishers) >= 2)


def _counterpoints(f: Facts, insights: list[Insight]) -> list[tuple[str, float, str, str]]:
    """(side, weight, text, topic) facts that argue a side whatever their effect on the score."""
    out: list[tuple[str, float, str, str]] = []
    ins = f.inputs.insiders
    if ins is not None and f.composite.parts["insiders"].available:
        if ins.sells and ins.sell_value >= INSIDER_SELL_MIN and ins.sell_value >= 5 * ins.buy_value:
            cap = f.market_cap
            bps = ins.sell_value / cap * 1e4 if cap else None
            share = f" ({cap_share(bps)} of market cap)" if bps is not None else ""
            bought = f"bought {money(ins.buy_value)}" if ins.buys else "bought nothing"
            since = f.now.date() - timedelta(days=ins.window_days)
            sells = [t for t in ins.transactions if t.kind == "sell" and t.value and t.date >= since]
            largest = max(sells, key=lambda t: t.value or 0.0, default=None)
            who = (f"; largest: {largest.insider}" + (f" ({largest.position})" if largest.position else "")
                   + f" {money(largest.value or 0.0)}") if largest is not None else ""
            routine = (" — routine-sized for the company; large holders' sales are often pre-scheduled (10b5-1)"
                       if bps is not None and bps < ROUTINE_SELL_BPS else "")
            out.append(("bear", 0.45, f"Insider selling: {count(ins.sells, 'open-market sale')} worth "
                                      f"{money(ins.sell_value)}{share} vs {bought} in {ins.window_days} days"
                                      f"{who}{routine}", "insiders"))
        elif ins.buys and ins.buy_value >= 100_000 and ins.buy_value >= ins.sell_value:
            out.append(("bull", 0.45, f"Insider buying: {count(ins.buys, 'open-market purchase')} worth "
                                      f"{money(ins.buy_value)} in {ins.window_days} days"
                                      + (f" vs {money(ins.sell_value)} sold" if ins.sells else ", no sales"),
                        "insiders"))
    a = f.inputs.analysts
    if a is not None and f.composite.parts["analysts"].available:
        name = consensus_name(a)
        rated = f" (despite a {name} consensus)" if name in ("Buy", "Strong Buy") else ""
        rev = f.composite.parts["analysts"].facts.get("revisions")
        cuts = int(getattr(rev, "cuts_30d", 0) or 0)
        against: list[str] = []
        if a.upside_pct is not None and a.target_mean is not None and a.upside_pct < 0:
            against.append(f"mean target {money(a.target_mean, price=True)} is "
                           f"{pct(abs(a.upside_pct), sign=False)} below the price")
        if a.downgrades_90d >= DOWNGRADES_MIN:
            against.append(f"{count(a.downgrades_90d, 'downgrade')} in 90 days")
        if cuts >= 3:
            against.append(f"{count(cuts, 'price-target cut')} in 30 days")
        if against:
            out.append(("bear", 0.4, f"Analysts turning cautious: {join_and(against)}{rated}", "analysts"))
        if a.upside_pct is not None and a.upside_pct >= 15 and name in ("Buy", "Strong Buy"):
            out.append(("bull", 0.4, f"Analysts see upside: {name} consensus" + (f" from {count(a.total, 'analyst')}"
                                     if a.total else "") + f", mean target {pct(a.upside_pct)} above the price",
                        "analysts"))
    t = f.inputs.technicals
    if t is not None and f.composite.parts["technicals"].available:
        if t.rsi_14 is not None and t.rsi_14 >= 70:
            run = f" after {pct(t.return_1m)} in a month" if t.return_1m is not None and t.return_1m > 0 else ""
            out.append(("bear", 0.3, f"Overbought: RSI {t.rsi_14:.0f}{run}", "technicals-extreme"))
        elif t.vs_200dma_pct is not None and t.vs_200dma_pct >= STRETCHED_200DMA:
            out.append(("bear", 0.25, f"Extended: price {pct(t.vs_200dma_pct)} above its 200-day average",
                        "technicals-extreme"))
        if t.rsi_14 is not None and t.rsi_14 <= 30:
            run = f" after {pct(t.return_1m)} in a month" if t.return_1m is not None and t.return_1m < 0 else ""
            out.append(("bull", 0.3, f"Oversold: RSI {t.rsi_14:.0f}{run} — washed-out levels often see relief",
                        "technicals-extreme"))
    for item in insights:  # crowding extremes argue the contrarian side
        if item.kind == "crowding" and item.polarity in ("bull", "bear"):
            out.append((item.polarity, 0.42, f"{item.title}: {item.detail}", f"insight:{item.title}"))
    for key, part in f.composite.parts.items():  # clear leans that move the score < 0.5 points
        if part.reason and abs(part.x) >= 0.1 and abs(f.composite.contributions.get(key, 0.0)) < 0.5:
            out.append(("bull" if part.x > 0 else "bear", 0.2, part.reason, key))
    return out


def _watch(f: Facts, insights: list[Insight], said: set[str] | None = None) -> list[str]:
    """Upcoming catalysts first, then the forward-looking flags (divergences, crowding, attention, data gaps).

    Lines already in the bull or bear case (`said`) are not repeated."""
    out: list[str] = []
    for c in f.catalysts:
        if c.upcoming:
            out.append(f"{c.title} ({short_date(c.date)})" + (f": {c.detail}" if c.detail else ""))
    for ins in insights:
        if ins.kind in WATCH_KINDS and ins.kind != "catalyst" and (ins.severity != "info" or ins.kind == "attention"):
            out.append(f"{ins.title}: {ins.detail}")
    return [w for w in dict.fromkeys(out) if w not in (said or set())][:MAX_POINTS]
