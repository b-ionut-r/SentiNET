"""The analyst brief: a short numbers-backed summary, the bull and bear case, and what to watch.

Deterministic templates fed only by evidence already computed for the
verdict, narratives, insights and catalysts — nothing generic is ever added.
"""
from __future__ import annotations

from app.analytics.composite import consensus_name
from app.analytics.crowd import reddit_move
from app.analytics.facts import Facts
from app.analytics.util import count, join_and, pct, polarity_of, quote, short_date, signed
from app.analytics.verdict import story_points
from app.schemas import Brief, Insight, Verdict

MAX_POINTS = 5
MAX_STORIES_PER_SIDE = 2


def build_brief(f: Facts, verdict: Verdict, insights: list[Insight]) -> Brief:
    bull, bear = _cases(f, insights)
    return Brief(summary=_summary(f, verdict), bull_points=bull, bear_points=bear, watch=_watch(f, insights))


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
    c = f.crowd
    if c is not None:
        tagged = (c.stocktwits_bullish or 0) + (c.stocktwits_bearish or 0)
        if c.stocktwits_bull_ratio is not None and tagged >= 5:
            crowd.append(f"StockTwits is {c.stocktwits_bull_ratio:.0%} bullish ({tagged} tagged)")
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


def _cases(f: Facts, insights: list[Insight]) -> tuple[list[str], list[str]]:
    scored: dict[str, list[tuple[float, str]]] = {"bull": [], "bear": []}
    for key, points in f.composite.contributions.items():
        part = f.composite.parts[key]
        side = polarity_of(part.score)
        if part.reason and abs(points) >= 0.5 and side != "neutral":
            scored[side].append((abs(points), part.reason))
    stories = {"bull": 0, "bear": 0}
    for story in f.stories:
        n = story.narrative
        points = story_points(f, story)
        side = "bull" if n.score > 0 else "bear"
        if points <= 0 or stories[side] >= MAX_STORIES_PER_SIDE:
            continue
        stories[side] += 1
        outlets = f" from {count(len(n.publishers), 'outlet')}" if n.publishers else ""
        scored[side].append((points, f"Story: {quote(n.headline)} — {count(n.count, 'article')}{outlets}, "
                                     f"tone {signed(n.score)}"))
    momentum_reason = bool(f.composite.parts["momentum"].reason) and \
        abs(f.composite.contributions.get("momentum", 0.0)) >= 0.5
    for ins in insights:
        if ins.kind not in CASE_KINDS or ins.polarity == "neutral":
            continue
        if ins.kind == "momentum" and momentum_reason and "48h" in ins.title:
            continue  # already stated by the momentum component's evidence line
        weight = {"alert": 6.0, "watch": 4.0, "info": 2.0}[ins.severity]
        scored[ins.polarity].append((weight, f"{ins.title}: {ins.detail}"))

    def top(side: str) -> list[str]:
        out: list[str] = []
        for _, text in sorted(scored[side], key=lambda x: -x[0]):
            if text not in out:
                out.append(text)
        return out[:MAX_POINTS]

    return top("bull"), top("bear")


def _watch(f: Facts, insights: list[Insight]) -> list[str]:
    """Upcoming catalysts first, then the forward-looking flags (divergences, crowding, attention, data gaps)."""
    out: list[str] = []
    for c in f.catalysts:
        if c.upcoming:
            out.append(f"{c.title} ({short_date(c.date)})" + (f": {c.detail}" if c.detail else ""))
    for ins in insights:
        if ins.kind in WATCH_KINDS and ins.kind != "catalyst" and (ins.severity != "info" or ins.kind == "attention"):
            out.append(f"{ins.title}: {ins.detail}")
    return list(dict.fromkeys(out))[:MAX_POINTS]
