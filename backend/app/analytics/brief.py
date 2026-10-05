"""The analyst brief: a short numbers-backed summary, the bull and bear case, and what to watch.

Deterministic templates fed only by evidence already computed for the
verdict, narratives, insights and catalysts — nothing generic is ever added.
"""
from __future__ import annotations

from datetime import timedelta

from app.analytics.composite import PLAN_RE, Part, Upside, consensus_name, upside
from app.analytics.crowd import reddit_breakout, reddit_change_pct, reddit_move
from app.analytics.facts import Facts
from app.analytics.util import cap_share, count, join_and, money, pct, polarity_of, quote, short_date, signed
from app.analytics.narratives import Story, featured
from app.analytics.verdict import story_points
from app.schemas import AnalystView, Brief, Catalyst, Insight, Verdict

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
    if f.deal is not None:
        d = f.deal
        sentences.append(f"A pending acquisition dominates: {f.name} agreed to be acquired{d.by} (merger agreement, "
                         f"8-K {d.when}), so its share price tracks the deal terms and the odds of closing.")
    ranked = featured(f.stories)
    if ranked:
        n = ranked[0].narrative
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
        said = _analyst_view(f, a)
        if said:
            smart.append(said)
    ins = f.inputs.insiders
    if ins is not None and f.composite.parts["insiders"].available:
        usd = f.insider_currency
        if ins.buys:
            smart.append(f"insiders bought {ins.buys}× ({money(ins.buy_value, currency=usd)}) vs sold {ins.sells}×"
                         + (f" ({money(ins.sell_value, currency=usd)})" if ins.sells else "") + f" in {ins.window_days}d")
        else:
            smart.append(f"insiders only sold ({ins.sells}×, {money(ins.sell_value, currency=usd)}) in {ins.window_days}d")
    c, tally = f.crowd, f.stocktwits
    if c is not None:
        if tally is not None and tally.ratio is not None and tally.n >= 5:
            crowd.append(f"StockTwits is {tally.ratio:.0%} bullish ({tally.sample})")
        move, change = reddit_move(c), reddit_change_pct(c)
        if reddit_breakout(c):
            crowd.append(f"Reddit rank jumped to #{c.reddit_rank}" + (
                f" from #{c.reddit_rank_prev}" if c.reddit_rank_prev is not None else "")
                + f" ({c.reddit_mentions} mentions)")
        elif move is not None and change is not None and abs(change) >= 50 and (c.reddit_mentions_prev or 0) >= 10:
            crowd.append(f"Reddit mentions {move}")  # smaller moves on a thin base are noise
        if c.wsb_label and c.wsb_sentiment is not None:
            crowd.append(f"WallStreetBets leans {c.wsb_label}")
    if not smart and not crowd:
        return None
    if smart and crowd:
        text = f"{join_and(smart)}, while {join_and(crowd)}"
    else:
        text = join_and(smart or crowd)
    return text[0].upper() + text[1:] + "."


def _analyst_view(f: Facts, a: AnalystView) -> str | None:
    """'analysts rate it Buy with +40% to the mean target' — or, when the price has already run past
    the targets, 'analysts rate it Buy, but the stock already trades above the mean target ($263, −11%)'."""
    name = consensus_name(a)
    up = _upside(f, a)
    which, target = ("median", a.target_median) if up.skewed else ("mean", a.target_mean)
    if up.split:
        tail = f"targets sit at about the price (mean {pct(up.mean or 0.0)}, median {pct(up.median or 0.0)})"
        return f"analysts rate it {name}, but {tail}" if name else f"analyst {tail}"
    value = up.value
    if value is None:
        return f"analysts rate it {name}" if name else None
    if name in ("Buy", "Strong Buy") and value <= 0 and target is not None:
        return (f"analysts rate it {name}, but the stock already trades above the {which} target "
                f"({money(target, price=True, currency=f.currency)}, {pct(value)})")
    if name:
        return f"analysts rate it {name} with {pct(value)} to the {which} target"
    return f"the {which} analyst target implies {pct(value)}"


def _upside(f: Facts, a: AnalystView) -> Upside:
    found = f.composite.parts["analysts"].facts.get("upside")
    return found if isinstance(found, Upside) else upside(a)


EARNINGS_HORIZON = 60  # days within which upcoming earnings are "the" next catalyst


def _payment_date(c: Catalyst) -> bool:
    """A dividend payment date: administrative (the ex-date is what matters to holders)."""
    return c.kind == "dividend" and c.title.lower().startswith("dividend payment")


def _next_catalyst(f: Facts) -> str | None:
    """The most material upcoming event, not merely the soonest: earnings within 60 days, else the
    first deal/index/other event, else the soonest ex-dividend date or later earnings (TGT: earnings
    Nov 18, not the Nov 10 ex-date)."""
    upcoming = [c for c in f.catalysts if c.upcoming and not _payment_date(c)]
    if not upcoming:
        return None

    def rank(c: Catalyst) -> int:
        if c.kind == "earnings":  # far-off earnings rank with the routine dates (soonest first)
            return 0 if (c.date.date() - f.now.date()).days <= EARNINGS_HORIZON else 2
        return 2 if c.kind == "dividend" else 1

    c = min(upcoming, key=lambda c: (rank(c), c.date))
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
        if part.reason and abs(points) >= 0.5 and side != "neutral" and not _settling(part):
            scored[side].append((abs(points), part.reason, key))
    stories = {"bull": 0, "bear": 0}
    for story in featured(f.stories):
        if story.insider_only:
            continue  # the insiders component states the Form 4 facts; the headlines would repeat them
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
        if ins.kind == "smart_money" and "insider" in ins.title.lower():  # incl. "Heavy insider selling"
            topic = "insiders"
        elif ins.kind == "momentum":  # every momentum insight restates the momentum component's evidence
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
            bps = f.composite.parts["insiders"].facts.get("sell_bps")  # None unless the cap is in USD too
            share = f" ({cap_share(bps)} of market cap)" if bps is not None else ""
            cash = f.insider_currency
            bought = f"bought {money(ins.buy_value, currency=cash)}" if ins.buys else "bought nothing"
            since = f.now.date() - timedelta(days=ins.window_days)
            sells = [t for t in ins.transactions if t.kind == "sell" and t.value and t.date >= since]
            largest = max(sells, key=lambda t: t.value or 0.0, default=None)
            who = (f"; largest: {largest.insider}" + (f" ({largest.position})" if largest.position else "")
                   + f" {money(largest.value or 0.0, currency=cash)}") if largest is not None else ""
            routine = " — routine-sized for the company" if bps is not None and bps < ROUTINE_SELL_BPS else ""
            # Pre-scheduled selling is stated only when the filings say so (Form 4's 10b5-1 box).
            planned = sum(1 for t in sells if t.text and PLAN_RE.search(t.text))
            plans = (f"; {planned} of {count(len(sells), 'sale')} under a pre-arranged 10b5-1 trading plan"
                     if planned else "")
            out.append(("bear", 0.45, f"Insider selling: {count(ins.sells, 'open-market sale')} worth "
                                      f"{money(ins.sell_value, currency=cash)}{share} vs {bought} in "
                                      f"{ins.window_days} days{who}{routine}{plans}", "insiders"))
        elif ins.buys and ins.buy_value >= 100_000 and ins.buy_value >= ins.sell_value:
            cash = f.insider_currency
            out.append(("bull", 0.45, f"Insider buying: {count(ins.buys, 'open-market purchase')} worth "
                                      f"{money(ins.buy_value, currency=cash)} in {ins.window_days} days"
                                      + (f" vs {money(ins.sell_value, currency=cash)} sold" if ins.sells
                                         else ", no sales"), "insiders"))
    a = f.inputs.analysts
    if a is not None and f.composite.parts["analysts"].available:
        out.extend(_analyst_points(f, a))
    t = f.inputs.technicals
    # While a deal is pending the price tracks its terms: stretched/washed-out levels say nothing.
    if t is not None and f.composite.parts["technicals"].available and f.deal is None:
        if t.rsi_14 is not None and t.rsi_14 >= 70:
            run = f" after {pct(t.return_1m)} in a month" if t.return_1m is not None and t.return_1m > 0 else ""
            out.append(("bear", 0.3, f"Overbought: RSI {t.rsi_14:.0f}{run}", "technicals-extreme"))
        elif t.vs_200dma_pct is not None and t.vs_200dma_pct >= STRETCHED_200DMA:
            out.append(("bear", 0.25, f"Extended: price {pct(t.vs_200dma_pct, sign=False)} above its 200-day average",
                        "technicals-extreme"))
        if t.rsi_14 is not None and t.rsi_14 <= 30:
            run = f" after {pct(t.return_1m)} in a month" if t.return_1m is not None and t.return_1m < 0 else ""
            out.append(("bull", 0.3, f"Oversold: RSI {t.rsi_14:.0f}{run} — washed-out levels often see relief",
                        "technicals-extreme"))
    for item in insights:  # crowding extremes argue the contrarian side
        if item.kind == "crowding" and item.polarity in ("bull", "bear"):
            out.append((item.polarity, 0.42, f"{item.title}: {item.detail}", f"insight:{item.title}"))
    for key, part in f.composite.parts.items():  # clear leans that move the score < 0.5 points
        if part.reason and abs(part.x) >= 0.1 and abs(f.composite.contributions.get(key, 0.0)) < 0.5 \
                and not _settling(part):
            out.append(("bull" if part.x > 0 else "bear", 0.2, part.reason, key))
    return out


def _settling(part: Part) -> bool:
    """A mild headline-tone drift back toward normal (no GDELT): context, not a bull or bear argument."""
    return bool(part.facts.get("normalizing")) and not part.strong


def _analyst_points(f: Facts, a: AnalystView) -> list[tuple[str, float, str, str]]:
    """Analyst facts for the cases. 'Turning cautious' needs actual downgrades or target cuts; a price
    that has merely run past the targets is said as such."""
    out: list[tuple[str, float, str, str]] = []
    name = consensus_name(a)
    rated = f", despite a {name} consensus" if name in ("Buy", "Strong Buy") else ""
    rev = f.composite.parts["analysts"].facts.get("revisions")
    cuts = int(getattr(rev, "cuts_30d", 0) or 0)
    up = _upside(f, a)
    which, target = ("median", a.target_median) if up.skewed else ("mean", a.target_mean)
    below = (f"{which} target {money(target, price=True, currency=f.currency)} is "
             f"{pct(abs(up.value), sign=False)} below the price") if (
        target is not None and up.value is not None and up.value < 0) else None
    revisions: list[str] = []
    if a.downgrades_90d >= DOWNGRADES_MIN:
        revisions.append(f"{count(a.downgrades_90d, 'downgrade')} in 90 days")
    if cuts >= 3:
        revisions.append(f"{count(cuts, 'price-target cut')} in 30 days")
    if revisions:
        out.append(("bear", 0.4, f"Analysts turning cautious: {join_and(revisions + ([below] if below else []))}"
                                 f"{rated}", "analysts"))
    elif below and target is not None and up.value is not None:
        t = f.inputs.technicals
        run = (f" after {pct(t.return_1m)} in 1M" if t is not None and t.return_1m is not None and t.return_1m > 0
               else "")
        out.append(("bear", 0.4, f"Price has run past the {which} analyst target "
                                 f"({money(target, price=True, currency=f.currency)}, {pct(up.value)}){run}{rated}",
                    "analysts"))
    if up.value is not None and up.value >= 15 and name in ("Buy", "Strong Buy"):
        out.append(("bull", 0.4, f"Analysts see upside: {name} consensus" + (f" from {count(a.total, 'analyst')}"
                                 if a.total else "") + f", {which} target {pct(up.value)} above the price",
                    "analysts"))
    return out


def _watch(f: Facts, insights: list[Insight], said: set[str] | None = None) -> list[str]:
    """Upcoming catalysts first, then the forward-looking flags (divergences, crowding, attention, data gaps).

    Lines already in the bull or bear case (`said`) are not repeated."""
    out: list[str] = []
    if f.deal is not None:
        d = f.deal
        out.append(f"Deal outcome: pending acquisition{d.by} (merger agreement, 8-K {d.when}) — the price tracks "
                   f"the deal terms until it closes or breaks")
    for c in f.catalysts:
        if c.upcoming and not _payment_date(c):
            out.append(f"{c.title} ({short_date(c.date)})" + (f": {c.detail}" if c.detail else ""))
    for ins in insights:
        if ins.kind in WATCH_KINDS and ins.kind != "catalyst" and (ins.severity != "info" or ins.kind == "attention"):
            out.append(f"{ins.title}: {ins.detail}")
    return [w for w in dict.fromkeys(out) if w not in (said or set())][:MAX_POINTS]
