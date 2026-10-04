"""Command line: `python -m app analyze NVDA`, `python -m app market`, `python -m app sources`,
`python -m app serve`.

`analyze` prints the intel at a glance (verdict, why, components, insights,
narratives, smart money vs crowd, catalysts, source health) with a live scan
status on stderr; `--json` writes the raw `Analysis` to stdout instead.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.schemas import Analysis, MarketOverview, ProgressEvent

BULL, BEAR, NEUTRAL, MUTED = "#0da293", "#e5533f", "#8a8984", "#85847e"
SEVERITY = {"alert": ("ALERT", "bold #d03b3b"), "watch": ("WATCH", "bold #fab219"), "info": ("INFO", "#3987e5")}
STATUS = {"ok": ("●", BULL), "empty": ("○", MUTED), "error": ("✕", BEAR), "disabled": ("–", MUTED),
          "unconfigured": ("+", "#fab219"), "skipped": ("–", MUTED), "running": ("…", MUTED)}

out = Console()
err = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def tone_color(score: float | None, band: float = 0.05) -> str:
    if score is None:
        return MUTED
    return BULL if score > band else BEAR if score < -band else NEUTRAL


def sentinel_color(score: float | None) -> str:
    if score is None:
        return MUTED
    return BULL if score >= 55 else BEAR if score <= 45 else NEUTRAL


def signed(value: float | None, fmt: str = "+.2f", suffix: str = "") -> Text:
    if value is None:
        return Text("n/a", style=MUTED)
    arrow = "▲" if value > 0 else "▼" if value < 0 else "•"
    return Text(f"{arrow} {value:{fmt}}{suffix}", style=tone_color(value, 0.0))


def money(value: float | None, currency: str | None = None) -> str:
    """Compact amount: $4.43T, -$3.30M (no symbol for non-USD currencies)."""
    if value is None:
        return "n/a"
    sign, mag = ("-" if value < 0 else ""), abs(value)
    sym = "$" if (currency or "USD") == "USD" else ""
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if mag >= div:
            return f"{sign}{sym}{mag / div:,.2f}{unit}"
    return f"{sign}{sym}{mag:,.2f}"


def bar(score: float | None, width: int = 20) -> Text:
    """Diverging bar for a 0-100 score centered at 50 (bear left, bull right)."""
    if score is None:
        return Text("n/a".center(width + 1), style=MUTED)
    half = width // 2
    n = round(abs(score - 50) / 50 * half)
    t = Text()
    t.append(("█" * n if score < 50 else "").rjust(half), style=BEAR)
    t.append("│", style=MUTED)
    t.append(("█" * n if score > 50 else "").ljust(half), style=BULL)
    return t


# --------------------------------------------------------------------------- #
# Analysis rendering
# --------------------------------------------------------------------------- #
def _header(a: Analysis) -> Panel:
    p, q, v = a.profile, a.quote, a.verdict
    title = Text(a.ticker, style="bold")
    if p is not None:
        title.append(f"  {p.name}", style="bold")
        meta = " · ".join(x for x in (p.exchange, p.sector, p.industry) if x)
        if meta:
            title.append(f"   {meta}", style=MUTED)
    lines: list[Text] = []
    if q is not None and q.price is not None:
        line = Text(f"{q.price:,.2f} {q.currency or ''}  ", style="bold")
        line.append_text(signed(q.change_pct, "+.2f", "%"))
        if q.day_low is not None and q.day_high is not None:
            line.append(f"   day {q.day_low:,.2f}–{q.day_high:,.2f}", style=MUTED)
        if q.market_cap:
            line.append(f"   cap {money(q.market_cap, q.currency)}", style=MUTED)
        lines.append(line)
    dial = Text("\n")
    dial.append(f" SentiNET {v.score:>3}", style=f"bold {sentinel_color(v.score)}")
    dial.append(" / 100  ")
    filled = round(v.score / 5)
    dial.append("█" * filled, style=sentinel_color(v.score))
    dial.append("░" * (20 - filled), style=MUTED)
    dial.append(f"  {v.label}", style=f"bold {sentinel_color(v.score)}")
    dial.append(f"   {v.confidence} confidence ({v.confidence_value:.2f})", style=MUTED)
    lines.append(dial)
    d = a.delta
    if d.sentinel_change is not None and d.previous_at is not None:
        delta = Text("  ")
        delta.append_text(signed(float(d.sentinel_change), "+.0f"))
        delta.append(f" since {d.previous_at:%b %d %H:%M} UTC", style=MUTED)
        if d.note:
            delta.append(f" · {d.note}", style=MUTED)
        lines.append(delta)
    lines.append(Text(f"\n {v.headline}", style="bold"))
    return Panel(Group(*lines), title=title, title_align="left", border_style=sentinel_color(v.score),
                 box=box.ROUNDED, padding=(0, 1))


def _reasons(a: Analysis) -> Table | None:
    if not a.verdict.reasons:
        return None
    t = Table(box=None, show_header=False, padding=(0, 1), title="Why", title_justify="left", title_style="bold")
    for r in a.verdict.reasons:
        mark, color = {"bull": ("▲", BULL), "bear": ("▼", BEAR)}.get(r.polarity, ("•", NEUTRAL))
        t.add_row(Text(mark, style=color), r.text)
    return t


def _components(a: Analysis) -> Table:
    t = Table(box=box.SIMPLE_HEAD, title="Components", title_justify="left", title_style="bold", expand=False)
    t.add_column("Component")
    t.add_column("Score", justify="right")
    t.add_column("bear ◂ 50 ▸ bull", justify="center")
    t.add_column("Weight", justify="right", style=MUTED)
    t.add_column("Detail", overflow="fold")
    for c in a.verdict.components:
        score = Text(f"{c.score:.0f}", style=sentinel_color(c.score)) if c.score is not None else Text("n/a", MUTED)
        t.add_row(c.label, score, bar(c.score if c.available else None), f"{c.weight:.0%}", c.detail)
    return t


def _insights(a: Analysis) -> Table | None:
    if not a.insights:
        return None
    t = Table(box=None, show_header=False, padding=(0, 1), title="Insights", title_justify="left", title_style="bold")
    for i in a.insights:
        tag, style = SEVERITY.get(i.severity, ("INFO", MUTED))
        mark = {"bull": ("▲", BULL), "bear": ("▼", BEAR)}.get(i.polarity, ("•", NEUTRAL))
        body = Text(i.title, style="bold")
        body.append(f"  {i.detail}", style=MUTED)
        t.add_row(Text(tag, style=style), Text(mark[0], style=mark[1]), body)
    return t


def _narratives(a: Analysis, limit: int = 8) -> Table | None:
    if not a.narratives:
        return None
    t = Table(box=box.SIMPLE_HEAD, title="What's moving it", title_justify="left", title_style="bold")
    t.add_column("#", justify="right", style=MUTED)
    t.add_column("Tone", justify="right")
    t.add_column("Items", justify="right")
    t.add_column("24h", justify="right")
    t.add_column("Story", overflow="fold")
    t.add_column("Outlets", style=MUTED, overflow="ellipsis", max_width=28)
    for i, n in enumerate(a.narratives[:limit], 1):
        story = Text(n.headline)
        if n.is_new:
            story.append("  NEW", style="bold #3987e5")
        t.add_row(str(i), signed(n.score), str(n.count), str(n.velocity_24h), story, ", ".join(n.publishers[:4]))
    return t


def _smart_vs_crowd(a: Analysis) -> Table | None:
    rows: list[tuple[str, Text]] = []
    an = a.analysts
    if an is not None and an.total:
        t = Text(f"{(an.consensus or 'n/a').replace('_', ' ')} · {an.total} analysts")
        if an.target_mean is not None:
            t.append(f" · mean target {money(an.target_mean)}")
        if an.upside_pct is not None:
            t.append(" (")
            t.append_text(signed(an.upside_pct, "+.0f", "%"))
            t.append(")")
        t.append(f" · 90d ▲{an.upgrades_90d}/▼{an.downgrades_90d} · 30d PT ▲{an.pt_raises_30d}/▼{an.pt_cuts_30d}",
                 style=MUTED)
        rows.append(("Analysts", t))
    ins = a.insiders
    if ins is not None and (ins.buys or ins.sells):
        t = Text(f"{ins.buys} buys {money(ins.buy_value)} · {ins.sells} sells {money(ins.sell_value)} "
                 f"({ins.window_days}d) · net ")
        t.append(money(ins.net_value), style=tone_color(ins.net_value, 0.0))
        rows.append(("Insiders", t))
    cr = a.crowd
    if cr is not None:
        if cr.stocktwits_bull_ratio is not None:
            tagged = (cr.stocktwits_bullish or 0) + (cr.stocktwits_bearish or 0)
            rows.append(("StockTwits", Text(f"{cr.stocktwits_bull_ratio:.0%} bullish of {tagged} tagged"
                                            + (f" · {cr.stocktwits_watchers:,} watchers"
                                               if cr.stocktwits_watchers else ""))))
        if cr.reddit_mentions is not None:
            t = Text(f"#{cr.reddit_rank or '?'} · {cr.reddit_mentions} mentions/24h")
            if cr.reddit_mentions_prev:
                chg = (cr.reddit_mentions - cr.reddit_mentions_prev) / cr.reddit_mentions_prev * 100
                t.append(" (")
                t.append_text(signed(chg, "+.0f", "%"))
                t.append(")")
            rows.append(("Reddit", t))
        if cr.wsb_sentiment is not None:
            rows.append(("WSB", Text(f"{cr.wsb_label or ''} {cr.wsb_sentiment:+.2f} · {cr.wsb_comments or 0} comments")))
    at = a.attention
    if at is not None:
        rows.append(("Attention", Text(f"{at.label} ({at.heat}/100) · {at.signals_24h} items in 24h")))
    if not rows:
        return None
    t = Table(box=None, show_header=False, padding=(0, 1), title="Smart money vs crowd", title_justify="left",
              title_style="bold")
    for k, v in rows:
        t.add_row(Text(k, style=MUTED), v)
    return t


def _catalysts(a: Analysis, limit: int = 8) -> Table | None:
    if not a.catalysts:
        return None
    upcoming = [c for c in a.catalysts if c.upcoming]
    recent = sorted((c for c in a.catalysts if not c.upcoming), key=lambda c: c.date, reverse=True)
    items = sorted(upcoming, key=lambda c: c.date) + recent
    t = Table(box=None, show_header=False, padding=(0, 1), title="Watch next / recent catalysts",
              title_justify="left", title_style="bold")
    for c in items[:limit]:
        mark = {"bull": ("▲", BULL), "bear": ("▼", BEAR)}.get(c.polarity, ("•", NEUTRAL))
        when = Text(f"{c.date:%b %d}", style="bold" if c.upcoming else MUTED)
        body = Text(c.title)
        if c.detail:
            body.append(f"  {c.detail}", style=MUTED)
        t.add_row(when, Text(mark[0], style=mark[1]), Text(c.kind, style=MUTED), body)
    return t


def _sources(a: Analysis) -> Table:
    t = Table(box=box.SIMPLE_HEAD, title="Sources", title_justify="left", title_style="bold")
    t.add_column("Source")
    t.add_column("Status")
    t.add_column("Kept/Fetched", justify="right")
    t.add_column("Tone", justify="right")
    t.add_column("Latency", justify="right", style=MUTED)
    t.add_column("Note", style=MUTED, overflow="fold")
    for s in a.sources:
        mark, color = STATUS.get(s.status, ("?", MUTED))
        note = s.error or ("add a free API key in .env" if s.status == "unconfigured" else "")
        t.add_row(s.label, Text(f"{mark} {s.status}", style=color),
                  f"{s.kept}/{s.fetched}" if s.fetched or s.kept else "—",
                  signed(s.score) if s.score is not None else Text("—", MUTED),
                  f"{s.latency_ms} ms" if s.latency_ms is not None else "—", note)
    return t


def _bull_bear(a: Analysis) -> Group | None:
    b = a.brief
    if not (b.summary or b.bull_points or b.bear_points or b.watch):
        return None
    parts: list[Text | Table] = []
    if b.summary:
        parts.append(Text(b.summary))
    if b.bull_points or b.bear_points:
        t = Table(box=None, show_header=True, padding=(0, 2), expand=True)
        t.add_column(Text("▲ Bull case", style=f"bold {BULL}"), ratio=1, overflow="fold")
        t.add_column(Text("▼ Bear case", style=f"bold {BEAR}"), ratio=1, overflow="fold")
        for i in range(max(len(b.bull_points), len(b.bear_points))):
            t.add_row(b.bull_points[i] if i < len(b.bull_points) else "",
                      b.bear_points[i] if i < len(b.bear_points) else "")
        parts.append(t)
    if b.watch:
        parts.append(Text("Watch: " + " · ".join(b.watch), style=MUTED))
    return Group(Text("Brief", style="bold"), *parts)


def render_analysis(a: Analysis, console: Console | None = None) -> None:
    """Most decisive first: verdict → why → insights → stories → bull/bear → smart money → catalysts → detail."""
    console = console or out
    console.print(_header(a))
    sections = (_reasons(a), _insights(a), _narratives(a), _bull_bear(a), _smart_vs_crowd(a), _catalysts(a),
                _components(a) if a.verdict.components else None, _sources(a) if a.sources else None)
    for part in sections:
        if part is not None:
            console.print(part)
            console.print()
    footer = f"{a.sentiment.n} signals · engine {a.engine} · generated {a.generated_at:%Y-%m-%d %H:%M} UTC · " \
             f"{a.elapsed_ms / 1000:.1f}s" + (" · cached" if a.cached else "")
    console.print(Text(footer, style=MUTED))


# --------------------------------------------------------------------------- #
# Market rendering
# --------------------------------------------------------------------------- #
def render_market(m: MarketOverview, console: Console | None = None) -> None:
    console = console or out
    body = [Text(m.regime, style="bold"), Text(m.regime_detail, style=MUTED)]
    for name, fg in (("CNN Fear & Greed", m.fear_greed), ("Crypto Fear & Greed", m.crypto_fear_greed)):
        if fg is not None:
            line = Text(f"{name}: ")
            line.append(f"{fg.score:.0f} {fg.rating}", style=f"bold {sentinel_color(fg.score)}")
            if fg.week_ago is not None:
                line.append(f"   1w ago {fg.week_ago:.0f}", style=MUTED)
            body.append(line)
    console.print(Panel(Group(*body), title="Market", title_align="left", box=box.ROUNDED))
    if m.indices:
        t = Table(box=box.SIMPLE_HEAD, title="Indices", title_justify="left", title_style="bold")
        t.add_column("Symbol")
        t.add_column("Name", style=MUTED)
        t.add_column("Price", justify="right")
        t.add_column("Change", justify="right")
        for i in m.indices:
            t.add_row(i.symbol, i.name, f"{i.price:,.2f}" if i.price is not None else "n/a",
                      signed(i.change_pct, "+.2f", "%"))
        console.print(t)
    if m.trending:
        t = Table(box=box.SIMPLE_HEAD, title="Trending", title_justify="left", title_style="bold")
        for col in ("#", "Symbol", "Source", "Mentions", "Δ 24h"):
            t.add_column(col, justify="right" if col in ("#", "Mentions", "Δ 24h") else "left")
        for tr in m.trending[:15]:
            t.add_row(str(tr.rank or ""), tr.symbol, tr.source, str(tr.mentions or ""), signed(tr.change_pct, "+.0f", "%"))
        console.print(t)
    if m.narratives:
        t = Table(box=box.SIMPLE_HEAD, title="Market narratives", title_justify="left", title_style="bold")
        t.add_column("Tone", justify="right")
        t.add_column("Items", justify="right")
        t.add_column("Story", overflow="fold")
        for n in m.narratives[:8]:
            t.add_row(signed(n.score), str(n.count), n.headline)
        console.print(t)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
class _Scan:
    """Live one-line scan status on stderr, fed by analyzer progress events."""

    def __init__(self, ticker: str, enabled: bool) -> None:
        self.ticker = ticker
        self.enabled = enabled
        self.total = 0
        self.finished = 0
        self.status = err.status(f"Scanning {ticker}…", spinner="dots") if enabled else None

    async def __call__(self, ev: ProgressEvent) -> None:
        if ev.stage in ("source", "intel"):
            if ev.status == "running":
                self.total += 1
            elif ev.status != "skipped":
                self.finished += 1
        if self.status is None:
            return
        if ev.status == "error" and ev.stage in ("source", "intel"):
            err.print(Text(f"  ✕ {ev.label}: {ev.detail or 'error'}", style=BEAR))
        label = f"{ev.label} {ev.status}" + (f" · {ev.detail}" if ev.detail else "")
        self.status.update(f"Scanning {self.ticker} [{self.finished}/{self.total}] {label}"[:110])


async def _cmd_analyze(ticker: str, as_json: bool, refresh: bool, quiet: bool) -> int:
    from app.core.http import close_client
    from app.services import alerts, analyzer
    from app.services.errors import ServiceError
    from app.storage import db

    scan = _Scan(ticker.upper(), enabled=not quiet)
    try:
        if scan.status is not None:
            scan.status.start()
        try:
            analysis = await analyzer.analyze(ticker, refresh=refresh, progress=scan)
        finally:
            if scan.status is not None:
                scan.status.stop()
    except ServiceError as exc:
        err.print(Text(f"✕ {exc}", style=BEAR))
        return 2 if exc.status_code < 500 else 1
    finally:
        await alerts.drain()  # alert webhooks of this run go out before the HTTP client closes
        await analyzer.shutdown()
        await close_client()
        db.close_db()
    if as_json:
        sys.stdout.write(analysis.model_dump_json(indent=2) + "\n")
    else:
        render_analysis(analysis)
    return 0


async def _cmd_market(as_json: bool) -> int:
    from app.core.http import close_client
    from app.services.errors import ServiceError
    from app.services.market import get_market_overview

    try:
        with err.status("Reading the market…", spinner="dots"):
            overview = await get_market_overview()
    except ServiceError as exc:
        err.print(Text(f"✕ {exc}", style=BEAR))
        return 1
    finally:
        await close_client()
    if as_json:
        sys.stdout.write(overview.model_dump_json(indent=2) + "\n")
    else:
        render_market(overview)
    return 0


def _cmd_sources() -> int:
    from app.api.routes_meta import source_infos

    t = Table(box=box.SIMPLE_HEAD, title="SentiNET sources", title_justify="left", title_style="bold")
    for col in ("Key", "Label", "Kind", "State", "Description"):
        t.add_column(col, overflow="fold")
    for s in source_infos():
        if not s.enabled:
            state = Text("disabled", style=MUTED)
        elif s.requires_key and not s.configured:
            state = Text("needs free key", style="#fab219")
        else:
            state = Text("ready", style=BULL)
        t.add_row(s.key, s.label, s.kind, state, s.description)
    out.print(t)
    return 0


def _cmd_serve(host: str, port: int, reload: bool) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=reload, proxy_headers=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app", description="SentiNET market-intelligence terminal")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="log provider warnings (-v) or everything (-vv) to stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="full intel for one ticker")
    p.add_argument("ticker")
    p.add_argument("--json", action="store_true", help="print the raw Analysis JSON to stdout")
    p.add_argument("--refresh", action="store_true", help="bypass the cache")
    p.add_argument("--quiet", action="store_true", help="no live scan status on stderr")

    p = sub.add_parser("market", help="market regime, fear & greed, indices, trending")
    p.add_argument("--json", action="store_true")

    sub.add_parser("sources", help="list data sources and their configuration state")

    p = sub.add_parser("serve", help="run the API + web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = {0: logging.CRITICAL, 1: logging.WARNING}.get(args.verbose, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)
    if args.command == "analyze":
        return asyncio.run(_cmd_analyze(args.ticker, args.json, args.refresh, args.quiet))
    if args.command == "market":
        return asyncio.run(_cmd_market(args.json))
    if args.command == "sources":
        return _cmd_sources()
    return _cmd_serve(args.host, args.port, args.reload)
