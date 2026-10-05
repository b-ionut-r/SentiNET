"""A dated catalyst list: what is coming (earnings, ex-dividend) and what just happened
(rating changes, material 8-Ks, insider trades, high-impact news events).

Upcoming events come first (soonest first), then recent ones (newest first).
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, UTC

from app.analytics import textkit
from app.analytics.facts import Facts
from app.analytics.util import count, money, pct, signed, tone_polarity
from app.schemas import AnalystAction, Catalyst, EarningsView, Polarity

ANALYST_WINDOW = timedelta(days=30)
INSIDER_WINDOW = timedelta(days=90)
FILING_WINDOW_HIGH = timedelta(days=120)
FILING_WINDOW_MEDIUM = timedelta(days=30)
LARGE_SELL = 1_000_000.0
MAX_ANALYST = 8
MAX_RECENT = 16
NEWS_MIN_IMPACT = 0.3

_BULL_GRADE = re.compile(r"\b(?:strong buy|buy|outperform|overweight|accumulate|positive|top pick|add|"
                         r"conviction buy)\b", re.IGNORECASE)
_BEAR_GRADE = re.compile(r"\b(?:sell|underperform|underweight|reduce|negative)\b", re.IGNORECASE)


def noon_utc(d: date) -> datetime:
    """Date-only events as 12:00 UTC (same calendar day across most time zones)."""
    return datetime.combine(d, time(12, 0), tzinfo=UTC)


def grade_polarity(grade: str | None) -> Polarity:
    if not grade:
        return "neutral"
    if _BEAR_GRADE.search(grade):
        return "bear"
    return "bull" if _BULL_GRADE.search(grade) else "neutral"


def build_catalysts(f: Facts) -> list[Catalyst]:
    today = f.now.date()
    upcoming: list[Catalyst] = []
    earnings = _earnings(f.inputs.earnings, today)
    if earnings is not None:
        upcoming.append(earnings)
    upcoming.extend(c for c in f.inputs.calendar_catalysts if c.upcoming and c.date.date() >= today)

    recent: list[Catalyst] = []
    if f.inputs.analysts is not None:
        recent.extend(_analyst_actions(f.inputs.analysts.actions, f.now))
    recent.extend(_filings(f))
    recent.extend(_insiders(f))
    recent.extend(_news(f))

    upcoming.sort(key=lambda c: c.date)
    recent.sort(key=lambda c: c.date, reverse=True)
    return upcoming + recent[:MAX_RECENT]


def earnings_detail(e: EarningsView) -> str | None:
    bits = []
    if e.eps_estimate is not None:
        rng = (f" (range {money(e.eps_low, price=True)}–{money(e.eps_high, price=True)})"
               if e.eps_low is not None and e.eps_high is not None and e.eps_high > e.eps_low else "")
        bits.append(f"EPS est. {money(e.eps_estimate, price=True)}{rng}")
    if e.revenue_estimate:
        bits.append(f"revenue est. {money(e.revenue_estimate)}")
    reported = [h for h in e.history if h.eps_actual is not None and h.eps_estimate is not None]
    if e.beat_rate is not None and reported:
        beats = round(e.beat_rate * len(reported))
        bits.append(f"beat {beats} of last {len(reported)}")
    last = next((h for h in e.history if h.surprise_pct is not None), None)
    if last is not None and last.surprise_pct is not None:
        bits.append(f"last surprise {pct(last.surprise_pct)}")
    return " · ".join(bits) or None


def _earnings(e: EarningsView | None, today: date) -> Catalyst | None:
    if e is None or e.next_date is None or e.next_date < today:
        return None
    days = (e.next_date - today).days
    when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
    return Catalyst(date=noon_utc(e.next_date), kind="earnings", title=f"Earnings {when}",
                    detail=earnings_detail(e), upcoming=True)


def describe_action(a: AnalystAction) -> tuple[str, str | None, Polarity]:
    """(title, detail, polarity) for one rating/target action."""
    grade = a.to_grade or ""
    pol: Polarity
    pt_change = None
    if a.price_target and a.prior_target and a.prior_target > 0 and a.action != "init":
        pt_change = a.price_target / a.prior_target - 1
    if a.action == "up":
        title, pol = f"{a.firm} upgrades to {grade or 'a higher rating'}", "bull"
    elif a.action == "down":
        title, pol = f"{a.firm} downgrades to {grade or 'a lower rating'}", "bear"
    elif a.action == "init":
        title, pol = f"{a.firm} initiates at {grade or 'coverage'}", grade_polarity(grade)
    elif pt_change is not None and abs(pt_change) > 0.001:
        verb = "raises" if pt_change > 0 else "cuts"
        title = f"{a.firm} {verb} target to {money(a.price_target or 0, price=True)}" + (f" · {grade}" if grade else "")
        pol = "bull" if pt_change > 0 else "bear"
    else:
        title, pol = f"{a.firm} reiterates {grade or 'rating'}", "neutral"
    detail = None
    if a.price_target:
        detail = f"PT {money(a.price_target, price=True)}"
        if pt_change is not None and abs(pt_change) > 0.001 and a.prior_target:
            detail = f"PT {money(a.prior_target, price=True)} → {money(a.price_target, price=True)} ({pct(pt_change * 100)})"
    if a.action in ("up", "down") and a.from_grade:
        detail = f"from {a.from_grade}" + (f" · {detail}" if detail else "")
    return title, detail, pol


def _analyst_actions(actions: list[AnalystAction], now: datetime) -> list[Catalyst]:
    out: list[Catalyst] = []
    for a in actions:
        if now - a.date > ANALYST_WINDOW or a.date > now + timedelta(days=1):
            continue
        title, detail, pol = describe_action(a)
        if pol == "neutral" and a.action in ("main", "reit", "other"):
            continue  # plain reiterations are noise on a timeline
        out.append(Catalyst(date=a.date, kind="analyst", title=title, detail=detail, polarity=pol))
        if len(out) >= MAX_ANALYST:
            break
    return out


def _filings(f: Facts) -> list[Catalyst]:
    out: list[Catalyst] = []
    for filing in f.inputs.filings:
        if not filing.form.upper().startswith("8-K"):
            continue
        age = f.now.date() - filing.date
        window = FILING_WINDOW_HIGH if filing.importance == "high" else FILING_WINDOW_MEDIUM
        if filing.importance == "low" or age > window or age < timedelta(days=-1):
            continue
        items = f" · items {', '.join(filing.items)}" if filing.items else ""
        out.append(Catalyst(date=noon_utc(filing.date), kind="filing", title=filing.title,
                            detail=f"Form {filing.form}{items}", polarity=filing.polarity, url=filing.url))
    return out


def _insiders(f: Facts) -> list[Catalyst]:
    view = f.inputs.insiders
    if view is None:
        return []
    since = (f.now - INSIDER_WINDOW).date()
    out: list[Catalyst] = []
    sells = []
    for t in view.transactions:
        if t.date < since:
            continue
        who = t.insider + (f" ({t.position})" if t.position else "")
        if t.kind == "buy":
            value = f" {money(t.value)}" if t.value else ""
            out.append(Catalyst(date=noon_utc(t.date), kind="insider", title=f"{who} bought{value}",
                                detail=_shares(t.shares), polarity="bull"))
        elif t.kind == "sell" and (t.value or 0) >= LARGE_SELL:
            sells.append(t)
    for t in sorted(sells, key=lambda t: -(t.value or 0))[:3]:
        who = t.insider + (f" ({t.position})" if t.position else "")
        out.append(Catalyst(date=noon_utc(t.date), kind="insider", title=f"{who} sold {money(t.value or 0)}",
                            detail=_shares(t.shares), polarity="bear"))
    return out


def _shares(shares: float | None) -> str | None:
    return f"{shares:,.0f} shares, open market" if shares else "open market"


ANALYST_EVENTS = frozenset({"analyst_upgrade", "analyst_downgrade", "analyst_initiate", "analyst_top_pick",
                            "pt_raise", "pt_cut"})


def _news(f: Facts) -> list[Catalyst]:
    """High-impact stories with a material event (analyst-only stories are already listed as actions)."""
    has_actions = f.inputs.analysts is not None and bool(f.inputs.analysts.actions)
    out: list[Catalyst] = []
    for story in f.stories:
        n = story.narrative
        if not story.material_events or n.impact < NEWS_MIN_IMPACT:
            continue
        if has_actions and set(story.material_events) <= ANALYST_EVENTS:
            continue
        when = n.first_seen or n.last_seen
        if when is None:
            continue
        labels = ", ".join(textkit.event_label(k) for k in story.material_events[:2])
        out.append(Catalyst(
            date=when, kind="news", title=n.headline,
            detail=f"{labels} · {count(n.count, 'article')}, tone {signed(n.score)}",
            polarity=tone_polarity(n.score, 0.1), url=n.url,
        ))
    return out
