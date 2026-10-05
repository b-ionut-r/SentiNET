"""A dated catalyst list: what is coming (earnings, ex-dividend) and what just happened
(rating changes, material 8-Ks, insider trades, high-impact news events).

Upcoming events come first (soonest first), then recent ones (newest first).
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, UTC

from app.analytics import textkit
from app.analytics.facts import Facts
from app.analytics.narratives import PRICE_EVENTS, same_firm
from app.analytics.util import count, filing_parts, money, pct, quote, signed, tone_polarity, trim
from app.schemas import AnalystAction, Catalyst, EarningsView, Polarity

ANALYST_WINDOW = timedelta(days=30)
INSIDER_WINDOW = timedelta(days=90)
FILING_WINDOW_HIGH = timedelta(days=120)
FILING_WINDOW_MEDIUM = timedelta(days=30)
LARGE_SELL = 1_000_000.0
MAX_ANALYST = 8
MAX_RECENT = 16
FILING_DETAIL = 140  # characters of an 8-K's description shown on the timeline
NEWS_MIN_IMPACT = 0.45
RETROSPECTIVE_DAYS = 7  # results stories first seen this long after the last report are look-backs

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
    earnings = _earnings(f.inputs.earnings, today, f.reporting_currency)
    if earnings is not None:
        upcoming.append(earnings)
    upcoming.extend(c for c in f.inputs.calendar_catalysts if c.upcoming and c.date.date() >= today)
    upcoming.extend(_deal_deadlines(f))

    recent: list[Catalyst] = []
    if f.inputs.analysts is not None:
        recent.extend(_analyst_actions(f.inputs.analysts.actions, f.now, f.action_currency))
    recent.extend(_filings(f))
    recent.extend(_insiders(f))
    listed: set[str] = set()
    recent.extend(_news(f, listed))
    recent.extend(_headline_analyst_actions(f, listed))

    upcoming.sort(key=lambda c: c.date)
    recent.sort(key=lambda c: c.date, reverse=True)
    return upcoming + recent[:MAX_RECENT]


def earnings_detail(e: EarningsView, currency: str | None = "USD") -> str | None:
    """EPS/revenue estimates (in the reporting `currency`; None: unknown, no symbol) and the beat record."""
    bits = []
    if e.eps_estimate is not None:
        rng = (f" (range {money(e.eps_low, price=True, currency=currency)}–"
               f"{money(e.eps_high, price=True, currency=currency)})"
               if e.eps_low is not None and e.eps_high is not None and e.eps_high > e.eps_low else "")
        bits.append(f"EPS est. {money(e.eps_estimate, price=True, currency=currency)}{rng}")
    if e.revenue_estimate:
        bits.append(f"revenue est. {money(e.revenue_estimate, currency=currency)}")
    reported = [h for h in e.history if h.eps_actual is not None and h.eps_estimate is not None]
    if e.beat_rate is not None and reported:
        beats = round(e.beat_rate * len(reported))
        bits.append(f"beat {beats} of last {len(reported)}")
    last = next((h for h in e.history if h.surprise_pct is not None), None)
    if last is not None and last.surprise_pct is not None:
        bits.append(f"last surprise {pct(last.surprise_pct)}")
    return " · ".join(bits) or None


def _earnings(e: EarningsView | None, today: date, currency: str | None = "USD") -> Catalyst | None:
    if e is None or e.next_date is None or e.next_date < today:
        return None
    days = (e.next_date - today).days
    when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
    return Catalyst(date=noon_utc(e.next_date), kind="earnings", title=f"Earnings {when}",
                    detail=earnings_detail(e, currency), upcoming=True)


def describe_action(a: AnalystAction, currency: str | None = "USD") -> tuple[str, str | None, Polarity]:
    """(title, detail, polarity) for one rating/target action.

    `currency` None (a non-USD listing): the broker feed may quote another line's
    targets (SHOP.TO's are Shopify's USD targets), so only the % change is shown."""
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
        to = f" to {money(a.price_target or 0, price=True)}" if currency == "USD" else f" ({pct(pt_change * 100)})"
        title = f"{a.firm} {verb} target{to}" + (f" · {grade}" if grade else "")
        pol = "bull" if pt_change > 0 else "bear"
    else:
        title, pol = f"{a.firm} reiterates {grade or 'rating'}", "neutral"
    detail = None
    if a.price_target and currency == "USD":
        detail = f"PT {money(a.price_target, price=True)}"
        if pt_change is not None and abs(pt_change) > 0.001 and a.prior_target:
            detail = f"PT {money(a.prior_target, price=True)} → {money(a.price_target, price=True)} ({pct(pt_change * 100)})"
    elif pt_change is not None and abs(pt_change) > 0.001:
        detail = f"PT {pct(pt_change * 100)}"
    if a.action in ("up", "down") and a.from_grade:
        detail = f"from {a.from_grade}" + (f" · {detail}" if detail else "")
    return title, detail, pol


def _analyst_actions(actions: list[AnalystAction], now: datetime, currency: str | None = "USD") -> list[Catalyst]:
    out: list[Catalyst] = []
    for a in actions:
        if now - a.date > ANALYST_WINDOW or a.date > now + timedelta(days=1):
            continue
        title, detail, pol = describe_action(a, currency)
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
        label, desc = filing_parts(filing.title)
        detail = f"Form {filing.form}{items}" + (f" · {trim(desc, FILING_DETAIL)}" if desc else "")
        out.append(Catalyst(date=noon_utc(filing.date), kind="filing", title=trim(label, 80),
                            detail=detail, polarity=filing.polarity, url=filing.url))
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
            value = f" {money(t.value, currency=f.insider_currency)}" if t.value else ""
            out.append(Catalyst(date=noon_utc(t.date), kind="insider", title=f"{who} bought{value}",
                                detail=_shares(t.shares), polarity="bull"))
        elif t.kind == "sell" and (t.value or 0) >= LARGE_SELL:
            sells.append(t)
    for t in sorted(sells, key=lambda t: -(t.value or 0))[:3]:
        who = t.insider + (f" ({t.position})" if t.position else "")
        out.append(Catalyst(date=noon_utc(t.date), kind="insider", title=f"{who} sold {money(t.value or 0, currency=f.insider_currency)}",
                            detail=_shares(t.shares), polarity="bear"))
    return out


def _shares(shares: float | None) -> str | None:
    return f"{shares:,.0f} shares, open market" if shares else "open market"


ANALYST_EVENTS = frozenset({"analyst_upgrade", "analyst_downgrade", "analyst_initiate", "analyst_top_pick",
                            "pt_raise", "pt_cut"})
# Listed from authoritative data instead (Form 4 trades) or merely describing the tape.
NOT_NEWS_CATALYSTS = PRICE_EVENTS | {"insider_buy", "insider_sell"}
RESULTS_EVENTS = frozenset({"earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut", "record_results"})


def last_report_date(e: EarningsView | None, today: date) -> date | None:
    """Date of the most recent reported quarter (None when unknown)."""
    if e is None:
        return None
    reported = [h.date for h in e.history if h.eps_actual is not None and h.date <= today]
    return max(reported) if reported else None


def _deal_deadlines(f: Facts) -> list[Catalyst]:
    """Explicit, dated deadlines stated in the deal-in-play coverage (see deals.py)."""
    play = f.deal_in_play
    if play is None:
        return []
    return [Catalyst(date=noon_utc(d.when), kind="news", title=f"Deal deadline: {d.what.lower()}",
                     detail=f"{d.item.publisher or 'Coverage'}: {quote(d.item.title, 110)}", upcoming=True,
                     url=d.item.url) for d in play.deadlines]


def _news(f: Facts, listed: set[str] | None = None) -> list[Catalyst]:
    """High-impact stories carrying a material development, and the deal in play whatever its impact.

    Skipped: analyst stories when the rating actions are already listed,
    insider and price-only events, and results/guidance stories that first
    appeared more than a week after the last report (retrospectives). The ids
    of the listed stories are added to `listed`."""
    has_actions = f.inputs.analysts is not None and bool(f.inputs.analysts.actions)
    last_report = last_report_date(f.inputs.earnings, f.now.date())
    play = f.deal_in_play
    out: list[Catalyst] = []
    for story in f.stories:
        n = story.narrative
        when = n.first_seen or n.last_seen
        events = [k for k in story.material_events if k not in NOT_NEWS_CATALYSTS]
        in_play = play is not None and play.story is story
        if not events or (n.impact < NEWS_MIN_IMPACT and not in_play) or when is None:
            continue
        if has_actions and (ANALYST_EVENTS & set(story.lead.event_keys) or set(events) <= ANALYST_EVENTS):
            continue
        if last_report is not None and (when.date() - last_report).days > RETROSPECTIVE_DAYS:
            events = [k for k in events if k not in RESULTS_EVENTS]
            if not events:
                continue
        labels = ", ".join(textkit.event_label(k) for k in events[:2])
        size = play.size if in_play and play is not None else None
        labels += f" · quoted {size}" if size else ""
        out.append(Catalyst(
            date=when, kind="news", title=n.headline,
            detail=f"{labels} · {count(n.count, 'article')}, tone {signed(n.score)}",
            polarity=tone_polarity(n.score, 0.1) if story.directional else "neutral", url=n.url,
        ))
        if listed is not None:
            listed.add(n.id)
    return out


HEADLINE_ACTION_WINDOW = timedelta(days=10)  # a broker action seen only in headlines, while still fresh
# A ratings-feed action by the same firm from a week before the headline (coverage lags the call: AAPL's
# 'Morgan Stanley trims target' ran 3.4 days after the feed's row) to 2 days after it is the same call.
HEADLINE_ACTION_LAG = timedelta(days=7)
HEADLINE_ACTION_LEAD = timedelta(days=2)
HEADLINE_ACTION_RELEVANCE = 0.8  # the headline is clearly about the company
MAX_HEADLINE_ACTIONS = 3


def _headline_analyst_actions(f: Facts, listed: set[str]) -> list[Catalyst]:
    """Fresh broker calls by a named firm that only the headlines carry.

    Citi's Buy/$365 initiation of AAPL was detected in a Moomoo headline, yet it
    was a single-outlet story below the narrative cut and Yahoo's action list
    lacked it, so it appeared nowhere. Kept: a news headline (not an opinion
    column, press release or post) clearly about the company, <= 10 days old,
    whose upgrade/downgrade/initiation/top-pick/target event names the firm,
    when the ratings feed has no action by that firm from 7 days before to 2
    days after it and no listed story already carries it. One catalyst per firm
    and call, from its heaviest headline."""
    if not f.is_equity:
        return []
    feed = f.inputs.analysts.actions if f.inputs.analysts is not None else []
    out: list[Catalyst] = []
    said: list[tuple[str, str, datetime]] = []
    for it in f.prepared.items:  # heaviest first
        when = it.timestamp
        if (it.kind != "news" or not it.scored or it.press_release or when is None
                or it.relevance < HEADLINE_ACTION_RELEVANCE or it.narrative_id in listed
                or not timedelta(0) <= f.now - when <= HEADLINE_ACTION_WINDOW):
            continue
        event = next((e for e in it.events if e.key in ANALYST_EVENTS and e.firm), None)
        if event is None or event.firm is None:
            continue
        firm = event.firm
        if any(same_firm(firm, a.firm) and -HEADLINE_ACTION_LEAD <= when - a.date <= HEADLINE_ACTION_LAG for a in feed):
            continue  # the ratings feed lists this call: not a second catalyst
        if any(same_firm(firm, s) and k == event.key and abs(t - when) <= HEADLINE_ACTION_LAG for s, k, t in said):
            continue
        said.append((firm, event.key, when))
        target = (f" · PT {money(event.value, price=True)}" if event.value and f.action_currency == "USD" else "")
        source = f" · {it.publisher}" if it.publisher else ""
        polarity: Polarity = event.polarity if event.polarity in ("bull", "bear") else tone_polarity(it.score, 0.1)
        out.append(Catalyst(date=when, kind="analyst", title=trim(it.title, 120),
                            detail=f"{textkit.event_label(event.key)} by {firm}{target}{source} · from the headlines "
                                   f"(not in the ratings feed)", polarity=polarity, url=it.url))
        if len(out) >= MAX_HEADLINE_ACTIONS:
            break
    return out
