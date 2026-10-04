"""Alert rules: evaluate every fresh analysis, record events, push them to a webhook.

Rule semantics (`threshold` meaning per kind; defaults applied on creation):

* `score_above` / `score_below` (SentiNET 0-100; default 70 / 30) — fires when
  the score *crosses* the line: the condition holds now and either the rule has
  never fired or the previous snapshot was on the other side. No re-fire while
  the condition simply persists.
* `score_change` (points; default 10) — the score moved ≥ threshold vs. a
  baseline: the snapshot ~24 h earlier (≤ 7 days old; else the oldest within
  24 h), advanced to the snapshot at the last trigger so a trend fires once per leg.
* `attention_spike` (heat 0-100; default 75) — attention heat crosses the line.
* `new_narrative` (min. items; default 3) — a story with ≥ threshold items that
  was not among the previous snapshot's narratives (headline similarity).
* `analyst_action` (no threshold) — rating/target actions (≤ 10 days old) that
  were not in the previous snapshot.

Score/attention rules also have a 2 h cooldown so a value hovering around the
threshold cannot spam. Evaluation is pure (`evaluate_rule`); persistence and
delivery live in `process_analysis` / `deliver`.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.schemas import AlertEvent, AlertRule, AlertRuleIn, Analysis, AnalystAction
from app.services.tasks import describe_error
from app.storage import db
from app.storage.db import SnapshotRecord, analyst_action_key

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[str, float | None] = {
    "score_above": 70.0,
    "score_below": 30.0,
    "score_change": 10.0,
    "attention_spike": 75.0,
    "new_narrative": 3.0,
    "analyst_action": None,
}
THRESHOLD_RANGES: dict[str, tuple[float, float]] = {
    "score_above": (1, 99),
    "score_below": (1, 99),
    "score_change": (1, 100),
    "attention_spike": (1, 100),
    "new_narrative": (1, 500),
}
COOLDOWN = timedelta(hours=2)
CHANGE_WINDOW = timedelta(hours=24)
CHANGE_MAX_BASELINE_AGE = timedelta(days=7)  # an older baseline says nothing about "now"
ACTION_MAX_AGE = timedelta(days=10)
SIMILAR_HEADLINE = 0.5  # token Jaccard at/above which two headlines are "the same story"
MAX_LISTED = 4
WEBHOOK_UNDELIVERED_WINDOW = timedelta(hours=24)

_ACTION_VERBS = {"up": "upgrade", "down": "downgrade", "init": "initiates", "main": "maintains",
                 "reit": "reiterates", "other": "update"}
_TOKEN = re.compile(r"[a-z0-9$%.]+")
_STOP = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "as", "at", "by", "from",
    "is", "are", "be", "its", "it", "this", "that",
})

_deliveries: set[asyncio.Task[int]] = set()


@dataclass(frozen=True)
class Alert:
    title: str
    detail: str


@dataclass(frozen=True)
class RuleContext:
    """Everything a rule may look at, for one ticker at one point in time."""

    now: datetime
    analysis: Analysis
    current: SnapshotRecord
    previous: SnapshotRecord | None  # immediately preceding snapshot


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def normalize_rule(rule: AlertRuleIn) -> AlertRuleIn:
    """Apply the kind's default threshold and range-check it (`ValueError` if invalid)."""
    threshold = rule.threshold if rule.threshold is not None else DEFAULT_THRESHOLDS[rule.kind]
    if rule.kind == "analyst_action":
        threshold = None
    elif threshold is not None:
        lo, hi = THRESHOLD_RANGES[rule.kind]
        if not lo <= threshold <= hi:
            raise ValueError(f"threshold for {rule.kind} must be between {lo:g} and {hi:g}")
    return rule.model_copy(update={"threshold": threshold})


# --------------------------------------------------------------------------- #
# Pure evaluation
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


def headline_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _cooling(rule: AlertRule, now: datetime) -> bool:
    return rule.last_triggered_at is not None and now - rule.last_triggered_at < COOLDOWN


def _when(dt: datetime) -> str:
    return dt.strftime("%b %d %H:%M UTC")


def _verdict_line(analysis: Analysis) -> str:
    v = analysis.verdict
    return f"{v.label} ({v.confidence} confidence). {v.headline}".strip()


def _crossing(rule: AlertRule, ctx: RuleContext, value: float | None, prev_value: float | None,
              above: bool) -> bool:
    """Edge-triggered threshold test with cooldown."""
    thr = rule.threshold
    if value is None or thr is None or _cooling(rule, ctx.now):
        return False
    holds = value >= thr if above else value <= thr
    if not holds:
        return False
    if rule.last_triggered_at is None or prev_value is None:
        return True
    return not (prev_value >= thr if above else prev_value <= thr)


def _action_text(a: AnalystAction) -> str:
    marker = {"up": "▲", "down": "▼"}.get(a.action, "•")
    text = f"{marker} {a.firm} {_ACTION_VERBS.get(a.action, a.action)}"
    if a.from_grade and a.to_grade and a.from_grade != a.to_grade:
        text += f": {a.from_grade} → {a.to_grade}"
    elif a.to_grade:
        text += f": {a.to_grade}"
    if a.price_target is not None:
        text += f" · PT ${a.price_target:,.0f}"
        if a.prior_target is not None and a.prior_target != a.price_target:
            text += f" (from ${a.prior_target:,.0f})"
    return text


def evaluate_rule(rule: AlertRule, ctx: RuleContext, reference: SnapshotRecord | None = None) -> Alert | None:
    """Return the alert this rule raises for `ctx`, or None. `reference` is the score_change baseline."""
    t = ctx.analysis.ticker
    cur, prev = ctx.current, ctx.previous
    thr = rule.threshold

    if rule.kind in ("score_above", "score_below"):
        above = rule.kind == "score_above"
        if not _crossing(rule, ctx, cur.score, prev.score if prev else None, above):
            return None
        sign = "≥" if above else "≤"
        was = f" Previously {prev.score} ({_when(prev.at)})." if prev else ""
        return Alert(f"{t} SentiNET {cur.score} {sign} {thr:g}", f"{_verdict_line(ctx.analysis)}{was}")

    if rule.kind == "score_change":
        if reference is None or thr is None or _cooling(rule, ctx.now):
            return None
        delta = cur.score - reference.score
        if abs(delta) < thr:
            return None
        hours = max(1, round((cur.at - reference.at).total_seconds() / 3600))
        reasons = ctx.analysis.verdict.reasons
        why = f" Top driver: {reasons[0].text}" if reasons else ""
        return Alert(
            f"{t} SentiNET {delta:+d} in {hours}h ({reference.score} → {cur.score})",
            f"{_verdict_line(ctx.analysis)}{why}",
        )

    if rule.kind == "attention_spike":
        att = ctx.analysis.attention
        if att is None or not _crossing(rule, ctx, cur.attention_heat,
                                        prev.attention_heat if prev else None, above=True):
            return None
        parts = []
        if att.news_volume_z is not None:
            parts.append(f"news volume z {att.news_volume_z:+.1f}")
        if att.reddit_change_pct is not None:
            parts.append(f"Reddit mentions {att.reddit_change_pct:+.0f}%")
        if att.wiki_views_z is not None:
            parts.append(f"Wikipedia views z {att.wiki_views_z:+.1f}")
        parts.append(f"{att.signals_24h} items in 24h")
        return Alert(f"{t} attention {att.label} ({att.heat}/100)", " · ".join(parts))

    if rule.kind == "new_narrative":
        if prev is None:  # first snapshot: everything would look "new"
            return None
        min_items = int(thr or 1)
        fresh = [
            n for n in ctx.analysis.narratives
            if n.count >= min_items
            and all(headline_similarity(n.headline, old) < SIMILAR_HEADLINE for old in prev.snapshot.narratives)
        ]
        if not fresh:
            return None
        lines = [f"• {n.headline} — {n.count} items, tone {n.score:+.2f}" for n in fresh[:MAX_LISTED]]
        noun = "story" if len(fresh) == 1 else "stories"
        return Alert(f"{t}: {len(fresh)} new {noun}", "\n".join(lines))

    if rule.kind == "analyst_action":
        analysts = ctx.analysis.analysts
        if prev is None or analysts is None:
            return None
        seen = set(prev.analyst_keys)
        new = [
            a for a in analysts.actions
            if ctx.now - _aware(a.date) <= ACTION_MAX_AGE
            and analyst_action_key(a.date, a.firm, a.action, a.to_grade, a.price_target) not in seen
        ]
        if not new:
            return None
        noun = "action" if len(new) == 1 else "actions"
        return Alert(f"{t}: {len(new)} new analyst {noun}", "\n".join(_action_text(a) for a in new[:MAX_LISTED]))

    return None


# --------------------------------------------------------------------------- #
# Persistence + delivery
# --------------------------------------------------------------------------- #
async def _change_reference(rule: AlertRule, current: SnapshotRecord) -> SnapshotRecord | None:
    """Baseline for score_change: ~24 h ago (≤ 7 d), advanced to the last trigger's snapshot."""
    t = current.snapshot.ticker
    base = await db.latest_record(t, before=current.at - CHANGE_WINDOW)
    if base is not None and current.at - base.at > CHANGE_MAX_BASELINE_AGE:
        base = None
    if base is None:
        base = await db.oldest_record_since(t, current.at - CHANGE_WINDOW, exclude_id=current.id)
    if rule.last_triggered_at is not None:
        at_trigger = await db.latest_record(t, before=rule.last_triggered_at, before_id=current.id)
        if at_trigger is not None and (base is None or at_trigger.at > base.at):
            base = at_trigger
    return base


async def process_analysis(analysis: Analysis, current: SnapshotRecord) -> list[AlertEvent]:
    """Evaluate the ticker's enabled rules against a freshly stored snapshot."""
    rules = await db.list_rules(ticker=analysis.ticker, enabled_only=True)
    if not rules:
        return []
    previous = await db.latest_record(analysis.ticker, before_id=current.id)
    ctx = RuleContext(now=current.at, analysis=analysis, current=current, previous=previous)
    events: list[AlertEvent] = []
    for rule in rules:
        reference = await _change_reference(rule, current) if rule.kind == "score_change" else None
        alert = evaluate_rule(rule, ctx, reference)
        if alert is None:
            continue
        events.append(await db.add_alert_event(rule.id, analysis.ticker, current.at, alert.title, alert.detail))
        await db.mark_rule_triggered(rule.id, current.at)
    if events and settings.alert_webhook_url:
        task = asyncio.create_task(deliver(events), name="alert-webhook")
        _deliveries.add(task)
        task.add_done_callback(_deliveries.discard)
    return events


def webhook_payload(event: AlertEvent) -> dict[str, str]:
    """Body accepted by both Discord (`content`) and Slack (`text`) incoming webhooks."""
    return {
        "username": "SentiNET",
        "content": f"**{event.title}**\n{event.detail}"[:1990],
        "text": f"*{event.title}*\n{event.detail}"[:3000],
    }


async def deliver(events: list[AlertEvent]) -> int:
    """POST events to the configured webhook in order; returns how many were delivered."""
    from app.core.http import fetch

    url = settings.alert_webhook_url.strip()
    if not url.startswith(("https://", "http://")) or not events:
        return 0
    done: list[int] = []
    for event in events:
        try:
            await fetch(url, method="POST", json=webhook_payload(event), timeout=8.0, retries=1)
        except Exception as exc:  # noqa: BLE001 - keep it queued; the monitor retries later
            logger.warning("alert webhook delivery failed: %s", describe_error(exc))
            break
        done.append(event.id)
    await db.mark_delivered(done)
    return len(done)


async def retry_undelivered(now: datetime) -> int:
    """Re-send events from the last 24 h that never reached the webhook."""
    if not settings.alert_webhook_url:
        return 0
    pending = await db.undelivered_events(now - WEBHOOK_UNDELIVERED_WINDOW)
    return await deliver(pending)


async def drain() -> None:
    """Wait for in-flight webhook deliveries (tests / shutdown)."""
    if _deliveries:
        await asyncio.gather(*list(_deliveries), return_exceptions=True)
