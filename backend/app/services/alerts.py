"""Alert rules: evaluate every fresh analysis, record events, push them to a webhook.

Rule semantics (`threshold` meaning per kind; defaults applied on creation):

* `score_above` / `score_below` (SentiNET 0-100; default 70 / 30) — fires when
  the score *crosses* the line: the condition holds now and either the rule has
  never fired or the previous snapshot was on the other side. No re-fire while
  the condition simply persists.
* `score_change` (points; default 10) — the score moved ≥ threshold vs. a
  baseline: the snapshot ~24 h earlier (≤ 7 days old; else the oldest within
  24 h), advanced to the snapshot at the last trigger so a trend fires once per leg.
* `attention_spike` (heat 0-100; default 75) — attention heat crosses the line,
  compared with the newest earlier snapshot that *had* attention data.
* `new_narrative` (min. items, whole number; default 3) — a fresh story (items in
  the last 24 h, first seen ≤ 72 h ago) with ≥ threshold items that matches no
  story stored in the last 48 h. Stories are matched by identity — shared member
  articles — or by a near-identical headline with compatible tone, never by loose
  word overlap (which re-announces rephrased stories and merges opposite ones).
* `analyst_action` (no threshold) — rating/target actions (≤ 10 days old) absent
  from every earlier snapshot of the last 14 days that had analyst data.

Every comparison skips earlier runs that lacked the data in question (a Yahoo
throttle or a news outage stores "unavailable", not "empty"), so a degraded run
never makes old facts look new afterwards. Composite numbers (SentiNET score,
attention heat) of a *degraded* run (several inputs failed; see
`analyzer.run_quality`) are not readings: score/attention rules neither fire
on one nor use one as the previous value or baseline. (Runs with no evidence
at all are never stored.) With no usable baseline yet, the rule stays quiet. Score/attention rules also have a 2 h cooldown so a value
hovering around the threshold cannot spam. Evaluation is pure (`evaluate_rule`);
persistence and delivery live in `process_analysis` / `deliver`.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import settings
from app.schemas import AlertEvent, AlertRule, AlertRuleIn, Analysis, AnalystAction, Narrative
from app.services.tasks import describe_error
from app.storage import db
from app.storage.db import SnapshotRecord, StoryRef, analyst_action_key

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
WHOLE_NUMBER_KINDS = frozenset({"new_narrative"})
# Rules on composite numbers, which a degraded run does not measure reliably.
_READING_KINDS = frozenset({"score_above", "score_below", "score_change", "attention_spike"})
COOLDOWN = timedelta(hours=2)
REARM_AFTER = timedelta(days=7)  # a crossing rule with no comparable earlier value re-arms after this
CHANGE_WINDOW = timedelta(hours=24)
CHANGE_MAX_BASELINE_AGE = timedelta(days=7)  # an older baseline says nothing about "now"
ACTION_MAX_AGE = timedelta(days=10)
ANALYST_LOOKBACK = timedelta(days=14)
ATTENTION_LOOKBACK = timedelta(days=7)
STORY_LOOKBACK = timedelta(hours=48)  # stories seen in this window are not "new"
STORY_FRESH = timedelta(hours=24)  # a new story has coverage this recent…
STORY_MAX_AGE = timedelta(hours=72)  # …and did not start longer ago than this
HISTORY_WINDOW = max(ANALYST_LOOKBACK, ATTENTION_LOOKBACK, STORY_LOOKBACK)
HISTORY_LIMIT = 400
SHARED_MEMBERS = 1 / 3  # share of the smaller story's members two runs must share to be one story
OPPOSITE_TONE = 0.15  # both tones at least this strong and of opposite sign → different stories
MAX_LISTED = 4
WEBHOOK_UNDELIVERED_WINDOW = timedelta(hours=24)

_ACTION_VERBS = {"up": "upgrade", "down": "downgrade", "init": "initiates", "main": "maintains",
                 "reit": "reiterates", "other": "update"}
_TOKEN = re.compile(r"[a-z0-9$%.]+")
_STOP = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "as", "at", "by", "from",
    "is", "are", "be", "its", "it", "this", "that", "stock", "stocks", "shares", "inc", "corp",
})
_DUP_JACCARD = 0.8  # fallback near-duplicate cut-off when textintel is unavailable

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
    history: tuple[SnapshotRecord, ...] = ()  # earlier snapshots, newest first (≤ HISTORY_WINDOW)

    @property
    def previous(self) -> SnapshotRecord | None:
        """The newest earlier *reading*: the preceding snapshot that was not degraded."""
        return _first(self.history, lambda r: not r.degraded)


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
        if rule.kind in WHOLE_NUMBER_KINDS and threshold != int(threshold):
            raise ValueError(f"threshold for {rule.kind} is a number of items: use a whole number")
    return rule.model_copy(update={"threshold": threshold})


# --------------------------------------------------------------------------- #
# Story identity
# --------------------------------------------------------------------------- #
def _tokens(text: str, ignore: frozenset[str] = frozenset()) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and t not in ignore and len(t) > 1}


def _near_duplicate(a: str, b: str, ignore: frozenset[str]) -> bool:
    """Syndicated copies / trivial rewrites of one headline (textintel's detector when available)."""
    try:
        from app.nlp.narratives import find_duplicates

        return len(find_duplicates([a, b])) == 1
    except Exception:  # noqa: BLE001 - textintel mid-edit: strict token fallback
        ta, tb = _tokens(a, ignore), _tokens(b, ignore)
        return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= _DUP_JACCARD


def _opposite(a: float, b: float) -> bool:
    return a * b < 0 and min(abs(a), abs(b)) >= OPPOSITE_TONE


def same_story(new: Narrative, old: StoryRef, ignore: frozenset[str] = frozenset()) -> bool:
    """True when `new` (this run) is the story `old` (an earlier run).

    Primary evidence is shared member articles (ids are stable per URL): one
    third of the smaller story's members. Without that, only a near-identical
    headline of compatible tone counts — "shares rise after launch" and
    "shares fall after launch" are different stories.
    """
    a, b = set(new.signal_ids), set(old.ids)
    if a and b:
        shared = len(a & b)
        if shared and shared >= SHARED_MEMBERS * min(len(a), len(b)):
            return True
    if _opposite(new.score, old.score):
        return False
    return _near_duplicate(new.headline, old.headline, ignore)


def _company_tokens(analysis: Analysis) -> frozenset[str]:
    names = [analysis.ticker]
    if analysis.profile is not None:
        names += [analysis.profile.name or "", analysis.profile.short_name or ""]
    return frozenset(t for n in names for t in _TOKEN.findall(n.lower()))


# --------------------------------------------------------------------------- #
# Baselines: the newest earlier snapshots that actually had the data
# --------------------------------------------------------------------------- #
def _within(ctx: RuleContext, window: timedelta) -> Iterable[SnapshotRecord]:
    return (r for r in ctx.history if ctx.now - r.at <= window)


def _first(records: Iterable[SnapshotRecord], pred: Callable[[SnapshotRecord], bool]) -> SnapshotRecord | None:
    return next((r for r in records if pred(r)), None)


def known_analyst_keys(ctx: RuleContext) -> set[str] | None:
    """Action keys seen in recent runs that had analyst data; None when no such run exists."""
    usable = [r.analyst_keys for r in _within(ctx, ANALYST_LOOKBACK) if r.analyst_keys is not None]
    return set().union(*usable) if usable else None


def known_stories(ctx: RuleContext) -> list[StoryRef] | None:
    """Stories of recent runs whose news sources answered; None when no such run exists."""
    usable = [r for r in _within(ctx, STORY_LOOKBACK)
              if r.news_ok is True or (r.news_ok is None and r.stories)]
    return [s for r in usable for s in r.stories] if usable else None


# --------------------------------------------------------------------------- #
# Pure evaluation
# --------------------------------------------------------------------------- #
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
    """Edge-triggered threshold test with cooldown.

    `prev_value` is the newest earlier value *available*; without one the rule
    fires only if it never fired (or last fired over a week ago) — an unknown
    previous value must not re-arm the edge.
    """
    thr = rule.threshold
    if value is None or thr is None or _cooling(rule, ctx.now):
        return False
    holds = value >= thr if above else value <= thr
    if not holds:
        return False
    if rule.last_triggered_at is None:
        return True
    if prev_value is None:
        return ctx.now - rule.last_triggered_at > REARM_AFTER
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


def _fresh_story(n: Narrative, now: datetime) -> bool:
    recent = n.last_seen is None or now - _aware(n.last_seen) <= STORY_FRESH
    young = n.first_seen is None or now - _aware(n.first_seen) <= STORY_MAX_AGE
    return recent and young


def evaluate_rule(rule: AlertRule, ctx: RuleContext, reference: SnapshotRecord | None = None) -> Alert | None:
    """Return the alert this rule raises for `ctx`, or None. `reference` is the score_change baseline."""
    t = ctx.analysis.ticker
    cur, prev = ctx.current, ctx.previous
    thr = rule.threshold

    if rule.kind in _READING_KINDS and cur.degraded:
        return None  # several inputs failed: this score/heat is not a reading

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
        if att is None:
            return None
        before = _first(_within(ctx, ATTENTION_LOOKBACK),
                        lambda r: r.attention_heat is not None and not r.degraded)
        if not _crossing(rule, ctx, cur.attention_heat, before.attention_heat if before else None, above=True):
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
        known = known_stories(ctx)
        if known is None:  # no earlier run with working news: everything would look "new"
            return None
        min_items = int(thr or 1)
        ignore = _company_tokens(ctx.analysis)
        fresh = [
            n for n in ctx.analysis.narratives
            if n.count >= min_items and _fresh_story(n, ctx.now)
            and not any(same_story(n, old, ignore) for old in known)
        ]
        if not fresh:
            return None
        lines = [f"• {n.headline} — {n.count} items, tone {n.score:+.2f}" for n in fresh[:MAX_LISTED]]
        noun = "story" if len(fresh) == 1 else "stories"
        return Alert(f"{t}: {len(fresh)} new {noun}", "\n".join(lines))

    if rule.kind == "analyst_action":
        analysts = ctx.analysis.analysts
        seen = known_analyst_keys(ctx)
        if analysts is None or seen is None:  # unavailable now, or no baseline yet
            return None
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
    """Baseline for score_change: ~24 h ago (≤ 7 d), advanced to the last trigger's snapshot.

    Only sound runs qualify: a degraded run's score is not a baseline."""
    t = current.snapshot.ticker
    base = await db.latest_record(t, before=current.at - CHANGE_WINDOW, sound_only=True)
    if base is not None and current.at - base.at > CHANGE_MAX_BASELINE_AGE:
        base = None
    if base is None:
        base = await db.oldest_record_since(t, current.at - CHANGE_WINDOW, exclude_id=current.id, sound_only=True)
    if rule.last_triggered_at is not None:
        at_trigger = await db.latest_record(t, before=rule.last_triggered_at, before_id=current.id,
                                            sound_only=True)
        if at_trigger is not None and (base is None or at_trigger.at > base.at):
            base = at_trigger
    return base


async def process_analysis(analysis: Analysis, current: SnapshotRecord) -> list[AlertEvent]:
    """Evaluate the ticker's enabled rules against a freshly stored snapshot."""
    rules = await db.list_rules(ticker=analysis.ticker, enabled_only=True)
    if not rules:
        return []
    history = await db.records_before(analysis.ticker, current.id, current.at - HISTORY_WINDOW, HISTORY_LIMIT)
    ctx = RuleContext(now=current.at, analysis=analysis, current=current, history=tuple(history))
    events: list[AlertEvent] = []
    for rule in rules:
        needs_reference = rule.kind == "score_change" and not current.degraded
        reference = await _change_reference(rule, current) if needs_reference else None
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


_DISCORD_MASS_MENTION = re.compile(r"@(everyone|here)\b", re.IGNORECASE)


def _discord_safe(text: str) -> str:
    """Defang @everyone/@here in provider text (belt and braces with `allowed_mentions`)."""
    return _DISCORD_MASS_MENTION.sub(lambda m: "@\u200b" + m.group(1), text)


def _slack_safe(text: str) -> str:
    """Slack's control characters: `<!channel>`, `<@U…>` and links only exist between < and >."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def webhook_payload(event: AlertEvent) -> dict[str, Any]:
    """Body accepted by both Discord (`content`) and Slack (`text`) incoming webhooks.

    Titles and details quote provider text (headlines, analyst firms), so the
    payload can never ping anyone: Discord parses no mentions, Slack gets its
    control characters escaped.
    """
    return {
        "username": "SentiNET",
        "content": _discord_safe(f"**{event.title}**\n{event.detail}")[:1990],
        "allowed_mentions": {"parse": []},
        "text": f"*{_slack_safe(event.title)}*\n{_slack_safe(event.detail)}"[:3000],
    }


async def deliver(events: list[AlertEvent]) -> int:
    """POST events to the configured webhook in order; returns how many were delivered.

    Each event is claimed in the database first and only claimed events are
    sent, so concurrent deliverers never post the same alert twice. Events left
    unsent (failure) are released back to pending for `retry_undelivered`.
    """
    from app.core.http import fetch

    url = settings.alert_webhook_url.strip()
    if not url.startswith(("https://", "http://")) or not events:
        return 0
    claimed = set(await db.claim_events([e.id for e in events]))
    batch = [e for e in events if e.id in claimed]
    sent: list[int] = []
    try:
        for event in batch:
            try:
                await fetch(url, method="POST", json=webhook_payload(event), timeout=8.0, retries=1)
            except Exception as exc:  # noqa: BLE001 - keep it queued; the monitor retries later
                logger.warning("alert webhook delivery failed: %s", describe_error(exc))
                break
            sent.append(event.id)
    finally:
        await db.release_events(sent, delivered=True)
        await db.release_events([e.id for e in batch if e.id not in sent], delivered=False)
    return len(sent)


async def retry_undelivered(now: datetime) -> int:
    """Re-send events from the last 24 h that never reached the webhook (after in-flight sends finish)."""
    if not settings.alert_webhook_url:
        return 0
    await drain()
    pending = await db.undelivered_events(now - WEBHOOK_UNDELIVERED_WINDOW)
    return await deliver(pending)


async def drain() -> None:
    """Wait for in-flight webhook deliveries (CLI exit, tests, shutdown)."""
    if _deliveries:
        await asyncio.gather(*list(_deliveries), return_exceptions=True)
