"""Alert rules: pure evaluation semantics, persistence and webhook delivery."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from app.analytics.inputs import AnalysisInputs
from app.config import settings
from app.schemas import (
    AlertRule,
    AlertRuleIn,
    AnalystAction,
    AnalystView,
    AttentionView,
    Narrative,
    Snapshot,
)
from app.services import alerts
from app.services.alerts import RuleContext, evaluate_rule, normalize_rule
from app.sources.base import CompanyRef
from app.storage import db
from app.storage.db import SnapshotRecord, analyst_action_key
from tests.services.fakes import FakeWorld

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def rule(kind: str, threshold: float | None = None, last: datetime | None = None, rid: int = 1) -> AlertRule:
    return AlertRule(id=rid, ticker="NVDA", kind=kind, threshold=threshold, created_at=NOW - timedelta(days=3),
                     last_triggered_at=last)


def record(score: int, at: datetime = NOW, heat: int | None = None, narratives: list[str] | None = None,
           keys: tuple[str, ...] = (), rid: int = 10) -> SnapshotRecord:
    snap = Snapshot(ticker="NVDA", at=at, sentinel_score=score, score=0.1, label="neutral", n_signals=10,
                    narratives=narratives or [])
    return SnapshotRecord(id=rid, snapshot=snap, attention_heat=heat, analyst_keys=keys)


def analysis(score: int = 72, **extra):
    world = FakeWorld(score=score)
    inputs = AnalysisInputs(company=CompanyRef(ticker="NVDA", name="NVIDIA", short_name="Nvidia"), now=NOW,
                            engine_name="sentinel")
    return world.build(inputs).model_copy(update=extra)


def ctx(current: SnapshotRecord, previous: SnapshotRecord | None, **extra) -> RuleContext:
    return RuleContext(now=current.at, analysis=analysis(current.score, **extra), current=current, previous=previous)


# ---- validation ------------------------------------------------------------------ #
def test_normalize_rule_defaults_and_ranges():
    assert normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_above")).threshold == 70
    assert normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_below")).threshold == 30
    assert normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_change")).threshold == 10
    assert normalize_rule(AlertRuleIn(ticker="NVDA", kind="analyst_action", threshold=5)).threshold is None
    with pytest.raises(ValueError, match="between 1 and 99"):
        normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_above", threshold=150))
    with pytest.raises(ValueError):
        normalize_rule(AlertRuleIn(ticker="NVDA", kind="new_narrative", threshold=0))


# ---- score thresholds --------------------------------------------------------------- #
def test_score_above_fires_on_first_eligible_snapshot():
    alert = evaluate_rule(rule("score_above", 70), ctx(record(72), record(68, NOW - timedelta(hours=1))))
    assert alert is not None
    assert alert.title == "NVDA SentiNET 72 ≥ 70"
    assert "Previously 68" in alert.detail and "Bullish" in alert.detail


def test_score_above_is_edge_triggered():
    fired_long_ago = NOW - timedelta(days=1)
    # Condition persisted since the last trigger → no re-fire.
    assert evaluate_rule(rule("score_above", 70, last=fired_long_ago),
                         ctx(record(75), record(73, NOW - timedelta(hours=1)))) is None
    # Dipped below, now crossed again → fires.
    assert evaluate_rule(rule("score_above", 70, last=fired_long_ago),
                         ctx(record(75), record(66, NOW - timedelta(hours=1)))) is not None
    # Below threshold → nothing.
    assert evaluate_rule(rule("score_above", 70), ctx(record(69), None)) is None


def test_score_below_and_cooldown():
    assert evaluate_rule(rule("score_below", 30), ctx(record(28), record(35))) is not None
    recently = NOW - timedelta(minutes=30)
    assert evaluate_rule(rule("score_below", 30, last=recently), ctx(record(28), record(35))) is None


def test_score_change_uses_reference():
    ref = record(55, NOW - timedelta(hours=24), rid=3)
    alert = evaluate_rule(rule("score_change", 10), ctx(record(68), None), reference=ref)
    assert alert is not None and alert.title == "NVDA SentiNET +13 in 24h (55 → 68)"
    assert "Top driver: 3 price-target raises" in alert.detail
    assert evaluate_rule(rule("score_change", 15), ctx(record(68), None), reference=ref) is None
    assert evaluate_rule(rule("score_change", 10), ctx(record(68), None), reference=None) is None
    down = evaluate_rule(rule("score_change", 10), ctx(record(40), None), reference=ref)
    assert down is not None and "-15" in down.title


# ---- attention / narratives / analysts ----------------------------------------------- #
def test_attention_spike_crossing():
    att = AttentionView(heat=88, label="Spiking", news_volume_z=2.6, reddit_change_pct=140.0, signals_24h=55)
    alert = evaluate_rule(rule("attention_spike", 75), ctx(record(60, heat=88), record(60, heat=40), attention=att))
    assert alert is not None and alert.title == "NVDA attention Spiking (88/100)"
    assert "news volume z +2.6" in alert.detail and "Reddit mentions +140%" in alert.detail
    assert evaluate_rule(rule("attention_spike", 75, last=NOW - timedelta(days=2)),
                         ctx(record(60, heat=88), record(60, heat=80), attention=att)) is None


def test_new_narrative_ignores_known_stories_and_small_ones():
    narratives = [
        Narrative(id="a", headline="Nvidia faces $1.05B lawsuit from 1993 advisor", count=9, score=-0.4),
        Narrative(id="b", headline="Nvidia hit with $1.05B lawsuit from 1993 advisor over options", count=7),
        Narrative(id="c", headline="Nvidia unveils Rubin Ultra roadmap at GTC Paris", count=5, score=0.3),
        Narrative(id="d", headline="Tiny blog post", count=1),
    ]
    prev = record(60, NOW - timedelta(hours=1), narratives=["Nvidia faces $1.05B lawsuit from 1993 advisor"])
    alert = evaluate_rule(rule("new_narrative", 3), ctx(record(60), prev, narratives=narratives))
    assert alert is not None and alert.title == "NVDA: 1 new story"
    assert "Rubin Ultra" in alert.detail and "lawsuit" not in alert.detail
    # First snapshot ever: nothing is "new" yet.
    assert evaluate_rule(rule("new_narrative", 3), ctx(record(60), None, narratives=narratives)) is None


def test_analyst_action_detects_only_unseen_recent_actions():
    seen = AnalystAction(date=NOW - timedelta(days=2), firm="Morgan Stanley", action="up",
                         from_grade="Equal-Weight", to_grade="Overweight", price_target=250, prior_target=220)
    new = AnalystAction(date=NOW - timedelta(hours=3), firm="Mizuho", action="down", from_grade="Buy",
                        to_grade="Neutral", price_target=150, prior_target=190)
    stale = AnalystAction(date=NOW - timedelta(days=40), firm="Old Bank", action="up", to_grade="Buy")
    keys = (analyst_action_key(seen.date, seen.firm, seen.action, seen.to_grade, seen.price_target),)
    view = AnalystView(actions=[new, seen, stale])
    alert = evaluate_rule(rule("analyst_action"), ctx(record(60), record(60, keys=keys), analysts=view))
    assert alert is not None and alert.title == "NVDA: 1 new analyst action"
    assert alert.detail == "▼ Mizuho downgrade: Buy → Neutral · PT $150 (from $190)"
    assert evaluate_rule(rule("analyst_action"), ctx(record(60), None, analysts=view)) is None


def test_headline_similarity():
    assert alerts.headline_similarity("Nvidia beats estimates", "Nvidia beats estimates") == 1.0
    assert alerts.headline_similarity("Nvidia beats estimates", "Apple cuts guidance") == 0.0
    assert alerts.headline_similarity("", "x") == 0.0


# ---- persistence + delivery ------------------------------------------------------------ #
async def _store(a, at: datetime) -> SnapshotRecord:
    return await db.save_snapshot(a.model_copy(update={"generated_at": at}))


async def test_process_analysis_records_events_and_marks_rules(store):
    above = await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_above")))
    change = await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_change")))
    await db.create_rule(normalize_rule(AlertRuleIn(ticker="AAPL", kind="score_above")))

    await _store(analysis(52), NOW - timedelta(hours=30))
    await _store(analysis(60), NOW - timedelta(hours=2))
    a = analysis(74)
    cur = await _store(a, NOW)
    events = await alerts.process_analysis(a, cur)
    titles = sorted(e.title for e in events)
    assert titles == ["NVDA SentiNET +22 in 30h (52 → 74)", "NVDA SentiNET 74 ≥ 70"]
    rules = {r.id: r for r in await db.list_rules()}
    assert rules[above.id].last_triggered_at == NOW and rules[change.id].last_triggered_at == NOW

    # Next refresh: still high, nothing new → no repeat alerts.
    a2 = analysis(75)
    cur2 = await _store(a2, NOW + timedelta(hours=3))
    assert await alerts.process_analysis(a2, cur2) == []


async def test_score_change_baseline_advances_to_last_trigger(store):
    r = await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_change", threshold=10)))
    await _store(analysis(50), NOW - timedelta(hours=5))
    a = analysis(62)
    cur = await _store(a, NOW - timedelta(hours=4))
    assert len(await alerts.process_analysis(a, cur)) == 1
    # +5 more points since the trigger: below threshold vs the new baseline (62), though +17 vs 24h.
    a2 = analysis(67)
    cur2 = await _store(a2, NOW)
    assert await alerts.process_analysis(a2, cur2) == []
    assert (await db.list_rules())[0].id == r.id


async def test_webhook_delivery_payload_and_retry(store, monkeypatch):
    monkeypatch.setattr(settings, "alert_webhook_url", "https://discord.example/api/webhooks/1/abc")
    await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_above", threshold=60)))
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://discord.example/api/webhooks/1/abc").mock(
            side_effect=[httpx.Response(500), httpx.Response(500), httpx.Response(204)])
        a = analysis(72)
        cur = await _store(a, NOW)
        assert len(await alerts.process_analysis(a, cur)) == 1
        await alerts.drain()
        assert route.call_count == 2  # first try + one retry, both failed
        assert (await db.list_alert_events(5))[0].delivered is False

        body = json.loads(route.calls[0].request.content)
        assert body["username"] == "SentiNET"
        assert body["content"].startswith("**NVDA SentiNET 72 ≥ 60**")
        assert body["text"].startswith("*NVDA SentiNET 72 ≥ 60*")

        assert await alerts.retry_undelivered(NOW + timedelta(minutes=30)) == 1
        assert (await db.list_alert_events(5))[0].delivered is True


async def test_no_webhook_configured_means_no_delivery(store, monkeypatch):
    monkeypatch.setattr(settings, "alert_webhook_url", "")
    assert await alerts.retry_undelivered(NOW) == 0


async def test_score_change_ignores_stale_baseline(store):
    await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_change", threshold=10)))
    await _store(analysis(40), NOW - timedelta(days=30))  # weeks old: not a baseline
    await _store(analysis(60), NOW - timedelta(hours=3))
    a = analysis(66)
    cur = await _store(a, NOW)
    assert await alerts.process_analysis(a, cur) == []  # +6 vs 3h ago, not +26 vs a month ago
