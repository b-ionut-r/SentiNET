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
from app.services.alerts import RuleContext, evaluate_rule, normalize_rule, same_story
from app.sources.base import CompanyRef
from app.storage import db
from app.storage.db import SnapshotRecord, StoryRef, analyst_action_key
from tests.services.fakes import FakeWorld

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def rule(kind: str, threshold: float | None = None, last: datetime | None = None, rid: int = 1) -> AlertRule:
    return AlertRule(id=rid, ticker="NVDA", kind=kind, threshold=threshold, created_at=NOW - timedelta(days=3),
                     last_triggered_at=last)


def record(score: int, at: datetime = NOW, heat: int | None = None, narratives: list[str] | None = None,
           keys: tuple[str, ...] | None = (), rid: int = 10, stories: tuple[StoryRef, ...] | None = None,
           news_ok: bool | None = True) -> SnapshotRecord:
    snap = Snapshot(ticker="NVDA", at=at, sentinel_score=score, score=0.1, label="neutral", n_signals=10,
                    narratives=narratives or [])
    if stories is None:
        stories = tuple(StoryRef(headline=h) for h in narratives or [])
    return SnapshotRecord(id=rid, snapshot=snap, attention_heat=heat, analyst_keys=keys, stories=stories,
                          news_ok=news_ok)


def analysis(score: int = 72, **extra):
    world = FakeWorld(score=score)
    inputs = AnalysisInputs(company=CompanyRef(ticker="NVDA", name="NVIDIA", short_name="Nvidia"), now=NOW,
                            engine_name="sentinel")
    return world.build(inputs).model_copy(update=extra)


def ctx(current: SnapshotRecord, previous: SnapshotRecord | None = None,
        history: tuple[SnapshotRecord, ...] | None = None, **extra) -> RuleContext:
    """Context with `history` (newest first), or just the immediately preceding snapshot."""
    if history is None:
        history = (previous,) if previous is not None else ()
    return RuleContext(now=current.at, analysis=analysis(current.score, **extra), current=current, history=history)


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
    with pytest.raises(ValueError, match="whole number"):  # 2.7 used to be silently truncated to 2
        normalize_rule(AlertRuleIn(ticker="NVDA", kind="new_narrative", threshold=2.7))
    assert normalize_rule(AlertRuleIn(ticker="NVDA", kind="new_narrative", threshold=4.0)).threshold == 4


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
        # Same story as stored, by shared member articles although the headline was rephrased.
        Narrative(id="a", headline="Nvidia hit with $1.05B lawsuit over options", count=9, score=-0.4,
                  signal_ids=["s1", "s2", "s3", "s9"]),
        # Same story by a near-identical (syndicated) headline.
        Narrative(id="b", headline="Nvidia unveils Rubin GPU at GTC keynote - Reuters", count=7, score=0.3),
        Narrative(id="c", headline="Nvidia wins $6B sovereign AI cloud contract in Saudi Arabia", count=5, score=0.3,
                  signal_ids=["s20", "s21"]),
        Narrative(id="d", headline="Tiny blog post", count=1),
    ]
    prev = record(60, NOW - timedelta(hours=1), stories=(
        StoryRef("Nvidia faces $1.05B suit from 1993 advisor", ids=("s1", "s2", "s4"), score=-0.35),
        StoryRef("Nvidia unveils Rubin GPU at GTC keynote", ids=("s7",), score=0.25),
    ))
    alert = evaluate_rule(rule("new_narrative", 3), ctx(record(60), prev, narratives=narratives))
    assert alert is not None and alert.title == "NVDA: 1 new story"
    assert "sovereign AI cloud" in alert.detail and "lawsuit" not in alert.detail and "Rubin" not in alert.detail
    # First snapshot ever: nothing is "new" yet.
    assert evaluate_rule(rule("new_narrative", 3), ctx(record(60), None, narratives=narratives)) is None


def test_new_narrative_skips_runs_without_news_and_needs_a_baseline():
    story = Narrative(id="a", headline="Nvidia unveils Rubin GPU at GTC", count=6, signal_ids=["s1", "s2"])
    stored = record(60, NOW - timedelta(hours=3), rid=1,
                    stories=(StoryRef("Nvidia unveils Rubin GPU at GTC", ids=("s1", "s2")),))
    outage = record(60, NOW - timedelta(hours=1), rid=2, stories=(), news_ok=False)
    # News outage in between: the story is still known from the earlier run → no alert.
    assert evaluate_rule(rule("new_narrative", 3), ctx(record(60), history=(outage, stored),
                                                      narratives=[story])) is None
    # Only outage runs in the window: no baseline → stay quiet rather than call everything new.
    assert evaluate_rule(rule("new_narrative", 3), ctx(record(60), history=(outage,), narratives=[story])) is None
    # A quiet-but-working run (news answered, no stories) is a valid empty baseline.
    quiet = record(60, NOW - timedelta(hours=1), rid=3, stories=(), news_ok=True)
    alert = evaluate_rule(rule("new_narrative", 3), ctx(record(60), history=(quiet,), narratives=[story]))
    assert alert is not None and alert.title == "NVDA: 1 new story"
    # Stories older than the 48 h window are forgotten.
    old = record(60, NOW - timedelta(hours=60), rid=4,
                 stories=(StoryRef("Nvidia unveils Rubin GPU at GTC", ids=("s1", "s2")),))
    assert evaluate_rule(rule("new_narrative", 3), ctx(record(60), history=(quiet, old), narratives=[story]))


def test_new_narrative_requires_fresh_coverage():
    quiet = record(60, NOW - timedelta(hours=1), stories=(), news_ok=True)
    resurfaced = Narrative(id="a", headline="Nvidia antitrust probe widens", count=8,
                           first_seen=NOW - timedelta(days=5), last_seen=NOW - timedelta(hours=2))
    stale = Narrative(id="b", headline="Nvidia insider sale filed", count=4,
                      first_seen=NOW - timedelta(hours=40), last_seen=NOW - timedelta(hours=30))
    assert evaluate_rule(rule("new_narrative", 3), ctx(record(60), quiet, narratives=[resurfaced, stale])) is None


def test_same_story_identity():
    def n(headline: str, ids: list[str] = (), score: float = 0.0) -> Narrative:
        return Narrative(id="x", headline=headline, count=5, signal_ids=list(ids), score=score)

    # The spec's same-story pair (rephrased): same story when member articles are shared.
    assert same_story(n("Nvidia hit with $1.05B lawsuit over options", ["a", "b", "c"]),
                      StoryRef("Nvidia faces $1.05B suit from 1993 advisor", ("a", "z")))
    # A story that grew: 2 of the old story's 3 members are still there.
    assert same_story(n("Nvidia's Rubin chips steal the show at GTC", [f"m{i}" for i in range(12)]),
                      StoryRef("Nvidia unveils Rubin GPU at GTC keynote", ("m1", "m2", "q9")))
    # Opposite developments sharing most words are different stories…
    assert not same_story(n("Apple stock rises after iPhone 18 launch", ["r1"], 0.4),
                          StoryRef("Apple stock falls after iPhone 18 launch", ("f1",), -0.4))
    assert not same_story(n("Tesla shares jump on record deliveries", ["t1"], 0.5),
                          StoryRef("Tesla shares slide on record deliveries miss", ("t2",), -0.5))
    # …and a strongly opposite tone overrides an identical headline (ids differ).
    assert not same_story(n("Nvidia guidance update", ["g1"], 0.5), StoryRef("Nvidia guidance update", ("g2",), -0.5))
    # Syndicated copy (publisher suffix, case) with different URLs: same story.
    assert same_story(n("NVIDIA unveils Rubin GPU at GTC keynote - Reuters", ["u1"], 0.3),
                      StoryRef("Nvidia unveils Rubin GPU at GTC keynote", ("u2",), 0.2))


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


def test_analyst_action_ignores_runs_without_analyst_data():
    a = AnalystAction(date=NOW - timedelta(days=2), firm="Morgan Stanley", action="up", to_grade="Overweight")
    keys = (analyst_action_key(a.date, a.firm, a.action, a.to_grade, a.price_target),)
    view = AnalystView(actions=[a])
    with_data = record(60, NOW - timedelta(hours=2), keys=keys, rid=1)
    outage = record(60, NOW - timedelta(hours=1), keys=None, rid=2)  # Yahoo throttled: unavailable, not empty
    assert evaluate_rule(rule("analyst_action"), ctx(record(60), history=(outage, with_data), analysts=view)) is None
    # Nothing but outages to compare with: no baseline → quiet.
    assert evaluate_rule(rule("analyst_action"), ctx(record(60), history=(outage,), analysts=view)) is None


def test_attention_spike_compares_with_last_available_heat():
    att = AttentionView(heat=90, label="Spiking", signals_24h=40)
    fired = NOW - timedelta(hours=3)  # past the cooldown
    hot = record(60, NOW - timedelta(hours=2), heat=90, rid=1)
    blank = record(60, NOW - timedelta(hours=1), heat=None, rid=2)
    # Heat never left the zone; a run without attention data in between must not re-arm the edge.
    assert evaluate_rule(rule("attention_spike", 75, last=fired),
                         ctx(record(60, heat=90), history=(blank, hot), attention=att)) is None
    cool = record(60, NOW - timedelta(hours=2), heat=40, rid=3)
    assert evaluate_rule(rule("attention_spike", 75, last=fired),
                         ctx(record(60, heat=90), history=(blank, cool), attention=att)) is not None
    # Never fired before: the first observation above the line fires.
    assert evaluate_rule(rule("attention_spike", 75), ctx(record(60, heat=90), attention=att)) is not None


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


async def test_concurrent_deliverers_never_post_the_same_event_twice(store, monkeypatch):
    import asyncio

    import app.core.http as http

    monkeypatch.setattr(settings, "alert_webhook_url", "https://hooks.slack.example/T/B/x")
    posts: list[str] = []

    async def slow_fetch(url, **kw):
        await asyncio.sleep(0.1)
        posts.append(kw["json"]["text"])

    monkeypatch.setattr(http, "fetch", slow_fetch)
    events = [await db.add_alert_event(None, "NVDA", NOW, f"alert {i}", "detail") for i in range(3)]
    sent = await asyncio.gather(alerts.deliver(events), alerts.deliver(events),
                                alerts.retry_undelivered(datetime.now(UTC)))
    assert sorted(posts) == ["*alert 0*\ndetail", "*alert 1*\ndetail", "*alert 2*\ndetail"]
    assert sum(sent) == 3
    assert all(e.delivered for e in await db.list_alert_events(5))


async def test_abandoned_claim_is_retried(store, monkeypatch):
    """A process that died mid-POST leaves a claim; after CLAIM_STALE it is pending again."""
    event = await db.add_alert_event(None, "NVDA", NOW, "t", "d")
    assert await db.claim_events([event.id]) == [event.id]
    assert await db.claim_events([event.id]) == []  # in flight: nobody else may send it
    assert await db.undelivered_events(NOW - timedelta(days=1)) == []

    def age_claim(conn):
        with conn:
            conn.execute("UPDATE alert_events SET claimed_at = ?",
                         (db.to_db_time(datetime.now(UTC) - db.CLAIM_STALE - timedelta(seconds=1)),))

    await db._run(age_claim)
    assert [e.id for e in await db.undelivered_events(NOW - timedelta(days=1))] == [event.id]
    assert await db.claim_events([event.id]) == [event.id]
    await db.release_events([event.id], delivered=False)
    assert (await db.list_alert_events(1))[0].delivered is False


# ---- degraded runs through the real pipeline (reviewer repros, inverted) ------------------ #
async def test_no_false_analyst_alert_after_analyst_outage(world):
    from tests.services.fakes import Sentinel

    now = datetime.now(UTC)
    world.analyst_actions = [AnalystAction(date=now - timedelta(days=d), firm=f"Firm {d}", action="up",
                                           to_grade="Buy") for d in (2, 5, 8)]
    await db.create_rule(AlertRuleIn(ticker="NVDA", kind="analyst_action"))
    from app.services import analyzer

    await analyzer.analyze("NVDA", refresh=True)  # baseline with analyst data
    world.intel["analysts"] = Sentinel(exc=RuntimeError("yahoo 429"))
    await analyzer.analyze("NVDA", refresh=True)  # analysts unavailable: stored as unknown, not "none"
    world.intel["analysts"] = "auto"
    await analyzer.analyze("NVDA", refresh=True)  # back to normal: nothing actually new
    assert await db.list_alert_events(10) == []
    # A genuinely new action is still caught.
    world.analyst_actions.insert(0, AnalystAction(date=now, firm="Mizuho", action="down", to_grade="Neutral"))
    await analyzer.analyze("NVDA", refresh=True)
    assert [e.title for e in await db.list_alert_events(10)] == ["NVDA: 1 new analyst action"]


async def test_no_false_story_alert_after_news_outage(world):
    from app.services import analyzer

    stories = [Narrative(id=f"n{i}", headline=h, count=6, signal_ids=[f"{i}a", f"{i}b"]) for i, h in enumerate([
        "Nvidia hit with $1.05B lawsuit over options", "Nvidia unveils Rubin GPU at GTC"])]
    world.narratives = stories
    await db.create_rule(AlertRuleIn(ticker="NVDA", kind="new_narrative", threshold=3))
    await analyzer.analyze("NVDA", refresh=True)
    world.narratives = []
    world.sources[0][0].error = RuntimeError("google news down")  # the news outage itself
    await analyzer.analyze("NVDA", refresh=True)
    world.sources[0][0].error = None
    world.narratives = stories
    await analyzer.analyze("NVDA", refresh=True)
    assert await db.list_alert_events(10) == []


async def test_attention_spike_does_not_refire_after_attention_outage(world):
    from app.services import analyzer

    hot = AttentionView(heat=90, label="Spiking", signals_24h=40)
    await db.create_rule(AlertRuleIn(ticker="NVDA", kind="attention_spike", threshold=75))

    def age_rule(conn):  # the last trigger was 3 h ago (past the 2 h cooldown)
        with conn:
            conn.execute("UPDATE alert_rules SET last_triggered_at = ?",
                         (db.to_db_time(datetime.now(UTC) - timedelta(hours=3)),))

    fired = []
    for att in (hot, hot, None, hot):
        world.attention = att
        await analyzer.analyze("NVDA", refresh=True)
        fired.append(len(await db.list_alert_events(10)))
        await db._run(age_rule)
    assert fired == [1, 1, 1, 1]  # heat never fell below 75: one alert only


# ---- evidence-free and degraded runs are not readings (final-review repros) ---------------- #
def no_read(**extra):
    """The real "No read" shape: score 50, no signals, no available score component."""
    from tests.services.fakes import make_verdict

    return analysis(50, verdict=make_verdict(evidence=False), **extra)


async def test_no_read_run_neither_fires_nor_anchors_score_rules(store):
    """Secops repro: a run where every source/feed failed fired 'score_below' and then a +14 'change'."""
    await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_below", threshold=55)))
    await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_change", threshold=10)))
    blank = no_read()
    assert blank.sentiment.n == 0 and blank.verdict.score == 50
    cur = await _store(blank, NOW - timedelta(hours=1))
    assert cur.degraded  # stored by a caller anyway: flagged by the store itself
    assert await alerts.process_analysis(blank, cur) == []
    later = analysis(64)
    cur2 = await _store(later, NOW)
    assert await alerts.process_analysis(later, cur2) == []  # no "+14 in 1h (50 → 64)"
    assert await db.list_alert_events(10) == []


async def test_degraded_runs_are_skipped_as_triggers_and_baselines(store):
    """E2E repro: 62 → 57 (4 sources timed out) → 61 → 60 (5 of 9 failed) fired 'SentiNET 60 ≤ 60'."""
    await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_below", threshold=60)))
    titles = []
    for minutes, score, degraded in ((-18, 62, False), (-14, 57, True), (-7, 61, False), (0, 60, True),
                                     (7, 59, False)):
        a = analysis(score)
        cur = await db.save_snapshot(a.model_copy(update={"generated_at": NOW + timedelta(minutes=minutes)}),
                                     degraded=degraded)
        titles += [e.title for e in await alerts.process_analysis(a, cur)]
    assert titles == ["NVDA SentiNET 59 ≤ 60"]
    detail = (await db.list_alert_events(1))[0].detail
    assert "Previously 61" in detail  # the last sound reading, not the degraded 60


async def test_score_change_baseline_skips_degraded_runs(store):
    await db.create_rule(normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_change", threshold=10)))
    await _store(analysis(62), NOW - timedelta(hours=26))
    await db.save_snapshot(analysis(40).model_copy(update={"generated_at": NOW - timedelta(hours=25)}),
                           degraded=True)
    a = analysis(64)
    cur = await _store(a, NOW)
    assert await alerts.process_analysis(a, cur) == []  # +2 vs the sound 62, not +24 vs the degraded 40


def test_reading_rules_ignore_degraded_records():
    from dataclasses import replace

    bad_prev = replace(record(55, NOW - timedelta(minutes=30), heat=30, rid=9), degraded=True)
    good_prev = record(66, NOW - timedelta(hours=1), heat=80, rid=8)
    # Previous *reading* is 66 (already ≥ 70? no: 66 < 70) → a crossing to 72 fires; the
    # degraded 55 is skipped, and the alert quotes 66.
    alert = evaluate_rule(rule("score_above", 70, last=NOW - timedelta(days=1)),
                          ctx(record(72), history=(bad_prev, good_prev)))
    assert alert is not None and "Previously 66" in alert.detail
    # A degraded *current* run never fires a reading rule…
    cur_bad = replace(record(80, heat=95), degraded=True)
    att = AttentionView(heat=95, label="Spiking", signals_24h=50)
    for kind, thr in (("score_above", 70), ("attention_spike", 75)):
        assert evaluate_rule(rule(kind, thr), ctx(cur_bad, good_prev, attention=att)) is None
    assert evaluate_rule(rule("score_change", 10), ctx(cur_bad, None), reference=good_prev) is None
    # …and attention compares with the last sound heat (80: no crossing), not the degraded 30.
    assert evaluate_rule(rule("attention_spike", 75, last=NOW - timedelta(days=1)),
                         ctx(record(60, heat=95), history=(bad_prev, good_prev), attention=att)) is None


def test_webhook_payload_never_pings_anyone():
    """Provider text (headlines, firms) lands in alerts: '@everyone' or '<!channel>' must stay inert."""
    from app.schemas import AlertEvent

    ev = AlertEvent(id=1, rule_id=1, ticker="NVDA", at=NOW, title="NVDA: 1 new story",
                    detail="• @everyone Nvidia <!channel> beats & raises <https://x.io|click> @HERE")
    body = alerts.webhook_payload(ev)
    assert body["allowed_mentions"] == {"parse": []}  # Discord: no mention is parsed at all
    assert "@everyone" not in body["content"] and "@\u200beveryone" in body["content"]
    assert "@HERE" not in body["content"]
    assert "<!channel>" not in body["text"] and "&lt;!channel&gt;" in body["text"] and "&amp; raises" in body["text"]
    assert body["text"].startswith("*NVDA: 1 new story*")
