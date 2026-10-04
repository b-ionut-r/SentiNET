"""SQLite store: schema, snapshots, watchlist (Δ baseline + sparkline), alert rules/events."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.config import settings
from app.schemas import (
    AlertRuleIn,
    AnalystAction,
    AnalystView,
    AttentionView,
    Narrative,
)
from app.storage import db
from tests.services.fakes import FakeWorld

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "nested" / "dir" / "s.db"))
    db.init_db()
    yield db
    db.close_db()


def analysis_at(at: datetime, score: int = 60, ticker: str = "NVDA", **extra):
    """A minimal valid Analysis via the fake builder (no network)."""
    from app.analytics.inputs import AnalysisInputs
    from app.sources.base import CompanyRef

    world = FakeWorld(score=score)
    inputs = AnalysisInputs(company=CompanyRef(ticker=ticker, name=ticker, short_name=ticker), now=at,
                            engine_name="sentinel", quote=world.intel["quote"])
    a = world.build(inputs)
    a = a.model_copy(update={"news": a.news.model_copy(update={"n": 12}),
                             "social": a.social.model_copy(update={"n": 5})})
    return a.model_copy(update=extra)


def test_init_creates_parent_dirs_and_wal(store, tmp_path):
    path = tmp_path / "nested" / "dir" / "s.db"
    assert path.exists()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"snapshots", "watchlist", "alert_rules", "alert_events"} <= tables


def test_reopen_is_idempotent(store):
    db.init_db()  # migrations must not fail on an existing schema
    db.init_db()


def test_time_roundtrip_is_utc_and_sortable():
    naive = datetime(2026, 1, 2, 3, 4, 5)  # noqa: DTZ001 - the point: naive input is taken as UTC
    assert db.to_db_time(naive).endswith("+00:00")
    est = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC).astimezone(ZoneInfo("America/New_York"))
    assert db.from_db_time(db.to_db_time(est)) == est
    stamps = [db.to_db_time(T0 + timedelta(seconds=s)) for s in (1, 10, 100, 1000, 0.5)]
    assert sorted(stamps) == [db.to_db_time(T0 + timedelta(seconds=s)) for s in (0.5, 1, 10, 100, 1000)]


async def test_snapshot_roundtrip(store):
    a = analysis_at(
        T0, score=71,
        narratives=[Narrative(id="n1", headline="Nvidia faces $1.05B lawsuit", count=14)],
        attention=AttentionView(heat=82, label="Spiking"),
        analysts=AnalystView(actions=[AnalystAction(date=T0, firm="Morgan Stanley", action="up",
                                                    to_grade="Overweight", price_target=250.0)]),
    )
    rec = await db.save_snapshot(a)
    snap = rec.snapshot
    assert (snap.ticker, snap.at, snap.sentinel_score, snap.label) == ("NVDA", T0, 71, "bullish")
    assert snap.price == 180.0
    assert snap.narratives == ["Nvidia faces $1.05B lawsuit"]
    assert snap.news_score == pytest.approx(a.news.score)
    assert rec.attention_heat == 82
    assert rec.analyst_keys == ("2026-10-01|morgan stanley|up|overweight|250.00",)
    assert rec.verdict is not None and rec.verdict.score == 71


async def test_empty_news_and_social_scores_are_null(store):
    a = analysis_at(T0)
    a = a.model_copy(update={"news": a.news.model_copy(update={"n": 0}),
                             "social": a.social.model_copy(update={"n": 0})})
    snap = (await db.save_snapshot(a)).snapshot
    assert snap.news_score is None and snap.social_score is None


async def test_latest_before_and_listing(store):
    for h in range(5):
        await db.save_snapshot(analysis_at(T0 + timedelta(hours=h), score=50 + h))
    await db.save_snapshot(analysis_at(T0, ticker="AAPL", score=10))

    latest = await db.latest_snapshot("NVDA")
    assert latest.sentinel_score == 54
    older = await db.latest_snapshot("NVDA", before=T0 + timedelta(hours=2, minutes=30))
    assert older.sentinel_score == 52
    assert await db.latest_snapshot("NVDA", before=T0 - timedelta(minutes=1)) is None
    assert await db.latest_snapshot("MSFT") is None

    listed = await db.list_snapshots("NVDA", limit=3)
    assert [s.sentinel_score for s in listed] == [54, 53, 52]  # newest first
    since = await db.list_snapshots("NVDA", limit=100, since=T0 + timedelta(hours=3))
    assert [s.sentinel_score for s in since] == [54, 53]

    rec = await db.latest_record("NVDA")
    prev = await db.latest_record("NVDA", before_id=rec.id)
    assert prev.score == 53


async def test_prune(store):
    old = datetime.now(UTC) - timedelta(days=500)
    await db.save_snapshot(analysis_at(old))
    await db.save_snapshot(analysis_at(datetime.now(UTC)))
    assert await db.prune_snapshots(keep_days=400) == 1
    assert len(await db.list_snapshots("NVDA", 10)) == 1


async def test_watchlist_add_remove_idempotent(store):
    assert await db.add_watch("NVDA", "Nvidia") is True
    assert await db.add_watch("NVDA", "Nvidia") is False
    await db.add_watch("AAPL")
    assert await db.watch_tickers() == ["NVDA", "AAPL"]
    items = await db.watch_items()
    assert [i.ticker for i in items] == ["NVDA", "AAPL"]
    assert items[0].name == "Nvidia" and items[0].last is None and items[0].spark == []
    assert await db.remove_watch("NVDA") is True
    assert await db.remove_watch("NVDA") is False
    assert await db.watch_tickers() == ["AAPL"]


async def test_watch_item_baseline_prefers_12h_old_snapshot(store):
    await db.add_watch("NVDA")
    for h, score in ((0, 40), (6, 45), (13, 50), (24, 58), (25, 60)):
        await db.save_snapshot(analysis_at(T0 + timedelta(hours=h), score=score))
    (item,) = await db.watch_items()
    assert item.last.sentinel_score == 60
    assert item.previous.sentinel_score == 50  # latest snapshot ≥ 12h before the last one
    assert item.spark == [40, 45, 50, 58, 60]


async def test_watch_item_baseline_falls_back_to_oldest(store):
    await db.add_watch("NVDA")
    for m, score in ((0, 40), (30, 45), (60, 52)):
        await db.save_snapshot(analysis_at(T0 + timedelta(minutes=m), score=score))
    (item,) = await db.watch_items()
    assert item.previous.sentinel_score == 40  # "since added"


def test_spark_downsamples_in_time():
    pts = [(T0 + timedelta(hours=i), i) for i in range(300)]
    spark = db._spark(pts, n=30)
    assert len(spark) == 30
    assert spark[0] < spark[-1] and spark[-1] == 299  # oldest → newest, last bin keeps the latest
    assert spark == sorted(spark)


async def test_alert_rules_and_events(store):
    r1 = await db.create_rule(AlertRuleIn(ticker="NVDA", kind="score_above", threshold=70))
    r2 = await db.create_rule(AlertRuleIn(ticker="AAPL", kind="analyst_action"))
    assert r1.id != r2.id and r1.enabled and r1.last_triggered_at is None
    assert [r.id for r in await db.list_rules()] == [r1.id, r2.id]
    assert [r.id for r in await db.list_rules(ticker="NVDA", enabled_only=True)] == [r1.id]
    assert await db.alert_tickers() == ["AAPL", "NVDA"]

    await db.mark_rule_triggered(r1.id, T0)
    assert (await db.list_rules(ticker="NVDA"))[0].last_triggered_at == T0

    e1 = await db.add_alert_event(r1.id, "NVDA", T0, "NVDA SentiNET 72 ≥ 70", "detail")
    e2 = await db.add_alert_event(r1.id, "NVDA", T0 + timedelta(hours=1), "second", "detail")
    events = await db.list_alert_events(10)
    assert [e.id for e in events] == [e2.id, e1.id]  # newest first
    assert await db.list_alert_events(10, ticker="AAPL") == []

    assert [e.id for e in await db.undelivered_events(T0 - timedelta(days=1))] == [e1.id, e2.id]
    await db.mark_delivered([e1.id])
    assert [e.id for e in await db.undelivered_events(T0 - timedelta(days=1))] == [e2.id]

    # Deleting a rule keeps its history (rule_id becomes NULL).
    assert await db.delete_rule(r1.id) is True
    assert await db.delete_rule(r1.id) is False
    assert all(e.rule_id is None for e in await db.list_alert_events(10))


async def test_snapshot_extra_separates_unavailable_from_empty(store):
    from app.schemas import SourceReport

    news = SourceReport(key="google_news", label="Google News", kind="news", status="empty")
    down = SourceReport(key="bing_news", label="Bing News", kind="news", status="error")
    a = analysis_at(T0, analysts=None, sources=[news, down],
                    narratives=[Narrative(id="n1", headline="Nvidia unveils Rubin", count=5, score=0.31,
                                          signal_ids=[f"s{i}" for i in range(40)])])
    rec = await db.save_snapshot(a)
    assert rec.analyst_keys is None  # Yahoo failed: unknown, not "no actions"
    assert rec.news_ok is True  # a news source answered (even "nothing found" is an answer)
    outage = await db.save_snapshot(analysis_at(T0, sources=[down]))
    assert outage.news_ok is False
    (story,) = rec.stories
    assert story.headline == "Nvidia unveils Rubin" and story.score == 0.31
    assert story.ids == tuple(f"s{i}" for i in range(db.STORY_IDS))
    empty = await db.save_snapshot(analysis_at(T0 + timedelta(hours=1), analysts=AnalystView()))
    assert empty.analyst_keys == () and empty.stories == ()


async def test_nan_scores_do_not_sink_the_snapshot(store):
    a = analysis_at(T0)
    a = a.model_copy(update={"sentiment": a.sentiment.model_copy(update={"score": float("nan")}),
                             "news": a.news.model_copy(update={"score": float("inf")})})
    rec = await db.save_snapshot(a)
    assert rec.snapshot.score == 0.0 and rec.snapshot.news_score is None


async def test_prune_drops_story_ids_from_old_snapshots(store):
    narr = [Narrative(id="n1", headline="Old story", count=5, signal_ids=["a", "b"])]
    await db.save_snapshot(analysis_at(datetime.now(UTC) - timedelta(days=10), narratives=narr))
    await db.save_snapshot(analysis_at(datetime.now(UTC), narratives=narr))
    await db.prune_snapshots(keep_days=400)
    old, new = sorted([await db.latest_record("NVDA", before=datetime.now(UTC) - timedelta(days=5)),
                       await db.latest_record("NVDA")], key=lambda r: r.at)
    assert old.stories[0].ids == () and old.stories[0].headline == "Old story"  # headline kept
    assert new.stories[0].ids == ("a", "b")


def test_migrates_v1_database(tmp_path):
    """A v1 store (no claim column) opens, gains the column and keeps its rows."""
    path = tmp_path / "v1.db"
    conn = sqlite3.connect(path)
    conn.executescript(db._SCHEMA.replace("    delivered  INTEGER NOT NULL DEFAULT 0,  -- 0 pending · 1 delivered · 2 "
                                          "claimed (POST in flight)\n    claimed_at TEXT\n",
                                          "    delivered  INTEGER NOT NULL DEFAULT 0\n"))
    conn.execute("INSERT INTO alert_events (ticker, at, title, detail) VALUES ('NVDA', ?, 't', 'd')",
                 (db.to_db_time(T0),))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    assert "claimed_at" not in {r[1] for r in conn.execute("PRAGMA table_info(alert_events)")}
    conn.close()
    store = db.Database(str(path))
    cols = store.call(lambda c: {r[1] for r in c.execute("PRAGMA table_info(alert_events)")})
    assert "claimed_at" in cols
    assert store.call(lambda c: c.execute("PRAGMA user_version").fetchone()[0]) == db.SCHEMA_VERSION
    assert store.call(lambda c: c.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0]) == 1
    store.close()
