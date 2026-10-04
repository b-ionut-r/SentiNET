"""SQLite persistence: analysis snapshots, watchlist, alert rules and alert events.

Stdlib `sqlite3` in WAL mode behind a single connection guarded by a lock.
Every public coroutine runs its blocking work in a worker thread
(`asyncio.to_thread`), so the event loop never stalls on disk I/O. The sync
`_…` helpers take the connection explicitly and are what the tests exercise.

Timestamps are stored as fixed-format UTC ISO-8601 strings, which sort
lexicographically in time order.

Snapshot `extra` keeps what alert rules compare against, distinguishing
*unavailable* (provider failed: `None`) from *empty* (answered, nothing there),
so a degraded run never makes old facts look new on the next run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from app.config import settings
from app.schemas import (
    AlertEvent,
    AlertRule,
    AlertRuleIn,
    Analysis,
    Snapshot,
    Verdict,
    WatchItem,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

SCHEMA_VERSION = 2
SNAPSHOT_NARRATIVES = 12  # narrative headlines kept per snapshot (for "is new?" checks)
STORY_IDS = 25  # member signal ids kept per narrative (story identity across runs)
STORY_RETENTION = timedelta(days=7)  # member ids are pruned from older snapshots
CLAIM_STALE = timedelta(minutes=5)  # an in-flight webhook claim older than this is abandoned
SPARK_POINTS = 30
SPARK_WINDOW = timedelta(days=30)
WATCH_BASELINE_GAP = timedelta(hours=12)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    at              TEXT    NOT NULL,
    sentinel_score  INTEGER NOT NULL,
    score           REAL    NOT NULL,
    label           TEXT    NOT NULL,
    n_signals       INTEGER NOT NULL,
    price           REAL,
    news_score      REAL,
    social_score    REAL,
    attention_heat  INTEGER,
    narratives      TEXT    NOT NULL DEFAULT '[]',
    verdict         TEXT,
    extra           TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_snapshots_ticker_at ON snapshots (ticker, at);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker    TEXT PRIMARY KEY,
    added_at  TEXT NOT NULL,
    name      TEXT
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT    NOT NULL,
    kind               TEXT    NOT NULL,
    threshold          REAL,
    enabled            INTEGER NOT NULL DEFAULT 1,
    created_at         TEXT    NOT NULL,
    last_triggered_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_alert_rules_ticker ON alert_rules (ticker);

CREATE TABLE IF NOT EXISTS alert_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id    INTEGER REFERENCES alert_rules (id) ON DELETE SET NULL,
    ticker     TEXT    NOT NULL,
    at         TEXT    NOT NULL,
    title      TEXT    NOT NULL,
    detail     TEXT    NOT NULL,
    delivered  INTEGER NOT NULL DEFAULT 0,  -- 0 pending · 1 delivered · 2 claimed (POST in flight)
    claimed_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_alert_events_at ON alert_events (at);
"""

# Incremental migrations: target version -> statements.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: ("ALTER TABLE alert_events ADD COLUMN claimed_at TEXT",),
}

# Webhook delivery states (alert_events.delivered).
PENDING, DELIVERED, CLAIMED = 0, 1, 2


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def utcnow() -> datetime:
    return datetime.now(UTC)


def to_db_time(dt: datetime) -> str:
    """Fixed-width UTC ISO string (naive datetimes are taken to be UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def from_db_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StoryRef:
    """What identifies a narrative across runs: its headline, member ids and tone."""

    headline: str
    ids: tuple[str, ...] = ()
    score: float = 0.0


@dataclass(frozen=True)
class SnapshotRecord:
    """A stored snapshot plus the extra state alert rules compare against.

    `analyst_keys` is None when analyst data was unavailable for that run (as
    opposed to an empty tuple: "no recent actions"); `news_ok` is False when no
    news source answered (so an empty narrative list says nothing) and None for
    rows written before it was recorded.
    """

    id: int
    snapshot: Snapshot
    attention_heat: int | None = None
    analyst_keys: tuple[str, ...] | None = None
    verdict: Verdict | None = None
    stories: tuple[StoryRef, ...] = ()
    news_ok: bool | None = None

    @property
    def at(self) -> datetime:
        return self.snapshot.at

    @property
    def score(self) -> int:
        return self.snapshot.sentinel_score


def analyst_action_key(date: datetime, firm: str, action: str, to_grade: str | None,
                       price_target: float | None) -> str:
    """Stable identity of an analyst action, used to spot new ones between snapshots."""
    pt = f"{price_target:.2f}" if price_target is not None else ""
    return f"{date.date().isoformat()}|{firm.strip().lower()}|{action}|{(to_grade or '').lower()}|{pt}"


def _finite(value: float | None, default: float | None = None) -> float | None:
    """`value` as a float, or `default` when missing / NaN / infinite (SQLite would store NULL)."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _stories(raw: Any) -> tuple[StoryRef, ...]:
    out = []
    for item in raw if isinstance(raw, list) else ():
        if isinstance(item, dict) and isinstance(item.get("h"), str):
            out.append(StoryRef(headline=item["h"], ids=tuple(str(i) for i in item.get("ids") or ()),
                                score=_finite(item.get("s"), 0.0) or 0.0))
    return tuple(out)


def _row_to_record(row: sqlite3.Row) -> SnapshotRecord:
    extra = json.loads(row["extra"] or "{}")
    verdict = None
    if row["verdict"]:
        try:
            verdict = Verdict.model_validate_json(row["verdict"])
        except ValueError:  # schema drift in an old row: keep the numbers, drop the blob
            verdict = None
    snap = Snapshot(
        ticker=row["ticker"],
        at=from_db_time(row["at"]),
        sentinel_score=row["sentinel_score"],
        score=row["score"],
        label=row["label"],
        n_signals=row["n_signals"],
        price=row["price"],
        news_score=row["news_score"],
        social_score=row["social_score"],
        narratives=json.loads(row["narratives"] or "[]"),
    )
    keys = extra.get("analyst_keys")
    headlines = snap.narratives
    stories = _stories(extra.get("stories")) or tuple(StoryRef(headline=h) for h in headlines)
    return SnapshotRecord(
        id=row["id"],
        snapshot=snap,
        attention_heat=row["attention_heat"],
        analyst_keys=tuple(keys) if isinstance(keys, list) else None,
        verdict=verdict,
        stories=stories,
        news_ok=extra.get("news_ok") if isinstance(extra.get("news_ok"), bool) else None,
    )


# --------------------------------------------------------------------------- #
# Connection management
# --------------------------------------------------------------------------- #
class Database:
    """One SQLite connection shared across worker threads, serialized by a lock."""

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
            self._migrate()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        with self._conn:
            if version == 0:  # fresh database: the current schema in one go
                self._conn.executescript(_SCHEMA)
            else:
                for target in range(version + 1, SCHEMA_VERSION + 1):
                    for stmt in _MIGRATIONS.get(target, ()):
                        self._conn.execute(stmt)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def call(self, fn: Callable[..., T], *args: Any) -> T:
        """Run `fn(conn, *args)` under the lock (blocking)."""
        with self._lock:
            return fn(self._conn, *args)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_db: Database | None = None
_db_guard = threading.Lock()


def get_db() -> Database:
    """The process-wide database (opened lazily at `settings.database_path`)."""
    global _db
    with _db_guard:
        if _db is None:
            _db = Database(settings.database_path)
        return _db


def init_db(path: str | None = None) -> Database:
    """(Re)open the database; `path` overrides `settings.database_path`."""
    global _db
    with _db_guard:
        if _db is not None:
            _db.close()
        _db = Database(path or settings.database_path)
        return _db


def close_db() -> None:
    global _db
    with _db_guard:
        if _db is not None:
            _db.close()
            _db = None


async def _run(fn: Callable[..., T], *args: Any) -> T:
    db = get_db()
    return await asyncio.to_thread(db.call, fn, *args)


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
def _snapshot_extra(analysis: Analysis) -> dict[str, Any]:
    """Alert-rule state: analyst action keys, narrative identities, news availability."""
    keys = None  # analysts unavailable this run: not "no actions"
    if analysis.analysts is not None:
        keys = [analyst_action_key(a.date, a.firm, a.action, a.to_grade, a.price_target)
                for a in analysis.analysts.actions]
    stories = [
        {"h": n.headline, "ids": n.signal_ids[:STORY_IDS], "s": round(_finite(n.score, 0.0) or 0.0, 3)}
        for n in analysis.narratives[:SNAPSHOT_NARRATIVES]
    ]
    news_ok = any(s.kind in ("news", "analysis") and s.status in ("ok", "empty") for s in analysis.sources)
    return {"analyst_keys": keys, "stories": stories, "news_ok": news_ok}


def _insert_snapshot(conn: sqlite3.Connection, analysis: Analysis) -> SnapshotRecord:
    verdict = analysis.verdict
    narratives = [n.headline for n in analysis.narratives[:SNAPSHOT_NARRATIVES]]
    at = analysis.generated_at
    with conn:
        cur = conn.execute(
            """INSERT INTO snapshots (ticker, at, sentinel_score, score, label, n_signals, price,
                   news_score, social_score, attention_heat, narratives, verdict, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis.ticker,
                to_db_time(at),
                int(verdict.score),
                _finite(analysis.sentiment.score, 0.0),  # NOT NULL: a NaN must not sink the snapshot
                verdict.stance,
                int(analysis.sentiment.n),
                _finite(analysis.quote.price) if analysis.quote else None,
                _finite(analysis.news.score) if analysis.news.n else None,
                _finite(analysis.social.score) if analysis.social.n else None,
                analysis.attention.heat if analysis.attention else None,
                json.dumps(narratives),
                verdict.model_dump_json(),
                json.dumps(_snapshot_extra(analysis)),
            ),
        )
    row = conn.execute("SELECT * FROM snapshots WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _row_to_record(row)


def _latest_record(conn: sqlite3.Connection, ticker: str, before: datetime | None,
                   before_id: int | None) -> SnapshotRecord | None:
    sql, args = "SELECT * FROM snapshots WHERE ticker = ?", [ticker]
    if before is not None:
        sql += " AND at <= ?"
        args.append(to_db_time(before))
    if before_id is not None:
        sql += " AND id < ?"
        args.append(before_id)
    row = conn.execute(sql + " ORDER BY at DESC, id DESC LIMIT 1", args).fetchone()
    return _row_to_record(row) if row else None


def _oldest_record_since(conn: sqlite3.Connection, ticker: str, since: datetime,
                         exclude_id: int | None) -> SnapshotRecord | None:
    row = conn.execute(
        "SELECT * FROM snapshots WHERE ticker = ? AND at >= ? AND id != ? ORDER BY at ASC, id ASC LIMIT 1",
        (ticker, to_db_time(since), exclude_id if exclude_id is not None else -1),
    ).fetchone()
    return _row_to_record(row) if row else None


def _records_before(conn: sqlite3.Connection, ticker: str, before_id: int, since: datetime,
                    limit: int) -> list[SnapshotRecord]:
    rows = conn.execute(
        "SELECT * FROM snapshots WHERE ticker = ? AND id < ? AND at >= ? ORDER BY at DESC, id DESC LIMIT ?",
        (ticker, before_id, to_db_time(since), limit),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def _list_snapshots(conn: sqlite3.Connection, ticker: str, limit: int,
                    since: datetime | None) -> list[Snapshot]:
    sql, args = "SELECT * FROM snapshots WHERE ticker = ?", [ticker]
    if since is not None:
        sql += " AND at >= ?"
        args.append(to_db_time(since))
    sql += " ORDER BY at DESC, id DESC LIMIT ?"
    args.append(limit)
    return [_row_to_record(r).snapshot for r in conn.execute(sql, args).fetchall()]


def _prune_snapshots(conn: sqlite3.Connection, older_than: datetime, stories_before: datetime) -> int:
    """Delete snapshots older than `older_than`; drop bulky story ids from those before `stories_before`."""
    with conn:
        cur = conn.execute("DELETE FROM snapshots WHERE at < ?", (to_db_time(older_than),))
        conn.execute(
            "UPDATE snapshots SET extra = json_remove(extra, '$.stories') "
            "WHERE at < ? AND json_extract(extra, '$.stories') IS NOT NULL",
            (to_db_time(stories_before),),
        )
    return cur.rowcount


async def save_snapshot(analysis: Analysis) -> SnapshotRecord:
    """Persist the headline numbers of a fresh analysis."""
    return await _run(_insert_snapshot, analysis)


async def latest_record(ticker: str, *, before: datetime | None = None,
                        before_id: int | None = None) -> SnapshotRecord | None:
    """Newest snapshot of `ticker` taken at/before `before` and/or older than row `before_id`."""
    return await _run(_latest_record, ticker, before, before_id)


async def latest_snapshot(ticker: str, before: datetime | None = None) -> Snapshot | None:
    rec = await latest_record(ticker, before=before)
    return rec.snapshot if rec else None


async def oldest_record_since(ticker: str, since: datetime,
                              exclude_id: int | None = None) -> SnapshotRecord | None:
    return await _run(_oldest_record_since, ticker, since, exclude_id)


async def records_before(ticker: str, before_id: int, since: datetime, limit: int = 100) -> list[SnapshotRecord]:
    """Snapshots of `ticker` older than row `before_id` and taken at/after `since`, newest first."""
    return await _run(_records_before, ticker, before_id, since, limit)


async def list_snapshots(ticker: str, limit: int = 50, since: datetime | None = None) -> list[Snapshot]:
    """Newest first."""
    return await _run(_list_snapshots, ticker, limit, since)


async def prune_snapshots(keep_days: int = 400) -> int:
    now = utcnow()
    return await _run(_prune_snapshots, now - timedelta(days=keep_days), now - STORY_RETENTION)


# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #
def _spark(points: list[tuple[datetime, int]], n: int = SPARK_POINTS) -> list[int]:
    """Downsample (time, score) pairs (oldest first) to ≤ n evenly spaced-in-time values."""
    if len(points) <= n:
        return [s for _, s in points]
    start, end = points[0][0], points[-1][0]
    span = (end - start).total_seconds() or 1.0
    bins: dict[int, int] = {}
    for t, s in points:
        idx = min(n - 1, int((t - start).total_seconds() / span * n))
        bins[idx] = s  # last value in each time bin
    return [bins[i] for i in sorted(bins)]


def _watch_item(conn: sqlite3.Connection, row: sqlite3.Row) -> WatchItem:
    ticker = row["ticker"]
    last = _latest_record(conn, ticker, None, None)
    previous = None
    spark: list[int] = []
    if last is not None:
        # Δ baseline: the latest snapshot ≥ 12h before `last` (a "daily" change);
        # with less history, the oldest snapshot we have ("since added").
        prev = _latest_record(conn, ticker, last.at - WATCH_BASELINE_GAP, None)
        if prev is None:
            row0 = conn.execute(
                "SELECT * FROM snapshots WHERE ticker = ? AND id != ? ORDER BY at ASC, id ASC LIMIT 1",
                (ticker, last.id),
            ).fetchone()
            prev = _row_to_record(row0) if row0 else None
        previous = prev.snapshot if prev else None
        rows = conn.execute(
            "SELECT at, sentinel_score FROM snapshots WHERE ticker = ? AND at >= ? ORDER BY at ASC, id ASC",
            (ticker, to_db_time(last.at - SPARK_WINDOW)),
        ).fetchall()
        spark = _spark([(from_db_time(r["at"]), r["sentinel_score"]) for r in rows])
    return WatchItem(
        ticker=ticker,
        added_at=from_db_time(row["added_at"]),
        name=row["name"],
        last=last.snapshot if last else None,
        previous=previous,
        spark=spark,
    )


def _watch_items(conn: sqlite3.Connection) -> list[WatchItem]:
    rows = conn.execute("SELECT * FROM watchlist ORDER BY added_at ASC").fetchall()
    return [_watch_item(conn, r) for r in rows]


def _add_watch(conn: sqlite3.Connection, ticker: str, name: str | None, at: datetime) -> bool:
    with conn:
        cur = conn.execute(
            "INSERT INTO watchlist (ticker, added_at, name) VALUES (?, ?, ?) ON CONFLICT(ticker) DO NOTHING",
            (ticker, to_db_time(at), name),
        )
    return cur.rowcount > 0


def _remove_watch(conn: sqlite3.Connection, ticker: str) -> bool:
    with conn:
        cur = conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
    return cur.rowcount > 0


def _watch_tickers(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT ticker FROM watchlist ORDER BY added_at ASC")]


async def watch_items() -> list[WatchItem]:
    """Watchlist rows with latest snapshot, Δ baseline and a score sparkline."""
    return await _run(_watch_items)


async def add_watch(ticker: str, name: str | None = None) -> bool:
    """True when newly added (idempotent)."""
    return await _run(_add_watch, ticker, name, utcnow())


async def remove_watch(ticker: str) -> bool:
    return await _run(_remove_watch, ticker)


async def watch_tickers() -> list[str]:
    return await _run(_watch_tickers)


# --------------------------------------------------------------------------- #
# Alert rules & events
# --------------------------------------------------------------------------- #
def _rule(row: sqlite3.Row) -> AlertRule:
    return AlertRule(
        id=row["id"],
        ticker=row["ticker"],
        kind=row["kind"],
        threshold=row["threshold"],
        enabled=bool(row["enabled"]),
        created_at=from_db_time(row["created_at"]),
        last_triggered_at=from_db_time(row["last_triggered_at"]),
    )


def _event(row: sqlite3.Row) -> AlertEvent:
    return AlertEvent(
        id=row["id"],
        rule_id=row["rule_id"],
        ticker=row["ticker"],
        at=from_db_time(row["at"]),
        title=row["title"],
        detail=row["detail"],
        delivered=row["delivered"] == DELIVERED,
    )


def _create_rule(conn: sqlite3.Connection, rule: AlertRuleIn, at: datetime) -> AlertRule:
    with conn:
        cur = conn.execute(
            "INSERT INTO alert_rules (ticker, kind, threshold, enabled, created_at) VALUES (?, ?, ?, 1, ?)",
            (rule.ticker, rule.kind, rule.threshold, to_db_time(at)),
        )
    return _rule(conn.execute("SELECT * FROM alert_rules WHERE id = ?", (cur.lastrowid,)).fetchone())


def _list_rules(conn: sqlite3.Connection, ticker: str | None, enabled_only: bool) -> list[AlertRule]:
    sql, args = "SELECT * FROM alert_rules WHERE 1 = 1", []
    if ticker is not None:
        sql += " AND ticker = ?"
        args.append(ticker)
    if enabled_only:
        sql += " AND enabled = 1"
    return [_rule(r) for r in conn.execute(sql + " ORDER BY id ASC", args).fetchall()]


def _delete_rule(conn: sqlite3.Connection, rule_id: int) -> bool:
    with conn:
        cur = conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
    return cur.rowcount > 0


def _mark_triggered(conn: sqlite3.Connection, rule_id: int, at: datetime) -> None:
    with conn:
        conn.execute("UPDATE alert_rules SET last_triggered_at = ? WHERE id = ?", (to_db_time(at), rule_id))


def _alert_tickers(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM alert_rules WHERE enabled = 1 ORDER BY ticker")]


def _add_event(conn: sqlite3.Connection, rule_id: int | None, ticker: str, at: datetime,
               title: str, detail: str) -> AlertEvent:
    with conn:
        cur = conn.execute(
            "INSERT INTO alert_events (rule_id, ticker, at, title, detail, delivered) VALUES (?, ?, ?, ?, ?, 0)",
            (rule_id, ticker, to_db_time(at), title, detail),
        )
    return _event(conn.execute("SELECT * FROM alert_events WHERE id = ?", (cur.lastrowid,)).fetchone())


def _list_events(conn: sqlite3.Connection, limit: int, ticker: str | None) -> list[AlertEvent]:
    sql, args = "SELECT * FROM alert_events", []
    if ticker is not None:
        sql += " WHERE ticker = ?"
        args.append(ticker)
    sql += " ORDER BY at DESC, id DESC LIMIT ?"
    args.append(limit)
    return [_event(r) for r in conn.execute(sql, args).fetchall()]


_CLAIMABLE = f"(delivered = {PENDING} OR (delivered = {CLAIMED} AND claimed_at < ?))"


def _undelivered(conn: sqlite3.Connection, since: datetime, stale_before: datetime) -> list[AlertEvent]:
    rows = conn.execute(
        f"SELECT * FROM alert_events WHERE at >= ? AND {_CLAIMABLE} ORDER BY at ASC, id ASC",
        (to_db_time(since), to_db_time(stale_before)),
    ).fetchall()
    return [_event(r) for r in rows]


def _claim_events(conn: sqlite3.Connection, event_ids: list[int], now: datetime) -> list[int]:
    """Atomically mark pending (or abandoned) events as in flight; returns the ids actually claimed.

    The conditional UPDATE runs under SQLite's write lock, so two deliverers
    (the post-analysis task and the monitor's retry, or two processes) can
    never both claim — and therefore never both POST — the same event.
    """
    claimed = []
    stale_before = to_db_time(now - CLAIM_STALE)
    with conn:
        for event_id in event_ids:
            cur = conn.execute(
                f"UPDATE alert_events SET delivered = {CLAIMED}, claimed_at = ? WHERE id = ? AND {_CLAIMABLE}",
                (to_db_time(now), event_id, stale_before),
            )
            if cur.rowcount:
                claimed.append(event_id)
    return claimed


def _release_events(conn: sqlite3.Connection, event_ids: list[int], delivered: bool) -> None:
    state = DELIVERED if delivered else PENDING
    with conn:
        conn.executemany(
            f"UPDATE alert_events SET delivered = {state}, claimed_at = NULL WHERE id = ? AND delivered = {CLAIMED}",
            [(i,) for i in event_ids],
        )


def _mark_delivered(conn: sqlite3.Connection, event_ids: list[int]) -> None:
    with conn:
        conn.executemany(f"UPDATE alert_events SET delivered = {DELIVERED}, claimed_at = NULL WHERE id = ?",
                         [(i,) for i in event_ids])


async def create_rule(rule: AlertRuleIn) -> AlertRule:
    return await _run(_create_rule, rule, utcnow())


async def list_rules(ticker: str | None = None, enabled_only: bool = False) -> list[AlertRule]:
    return await _run(_list_rules, ticker, enabled_only)


async def delete_rule(rule_id: int) -> bool:
    return await _run(_delete_rule, rule_id)


async def mark_rule_triggered(rule_id: int, at: datetime) -> None:
    await _run(_mark_triggered, rule_id, at)


async def alert_tickers() -> list[str]:
    """Tickers with at least one enabled rule."""
    return await _run(_alert_tickers)


async def add_alert_event(rule_id: int | None, ticker: str, at: datetime, title: str, detail: str) -> AlertEvent:
    return await _run(_add_event, rule_id, ticker, at, title, detail)


async def list_alert_events(limit: int = 50, ticker: str | None = None) -> list[AlertEvent]:
    """Newest first."""
    return await _run(_list_events, limit, ticker)


async def undelivered_events(since: datetime) -> list[AlertEvent]:
    """Events since `since` not delivered and not currently being delivered (oldest first)."""
    return await _run(_undelivered, since, utcnow() - CLAIM_STALE)


async def claim_events(event_ids: list[int]) -> list[int]:
    """Claim events for webhook delivery; only claimed ids may be sent (see `_claim_events`)."""
    return await _run(_claim_events, event_ids, utcnow()) if event_ids else []


async def release_events(event_ids: list[int], delivered: bool) -> None:
    """End a claim: mark delivered, or return to pending for a later retry."""
    if event_ids:
        await _run(_release_events, event_ids, delivered)


async def mark_delivered(event_ids: list[int]) -> None:
    if event_ids:
        await _run(_mark_delivered, event_ids)
