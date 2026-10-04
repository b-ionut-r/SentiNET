"""API routes over the real app with fake providers (no network)."""
from __future__ import annotations

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from app.schemas import Analysis, Narrative, ProgressEvent
from tests.services.fakes import FakeSource, FakeWorld, Sentinel


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, json data) pairs, skipping comments/keep-alives."""
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith(":") or not line:
                continue
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


# ---- analysis -------------------------------------------------------------------- #
def test_analyze_json_and_cache_flag(client: TestClient, world: FakeWorld):
    r = client.get("/api/analyze/nvda")
    assert r.status_code == 200
    a = Analysis.model_validate(r.json())
    assert a.ticker == "NVDA" and a.cached is False and len(a.signals) == 5
    r2 = client.get("/api/analyze/NVDA")
    assert r2.json()["cached"] is True
    r3 = client.get("/api/analyze/NVDA", params={"refresh": "true"})
    assert r3.json()["cached"] is False and len(world.inputs) == 2


@pytest.mark.parametrize("bad", ["NV%20DA!", "$$$", "A" * 30])
def test_invalid_ticker_is_400_with_help(client: TestClient, bad: str):
    r = client.get(f"/api/analyze/{bad}")
    assert r.status_code == 400
    assert "not a valid ticker" in r.json()["detail"] and "BTC-USD" in r.json()["detail"]


def test_unknown_symbol_is_404(client: TestClient, world: FakeWorld):
    from app.sources.base import CompanyRef

    world.resolve = CompanyRef(ticker="ZZZZZ", name="ZZZZZ", short_name="ZZZZZ")
    world.sources = [(FakeSource("google_news", signals=0), "enabled")]
    for key in ("profile", "quote", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    r = client.get("/api/analyze/ZZZZZ")
    assert r.status_code == 404 and "No market data" in r.json()["detail"]


def test_synthesis_failure_is_503_json(client: TestClient, world: FakeWorld):
    world.build_error = KeyError("components")
    r = client.get("/api/analyze/NVDA")
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/json")
    assert "Synthesis failed" in r.json()["detail"]


def test_provider_failures_never_500(client: TestClient, world: FakeWorld):
    world.sources = [(FakeSource("broken", error=RuntimeError("boom")), "enabled")]
    for key in ("profile", "technicals", "analysts", "earnings", "insiders", "tone", "wiki"):
        world.intel[key] = Sentinel(exc=RuntimeError(f"{key} down"))
    r = client.get("/api/analyze/NVDA")
    assert r.status_code == 200
    statuses = {s["key"]: s["status"] for s in r.json()["sources"]}
    assert statuses == {"broken": "error"}


# ---- SSE ------------------------------------------------------------------------------ #
def test_stream_emits_progress_then_result(client: TestClient, world: FakeWorld):
    with client.stream("GET", "/api/analyze/NVDA/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("cache-control") == "no-cache"
        assert "content-encoding" not in r.headers  # never gzip a stream
        body = "".join(r.iter_text())
    events = parse_sse(body)
    names = [n for n, _ in events]
    assert names[-1] == "result" and names.count("result") == 1
    assert set(names[:-1]) == {"progress"}
    progress = [ProgressEvent.model_validate(d) for n, d in events if n == "progress"]
    keys = {p.key for p in progress}
    assert {"resolve", "google_news", "stocktwits", "quote", "tone", "synthesis", "done"} <= keys
    assert progress[0].key == "resolve" and progress[0].status == "running"
    assert progress[-1].stage == "done"
    result = Analysis.model_validate(events[-1][1])
    assert result.ticker == "NVDA" and result.cached is False


def test_stream_cached_result_is_immediate(client: TestClient, world: FakeWorld):
    client.get("/api/analyze/NVDA")
    with client.stream("GET", "/api/analyze/NVDA/stream") as r:
        events = parse_sse("".join(r.iter_text()))
    assert [n for n, _ in events] == ["progress", "result"]
    assert events[0][1]["label"] == "Served from cache"
    assert events[1][1]["cached"] is True


def test_stream_error_event(client: TestClient, world: FakeWorld):
    world.build_error = RuntimeError("bad math")
    with client.stream("GET", "/api/analyze/NVDA/stream") as r:
        assert r.status_code == 200
        events = parse_sse("".join(r.iter_text()))
    name, data = events[-1]
    assert name == "error" and data["status"] == 503 and "Synthesis failed" in data["detail"]
    assert "result" not in [n for n, _ in events]


def test_stream_invalid_ticker_is_plain_400(client: TestClient):
    r = client.get("/api/analyze/!!!/stream")
    assert r.status_code == 400 and "not a valid ticker" in r.json()["detail"]


# ---- price / history / market / search -------------------------------------------------- #
def test_price(client: TestClient, world: FakeWorld):
    r = client.get("/api/price/NVDA", params={"range": "1y"})
    assert r.status_code == 200 and r.json()["range"] == "1Y" and len(r.json()["candles"]) == 1
    assert world.calls["price"] == [("NVDA", "1Y")]
    bad = client.get("/api/price/NVDA", params={"range": "2W"})
    assert bad.status_code == 400 and "1D, 5D, 1M" in bad.json()["detail"]


def test_price_provider_failure_is_unavailable_not_500(client: TestClient):
    r = client.get("/api/price/FAIL", params={"range": "5D"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False and body["interval"] == "30m"
    assert "SUPERSECRET123" not in body["error"]


def test_history(client: TestClient, world: FakeWorld):
    client.get("/api/analyze/NVDA")  # creates a snapshot
    r = client.get("/api/history/NVDA", params={"days": 30})
    assert r.status_code == 200 and r.json()["days"] == 30
    (args,) = world.calls["build_history"]
    ticker, days, tone, closes, snapshots, _wiki, status = args
    assert (ticker, days) == ("NVDA", 30)
    assert tone.tone_7d == 1.2 and closes[0][1] == 100.0 and len(snapshots) == 1
    assert status == {"tone": "ok", "price": "ok", "snapshots": "ok", "wiki": "ok"}
    # GDELT queried with the same args as the analyzer (shared provider cache).
    assert world.calls["tone"][-1][1] == 90
    assert client.get("/api/history/NVDA", params={"days": 3}).status_code == 422


def test_market(client: TestClient, world: FakeWorld):
    r = client.get("/api/market")
    assert r.status_code == 200
    body = r.json()
    assert body["regime"] == "Risk-off: Fear" and body["fear_greed"]["score"] == 38
    assert body["crypto_fear_greed"] is None
    assert body["status"]["crypto_fear_greed"].startswith("error: RuntimeError")
    assert body["status"]["fear_greed"] == "ok"
    client.get("/api/market")
    assert len(world.calls["build_market"]) == 1  # cached


def test_search(client: TestClient, world: FakeWorld):
    r = client.get("/api/search", params={"q": " $nvd "})
    assert r.status_code == 200 and r.json()[0]["symbol"] == "NVDA"
    assert world.calls["search"][0] == ("nvd", 8)
    assert client.get("/api/search", params={"q": "boom"}).json() == []
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": "$"}).json() == []


# ---- lab --------------------------------------------------------------------------------- #
def test_lab_scores_texts(client: TestClient):
    texts = ["Nvidia beats estimates on data center demand", "Nvidia hit with lawsuit", "  ", "Plain update"]
    r = client.post("/api/lab/score", json={"texts": texts, "ticker": "nvda"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "sentinel" and len(body["results"]) == 3  # blank dropped
    first = body["results"][0]
    assert first["label"] == "bullish" and first["events"] == ["earnings_beat"] and first["relevance"] == 1.0
    assert first["drivers"] == [{"term": "beat", "impact": 0.5}]
    assert body["summary"]["n"] == 3 and body["summary"]["bullish"] == 1 and body["summary"]["bearish"] == 1
    themes = {t["theme"]: t for t in body["themes"]}
    assert themes["earnings"]["label"] == "Earnings" and themes["earnings"]["share"] == pytest.approx(1 / 3, abs=1e-3)


def test_lab_validation(client: TestClient):
    assert client.post("/api/lab/score", json={"texts": []}).status_code == 422
    r = client.post("/api/lab/score", json={"texts": ["  ", ""]})
    assert r.status_code == 400 and "non-empty" in r.json()["detail"]
    r = client.post("/api/lab/score", json={"texts": ["x" * 10_001]})
    assert r.status_code == 400 and "10,000" in r.json()["detail"]
    r = client.post("/api/lab/score", json={"texts": ["ok"], "ticker": "!!"})
    assert r.status_code == 400
    assert client.post("/api/lab/score", json={"texts": ["a"] * 501}).status_code == 422


# ---- watchlist / snapshots / alerts -------------------------------------------------------- #
def test_watchlist_crud(client: TestClient, world: FakeWorld):
    assert client.get("/api/watchlist").json() == []
    r = client.post("/api/watchlist", json={"ticker": "$nvda"})
    assert r.status_code == 200 and [w["ticker"] for w in r.json()] == ["NVDA"]
    assert r.json()[0]["name"] == "NVIDIA Corporation"
    client.post("/api/watchlist", json={"ticker": "NVDA"})  # idempotent
    client.get("/api/analyze/NVDA")
    (item,) = client.get("/api/watchlist").json()
    assert item["last"]["sentinel_score"] == world.score and item["spark"] == [world.score]
    assert client.post("/api/watchlist", json={"ticker": "??"}).status_code == 400
    assert client.delete("/api/watchlist/NVDA").json() == []
    assert client.delete("/api/watchlist/NVDA").status_code == 200


def test_snapshots(client: TestClient):
    client.get("/api/analyze/NVDA")
    client.get("/api/analyze/NVDA", params={"refresh": True})
    snaps = client.get("/api/snapshots/NVDA", params={"limit": 1}).json()
    assert len(snaps) == 1 and snaps[0]["ticker"] == "NVDA"
    assert client.get("/api/snapshots/NVDA", params={"limit": 0}).status_code == 422


def test_alerts_crud_and_events(client: TestClient, world: FakeWorld):
    r = client.post("/api/alerts", json={"ticker": "nvda", "kind": "score_above"})
    assert r.status_code == 201
    rule = r.json()
    assert rule["ticker"] == "NVDA" and rule["threshold"] == 70 and rule["enabled"] is True
    bad = client.post("/api/alerts", json={"ticker": "NVDA", "kind": "score_above", "threshold": 120})
    assert bad.status_code == 400 and "between" in bad.json()["detail"]
    assert client.post("/api/alerts", json={"ticker": "NVDA", "kind": "moon"}).status_code == 422
    assert [x["id"] for x in client.get("/api/alerts").json()] == [rule["id"]]

    client.post("/api/alerts", json={"ticker": "NVDA", "kind": "score_above", "threshold": 60})
    client.get("/api/analyze/NVDA")
    events = client.get("/api/alerts/events").json()
    assert [e["title"] for e in events] == ["NVDA SentiNET 64 ≥ 60"]
    assert client.get("/api/alerts/events", params={"ticker": "AAPL"}).json() == []

    assert client.delete(f"/api/alerts/{rule['id']}").status_code == 204
    missing = client.delete(f"/api/alerts/{rule['id']}")
    assert missing.status_code == 404 and "does not exist" in missing.json()["detail"]


# ---- export -------------------------------------------------------------------------------- #
def test_export_csv_and_json(client: TestClient, world: FakeWorld):
    world.narratives = [Narrative(id="n1", headline="x", count=2)]
    r = client.get("/api/export/nvda.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert 'filename="sentinet_NVDA_' in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0][:3] == ["id", "timestamp", "source"] and len(rows) == 6

    j = client.get("/api/export/NVDA.json")
    body = j.json()
    assert body["ticker"] == "NVDA" and body["count"] == 5 and body["verdict"]["score"] == world.score
    assert len(world.inputs) == 1  # export reused the analysis
    assert client.get("/api/export/NVDA.xlsx").status_code == 404
    assert client.get("/api/export/NVDA").status_code == 404


def test_export_neutralizes_formula_injection(client: TestClient, world: FakeWorld, monkeypatch):
    from app.api import routes_export

    assert routes_export._safe_cell("=HYPERLINK(\"http://evil\")").startswith("'=")
    assert routes_export._safe_cell("+1 Nvidia") == "'+1 Nvidia"
    assert routes_export._safe_cell("Nvidia rallies") == "Nvidia rallies"


# ---- meta / SPA / middleware --------------------------------------------------------------- #
def test_health_and_sources(client: TestClient):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["version"] == "2.0.0" and h["engine"]
    assert h["features"]["database"] is True and h["features"]["monitor"] is False
    keys = {s["key"]: s for s in h["sources"]}
    assert keys["finnhub"]["requires_key"] is True and keys["finnhub"]["configured"] is False
    assert client.get("/api/sources").json() == h["sources"]


def test_unknown_api_route_is_json_404(client: TestClient):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")


def test_gzip_for_large_json(client: TestClient):
    r = client.get("/api/analyze/NVDA", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"


def test_cors_allows_dev_origin_only(client: TestClient):
    ok = client.options("/api/health", headers={"Origin": "http://localhost:5173",
                                                "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = client.options("/api/watchlist", headers={"Origin": "https://evil.example",
                                                     "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in evil.headers


def test_spa_serving(world: FakeWorld, monkeypatch, tmp_path):
    from app.api import spa
    from app.config import settings
    from app.main import create_app

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>SentiNET</title>")
    (dist / "assets" / "app-123.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("nope")
    monkeypatch.setattr(spa, "FRONTEND_DIST", dist)
    monkeypatch.setattr(settings, "monitor_enabled", False)
    with TestClient(create_app()) as c:
        root = c.get("/")
        assert root.status_code == 200 and "SentiNET" in root.text and root.headers["cache-control"] == "no-cache"
        assert "SentiNET" in c.get("/t/NVDA").text  # client-side route
        assert c.get("/favicon.svg").text == "<svg/>"
        asset = c.get("/assets/app-123.js")
        assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
        assert c.get("/assets/missing.js").status_code == 404
        assert "nope" not in c.get("/..%2Fsecret.txt").text
        assert c.get("/api/health").json()["features"]["frontend"] is True
        assert c.get("/api/does-not-exist").status_code == 404


def test_spa_not_built_page(world: FakeWorld, monkeypatch, tmp_path):
    from app.api import spa
    from app.config import settings
    from app.main import create_app

    monkeypatch.setattr(spa, "FRONTEND_DIST", tmp_path / "missing")
    monkeypatch.setattr(settings, "monitor_enabled", False)
    with TestClient(create_app()) as c:
        r = c.get("/")
        assert r.status_code == 200 and "make build" in r.text
        assert c.get("/t/NVDA").status_code == 404


def test_validation_errors_have_readable_detail(client: TestClient):
    r = client.get("/api/history/NVDA", params={"days": 3})
    body = r.json()
    assert r.status_code == 422
    assert body["detail"] == "days: Input should be greater than or equal to 7"
    assert body["errors"][0]["loc"] == ["query", "days"]
    r = client.post("/api/alerts", json={"ticker": "NVDA", "kind": "moon"})
    assert r.status_code == 422 and r.json()["detail"].startswith("kind: Input should be")
