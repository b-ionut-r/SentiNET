"""Edge hardening: body size limits, no input echo on 422, Host allow-list, lab admission."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.limits import DEFAULT_MAX_BODY, BodySizeLimitMiddleware
from app.api.web_settings import WebSettings
from app.services import lab
from app.services.errors import Busy
from tests.services.fakes import FakeWorld


# ---- 422 never echoes the rejected input ------------------------------------------------ #
def test_validation_error_does_not_echo_the_body(client: TestClient):
    marker = "SECRET-PAYLOAD-" + "x" * 5_000
    r = client.post("/api/lab/score", json={"texts": [marker] * 501})  # too many items
    assert r.status_code == 422
    assert "SECRET-PAYLOAD" not in r.text and len(r.content) < 2_000
    err = r.json()["errors"][0]
    assert set(err) <= {"type", "loc", "msg"} and err["loc"] == ["body", "texts"]


def test_validation_error_lists_at_most_ten_problems(client: TestClient):
    r = client.post("/api/lab/score", json={"texts": list(range(40))})  # 40 non-strings
    body = r.json()
    assert r.status_code == 422 and len(body["errors"]) == 10
    assert body["detail"].endswith("… and 30 more") and "texts.0: Input should be a valid string" in body["detail"]


# ---- request body size ---------------------------------------------------------------- #
def test_oversized_body_is_413_before_parsing(client: TestClient):
    body = json.dumps({"ticker": "NVDA", "pad": "x" * (DEFAULT_MAX_BODY + 10)})
    r = client.post("/api/watchlist", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413 and "64 KB" in r.json()["detail"]


def test_streamed_body_without_length_is_counted(client: TestClient):
    def chunks():
        for _ in range(100):
            yield b"x" * 1024  # 100 KB, chunked: no Content-Length

    r = client.post("/api/alerts", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_lab_accepts_its_full_character_budget(client: TestClient):
    texts = ["Nvidia beats estimates. " + "y" * (lab.MAX_TEXT_CHARS - 30)] * (lab.MAX_TOTAL_CHARS // lab.MAX_TEXT_CHARS)
    raw = json.dumps({"texts": texts})  # ~250 KB: far over the default limit, within the lab's
    assert len(raw) > DEFAULT_MAX_BODY
    r = client.post("/api/lab/score", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 200 and r.json()["summary"]["n"] == len(texts)


def test_lab_over_character_budget_is_400(client: TestClient):
    texts = ["z" * lab.MAX_TEXT_CHARS] * (lab.MAX_TOTAL_CHARS // lab.MAX_TEXT_CHARS + 1)
    r = client.post("/api/lab/score", json={"texts": texts})
    assert r.status_code == 400 and "250,000" in r.json()["detail"]


def test_lab_busy_is_503_with_retry_after(client: TestClient, monkeypatch):
    async def busy(_req):
        raise Busy("The sentiment lab is busy scoring other requests; retry in a few seconds.")

    monkeypatch.setattr(lab, "score_texts", busy)
    r = client.post("/api/lab/score", json={"texts": ["hello"]})
    assert r.status_code == 503 and r.headers["retry-after"] == "5" and "busy" in r.json()["detail"]


async def test_limit_middleware_per_path_and_trailing_slash():
    mw = BodySizeLimitMiddleware(app=None, default=100, overrides={"/api/lab/score": 1000})  # type: ignore[arg-type]
    assert mw.limit_for("/api/lab/score") == 1000 and mw.limit_for("/api/lab/score/") == 1000
    assert mw.limit_for("/api/watchlist") == 100 and mw.limit_for("/") == 100


# ---- Host allow-list (DNS rebinding, LAN exposure) ---------------------------------------- #
def test_foreign_host_header_is_rejected(world: FakeWorld, monkeypatch):
    from app.config import settings
    from app.main import create_app

    monkeypatch.setattr(settings, "monitor_enabled", False)
    with TestClient(create_app(), base_url="http://evil.example") as c:
        assert c.get("/api/health").status_code == 400
        assert c.post("/api/watchlist", json={"ticker": "NVDA"}).status_code == 400
    for host in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000"):
        with TestClient(create_app(), base_url=f"http://{host}") as c:
            assert c.get("/api/health").status_code == 200, host


@pytest.mark.parametrize("raw, expected", [
    ("", ["localhost", "127.0.0.1", "[::1]"]),
    ("sentinet.lan, *.example.com", ["sentinet.lan", "*.example.com"]),
    ("*", ["*"]),
    ("bad*host,ok.lan", ["ok.lan"]),  # malformed wildcard dropped, not a startup crash
    ("*bad", ["localhost", "127.0.0.1", "[::1]"]),
])
def test_allowed_hosts_setting(raw: str, expected: list[str]):
    assert WebSettings(allowed_hosts=raw).allowed_host_list == expected
