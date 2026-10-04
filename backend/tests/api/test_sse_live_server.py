"""SSE over a real uvicorn server: progress must arrive *while* the analysis runs.

TestClient buffers whole responses, so it cannot catch buffering (e.g. a
compression middleware swallowing chunks). This test serves the app on a
loopback port and timestamps each event as the client receives it.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn

from app.config import settings
from tests.services.fakes import FakeSource, FakeWorld

SLOW = 1.2  # seconds the slow source holds the run open


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(world: FakeWorld, monkeypatch) -> Iterator[str]:
    monkeypatch.setattr(settings, "monitor_enabled", False)
    world.sources = [
        (FakeSource("fast_news"), "enabled"),
        (FakeSource("slow_social", "social", delay=SLOW), "enabled"),
    ]
    from app.main import create_app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_progress_streams_before_result(live_server: str):
    arrivals: list[tuple[float, str, dict]] = []
    t0 = time.monotonic()
    with httpx.Client(trust_env=False, timeout=20) as client, client.stream(
        "GET", f"{live_server}/api/analyze/NVDA/stream", headers={"Accept-Encoding": "gzip"}
    ) as r:
        assert r.status_code == 200
        assert "content-encoding" not in r.headers
        event = "message"
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                arrivals.append((time.monotonic() - t0, event, json.loads(line.split(":", 1)[1])))
    names = [e for _, e, _ in arrivals]
    assert names[-1] == "result"
    result_at = arrivals[-1][0]
    fast_done = next(t for t, e, d in arrivals if e == "progress" and d["key"] == "fast_news" and d["status"] == "ok")
    first = arrivals[0][0]
    assert result_at >= SLOW
    # Fast events reached the client long before the slow source let the run finish.
    assert first < SLOW / 2 and fast_done < SLOW / 2, arrivals[:5]


def test_disconnect_mid_stream_does_not_abort_the_run(live_server: str, world: FakeWorld):
    with httpx.Client(trust_env=False, timeout=20) as client:
        with client.stream("GET", f"{live_server}/api/analyze/NVDA/stream") as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    break  # got the first event; hang up while the slow source is still running
        time.sleep(SLOW + 0.5)
        body = client.get(f"{live_server}/api/analyze/NVDA").json()
    assert body["cached"] is True  # the abandoned run finished and was cached…
    assert len(world.inputs) == 1  # …exactly once
