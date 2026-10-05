"""Intel test isolation: per-test HTTP client, provider state and disk caches."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.core.http import close_client
from app.intel import attention, gdelt
from app.intel.diskcache import DiskCache


@pytest.fixture(autouse=True)
def _private_disk_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Never read or write the real cache next to the database: each test starts cold.
    monkeypatch.setattr(gdelt, "_disk", DiskCache("gdelt", root=tmp_path / "cache"))
    monkeypatch.setattr(attention, "_disk", DiskCache("wikipedia", root=tmp_path / "cache"))


@pytest.fixture(autouse=True)
async def _fresh_http_client() -> AsyncIterator[None]:
    # pytest-asyncio gives each test its own loop; a pooled client from a previous
    # test would fail with "Event loop is closed". Close it within the test's loop.
    gdelt.reset_state()
    attention.reset_state()
    yield
    await gdelt.drain()
    gdelt.reset_state()
    attention.reset_state()
    await close_client()
