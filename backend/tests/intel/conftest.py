"""Intel test isolation: per-test HTTP client and provider state."""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.core.http import close_client
from app.intel import attention, gdelt


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
