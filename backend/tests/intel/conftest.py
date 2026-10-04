"""Intel test isolation: the shared HTTP client is bound to one event loop."""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.core.http import close_client


@pytest.fixture(autouse=True)
async def _fresh_http_client() -> AsyncIterator[None]:
    # pytest-asyncio gives each test its own loop; a pooled client from a previous
    # test would fail with "Event loop is closed". Close it within the test's loop.
    yield
    await close_client()
