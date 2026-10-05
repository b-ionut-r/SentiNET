"""Fixtures for API tests: a TestClient over the real app wired to the fake world."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from tests.services.conftest import store, world  # noqa: F401 - re-exported fixtures
from tests.services.fakes import FakeWorld


@pytest.fixture
def client(world: FakeWorld, monkeypatch) -> Iterator[TestClient]:  # noqa: F811
    monkeypatch.setattr(settings, "monitor_enabled", False)
    from app.main import create_app

    # A loopback Host: the app only serves allowed hosts (SENTINET_ALLOWED_HOSTS; "testserver" is not one).
    with TestClient(create_app(), base_url="http://localhost") as c:
        yield c
