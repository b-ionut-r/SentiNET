"""Fixtures for platform service tests: isolated SQLite store + fake provider world."""
from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import pytest

from app.config import settings
from app.services import analyzer
from app.storage import db
from tests.services.fakes import FakeWorld


@pytest.fixture
def store(tmp_path, monkeypatch) -> Iterator[ModuleType]:
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "sentinet.db"))
    db.init_db()
    yield db
    db.close_db()


@pytest.fixture
def world(monkeypatch, store) -> Iterator[FakeWorld]:
    analyzer.reset()
    monkeypatch.setattr(settings, "alert_webhook_url", "")
    monkeypatch.setattr(analyzer, "MIN_REFRESH_SECONDS", 0.0)  # tests refresh back-to-back
    yield FakeWorld().install(monkeypatch)
    analyzer.reset()
