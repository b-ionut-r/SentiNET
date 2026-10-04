"""Shared test fixtures. Tests are offline by default (recorded fixtures)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def load_json_fixture(name: str):
    return json.loads(load_fixture(name))


@pytest.fixture(autouse=True)
def _reset_shared_state():
    from app.core import cache
    from app.core.ratelimit import limiter

    cache.clear_all()
    limiter.reset()
    yield
    cache.clear_all()
    limiter.reset()
