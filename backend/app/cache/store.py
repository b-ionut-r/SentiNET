"""In-process TTL caches (no external infra; ephemeral container friendly)."""
from __future__ import annotations

from typing import Optional

from cachetools import TTLCache

from app.config import settings

# Full analysis responses, keyed by ticker.
_analyze_cache: TTLCache = TTLCache(maxsize=256, ttl=settings.analyze_cache_ttl)
# Price responses, keyed by "{ticker}:{range}". Shorter TTL for freshness.
_price_cache: TTLCache = TTLCache(maxsize=512, ttl=120)


def get_analysis(ticker: str):
    return _analyze_cache.get(ticker.upper())


def set_analysis(ticker: str, value) -> None:
    _analyze_cache[ticker.upper()] = value


def get_price(ticker: str, rng: str):
    return _price_cache.get(f"{ticker.upper()}:{rng.upper()}")


def set_price(ticker: str, rng: str, value) -> None:
    _price_cache[f"{ticker.upper()}:{rng.upper()}"] = value


def clear() -> None:
    _analyze_cache.clear()
    _price_cache.clear()
