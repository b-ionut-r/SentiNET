"""In-process TTL caching with single-flight for async functions.

`@cached(ttl=...)` memoizes an async function's result per argument tuple and
collapses concurrent identical calls into one upstream request (important for
rate-limited providers: ten users opening NVDA must cost one GDELT call).

Exceptions are not cached. `None` results are cached for `none_ttl` seconds
(short), so a temporarily failing provider is retried soon.
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")

_registry: list["_AsyncTTLCache"] = []


class _AsyncTTLCache:
    def __init__(self, ttl: float, none_ttl: float, maxsize: int) -> None:
        self.ttl = ttl
        self.none_ttl = none_ttl
        self.maxsize = maxsize
        self._data: dict[Any, tuple[float, Any]] = {}
        self._inflight: dict[Any, asyncio.Future] = {}

    def get(self, key: Any) -> tuple[bool, Any]:
        hit = self._data.get(key)
        if hit is None:
            return False, None
        expires, value = hit
        if expires < time.monotonic():
            self._data.pop(key, None)
            return False, None
        return True, value

    def set(self, key: Any, value: Any) -> None:
        if len(self._data) >= self.maxsize:
            # Drop the soonest-expiring ~10% (cheap approximate LRU).
            for k, _ in sorted(self._data.items(), key=lambda kv: kv[1][0])[: max(1, self.maxsize // 10)]:
                self._data.pop(k, None)
        ttl = self.none_ttl if value is None else self.ttl
        self._data[key] = (time.monotonic() + ttl, value)

    def clear(self) -> None:
        self._data.clear()


def cached(ttl: float, none_ttl: float = 60.0, maxsize: int = 512) -> Callable[
    [Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]
]:
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        cache = _AsyncTTLCache(ttl, none_ttl, maxsize)
        _registry.append(cache)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            key = (args, tuple(sorted(kwargs.items())))
            hit, value = cache.get(key)
            if hit:
                return value
            pending = cache._inflight.get(key)
            if pending is not None:
                return await asyncio.shield(pending)
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            cache._inflight[key] = fut
            try:
                value = await fn(*args, **kwargs)
            except BaseException as exc:
                if not fut.done():
                    if isinstance(exc, asyncio.CancelledError):
                        fut.cancel()
                    else:
                        fut.set_exception(exc)
                        fut.exception()  # mark retrieved; avoid "never retrieved" warnings
                raise
            else:
                cache.set(key, value)
                if not fut.done():
                    fut.set_result(value)
                return value
            finally:
                cache._inflight.pop(key, None)

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


def clear_all() -> None:
    """Test helper: drop every cached value."""
    for c in _registry:
        c.clear()


class TTLStore:
    """Tiny sync TTL dict for whole-object caches (e.g. analyses by ticker)."""

    def __init__(self, ttl: float, maxsize: int = 256) -> None:
        self._c = _AsyncTTLCache(ttl, ttl, maxsize)
        _registry.append(self._c)

    def get(self, key: Any) -> Optional[Any]:
        hit, value = self._c.get(key)
        return value if hit else None

    def set(self, key: Any, value: Any) -> None:
        self._c.set(key, value)

    def clear(self) -> None:
        self._c.clear()
