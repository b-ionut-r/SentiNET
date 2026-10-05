"""Time-boxed execution of provider calls with honest, secret-free status reporting.

Every network-facing task in an analysis goes through `run_bounded`, which
never raises (except on cancellation): it returns an `Outcome` carrying the
value or a short, human-readable error with API keys redacted.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit

import httpx

from app.config import settings
from app.core.http import UpstreamError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SECRET_PARAM = re.compile(
    r"(?i)\b(token|api_?token|api_?key|apikey|key|access_token|client_secret|password|app_password)=([^&\s'\"]+)"
)

# Tasks that outlived their time box but were allowed to finish so their
# result lands in the provider cache (see `run_bounded(keep_alive=True)`).
_background: set[asyncio.Task[Any]] = set()


@dataclass
class Outcome(Generic[T]):
    """Result of one bounded task. `pending` is the still-running call when a
    kept-alive task outlived its time box (callers may watch it complete)."""

    value: T | None = None
    error: str | None = None
    ms: int = 0
    timed_out: bool = False
    pending: asyncio.Task[Any] | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.error is None


def _secrets() -> list[str]:
    vals = [
        settings.finnhub_api_key, settings.alphavantage_api_key, settings.marketaux_api_key,
        settings.reddit_client_secret, settings.bluesky_app_password, settings.hf_token,
    ]
    return [v for v in vals if v and len(v) >= 6]


def redact(text: str) -> str:
    """Strip credentials from a message (query-string keys and configured secrets)."""
    text = _SECRET_PARAM.sub(lambda m: f"{m.group(1)}=***", text)
    for secret in _secrets():
        text = text.replace(secret, "***")
    return text


def describe_error(exc: BaseException) -> str:
    """Short, user-facing description of a provider failure (no URLs with keys)."""
    if isinstance(exc, httpx.HTTPStatusError):
        host = urlsplit(str(exc.request.url)).hostname or "upstream"
        code = exc.response.status_code
        hint = {401: "unauthorized (check API key)", 403: "forbidden", 404: "not found",
                429: "rate limited"}.get(code, exc.response.reason_phrase or "error")
        return f"HTTP {code} {hint} from {host}"
    if isinstance(exc, httpx.TimeoutException):
        return "upstream timeout"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timed out"
    name, msg = type(exc).__name__, str(exc).strip()
    if isinstance(exc, UpstreamError):
        text = msg or name  # already phrased for humans
    else:
        text = f"{name}: {msg}" if msg else name
    return redact(text)[:180]


def _secs(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds >= 2 else f"{seconds:.1f}s"


def _retrieve(task: asyncio.Task[Any]) -> None:
    """Mark a cancelled task's outcome as retrieved (no "exception never retrieved" noise)."""
    if not task.cancelled():
        task.exception()


def _reap(task: asyncio.Task[Any]) -> None:
    _background.discard(task)
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.debug("background task %s failed: %s", task.get_name(), describe_error(exc))


async def _await_task(task: asyncio.Task[T], timeout: float, until: asyncio.Event | None, grace: float) -> T:
    """`task`'s result within `timeout` — cut to `grace` seconds after `until` is set.

    Never cancels `task`; raises `TimeoutError` when it is still running.
    """
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    if until is None:
        await asyncio.wait({task}, timeout=timeout)
    else:
        waiter = asyncio.ensure_future(until.wait())
        try:
            await asyncio.wait({task, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if not task.done() and until.is_set():
            await asyncio.wait({task}, timeout=max(0.0, min(grace, end - loop.time())))
    if not task.done():
        raise TimeoutError
    return task.result()


async def run_bounded(
    factory: Callable[[], Awaitable[T]],
    timeout: float,
    *,
    keep_alive: bool = False,
    name: str = "task",
    until: asyncio.Event | None = None,
    grace: float = 0.0,
) -> Outcome[T]:
    """Await `factory()` for at most `timeout` seconds; never raises (except cancellation).

    `until` + `grace` implement a tail rule: once the event is set (e.g. "all
    core data is in"), wait at most `grace` more seconds instead of the full
    time box. With `keep_alive=True` a call that times out is *not* cancelled:
    it keeps running in the background (returned as `Outcome.pending`) so a
    cached provider call still completes and warms its cache — GDELT's
    1 request / 5 s budget makes this valuable.
    """
    started = time.perf_counter()

    def elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    task: asyncio.Task[T] = asyncio.ensure_future(factory())
    task.set_name(f"bounded:{name}")
    try:
        value = await _await_task(task, timeout, until, grace)
        return Outcome(value=value, ms=elapsed())
    except asyncio.CancelledError:
        task.cancel()
        raise
    except TimeoutError:
        waited = _secs(elapsed() / 1000)
        if keep_alive and not task.done():
            _background.add(task)
            task.add_done_callback(_reap)
            return Outcome(error=f"still loading after {waited}; continuing in the background (reload to include)",
                           ms=elapsed(), timed_out=True, pending=task)
        task.cancel()
        task.add_done_callback(_retrieve)
        return Outcome(error=f"timed out after {waited}", ms=elapsed(), timed_out=True)
    except Exception as exc:  # noqa: BLE001 - a provider failure must never sink the analysis
        logger.info("%s failed: %s", name, describe_error(exc))
        return Outcome(error=describe_error(exc), ms=elapsed())


def pending_background() -> list[asyncio.Task[Any]]:
    """Kept-alive stragglers still running on this loop (a one-shot caller may let them finish)."""
    loop = asyncio.get_running_loop()
    return [t for t in _background if not t.done() and t.get_loop() is loop]


async def cancel_background() -> None:
    """Cancel kept-alive stragglers on the running loop (application shutdown / tests)."""
    loop = asyncio.get_running_loop()
    mine = [t for t in _background if t.get_loop() is loop]
    for t in mine:
        t.cancel()
    if mine:
        await asyncio.gather(*mine, return_exceptions=True)
    _background.clear()  # tasks of other (closed) loops are gone with their loop


def deferred(module: str, fn: str, *args: Any) -> Callable[[], Awaitable[Any]]:
    """`lambda: module.fn(*args)` with the import done at call time.

    Provider modules may be mid-edit or broken; importing inside the bounded
    call turns an ImportError into a reported task error instead of a crash.
    """

    async def call() -> Any:
        mod = importlib.import_module(module)
        return await getattr(mod, fn)(*args)

    call.__name__ = f"{module.rsplit('.', 1)[-1]}.{fn}"
    return call


def has_data(value: Any) -> bool:
    """True when a provider result carries something (empty containers/trends don't count)."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    series = getattr(value, "series", None)  # ToneTrend-like
    if isinstance(series, list) and not series:
        return getattr(value, "tone_7d", None) is not None
    return True


def status_of(out: Outcome[Any], expect: type | tuple[type, ...] | None = None) -> str:
    """"ok" | "empty" | "error: …" for a provider outcome (wrong payload types count as errors)."""
    if not out.ok:
        return f"error: {out.error}"
    if expect is not None and out.value is not None and not isinstance(out.value, expect):
        return f"error: unexpected payload ({type(out.value).__name__})"
    return "ok" if has_data(out.value) else "empty"
