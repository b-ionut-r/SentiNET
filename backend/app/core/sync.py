"""Run blocking third-party calls (yfinance, feedparser) without stalling the loop.

yfinance is synchronous and Yahoo throttles bursts, so its calls go through a
small dedicated thread pool with a concurrency cap.
"""
from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_yahoo_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yahoo")
_cpu_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu")


async def run_yahoo(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_yahoo_pool, functools.partial(fn, *args, **kwargs))


async def run_cpu(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_cpu_pool, functools.partial(fn, *args, **kwargs))
