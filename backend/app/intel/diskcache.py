"""Tiny JSON disk cache for expensive, rate-limited provider payloads.

GDELT allows one request per 5 s per IP, and shared cloud IPs are often refused
for minutes at a time; Wikimedia budgets anonymous clients per IP too. A 90-day
timeline barely changes within hours, so losing the warm cache on every restart
is pure cost. Payloads are kept as small JSON files next to the SQLite database
(`settings.database_path`'s directory, i.e. the Docker data volume).

Best effort by design: every I/O or decode problem is logged and treated as a
miss, and writes are atomic (temp file + rename) so a crash never leaves a
half-written entry behind.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# Callers serve entries for at most a day; anything older is dead weight on the volume.
PRUNE_AFTER = 3 * 24 * 3600.0
PRUNE_EVERY = 3600.0  # seconds between sweeps (one directory listing)


class DiskCache:
    """Namespaced key -> JSON value store with wall-clock ages (old entries swept on save)."""

    def __init__(self, namespace: str, root: Path | None = None) -> None:
        self.namespace = namespace
        self._root = root
        self._last_prune = 0.0

    @property
    def directory(self) -> Path:
        root = self._root or Path(settings.database_path).expanduser().parent / "cache"
        return root / self.namespace

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()[:24]}.json"

    def load(self, key: str, max_age: float) -> tuple[Any, float] | None:
        """(value, age in seconds) when a fresh-enough entry exists, else None."""
        path = self._path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.debug("disk cache %s: unreadable %s (%s)", self.namespace, path.name, exc)
            return None
        if not isinstance(entry, dict) or entry.get("key") != key:  # hash collision or foreign file
            return None
        age = time.time() - float(entry.get("saved") or 0.0)
        if not 0 <= age <= max_age:
            return None
        return entry.get("value"), age

    def save(self, key: str, value: Any) -> None:
        """Store `value` (JSON-serializable) under `key`; failures are logged, never raised."""
        path = self._path(key)
        tmp: str | None = None
        try:
            payload = json.dumps({"key": key, "saved": time.time(), "value": value}, separators=(",", ":"))
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError) as exc:
            logger.info("disk cache %s: could not save (%s)", self.namespace, exc)
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
        if time.time() - self._last_prune >= PRUNE_EVERY:
            self.prune(PRUNE_AFTER)

    def prune(self, max_age: float) -> int:
        """Delete entries (and orphaned temp files) older than `max_age` seconds; returns the count."""
        self._last_prune = time.time()
        removed = 0
        try:
            for path in [*self.directory.glob("*.json"), *self.directory.glob(".tmp-*.part")]:
                if self._last_prune - path.stat().st_mtime > max_age:
                    path.unlink(missing_ok=True)
                    removed += 1
        except OSError as exc:
            logger.info("disk cache %s: could not prune (%s)", self.namespace, exc)
        return removed

    def clear(self) -> None:
        """Delete every entry of this namespace (tests, manual resets)."""
        try:
            for path in self.directory.glob("*.json"):
                path.unlink(missing_ok=True)
        except OSError as exc:
            logger.info("disk cache %s: could not clear (%s)", self.namespace, exc)
