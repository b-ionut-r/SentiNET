"""Request body size limits, enforced before anything parses the body.

Every SentiNET request body is small (a ticker, an alert rule) except the
sentiment lab's batch of texts, so bodies are capped per path: oversized
requests get a JSON 413 without the server buffering or parsing them. Both a
declared `Content-Length` and the bytes actually streamed (chunked uploads) are
checked.
"""
from __future__ import annotations

from collections.abc import Mapping

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_MAX_BODY = 64 * 1024


def _message(limit: int) -> str:
    size = f"{limit // (1024 * 1024)} MB" if limit >= 1024 * 1024 else f"{limit // 1024} KB"
    return f"Request body too large (limit {size} for this endpoint)."


class BodyTooLarge(HTTPException):
    """Raised from `receive` once a streamed body passes the limit (FastAPI re-raises HTTPExceptions)."""

    def __init__(self, limit: int) -> None:
        super().__init__(status_code=413, detail=_message(limit))


class BodySizeLimitMiddleware:
    """413 for request bodies over `default` bytes (or the per-path override)."""

    def __init__(self, app: ASGIApp, default: int = DEFAULT_MAX_BODY,
                 overrides: Mapping[str, int] | None = None) -> None:
        self.app = app
        self.default = default
        self.overrides = dict(overrides or {})

    def limit_for(self, path: str) -> int:
        return self.overrides.get(path.rstrip("/") or "/", self.default)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope.get("path", "/"))
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                too_big = int(declared) > limit
            except ValueError:
                too_big = False  # malformed: the server rejects it; the streamed count still applies
            if too_big:
                await JSONResponse({"detail": _message(limit)}, status_code=413)(scope, receive, send)
                return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(limit)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge as exc:  # raised outside FastAPI's handlers (e.g. a non-API route)
            if started:
                raise
            await JSONResponse({"detail": exc.detail}, status_code=413)(scope, receive, send)
