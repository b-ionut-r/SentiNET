"""Serve the built frontend (`frontend/dist`) as a single-page app.

* `/assets/*` — hashed bundles, served with long-lived immutable caching.
* any other non-API GET — a real file from dist if it exists (favicon…),
  otherwise `index.html` (client-side routing), never cached.
* `/api/*` is never shadowed: unknown API paths get a JSON 404.

The dist folder is looked up per request, so building the frontend while the
server runs just works. Override the location with `SENTINET_FRONTEND_DIST`.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from app.api.web_settings import web_settings

FRONTEND_DIST = web_settings.frontend_dist_path

_NOT_BUILT = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SentiNET API</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{{font:15px/1.6 system-ui,sans-serif;background:#0b0b0a;color:#f4f4f2;max-width:40rem;margin:4rem auto;padding:0 1rem}}
code{{background:#1a1a19;padding:.1rem .35rem;border-radius:4px}}a{{color:#3987e5}}</style></head><body>
<h1>SentiNET API is running</h1><p>The web app is not built at <code>{dist}</code>.</p>
<p>Build it with <code>make build</code> (or <code>npm run build</code> in <code>frontend/</code>), or run the dev
server with <code>npm run dev</code> and open <a href="http://localhost:5173">localhost:5173</a>.</p>
<p>API docs: <a href="/docs">/docs</a> · health: <a href="/api/health">/api/health</a></p></body></html>"""


class ImmutableStaticFiles(StaticFiles):
    """StaticFiles with far-future caching (Vite's asset names are content-hashed)."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def frontend_available() -> bool:
    return (FRONTEND_DIST / "index.html").is_file()


async def spa_fallback(full_path: str) -> Response:
    """Static file or index.html for client routes; JSON 404 for unknown `/api/*` paths."""
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"Unknown API endpoint: /{full_path}")
    if not frontend_available():
        return HTMLResponse(_NOT_BUILT.format(dist=FRONTEND_DIST), status_code=200 if not full_path else 404)
    if full_path:
        candidate = (FRONTEND_DIST / full_path).resolve()
        if candidate.is_relative_to(FRONTEND_DIST) and candidate.is_file():
            return FileResponse(candidate)
    return FileResponse(FRONTEND_DIST / "index.html", headers={"Cache-Control": "no-cache"})


def mount_frontend(app: FastAPI) -> None:
    """Register the asset mount and the SPA catch-all. Call after all API routes."""
    app.mount("/assets", ImmutableStaticFiles(directory=FRONTEND_DIST / "assets", check_dir=False), name="assets")
    app.add_api_route("/{full_path:path}", spa_fallback, methods=["GET", "HEAD"], include_in_schema=False)
