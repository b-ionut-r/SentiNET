"""Web-server settings (CORS, frontend location), loaded like `app.config`: env vars or `.env`.

Kept separate from the shared `Settings` contract; variables use the `SENTINET_` prefix.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import BACKEND_DIR

logger = logging.getLogger(__name__)

# Dev origins only: the Vite dev server proxies /api, and production is same-origin.
DEFAULT_CORS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"
# The API has no authentication, so by default it only answers requests addressed to this
# machine: that also defeats DNS rebinding (a web page pointing its own domain at 127.0.0.1).
DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1,[::1]"


class WebSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINET_", extra="ignore")

    cors_origins: str = DEFAULT_CORS  # comma-separated
    allowed_hosts: str = DEFAULT_ALLOWED_HOSTS  # comma-separated Host names; "*.example.com" or "*"
    frontend_dist: str = ""  # default: <repo>/frontend/dist

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_host_list(self) -> list[str]:
        """Host header values served (port ignored): exact names, `*.domain` wildcards or `*` (any).

        Malformed wildcards are dropped (with a warning) instead of failing startup; an empty
        list falls back to the loopback default.
        """
        hosts = []
        for raw in self.allowed_hosts.split(","):
            host = raw.strip().lower()
            if not host:
                continue
            if "*" in host[1:] or (host.startswith("*") and host != "*" and not host.startswith("*.")):
                logger.warning("ignoring SENTINET_ALLOWED_HOSTS entry %r (use '*.example.com' or '*')", host)
                continue
            hosts.append(host)
        return hosts or DEFAULT_ALLOWED_HOSTS.split(",")

    @property
    def frontend_dist_path(self) -> Path:
        return Path(self.frontend_dist or BACKEND_DIR.parent / "frontend" / "dist").resolve()


web_settings = WebSettings()
