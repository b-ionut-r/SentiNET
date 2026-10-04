"""Web-server settings (CORS, frontend location), loaded like `app.config`: env vars or `.env`.

Kept separate from the shared `Settings` contract; variables use the `SENTINET_` prefix.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import BACKEND_DIR

# Dev origins only: the Vite dev server proxies /api, and production is same-origin.
DEFAULT_CORS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"


class WebSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINET_", extra="ignore")

    cors_origins: str = DEFAULT_CORS  # comma-separated
    frontend_dist: str = ""  # default: <repo>/frontend/dist

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def frontend_dist_path(self) -> Path:
        return Path(self.frontend_dist or BACKEND_DIR.parent / "frontend" / "dist").resolve()


web_settings = WebSettings()
