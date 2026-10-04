"""Application settings, loaded from environment / .env.

Everything is optional: SentiNET runs with zero configuration on keyless
sources. Free API keys unlock extra sources (see `.env.example`).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Engine -------------------------------------------------------------
    # "sentinel" (finance lexicon + rules + VADER ensemble, default) or
    # "finbert" (local transformers model, ensembled with sentinel) or
    # "finbert-api" (Hugging Face Inference API; needs HF_TOKEN).
    sentiment_engine: str = "sentinel"
    hf_token: str = ""

    # ---- Caching / timeouts -------------------------------------------------
    analyze_cache_ttl: int = 600       # seconds a full analysis is reused
    price_cache_ttl: int = 120
    intel_cache_ttl: int = 1800        # analysts, earnings, insiders, filings
    history_cache_ttl: int = 7200      # GDELT timelines, pageviews
    source_timeout: float = 10.0       # per text source
    intel_timeout: float = 15.0        # per intel task (GDELT can queue)

    # ---- Storage ------------------------------------------------------------
    database_path: str = str(BACKEND_DIR / "data" / "sentinet.db")

    # ---- Watchlist monitor / alerts ----------------------------------------
    monitor_enabled: bool = True
    monitor_interval_minutes: int = 30
    alert_webhook_url: str = ""        # Discord/Slack-compatible incoming webhook

    # ---- Identity (some free APIs require a contact-bearing User-Agent) ----
    contact_email: str = "sentinet@example.com"
    browser_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )

    # ---- Optional free-key sources -----------------------------------------
    finnhub_api_key: str = ""          # https://finnhub.io/register (free)
    alphavantage_api_key: str = ""     # https://www.alphavantage.co/support/#api-key (free)
    marketaux_api_key: str = ""        # https://www.marketaux.com/register (free)
    bluesky_handle: str = ""           # optional: improves Bluesky search reliability
    bluesky_app_password: str = ""     # https://bsky.app/settings/app-passwords
    reddit_client_id: str = ""         # only if you already hold an approved Reddit app
    reddit_client_secret: str = ""

    disabled_sources: str = ""         # comma-separated source keys

    @property
    def disabled_source_set(self) -> set[str]:
        return {s.strip().lower() for s in self.disabled_sources.split(",") if s.strip()}

    @property
    def api_user_agent(self) -> str:
        """Descriptive UA for APIs whose policy asks for contact info (SEC, Wikimedia)."""
        return f"SentiNET/2.0 (+https://github.com/b-ionut-r/SentiNET; {self.contact_email})"


settings = Settings()
