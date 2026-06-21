"""Application settings, loaded from environment / .env (all optional)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sentiment_engine: str = "vader"
    analyze_cache_ttl: int = 600
    source_timeout: float = 8.0

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "sentiment-platform/1.0"

    disabled_sources: str = ""

    user_agent: str = (
        "Mozilla/5.0 (compatible; SentimentPlatform/1.0; +https://github.com/)"
    )

    @property
    def disabled_source_set(self) -> set[str]:
        return {s.strip().lower() for s in self.disabled_sources.split(",") if s.strip()}


settings = Settings()
