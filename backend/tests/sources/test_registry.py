"""Registry contract: every source satisfies the protocol and reports honest availability."""
from __future__ import annotations

import pytest

from app.config import settings
from app.sources.base import CompanyRef, Source
from app.sources.registry import ALL_SOURCES, all_sources, enabled_sources, get_source
from tests.sources.conftest import company

KEYLESS = {
    "google_news", "bing_news", "yahoo_news", "seeking_alpha", "nasdaq",
    "stocktwits", "bluesky", "hackernews", "apewisdom", "tradestie",
}  # fmt: skip
KEYED = {"finnhub", "marketaux", "alphavantage", "reddit"}


def test_registry_lists_every_source_once():
    keys = [s.key for s in ALL_SOURCES]
    assert len(keys) == len(set(keys))
    assert set(keys) == KEYLESS | KEYED
    assert all_sources() == ALL_SOURCES and all_sources() is not ALL_SOURCES


@pytest.mark.parametrize("source", ALL_SOURCES, ids=lambda s: s.key)
def test_source_protocol_and_metadata(source):
    assert isinstance(source, Source)
    assert source.kind in {"news", "social"}
    assert 0.3 <= source.weight <= 1.3
    assert source.label and len(source.description) > 20
    assert source.docs_url and source.docs_url.startswith("https://")
    assert source.requires_key == (source.key in KEYED)


def test_get_source_is_case_insensitive():
    assert get_source("Google_News").key == "google_news"
    assert get_source("lemmy") is None  # dropped after live evaluation


def test_enabled_sources_statuses(no_keys, monkeypatch):
    monkeypatch.setattr(settings, "disabled_sources", "bing_news")
    status = {s.key: st for s, st in enabled_sources(company("NVDA"))}
    assert status["bing_news"] == "disabled"
    assert all(status[k] == "unconfigured" for k in KEYED)
    assert all(status[k] == "enabled" for k in KEYLESS - {"bing_news"})


def test_enabled_sources_respect_asset_coverage(no_keys):
    crypto = {s.key: st for s, st in enabled_sources(company("BTC-USD"))}
    assert crypto["nasdaq"] == "unsupported" and crypto["tradestie"] == "unsupported"
    assert crypto["stocktwits"] == crypto["apewisdom"] == crypto["google_news"] == "enabled"
    etf = {s.key: st for s, st in enabled_sources(company("SPY"))}
    assert etf["hackernews"] == "unsupported" and etf["seeking_alpha"] == "enabled"
    foreign = {s.key: st for s, st in enabled_sources(company("SHOP.TO"))}
    assert foreign["stocktwits"] == foreign["nasdaq"] == foreign["seeking_alpha"] == "unsupported"
    assert foreign["google_news"] == foreign["yahoo_news"] == "enabled"


def test_configured_keyed_source_becomes_enabled(no_keys, monkeypatch):
    monkeypatch.setattr(settings, "finnhub_api_key", "k")
    status = {s.key: st for s, st in enabled_sources(company("AAPL"))}
    assert status["finnhub"] == "enabled"


def test_word_tickers_and_everyday_names_are_honestly_unsupported(no_keys):
    you = CompanyRef(ticker="YOU", name="Clear Secure, Inc.", short_name="Clear Secure")
    status = {s.key: st for s, st in enabled_sources(you)}
    assert status["apewisdom"] == status["tradestie"] == "unsupported"  # boards count the word "YOU"
    assert status["google_news"] == status["stocktwits"] == "enabled"
    target = {s.key: st for s, st in enabled_sources(company("TGT"))}
    assert target["hackernews"] == "unsupported"  # HN "target" is never the retailer


def test_futures_and_fx_get_theme_searches(no_keys):
    gold = CompanyRef(ticker="GC=F", name="Gold Dec 26", short_name="Gold Dec 26", quote_type="FUTURE")
    status = {s.key: st for s, st in enabled_sources(gold)}
    assert status["google_news"] == status["bing_news"] == status["yahoo_news"] == "enabled"
    assert status["stocktwits"] == status["nasdaq"] == "unsupported"

