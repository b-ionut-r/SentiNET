"""Analytics on real captured headlines with the real NLP modules (no fakes).

Uses textintel's hand-labeled Google News capture for NVDA (2026-10-04): the
'story' labels let us check that the top narratives are the stories a human
would pick. Skipped if the NLP package cannot be imported.
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import UTC, datetime

import pytest

from app.analytics.build import build_analysis
from app.sources.base import CompanyRef, RawSignal
from tests.analytics.factories import GOOGLE, STOCKTWITS, inputs, run
from tests.conftest import load_json_fixture

pytestmark = pytest.mark.real_nlp
pytest.importorskip("app.nlp.pipeline")
pytest.importorskip("app.nlp.narratives")

CAPTURED = datetime(2026, 10, 4, 23, 0, tzinfo=UTC)


def nvda_inputs():
    news = load_json_fixture("nlp/headlines_nvda.json")
    posts = load_json_fixture("nlp/stocktwits_nvda.json")
    c = news["company"]
    company = CompanyRef(ticker=c["ticker"], name=c["name"], short_name=c["short_name"], aliases=c["aliases"])
    headlines = [RawSignal(title=i["title"], publisher=i.get("publisher"),
                           timestamp=datetime.fromisoformat(i["timestamp"]), url=f"https://news.example/{n}")
                 for n, i in enumerate(news["items"])]
    social = [RawSignal(title=p["title"], publisher="StockTwits", timestamp=datetime.fromisoformat(
                  p["timestamp"].replace("Z", "+00:00")), user_label=p.get("user_label"),
                        ticker_specific=p.get("symbols") == ["NVDA"], extra={"symbols": len(p.get("symbols") or [])},
                        url=f"https://stocktwits.example/{n}")
              for n, p in enumerate(posts["items"])]
    stories = {f"https://news.example/{n}": i.get("story") for n, i in enumerate(news["items"])}
    built = inputs(company, [run(GOOGLE, headlines), run(STOCKTWITS, social)], now=CAPTURED)
    return built, stories


def test_real_headlines_produce_a_sound_analysis() -> None:
    built, stories = nvda_inputs()
    t0 = time.perf_counter()
    a = build_analysis(built)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
    assert a.sentiment.n >= 60 and len(a.narratives) >= 3
    assert all(s.relevance >= 0.35 for s in a.signals)
    news = next(c for c in a.verdict.components if c.key == "news")
    assert news.available and news.score > 50  # a week of buyback and record-high coverage

    by_id = {s.id: s for s in a.signals}

    def dominant(n) -> str | None:
        labels = Counter(stories.get(by_id[i].url) for i in n.signal_ids if i in by_id)
        labels.pop("?", None)
        labels.pop(None, None)
        return labels.most_common(1)[0][0] if labels else None

    top3 = [dominant(n) for n in a.narratives[:3]]
    assert "buyback" in top3, top3
    buyback = next(n for n in a.narratives if dominant(n) == "buyback")
    assert buyback.count >= 5 and len(buyback.publishers) >= 4 and buyback.label == "bullish"
