"""app.nlp.pipeline: engine score + events + themes per text."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.nlp import pipeline
from app.nlp.types import DetectedEvent, TextAnalysis

BACKEND = Path(__file__).resolve().parents[2]


class _FakeEngine:
    name = "fake"

    def __init__(self, drop: int = 0):
        self.calls: list[tuple[list[str], list[str] | None]] = []
        self.drop = drop

    def score(self, texts, kinds=None):
        self.calls.append((list(texts), kinds))
        out = [TextAnalysis(score=0.1 * (i + 1), label="bullish", confidence=0.5, drivers=[("x", 0.1)])
               for i in range(len(texts))]
        if out and texts[0].startswith("engine-event"):
            out[0].events = [DetectedEvent(key="buyback", polarity="bull")]
            out[0].themes = ["capital_return"]
        return out[: len(out) - self.drop]


@pytest.fixture
def fake_engine(monkeypatch):
    import app.nlp.engine as engine_module

    engine = _FakeEngine()
    monkeypatch.setattr(engine_module, "get_engine", lambda: engine)
    return engine


def test_enriches_engine_results_in_order(fake_engine):
    texts = [
        "Morgan Stanley raises Nvidia price target to $250 from $220",
        "Apple hit with $5.7 billion jury verdict over haptic patents",
        "$NVDA 🚀🚀🚀",
    ]
    results = pipeline.analyze_texts(texts, ["news", "news", "social"])
    assert len(results) == 3
    assert fake_engine.calls == [(texts, ["news", "news", "social"])]
    first, second, third = results
    assert first.score == pytest.approx(0.1) and first.drivers == [("x", 0.1)]  # engine output preserved
    pt = next(e for e in first.events if e.key == "pt_raise")
    assert (pt.firm, pt.value, pt.polarity) == ("Morgan Stanley", 250.0, "bull")
    assert first.themes[0] == "analyst"  # event-implied theme first
    assert [e.key for e in second.events] == ["lawsuit"] and second.themes[0] == "legal"
    assert third.events == [] and third.themes == []


def test_engine_events_and_themes_are_kept_and_not_duplicated(fake_engine):
    result = pipeline.analyze_text("engine-event: Nvidia adds $150 billion to its share buyback")
    assert [e.key for e in result.events] == ["buyback"]
    assert result.themes[0] == "capital_return"
    assert len(result.themes) == len(set(result.themes)) <= pipeline.MAX_THEMES


def test_edges(fake_engine):
    assert pipeline.analyze_texts([]) == []
    out = pipeline.analyze_texts([None, ""])  # type: ignore[list-item]
    assert len(out) == 2 and all(r.events == [] for r in out)
    assert fake_engine.calls[-1] == (["", ""], None)


def test_engine_contract_violation_is_loud(monkeypatch):
    import app.nlp.engine as engine_module

    monkeypatch.setattr(engine_module, "get_engine", lambda: _FakeEngine(drop=1))
    with pytest.raises(RuntimeError, match="returned 1 results for 2 texts"):
        pipeline.analyze_texts(["a b", "c d"])


def test_engine_is_imported_lazily():
    """Importing the pipeline must not import the (heavier, optional) engine."""
    code = "import sys, app.nlp.pipeline; print('app.nlp.engine' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_with_the_configured_engine():
    """Integration with the real engine (written by another agent): shape and
    enrichment only — sentiment quality is tested in test_engine*.py."""
    pytest.importorskip("app.nlp.engine")
    from app.nlp.engine import reset_engine

    reset_engine()
    texts = ["Nvidia stock hits record high after $150 billion buyback",
             "Target cuts prices on thousands of products as value war heats up",
             "$TSLA calls printing, to the moon"]
    results = pipeline.analyze_texts(texts, ["news", "news", "social"])
    assert len(results) == 3
    for r in results:
        assert -1.0 <= r.score <= 1.0 and 0.0 <= r.confidence <= 1.0
        assert r.label in {"bullish", "bearish", "neutral"}
    assert {"all_time_high", "buyback"} <= {e.key for e in results[0].events}
    assert "competition" in results[1].themes and not any(e.key == "pt_cut" for e in results[1].events)


@pytest.mark.live
def test_live_google_news_end_to_end():
    """Live smoke: today's Google News feed through relevance, dedup,
    scoring, clustering and keywords (run with `pytest -m live`)."""
    from urllib.parse import quote_plus

    import feedparser
    import httpx

    from app.nlp.keywords import extract_keywords
    from app.nlp.narratives import cluster_narratives, find_duplicates
    from app.nlp.relevance import relevance
    from app.nlp.text import is_meaningful, strip_publisher_suffix
    from app.nlp.types import ClusterItem
    from app.sources.base import CompanyRef

    company = CompanyRef(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia", industry="Semiconductors")
    query = quote_plus('"Nvidia" (stock OR shares OR NVDA) when:7d')
    response = httpx.get(f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=25, follow_redirects=True)
    entries = feedparser.parse(response.content).entries
    titles = [strip_publisher_suffix(e.title, getattr(getattr(e, "source", None), "title", None)) for e in entries]
    kept = [t for t in titles if is_meaningful(t) and relevance(t, company) >= 0.35]
    assert len(kept) >= 20
    analyses = pipeline.analyze_texts(kept, ["news"] * len(kept))
    items = [ClusterItem(id=str(i), title=t, score=a.score) for i, (t, a) in enumerate(zip(kept, analyses, strict=True))]
    clusters = cluster_narratives(items, company, max_clusters=8)
    assert clusters and len(clusters[0].item_ids) >= 3
    assert sum(len(g) for g in find_duplicates(kept)) == len(kept)
    assert extract_keywords(kept, [a.score for a in analyses], company)
