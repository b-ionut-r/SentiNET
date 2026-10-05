"""Analytics tests run against a small, deterministic fake of the NLP layer.

The real NLP modules are developed (and tested) separately; faking them here
keeps these tests about the analytics logic and stable while the lexicon,
rules and clustering evolve. Tests marked `real_nlp` use the real modules.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import pytest

from app.analytics import textkit
from app.nlp.types import Cluster, ClusterItem, DetectedEvent, TextAnalysis
from app.sources.base import CompanyRef

POS = {
    "beat": 1.0, "beats": 1.0, "upgrade": 1.2, "upgrades": 1.2, "upgraded": 1.2, "raises": 0.8, "record": 0.8,
    "surge": 1.0, "surges": 1.0, "soars": 1.2, "jumps": 1.0, "rally": 0.8, "strong": 0.6, "bullish": 1.0,
    "tops": 0.8, "outperform": 1.0, "moon": 1.0, "🚀": 1.0, "calls": 0.6, "gains": 0.6, "wins": 0.8,
    "approval": 0.8, "expands": 0.5, "growth": 0.4, "squeeze": 0.6, "buyback": 0.8, "love": 0.6,
}
NEG = {
    "miss": 1.0, "misses": 1.0, "downgrade": 1.2, "downgrades": 1.2, "downgraded": 1.2, "cuts": 0.8,
    "lawsuit": 1.0, "sues": 1.0, "sued": 1.0, "probe": 1.0, "investigation": 0.8, "plunge": 1.2, "plunges": 1.2,
    "falls": 0.8, "drops": 0.8, "weak": 0.6, "bearish": 1.0, "fraud": 1.4, "recall": 1.0, "crash": 1.2,
    "puts": 0.6, "dilution": 1.0, "offering": 0.6, "bankruptcy": 1.5, "slump": 1.0, "warning": 0.8,
    "dumping": 1.0, "bagholders": 1.0, "overvalued": 0.8, "scam": 1.2,
}
EVENTS = (
    ("analyst_upgrade", r"\bupgrade", "bull", "analyst"),
    ("analyst_downgrade", r"\bdowngrade", "bear", "analyst"),
    ("pt_raise", r"raises .*price target", "bull", "analyst"),
    ("pt_cut", r"cuts .*price target", "bear", "analyst"),
    ("earnings_beat", r"\bbeats?\b|\btops\b", "bull", "earnings"),
    ("earnings_miss", r"\bmiss(es)?\b", "bear", "earnings"),
    ("lawsuit", r"lawsuit|\bsues\b|\bsued\b", "bear", "legal"),
    ("investigation", r"\bprobe\b|investigat", "bear", "legal"),
    ("offering", r"offering|dilution", "bear", "trading"),
    ("bankruptcy", r"bankruptcy|going concern", "bear", "legal"),
    ("short_report", r"short[- ]seller", "bear", "trading"),
    ("buyback", r"buyback", "bull", "capital_return"),
    ("price_up", r"\b(soars|jumps|surges)\b", "bull", None),
    ("price_down", r"\b(plunges|falls|drops)\b", "bear", None),
)
THEME_WORDS = (("ai", r"\bAI\b"), ("earnings", r"earnings|revenue|quarter"), ("product", r"launch|product|chip"),
               ("regulatory", r"\bSEC\b|regulat"))
TRUST = {"Reuters": 1.25, "Bloomberg": 1.25, "CNBC": 1.2, "MarketWatch": 1.2, "Barron's": 1.2,
         "The Motley Fool": 0.9, "Benzinga": 0.95, "Yahoo Finance": 1.0, "Zacks": 0.9, "TipRanks": 0.9,
         "Seeking Alpha": 0.95, "Investopedia": 0.95, "PR Newswire": 0.55, "GlobeNewswire": 0.55}
PR_WIRES = {"PR Newswire", "GlobeNewswire", "Business Wire"}
STOP = set("about after again against amid among their there these those which while would could should".split())


class FakeNLP:
    """Keyword-lexicon scorer + regex events + token-overlap clustering."""

    calls = 0

    def analyze(self, texts, kinds=None, company=None):
        FakeNLP.calls += 1
        out = []
        for text in texts:
            words = re.findall(r"[\w']+|🚀", text.lower())
            drivers = [(w, POS[w]) for w in words if w in POS] + [(w, -NEG[w]) for w in words if w in NEG]
            raw = sum(v for _, v in drivers)
            score = math.tanh(0.6 * raw)
            label = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
            events = [DetectedEvent(key=k, polarity=p) for k, pat, p, _ in EVENTS if re.search(pat, text, re.I)]
            themes = [t for k, pat, _, t in EVENTS if t and re.search(pat, text, re.I)]
            themes += [t for t, pat in THEME_WORDS if re.search(pat, text, re.I)]
            out.append(TextAnalysis(score=score, label=label, confidence=min(1.0, 0.4 + 0.15 * len(drivers)),
                                    drivers=drivers[:5], themes=list(dict.fromkeys(themes)), events=events))
        return out

    @staticmethod
    def relevance(text: str, company: CompanyRef) -> float:
        base = re.escape(company.base_symbol)
        if re.search(rf"\${base}\b", text, re.I):
            return 1.0
        if re.search(rf"\b{base}\b", text):
            return 0.9
        names = [company.short_name, *company.aliases]
        return 0.85 if any(n and re.search(rf"\b{re.escape(n)}\b", text, re.I) for n in names) else 0.0

    @staticmethod
    def duplicates(titles):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(titles):
            groups[" ".join(re.findall(r"[a-z0-9]+", t.lower())) or f"#{i}"].append(i)
        return sorted(groups.values())

    @staticmethod
    def _story_tokens(title: str, company: CompanyRef | None) -> set[str]:
        own = {company.short_name.lower(), company.ticker.lower()} if company else set()
        return {w for w in re.findall(r"[a-z$0-9.]{5,}", title.lower()) if w not in own and w not in STOP}

    def clusters(self, items: list[ClusterItem], company, max_clusters):
        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        toks = [self._story_tokens(it.title, company) for it in items]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if len(toks[i] & toks[j]) >= 2:
                    parent[find(j)] = find(i)
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(items)):
            groups[find(i)].append(i)
        ranked = sorted(groups.values(), key=lambda g: -sum(items[i].weight for i in g))[:max_clusters]
        out = []
        for g in ranked:
            rep = max(g, key=lambda i: items[i].weight)
            out.append(Cluster(item_ids=[items[rep].id] + [items[i].id for i in g if i != rep],
                               representative_id=items[rep].id, title=items[rep].title))
        return out

    @staticmethod
    def keywords(texts, scores, company, top_n=15):
        counts: dict[str, list[float]] = defaultdict(list)
        for text, score in zip(texts, scores, strict=True):
            for w in set(re.findall(r"[a-z]{6,}", text.lower())):
                counts[w].append(score)
        ranked = sorted(counts.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top_n]
        return [(w, len(v), round(sum(v) / len(v), 3)) for w, v in ranked if len(v) >= 2]

    @staticmethod
    def trust(publisher):
        return TRUST.get(publisher or "", 0.8)

    @staticmethod
    def press_release(publisher, title):
        return (publisher in PR_WIRES) or bool(re.search(r"shareholder alert|investor alert|class action", title or "",
                                                         re.I))


def pytest_configure(config):
    config.addinivalue_line("markers", "real_nlp: run analytics against the real app.nlp modules")


@pytest.fixture(autouse=True)
def fake_nlp(request, monkeypatch):
    """Swap the NLP layer for `FakeNLP` (skipped for tests marked `real_nlp`)."""
    if request.node.get_closest_marker("real_nlp"):
        yield None
        return
    fake = FakeNLP()
    monkeypatch.setattr(textkit, "analyze", fake.analyze)
    monkeypatch.setattr(textkit, "relevance", fake.relevance)
    monkeypatch.setattr(textkit, "duplicates", fake.duplicates)
    monkeypatch.setattr(textkit, "clusters", fake.clusters)
    monkeypatch.setattr(textkit, "keywords", fake.keywords)
    monkeypatch.setattr(textkit, "publisher_trust", fake.trust)
    monkeypatch.setattr(textkit, "press_release", fake.press_release)
    monkeypatch.setattr(textkit, "publisher_name", lambda p: p.strip() if p else None)
    monkeypatch.setattr(textkit, "clean", lambda t: " ".join((t or "").split()))
    monkeypatch.setattr(textkit, "strip_suffix", lambda t, p: t)
    monkeypatch.setattr(textkit, "meaningful", lambda t: len(re.findall(r"[A-Za-z]{2,}", t or "")) >= 2)
    monkeypatch.setattr(textkit, "theme_label", lambda k: k.replace("_", " ").capitalize())
    monkeypatch.setattr(textkit, "event_label", lambda k: k.replace("_", " ").capitalize())
    yield fake
