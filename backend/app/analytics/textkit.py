"""The analytics layer's single doorway to the NLP package (`app.nlp.*`).

Every NLP call goes through here so that
* NLP modules are imported lazily (they are developed independently and the
  analytics package must import cleanly regardless);
* a failure inside a helper (bad input, a module mid-edit) degrades to a
  simple, honest fallback instead of failing the whole analysis — except
  sentiment scoring, whose failure the caller must surface (no fake scores);
* tests can swap the whole NLP layer by monkeypatching this module.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence

from app.nlp.types import Cluster, ClusterItem, TextAnalysis
from app.sources.base import CompanyRef

logger = logging.getLogger(__name__)
_warned: set[str] = set()

_TAG_RE = re.compile(r"<[^>]{0,400}>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'&-]*[A-Za-z]")


def _fallback(name: str, exc: BaseException) -> None:
    if name not in _warned:
        _warned.add(name)
        logger.warning("nlp.%s unavailable, using fallback: %s: %s", name, type(exc).__name__, exc)


# --------------------------------------------------------------------------- #
# Scoring (no fallback: the caller decides how to report an engine failure)
# --------------------------------------------------------------------------- #
def analyze(texts: Sequence[str], kinds: Sequence[str] | None = None,
            company: CompanyRef | None = None) -> list[TextAnalysis]:
    """Engine score + themes + events for each text, in order. With `company`,
    events and price moves that belong to another entity are attributed away."""
    from app.nlp.pipeline import analyze_texts

    return analyze_texts(list(texts), list(kinds) if kinds is not None else None, company)


# --------------------------------------------------------------------------- #
# Text hygiene
# --------------------------------------------------------------------------- #
def clean(text: str | None) -> str:
    if not text:
        return ""
    try:
        from app.nlp.text import clean_text

        return clean_text(text)
    except Exception as exc:  # noqa: BLE001
        _fallback("clean_text", exc)
        return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def strip_suffix(title: str, publisher: str | None) -> str:
    """Drop a trailing " - Publisher" from a headline."""
    try:
        from app.nlp.text import strip_publisher_suffix

        return strip_publisher_suffix(title, publisher)
    except Exception as exc:  # noqa: BLE001
        _fallback("strip_publisher_suffix", exc)
        if publisher and title.endswith(f" - {publisher}"):
            return title[: -len(publisher) - 3].rstrip()
        return title


def meaningful(text: str) -> bool:
    """False for bare ticker lists, emoji-only posts and quote-page stubs."""
    try:
        from app.nlp.text import is_meaningful

        return is_meaningful(text)
    except Exception as exc:  # noqa: BLE001
        _fallback("is_meaningful", exc)
        return len(_WORD_RE.findall(text or "")) >= 2


# --------------------------------------------------------------------------- #
# Relevance, publishers
# --------------------------------------------------------------------------- #
def relevance(text: str, company: CompanyRef) -> float:
    """0..1: how clearly `text` is about `company`."""
    try:
        from app.nlp.relevance import relevance as _relevance

        value = float(_relevance(text, company))
        return value if math.isfinite(value) else 0.0
    except Exception as exc:  # noqa: BLE001
        _fallback("relevance", exc)
        if re.search(rf"\${re.escape(company.base_symbol)}\b", text, re.IGNORECASE):
            return 1.0
        if re.search(rf"\b{re.escape(company.base_symbol)}\b", text):
            return 0.85
        names = [company.short_name, *company.aliases]
        return 0.8 if any(n and re.search(rf"\b{re.escape(n)}\b", text) for n in names) else 0.0


def publisher_name(name_or_domain: str | None) -> str | None:
    """Canonical outlet name ("Barron's on MSN" -> "Barron's")."""
    if not name_or_domain:
        return None
    try:
        from app.nlp.publishers import canonical_publisher

        return canonical_publisher(name_or_domain)
    except Exception as exc:  # noqa: BLE001
        _fallback("canonical_publisher", exc)
        return name_or_domain.strip() or None


def publisher_trust(name_or_domain: str | None) -> float:
    """Outlet trust tier (~0.55 press-release wires … 1.25 wires/majors)."""
    try:
        from app.nlp.publishers import publisher_trust as _trust

        value = float(_trust(name_or_domain))
        return value if math.isfinite(value) and value > 0 else 0.8
    except Exception as exc:  # noqa: BLE001
        _fallback("publisher_trust", exc)
        return 0.8


def press_release(publisher: str | None, title: str | None) -> bool:
    """Press-release wire / company channel / class-action solicitation."""
    try:
        from app.nlp.publishers import is_press_release

        return bool(is_press_release(publisher, title))
    except Exception as exc:  # noqa: BLE001
        _fallback("is_press_release", exc)
        return False


# --------------------------------------------------------------------------- #
# Grouping & vocabulary
# --------------------------------------------------------------------------- #
def duplicates(titles: list[str]) -> list[list[int]]:
    """Partition into syndicated near-copies (representative = lowest index)."""
    try:
        from app.nlp.narratives import find_duplicates

        return find_duplicates(titles)
    except Exception as exc:  # noqa: BLE001
        _fallback("find_duplicates", exc)
        groups: dict[str, list[int]] = {}
        for i, title in enumerate(titles):
            key = " ".join(w.lower() for w in _WORD_RE.findall(title)) or f"#{i}"
            groups.setdefault(key, []).append(i)
        return sorted(groups.values())


def clusters(items: list[ClusterItem], company: CompanyRef | None, max_clusters: int) -> list[Cluster]:
    """Story clusters (falls back to one cluster per item)."""
    try:
        from app.nlp.narratives import cluster_narratives

        return cluster_narratives(items, company, max_clusters=max_clusters)
    except Exception as exc:  # noqa: BLE001
        _fallback("cluster_narratives", exc)
        ranked = sorted(items, key=lambda it: -it.weight)[:max_clusters]
        return [Cluster(item_ids=[it.id], representative_id=it.id, title=it.title) for it in ranked]


def keywords(texts: list[str], scores: list[float], company: CompanyRef | None,
             top_n: int = 15) -> list[tuple[str, int, float]]:
    """(term, document count, mean score) — empty when unavailable."""
    try:
        from app.nlp.keywords import extract_keywords

        return extract_keywords(texts, scores, company, top_n=top_n)
    except Exception as exc:  # noqa: BLE001
        _fallback("extract_keywords", exc)
        return []


def company_terms(company: CompanyRef | None) -> frozenset[str]:
    """Lower-case tokens naming the company (ticker, cashtag, name words, aliases)."""
    if company is None:
        return frozenset()
    try:
        from app.nlp.relevance import company_terms as _terms

        return frozenset(_terms(company))
    except Exception as exc:  # noqa: BLE001
        _fallback("company_terms", exc)
        base = company.base_symbol.lower()
        words = {company.ticker.lower(), base, f"${base}"}
        for name in (company.name, company.short_name, *company.aliases):
            words.update(w for w in re.findall(r"[a-z0-9&]+", (name or "").lower()) if len(w) > 1)
        return frozenset(words)


def stemmed_tokens(text: str, drop: frozenset[str] = frozenset()) -> list[str]:
    """Lower-cased, lightly stemmed tokens ('raises'/'raised' -> 'rais'); raw tokens in `drop` are skipped."""
    try:
        from app.nlp.text import stem, tokenize

        return [stem(t) for t in tokenize(text) if t not in drop]
    except Exception as exc:  # noqa: BLE001
        _fallback("tokenize/stem", exc)
        raw = (t.strip(".'-") for t in re.findall(r"\$?[a-z0-9][a-z0-9.'%-]*", text.lower()))
        return [_crude_stem(t) for t in raw if t and t not in drop]


def _crude_stem(word: str) -> str:
    w = word
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def theme_label(key: str) -> str:
    try:
        from app.nlp.themes import THEMES

        return THEMES.get(key) or key.replace("_", " ").capitalize()
    except Exception as exc:  # noqa: BLE001
        _fallback("THEMES", exc)
        return key.replace("_", " ").capitalize()


def event_label(key: str) -> str:
    try:
        from app.nlp.events import EVENT_LABELS

        return EVENT_LABELS.get(key) or key.replace("_", " ").capitalize()
    except Exception as exc:  # noqa: BLE001
        _fallback("EVENT_LABELS", exc)
        return key.replace("_", " ").capitalize()
