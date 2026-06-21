"""Text cleaning, dedup and lightweight keyword extraction helpers."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_NONWORD = re.compile(r"[^a-z0-9$&'\- ]+")
_WORD = re.compile(r"[a-z][a-z'\-]{2,}|\$[a-z]{1,5}")

# Common English + finance stopwords we never want surfaced as "trending".
STOPWORDS: set[str] = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its",
    "let", "put", "say", "she", "too", "use", "this", "that", "with", "from",
    "they", "have", "will", "your", "what", "when", "them", "than", "then",
    "into", "just", "like", "over", "also", "back", "after", "more", "most",
    "much", "some", "such", "only", "very", "want", "going", "stock", "stocks",
    "share", "shares", "market", "today", "company", "price", "buy", "sell",
    "hold", "https", "http", "www", "com", "amp", "would", "could", "should",
    "about", "there", "their", "been", "were", "said", "says", "year", "years",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _URL.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def normalize_for_dedup(text: str) -> str:
    text = text.lower()
    text = _URL.sub("", text)
    text = _NONWORD.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text[:160]


def is_meaningful(text: str) -> bool:
    """Drop empty / too-short / link-only content."""
    cleaned = clean_text(text)
    return len(cleaned) >= 8 and bool(re.search(r"[a-zA-Z]", cleaned))


def extract_keywords(texts: Iterable[str], top_n: int = 12, ticker: str = "") -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    ticker_l = ticker.lower()
    for text in texts:
        for match in _WORD.findall(text.lower()):
            word = match.strip("'-")
            if len(word) < 3:
                continue
            if word in STOPWORDS:
                continue
            if word == ticker_l or word == f"${ticker_l}":
                continue
            counter[word] += 1
    return counter.most_common(top_n)
