"""Small numeric, labelling and formatting helpers shared by the analytics modules.

Formatting lives here so every number the user reads is rendered the same way
("+0.31", "−1.7%", "$1.37B", "61 analysts").
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from datetime import date, datetime

from app.schemas import Polarity, SentimentLabel

MINUS = "−"  # typographic minus for signed numbers in prose

# Neutral band for aggregated tone (same as the engine's per-item band).
NEUTRAL_BAND = 0.05

# SentiNET label bands (score -> label), checked top-down.
_BANDS: tuple[tuple[int, str], ...] = (
    (75, "Strongly Bullish"),
    (62, "Bullish"),
    (55, "Leaning Bullish"),
    (46, "Neutral"),
    (39, "Leaning Bearish"),
    (26, "Bearish"),
)


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #
def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def finite(value: object, default: float = 0.0) -> float:
    """`value` as a float, or `default` when missing, non-numeric, NaN or infinite."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def to_100(x: float) -> float:
    """Map a signed strength in [-1, 1] onto the 0..100 scale (50 = neutral)."""
    return clamp(50.0 + 50.0 * x, 0.0, 100.0)


def squash(value: float, scale: float) -> float:
    """tanh(value / scale): a signed, saturating strength in (-1, 1)."""
    return math.tanh(value / scale) if scale else 0.0


def weighted_mean(pairs: Iterable[tuple[float, float]]) -> tuple[float | None, float]:
    """(mean, weight sum) of (value, weight) pairs; mean is None without weight."""
    total = acc = 0.0
    for value, weight in pairs:
        if weight > 0:
            total += weight
            acc += value * weight
    return (acc / total if total > 0 else None), total


def effective_n(weights: Iterable[float]) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2 (1 heavy item ~ n_eff 1)."""
    ws = [w for w in weights if w > 0]
    sq = sum(w * w for w in ws)
    return (sum(ws) ** 2 / sq) if sq > 0 else 0.0


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def stdev(values: Iterable[float]) -> float | None:
    vals = list(values)
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def stable_id(*parts: object, n: int = 12) -> str:
    """Deterministic short id from the given parts."""
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def band_label(score: float) -> str:
    """SentiNET 0..100 score -> "Strongly Bullish" … "Strongly Bearish"."""
    s = round(score)
    for floor, label in _BANDS:
        if s >= floor:
            return label
    return "Strongly Bearish"


def stance_of(score: float) -> SentimentLabel:
    """0..100 -> bullish (>= 55) / bearish (<= 45) / neutral."""
    s = round(score)
    return "bullish" if s >= 55 else "bearish" if s <= 45 else "neutral"


def polarity_of(score: float | None, margin: float = 5.0) -> Polarity:
    """0..100 -> bull/bear when at least `margin` points away from 50."""
    if score is None:
        return "neutral"
    return "bull" if score >= 50 + margin else "bear" if score <= 50 - margin else "neutral"


def tone_label(score: float | None) -> SentimentLabel:
    """-1..1 tone -> bullish/bearish/neutral using the shared neutral band."""
    if score is None:
        return "neutral"
    return "bullish" if score > NEUTRAL_BAND else "bearish" if score < -NEUTRAL_BAND else "neutral"


AGGREGATE_CLEAR = 0.10  # an aggregate mean this far from 0 is labelled on its own
AGGREGATE_LEAN = 0.10  # otherwise (bullish − bearish) / n must lean the same way by this much


def aggregate_label(score: float, bullish: int, bearish: int, n: int) -> SentimentLabel:
    """Label of a set of items: the per-item band alone would call a perfectly balanced set
    ('+0.05', 14 bullish vs 14 bearish of 51) bullish, so a small mean also needs the counts
    to lean its way."""
    label = tone_label(score)
    if label == "neutral" or abs(score) >= AGGREGATE_CLEAR or n <= 0:
        return label
    lean = (bullish - bearish) / n
    return label if (lean >= AGGREGATE_LEAN if score > 0 else lean <= -AGGREGATE_LEAN) else "neutral"


def tone_polarity(score: float | None, band: float = NEUTRAL_BAND) -> Polarity:
    if score is None:
        return "neutral"
    return "bull" if score > band else "bear" if score < -band else "neutral"


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def signed(value: float, fmt: str = ".2f") -> str:
    """+0.31 / −0.42 (typographic minus); zero renders unsigned."""
    text = format(abs(value), fmt)
    if float(text.rstrip("%") or 0) == 0:
        return text
    return ("+" if value > 0 else MINUS) + text


def pct(value: float, digits: int = 1, sign: bool = True) -> str:
    """Percent with adaptive precision: +40% / −1.7% / +0.6%."""
    d = 0 if abs(value) >= 10 else digits
    body = f"{abs(value):.{d}f}%"
    if not sign:
        return (MINUS if value < 0 and float(body[:-1]) else "") + body
    if float(body[:-1]) == 0:
        return body
    return ("+" if value > 0 else MINUS) + body


# Insider trade values are US dollars from every source: Form 4 is filed in USD and Yahoo
# converts non-US filings (VOD.L sales at $1.70 = 126.15p × 1.3358 $/£; see intel insider_view).
INSIDER_CURRENCY = "USD"

# Display symbols by ISO currency code; unknown codes are written out ("NOK 12.40").
_SYMBOLS = {
    "USD": "$", "CAD": "C$", "AUD": "A$", "NZD": "NZ$", "HKD": "HK$", "SGD": "S$", "TWD": "NT$", "MXN": "MX$",
    "BRL": "R$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "CN¥", "INR": "₹", "KRW": "₩", "ILS": "₪", "ZAR": "R",
}
# Quotes in a minor unit (London pence, Johannesburg cents, Tel Aviv agorot): (major code, suffix, per major).
_MINOR_UNITS = {"GBp": ("GBP", "p"), "GBX": ("GBP", "p"), "ZAc": ("ZAR", "c"), "ZAC": ("ZAR", "c"),
                "ILA": ("ILS", " ag")}
_NO_CENTS = frozenset({"JPY", "KRW"})


# Minor units whose market cap is quoted in the major unit (VOD.L: 126.8p × 23.2B shares = the £29.4B cap).
_CAP_IN_MAJOR = frozenset({"GBp", "GBX"})


def major_currency(code: str | None) -> str | None:
    """ISO code of the major unit: 'GBp' (London pence) -> 'GBP'; market caps are quoted in it."""
    if not code:
        return None
    return _MINOR_UNITS[code][0] if code in _MINOR_UNITS else code.upper()


def shares_outstanding(market_cap: float | None, price: float | None, currency: str | None) -> float | None:
    """Market cap / price with both in the same unit (a pence price is converted to pounds); None
    when either is missing or the minor-unit convention of the listing is not verified."""
    if not market_cap or not price or market_cap <= 0 or price <= 0:
        return None
    if currency in _MINOR_UNITS:
        if currency not in _CAP_IN_MAJOR:
            return None
        price /= 100.0
    return market_cap / price


def usd_market_cap(market_cap: float | None, currency: str | None, usd_rate: float | None = None) -> float | None:
    """Market cap in USD, for comparing with USD amounts (insider trades, quoted deal values).

    As quoted for a USD listing (an unknown currency is taken as USD, like everywhere
    else in analytics); converted with `usd_rate` (USD per unit of the listing's major
    currency) otherwise; None when it cannot be compared — never at a guessed rate."""
    if not market_cap or market_cap <= 0:
        return None
    major = major_currency(currency) or "USD"
    if major == "USD":
        return market_cap
    return market_cap * usd_rate if usd_rate and usd_rate > 0 and math.isfinite(usd_rate) else None


def money(value: float, price: bool = False, currency: str | None = "USD") -> str:
    """$1.37B / $749K / $6.8M; `price=True` keeps cents ($327.70).

    `currency` is the ISO code of the value ("GBp" pence prices render as
    "121.82p", larger pence amounts in pounds); None means the currency is not
    known (a non-USD listing's reporting currency) and no symbol is shown."""
    v = abs(value)
    sign = MINUS if value < 0 else ""
    code = currency
    if code in _MINOR_UNITS:
        major, suffix = _MINOR_UNITS[code]
        if price:
            return f"{sign}{v:,.2f}{suffix}" if v < 10_000 else f"{sign}{v:,.0f}{suffix}"
        code, v = major, v / 100.0
    if code is None:
        symbol, tail = "", ""
    else:
        known = _SYMBOLS.get(code.upper())
        symbol, tail = (known, "") if known else ("", f" {code.upper()}")
    if price:
        digits = 0 if v >= 10_000 or (code or "").upper() in _NO_CENTS else 2
        return f"{sign}{symbol}{v:,.{digits}f}{tail}"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            q = v / div
            return f"{sign}{symbol}{q:.3g}{suffix}{tail}" if q < 100 else f"{sign}{symbol}{q:.0f}{suffix}{tail}"
    return f"{sign}{symbol}{v:,.0f}{tail}"


def count(n: int, noun: str, plural: str | None = None) -> str:
    """'1 article' / '54 articles'."""
    return f"{n} {noun if n == 1 else (plural or noun + 's')}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def cap_share(bps: float) -> str:
    """A share of market cap given in basis points: '0.06%', or '<0.01%' for a sliver."""
    return "<0.01%" if bps < 1 else f"{bps / 100:.2f}%"


def trim(text: str, limit: int = 140) -> str:
    """`text` with whitespace collapsed, cut on a word boundary with '…' when longer than `limit`."""
    t = " ".join(text.split())
    if len(t) > limit:
        t = t[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-") + "…"
    return t


_LEGAL_DATE = re.compile(r"^(?:as previously (?:disclosed|reported)[^,]*,\s*)?(?:on|effective(?:\s+as\s+of)?|as\s+of)\s+"
                         r"(?:[A-Z][a-z]+\.?\s+\d{1,2}|\d{1,2}\s+[A-Z][a-z]+\.?),?\s+\d{4},?\s*", re.IGNORECASE)
# ", a Delaware corporation," / ", a Cayman Islands exempted company (the “Company”),"
_LEGAL_APPOSITIVE = re.compile(r",\s+an?\s+(?:[A-Z][\w.]*\s+){1,3}(?i:(?:exempted\s+)?(?:corporation|company|limited "
                               r"liability company|limited partnership|public limited company|statutory trust))\b[^,.;]{0,60}(?:,|(?=[.;]|$))")
_LEGAL_PAREN = re.compile(r"\s*\((?:the |each, a |collectively,? the )?[\"“”'‘’]+[^)]{1,40}[\"“”'‘’]+[^)]{0,40}\)")


_PERIOD = re.compile(r"\.\s+(?=[A-Z])")
_ABBREV = re.compile(r"(?:\b(?i:inc|corp|co|ltd|no|st|mr|mrs|ms|dr|jr|sr|vs|al|approx)|\b[A-Z]|\.[A-Za-z])$")
_STARTER = re.compile(r"(?:The|This|That|These|Those|It|Its|Such|Each|Under|Pursuant|As|In|On|Upon|A|An|We|He|She|They)\b")


def _first_sentence(text: str) -> str:
    """`text` up to its first sentence end: a period before a capital, unless it closes an
    abbreviation ('Inc.', 'Dr.', 'U.S.') that the next word does not show to end a sentence."""
    for m in _PERIOD.finditer(text):
        if not _ABBREV.search(text[: m.start()]) or _STARTER.match(text, m.end()):
            return text[: m.start() + 1]
    return text


def gist(text: str, limit: int = 140, *, first_sentence: bool = True) -> str:
    """A filing excerpt as a reader-friendly line: the leading 'On May 4, 2026,' preamble,
    defined-term parentheticals ('(the “Company”)') and ', a Delaware corporation,' dropped,
    the first sentence only (unless `first_sentence=False`), trimmed to `limit`. A closing
    period is dropped unless it ends an abbreviation ('… Hugging Face, Inc.')."""
    t = _LEGAL_PAREN.sub("", " ".join(text.split()))
    t = _LEGAL_APPOSITIVE.sub("", t)
    t = _LEGAL_DATE.sub("", t)
    t = t[:1].upper() + t[1:]
    out = trim(_first_sentence(t) if first_sentence else t, limit)
    if out.endswith(".") and not _ABBREV.search(out[:-1]):
        out = out[:-1]
    return out


def quote(text: str, limit: int = 90) -> str:
    """'Headline…' trimmed on a word boundary, in single quotes."""
    return f"‘{trim(text, limit)}’"


def filing_parts(title: str) -> tuple[str, str | None]:
    """(label, description) of a decoded 8-K title.

    'Other material event: NVIDIA entered into…' -> ('Other material event', 'NVIDIA entered into…');
    an 'Amended: ' prefix becomes ' (amended)' on the label."""
    t = " ".join(title.split())
    amended = t.lower().startswith("amended:")
    if amended:
        t = t[len("amended:"):].strip()
    label, sep, desc = t.partition(": ")
    label = label.strip() or "Filing"
    return label + (" (amended)" if amended else ""), (desc.strip() or None) if sep else None


def short_date(value: date | datetime) -> str:
    """'Oct 29' (no leading zero)."""
    return f"{value:%b} {value.day}"


def join_and(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]
