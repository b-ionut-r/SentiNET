"""Pending acquisition of the company itself, read from its own 8-K filings.

A signed merger agreement in which the company is the *target* changes what
every other signal means: the share price tracks the deal terms (and the odds
of closing) until the deal closes or breaks, so analyst targets set before it
and the price trend that jumped on it no longer measure sentiment.

Detection (filings only — authoritative, never inferred from headlines):

* an 8-K filed within the last 180 days carries merger-agreement wording
  ("Agreement and Plan of Merger", "merger agreement", a tender offer for all
  outstanding shares, a scheme/arrangement agreement), and
* an 8-K in that window places the company on the target side: Merger Sub
  "will merge with and into" the company, the company "continuing as the
  surviving corporation", Merger Sub "a wholly owned subsidiary of Parent",
  the company "to be acquired by" someone, or a tender offer for the company's
  shares;
* no later 8-K reports the deal completed (item 2.01 / "consummated the
  merger") or terminated (item 1.02 / "terminated the merger agreement").

An acquirer's own 8-K ("Merger Sub, a wholly owned subsidiary of the Company,
will merge with and into Target") never matches: the company is neither the
merged-into party nor a subsidiary of "Parent".

A *deal in play* is read from news instead: a deal story (the company's own M&A
event, as bidder, target or partner — see narratives.py) that is fresh (last
item <= 7 days old), prominent (impact >= 0.35) and corroborated (m_and_a
carried by >= 2 articles from >= 2 outlets, or by one major outlet clearly about
the company). Its size is the deal value the headlines quote most often next to
a deal word ("$56B eBay Bid", "$56B Takeover Vision"), compared with the USD
market cap when both are in USD (GME: $56B = 4.5× its $12.5B cap). Members
reporting new shares for it (an offering event) are a dilution risk, and a
deadline the coverage states explicitly ("warrants expire on Nov 2", "the offer
expires in 26 days" in an article dated Oct 2) becomes an upcoming catalyst.
Social posts never date a deadline.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from collections import Counter
from datetime import date, datetime, timedelta

from app.analytics.narratives import STORY_MIN_IMPACT, Story
from app.analytics.prepare import Item
from app.analytics.util import filing_parts, money, pct, short_date, trim
from app.schemas import Filing
from app.sources.base import CompanyRef

DEAL_WINDOW = timedelta(days=180)
DEAL_FRESH = timedelta(days=7)  # a deal story whose latest item is older is no longer "in play"
MAJOR_TRUST = 1.1  # wires and majors (Reuters, Bloomberg, WSJ …)
MATERIAL_DEAL = 0.10  # quoted deal value / market cap from which a deal is material for the company
MINOR_DEAL = 0.01  # below this share of the company's market cap a deal is not worth an insight
DEADLINE_HORIZON = timedelta(days=366)

_AGREEMENT_RE = re.compile(
    r"agreement and plan of (?:merger|reorganization)|\bmerger agreement\b|\bplan of merger\b|"
    r"\barrangement agreement\b|\bscheme of arrangement\b|"
    r"\btender offer (?:to (?:purchase|acquire)|for) all\b", re.IGNORECASE)
_DONE_RE = re.compile(r"\bconsummated the (?:merger|acquisition)\b|\bcompleted (?:its|the) (?:merger|acquisition)\b|"
                      r"\bcompletion of (?:the )?(?:merger|acquisition)\b", re.IGNORECASE)
_BROKEN_RE = re.compile(r"\bterminat\w* (?:the |its )?(?:merger agreement|agreement and plan of merger)\b|"
                        r"\b(?:merger agreement|agreement and plan of merger)\b[^.]{0,80}\bterminated\b", re.IGNORECASE)
_BUYER_RE = re.compile(
    r"(?i:agreement and plan of merger|merger agreement)[^.]{0,60}?\b(?i:with)\s+"
    r"(?P<buyer>[A-Z][\w&.'\-]*(?:\s+[A-Z&][\w&.'\-]*){0,5}?,?\s+(?:LLC|L\.L\.C\.|Inc\.?|Corp\.?|Corporation|"
    r"Company|Holdings|L\.P\.|LP|Ltd\.?|Limited|plc|N\.V\.|S\.A\.|AG|SE|Group|Parent))(?=[\s,.;(]|$)|"
    r"\b(?i:to be acquired by)\s+(?P<buyer2>[A-Z][\w&.'\-]*(?:\s+[A-Z&][\w&.'\-]*){0,5})")
_SUFFIX_RE = re.compile(r",?\s+(?:Inc\.?|Corp\.?|Corporation|Co\.?|Ltd\.?|Limited|plc|N\.V\.|S\.A\.|AG|SE|Holdings|"
                        r"Group|Company)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Deal:
    """A signed agreement under which the company is to be acquired."""

    filed: date  # the 8-K quoted as the announcement (the item 1.01 agreement entry when present)
    form: str
    items: tuple[str, ...]
    buyer: str | None
    excerpt: str  # what the announcing filing says

    @property
    def when(self) -> str:
        return short_date(self.filed)

    @property
    def by(self) -> str:
        """' by Action Acquisitions LLC' (or '' when the filing does not name the buyer)."""
        return f" by {self.buyer}" if self.buyer else ""


def _names(company: CompanyRef) -> list[str]:
    """Ways a filing names the company itself ('GoPro', 'GoPro, Inc.', 'the Company')."""
    out = {"the company"}
    for n in (company.name, company.short_name):
        if n:
            out.add(n.lower())
            out.add(_SUFFIX_RE.sub("", n).strip().lower())
    return sorted((n for n in out if len(n) >= 3), key=len, reverse=True)


def _target_re(company: CompanyRef) -> re.Pattern[str]:
    names = "|".join(re.escape(n) for n in _names(company))
    return re.compile(
        rf"\bmerge (?:with and )?into (?:{names})\b|"
        rf"\b(?:{names}),? (?:continuing|surviving|will survive|to survive) as the surviving (?:corporation|company)\b|"
        rf"\b(?:{names})[^.]{{0,40}}\bcontinuing as the surviving (?:corporation|company)\b|"
        rf"\bsubsidiary of parent\b|"
        rf"\b(?:{names}) (?:has agreed |agreed |will |is )?(?:to )?be acquired by\b|"
        rf"\btender offer (?:to (?:purchase|acquire)|for) all (?:of )?(?:the )?(?:issued and )?outstanding "
        rf"(?:shares|common stock) of (?:{names})\b", re.IGNORECASE)


def _text(f: Filing) -> str:
    return " ".join(f.title.split())


def pending_deal(filings: Sequence[Filing], company: CompanyRef, today: date) -> Deal | None:
    """The company's pending acquisition (see module docstring), or None."""
    recent = sorted((f for f in filings if f.form.upper().startswith("8-K")
                     and timedelta(days=-1) <= today - f.date <= DEAL_WINDOW), key=lambda f: f.date)
    if not recent:
        return None
    target = _target_re(company)
    agreements = [f for f in recent if _AGREEMENT_RE.search(_text(f))]
    if not agreements or not any(target.search(_text(f)) for f in recent):
        return None
    first = agreements[0]
    for f in recent:
        if f.date < first.date or f is first:
            continue
        text = _text(f)
        about_deal = bool(_AGREEMENT_RE.search(text) or target.search(text))
        if _BROKEN_RE.search(text) or ("1.02" in f.items and about_deal):
            return None  # terminated
        if _DONE_RE.search(text) or ("2.01" in f.items and about_deal):
            return None  # closed: the stock no longer trades on its own story
    buyer = None
    for f in agreements:
        m = _BUYER_RE.search(_text(f))
        if m:
            buyer = (m.group("buyer") or m.group("buyer2") or "").strip(" ,.") or None
            break
    # The filing that reads best as the announcement: the 1.01 agreement entry when present.
    lead = next((f for f in agreements if "1.01" in f.items), first)
    label, desc = filing_parts(lead.title)
    return Deal(filed=lead.date, form=lead.form, items=tuple(lead.items), buyer=buyer,
                excerpt=trim(desc or label, 200))


# --------------------------------------------------------------------------- #
# A deal in play (news)
# --------------------------------------------------------------------------- #
_AMOUNT = (r"(?P<cur>US\$|\$|£|€)\s?(?P<num>\d+(?:[.,]\d+)?)\s?"
           r"(?P<unit>trillion|tn|billion|bn|b|million|mln|mn|m)\b")
_DEAL_NOUN = r"(?:bid|offer|deal|takeover|acquisition|buyout|merger|approach|proposal|tie-up|transaction)s?\b"
# "$56B eBay Bid", "$56B Takeover Vision" (never "$10.6M stock purchase" or a "$2B debt offer").
_AMOUNT_FIRST_RE = re.compile(
    _AMOUNT + r"(?:[\s-]+(?!(?:shares?|stock|equity|debt|notes?|bonds?|convertible|loan|credit|cash|"
    r"financing|buyback|repurchase|dividend)\b)[\w&.'’-]+){0,3}?[\s-]+" + _DEAL_NOUN, re.IGNORECASE)
# "takeover offer valued at $9 billion", "deal worth $2.1B", "to acquire Rival for $1.2 billion".
_NOUN_FIRST_RE = re.compile(
    r"\b(?:bid|offer|deal|takeover|acquisition|buyout|merger|acquire|buy)\b[^.;:!?$£€]{0,40}?"
    r"\b(?:valued at|worth|for|of)\s+" + _AMOUNT, re.IGNORECASE)
_UNITS = {"t": 1e12, "b": 1e9, "m": 1e6}
_CURRENCIES = {"$": "USD", "us$": "USD", "£": "GBP", "€": "EUR"}

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DEADLINE_RE = re.compile(
    r"\b(?P<what>warrants?|(?:tender )?offer|deadline|(?:shareholder )?vote|bid|exclusivity|go-shop(?: period)?)"
    r"(?:\s+(?!(?:on|in|within)\b)[\w'’-]+){0,2}?\s+"
    r"(?P<verb>expire|expires|expiring|end|ends|ending|close|closes|closing|lapse|lapses|is due|are due|falls?)\s+"
    r"(?:(?:in|within)\s+(?P<days>\d{1,3})\s+days\b|on\s+(?P<date>" + _MONTH +
    r"\s+\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>20\d\d))?\b)", re.IGNORECASE)
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


@dataclass(frozen=True)
class Deadline:
    """An explicit, dated deadline stated in deal coverage."""

    when: date
    what: str  # "Warrants expire"
    item: Item  # the article stating it


@dataclass(frozen=True)
class DealInPlay:
    """A fresh, corroborated deal story involving the company (see module docstring)."""

    story: Story
    coverage: tuple[Item, ...]  # the story's members carrying the M&A event, heaviest first
    amount: float | None  # the quoted deal value, in `currency`
    currency: str | None  # ISO code of the quoted value
    ratio: float | None  # amount / USD market cap (USD amounts only)
    dilution: tuple[Item, ...]  # members reporting new shares for the deal
    deadlines: tuple[Deadline, ...]

    @property
    def material(self) -> bool:
        return self.ratio is not None and self.ratio >= MATERIAL_DEAL

    @property
    def lead(self) -> Item:
        """The headline to quote: the heaviest deal article naming the quoted value (else the heaviest)."""
        quoting = [m for m in self.coverage if self.amount is not None and quoted_amount(m.titles())]
        return (quoting or list(self.coverage))[0]

    @property
    def articles(self) -> int:
        return sum(m.coverage for m in self.coverage)

    @property
    def outlets(self) -> int:
        return len({o for m in self.coverage for o in m.outlets()})

    @property
    def latest(self) -> datetime | None:
        return max((t for m in self.coverage for t in m.times()), default=None)

    @property
    def amount_text(self) -> str | None:
        """'$56B' / '£9B' (None: no value quoted)."""
        return money(self.amount, currency=self.currency) if self.amount is not None else None

    @property
    def share_text(self) -> str | None:
        """'4.5× its market cap' / '9.6% of its market cap' (None: not comparable)."""
        if self.ratio is None:
            return None
        return f"{self.ratio:.1f}× its market cap" if self.ratio >= 1 else \
            f"{pct(self.ratio * 100, sign=False)} of its market cap"

    @property
    def size(self) -> str | None:
        """'$56B, 4.5× its market cap' / '£9B' (None: no value quoted)."""
        if self.amount_text is None:
            return None
        return self.amount_text + (f", {self.share_text}" if self.share_text else "")


def quoted_amount(titles: list[str]) -> tuple[float, str] | None:
    """(value, ISO currency) of the deal value the titles quote most often (ties: the largest)."""
    found: Counter[tuple[float, str]] = Counter()
    for title in titles:
        for rx in (_AMOUNT_FIRST_RE, _NOUN_FIRST_RE):
            for m in rx.finditer(title):
                try:
                    value = float(m.group("num").replace(",", ".")) * _UNITS[m.group("unit")[0].lower()]
                except ValueError:
                    continue
                found[(float(f"{value:.3g}"), _CURRENCIES[m.group("cur").lower()])] += 1
    if not found:
        return None
    (value, currency), _ = max(found.items(), key=lambda kv: (kv[1], kv[0][0]))
    return value, currency


def deadlines(items: list[Item], today: date) -> list[Deadline]:
    """Explicit deadlines in published coverage: 'in N days' counted from the article's date,
    'on Nov 2' in the article's year (the next one for a date months before it); never from posts."""
    out: dict[tuple[date, str], Deadline] = {}
    for it in items:
        if it.group != "news" or it.timestamp is None:
            continue
        published = it.timestamp.date()
        for text in (it.title, it.body or ""):
            for m in _DEADLINE_RE.finditer(text):
                when = _deadline_date(m, published)
                if when is None or not today <= when <= today + DEADLINE_HORIZON:
                    continue
                what = f"{m.group('what')} {m.group('verb')}".strip()
                key = (when, what.lower())
                out.setdefault(key, Deadline(when=when, what=what[:1].upper() + what[1:], item=it))
    return sorted(out.values(), key=lambda d: d.when)


def _deadline_date(m: re.Match[str], published: date) -> date | None:
    if m.group("days"):
        return published + timedelta(days=int(m.group("days")))
    month_day = (m.group("date") or "").replace(".", " ").split()
    if len(month_day) != 2:
        return None
    month = _MONTHS.index(month_day[0][:3].lower()) + 1
    year = int(m.group("year")) if m.group("year") else published.year
    try:
        when = date(year, month, int(month_day[1]))
    except ValueError:
        return None
    # A yearless date well before the article is next year's ("on Jan 15" written in October); one just
    # before it is a past date, never pushed a year out.
    if not m.group("year") and when < published - timedelta(days=183):
        try:
            when = date(year + 1, month, int(month_day[1]))
        except ValueError:
            return None
    return when


def _corroborated(carriers: list[Item]) -> bool:
    outlets = {o for it in carriers for o in it.outlets()}
    return (sum(it.coverage for it in carriers) >= 2 and len(outlets) >= 2) or any(
        it.trust >= MAJOR_TRUST and it.relevance >= 0.8 for it in carriers)


def deal_in_play(stories: list[Story], market_cap_usd: float | None, now: datetime) -> DealInPlay | None:
    """The most prominent qualifying deal story (see module docstring), or None."""
    for story in sorted(stories, key=lambda s: -s.narrative.impact):
        n = story.narrative
        carriers = sorted((m for m in story.members if "m_and_a" in m.event_keys), key=lambda m: (-m.weight, m.id))
        latest = max((t for m in carriers for t in m.times()), default=None)
        if (not story.deal or n.impact < STORY_MIN_IMPACT or latest is None
                or now - latest > DEAL_FRESH or not _corroborated(carriers)):
            continue
        quoted = quoted_amount([t for m in carriers for t in m.titles()])
        amount, currency = quoted if quoted else (None, None)
        ratio = amount / market_cap_usd if amount and currency == "USD" and market_cap_usd else None
        if ratio is not None and ratio < MINOR_DEAL:
            continue  # a bolt-on: the catalyst list still carries the story
        dilution = tuple(m for m in story.members if "offering" in m.event_keys and m.relevance >= 0.6)
        return DealInPlay(story=story, coverage=tuple(carriers), amount=amount, currency=currency, ratio=ratio,
                          dilution=dilution,
                          deadlines=tuple(deadlines(story.members, now.date())))
    return None
