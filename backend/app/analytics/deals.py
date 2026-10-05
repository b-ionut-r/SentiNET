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
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from app.analytics.util import filing_parts, short_date, trim
from app.schemas import Filing
from app.sources.base import CompanyRef

DEAL_WINDOW = timedelta(days=180)

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
