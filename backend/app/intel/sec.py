"""SEC EDGAR: ticker->CIK map, recent filings (decoded), Form 4 insider fallback.

EDGAR is free and authoritative but strict about etiquette: <= 10 requests/s
(enforced by `app.core.ratelimit`) and a descriptive User-Agent with contact
info. Its firewall rejects UAs that contain a URL (403 "Request Rate Threshold
Exceeded" — verified 2026-10-04), so EDGAR gets its own UA without one.

Filings are turned into intel, not a raw list: 8-K item codes are decoded into
plain titles with importance/polarity priors (4.02 non-reliance and 1.03
bankruptcy are red flags; 2.02 is the earnings release), and floods of routine
forms (Form 4/144 insider paperwork, bank structured-note prospectuses) are
collapsed into one summary row each. Recent narrative 8-Ks are re-judged from
their own text, which can raise a prior (a deal, a CEO exit) or explain one
away (stock paid for an acquisition under 3.02; regained compliance under
3.01); a red flag (high + bear) always needs high, bear evidence.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC
from typing import Any, Literal
from urllib.parse import quote

from defusedxml import ElementTree as ET

from app.config import settings
from app.core.cache import cached
from app.core.http import UpstreamError, fetch, fetch_json
from app.intel.transforms import classify_insider, insider_view, pretty_insider_name
from app.schemas import Filing, InsiderTxn, InsiderView
from app.sources.base import CompanyRef

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}-index.htm"
BROWSE_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form}&dateb=&owner=include&count=40"

Importance = Literal["high", "medium", "low"]
Pol = Literal["bull", "bear", "neutral"]


def sec_headers() -> dict[str, str]:
    """EDGAR-compliant UA: product + contact email, no URL (EDGAR's WAF blocks those)."""
    return {"User-Agent": f"SentiNET/2.0 ({settings.contact_email})", "Accept-Encoding": "gzip, deflate"}


# --------------------------------------------------------------------------- #
# 8-K items & form catalog
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ItemInfo:
    title: str
    importance: Importance
    polarity: Pol


ITEMS_8K: dict[str, ItemInfo] = {
    "1.01": ItemInfo("Entered a material agreement", "medium", "neutral"),
    "1.02": ItemInfo("Terminated a material agreement", "medium", "neutral"),  # often a refinancing
    "1.03": ItemInfo("Bankruptcy or receivership", "high", "bear"),
    "1.04": ItemInfo("Mine safety violation", "low", "bear"),
    "1.05": ItemInfo("Material cybersecurity incident", "high", "bear"),
    "2.01": ItemInfo("Completed an acquisition or disposition", "medium", "neutral"),
    "2.02": ItemInfo("Results of operations (earnings release)", "medium", "neutral"),
    "2.03": ItemInfo("New debt or financial obligation", "medium", "neutral"),
    "2.04": ItemInfo("Debt acceleration triggered", "high", "bear"),
    "2.05": ItemInfo("Exit/restructuring costs (layoffs, closures)", "high", "bear"),
    "2.06": ItemInfo("Material impairment", "high", "bear"),
    "3.01": ItemInfo("Delisting notice / listing-rule failure", "high", "bear"),
    "3.02": ItemInfo("Unregistered sale of equity (dilution)", "medium", "bear"),
    "3.03": ItemInfo("Change to shareholder rights", "medium", "neutral"),
    "4.01": ItemInfo("Auditor change", "high", "bear"),
    "4.02": ItemInfo("Non-reliance on prior financials (restatement)", "high", "bear"),
    "5.01": ItemInfo("Change in control", "high", "neutral"),
    "5.02": ItemInfo("Director/officer departure or appointment", "medium", "neutral"),
    "5.03": ItemInfo("Bylaw or fiscal-year change", "low", "neutral"),
    "5.04": ItemInfo("Benefit-plan trading suspension", "low", "neutral"),
    "5.05": ItemInfo("Code of ethics change or waiver", "low", "neutral"),
    "5.06": ItemInfo("Change in shell company status", "medium", "neutral"),
    "5.07": ItemInfo("Shareholder vote results", "low", "neutral"),
    "5.08": ItemInfo("Shareholder director nominations", "low", "neutral"),
    "6.01": ItemInfo("ABS informational material", "low", "neutral"),
    "6.02": ItemInfo("ABS servicer/trustee change", "low", "neutral"),
    "6.03": ItemInfo("ABS credit enhancement change", "low", "neutral"),
    "6.04": ItemInfo("ABS failure to make distribution", "high", "bear"),
    "6.05": ItemInfo("ABS securities act updating", "low", "neutral"),
    "7.01": ItemInfo("Regulation FD disclosure", "low", "neutral"),
    "8.01": ItemInfo("Other material event", "low", "neutral"),
    "9.01": ItemInfo("Financial statements and exhibits", "low", "neutral"),
}


@dataclass(frozen=True)
class FormInfo:
    title: str
    importance: Importance
    polarity: Pol = "neutral"
    family: str | None = None  # forms that flood are collapsed per family


FORMS: dict[str, FormInfo] = {
    "10-K": FormInfo("Annual report", "medium"),
    "10-K/A": FormInfo("Amended annual report", "medium"),
    "10-Q": FormInfo("Quarterly report", "medium"),
    "10-Q/A": FormInfo("Amended quarterly report", "medium"),
    "20-F": FormInfo("Annual report (foreign issuer)", "medium"),
    "40-F": FormInfo("Annual report (Canadian issuer)", "medium"),
    "6-K": FormInfo("Foreign issuer report", "low", family="6-K"),
    "NT 10-K": FormInfo("Late annual report notice", "high", "bear"),
    "NT 10-Q": FormInfo("Late quarterly report notice", "high", "bear"),
    "NT 20-F": FormInfo("Late annual report notice", "high", "bear"),
    "S-1": FormInfo("Registration statement (IPO/offering)", "medium"),
    "S-1/A": FormInfo("Amended registration statement", "low"),
    "F-1": FormInfo("Registration statement (foreign IPO/offering)", "medium"),
    "S-3": FormInfo("Shelf registration (future securities offerings)", "medium"),
    "S-3ASR": FormInfo("Automatic shelf registration (future offerings)", "medium"),
    "F-3": FormInfo("Shelf registration (foreign issuer)", "medium"),
    "424B1": FormInfo("Prospectus: securities offering", "medium", family="424B"),
    "424B3": FormInfo("Prospectus: securities offering", "medium", family="424B"),
    "424B4": FormInfo("Prospectus: securities offering priced", "medium", family="424B"),
    "424B5": FormInfo("Prospectus supplement: securities offering", "medium", family="424B"),
    "424B7": FormInfo("Prospectus: resale by holders", "medium", family="424B"),
    "424B8": FormInfo("Prospectus (late-filed supplement)", "low", family="424B2"),
    "424B2": FormInfo("Pricing supplement (notes/debt program)", "low", family="424B2"),
    "FWP": FormInfo("Free-writing prospectus (offering materials)", "low", family="FWP"),
    "S-8": FormInfo("Employee stock plan registration", "low"),
    "S-4": FormInfo("Merger/exchange registration", "high"),
    "SC 13D": FormInfo("Activist/control stake disclosed (13D)", "high"),
    "SC 13D/A": FormInfo("Activist/control stake updated (13D/A)", "medium"),
    "SCHEDULE 13D": FormInfo("Activist/control stake disclosed (13D)", "high"),
    "SCHEDULE 13D/A": FormInfo("Activist/control stake updated (13D/A)", "medium"),
    "SC 13G": FormInfo("Passive 5%+ holder disclosed (13G)", "low", family="13G"),
    "SC 13G/A": FormInfo("Passive 5%+ holder update (13G/A)", "low", family="13G"),
    "SCHEDULE 13G": FormInfo("Passive 5%+ holder disclosed (13G)", "low", family="13G"),
    "SCHEDULE 13G/A": FormInfo("Passive 5%+ holder update (13G/A)", "low", family="13G"),
    "SC TO-T": FormInfo("Third-party tender offer (takeover bid)", "high", "bull"),
    "SC TO-T/A": FormInfo("Tender offer amendment", "medium"),
    "SC TO-I": FormInfo("Issuer tender offer (buyback)", "high", "bull"),
    "SC TO-I/A": FormInfo("Issuer tender offer amendment", "medium"),
    "SC 14D9": FormInfo("Board response to tender offer", "high"),
    "DEFM14A": FormInfo("Definitive merger proxy", "high"),
    "PREM14A": FormInfo("Preliminary merger proxy", "high"),
    "425": FormInfo("Merger communication", "medium", family="425"),
    "DEF 14A": FormInfo("Proxy statement (shareholder meeting)", "low"),
    "PRE 14A": FormInfo("Preliminary proxy statement", "low"),
    "DEFA14A": FormInfo("Additional proxy materials", "low", family="DEFA14A"),
    "DFAN14A": FormInfo("Dissident proxy materials (proxy fight)", "medium", family="DFAN14A"),
    "PX14A6G": FormInfo("Shareholder exempt solicitation", "low", family="PX14A6G"),
    "25-NSE": FormInfo("Exchange delisting of a security class", "medium"),
    "25": FormInfo("Voluntary delisting of a security class", "medium", "bear"),
    "15-12B": FormInfo("Deregistration (going dark / post-deal)", "high", "bear"),
    "15-12G": FormInfo("Deregistration (going dark / post-deal)", "high", "bear"),
    "15-15D": FormInfo("Suspension of reporting duty", "high", "bear"),
    "CORRESP": FormInfo("Company letter to SEC staff", "low", family="CORRESP"),
    "UPLOAD": FormInfo("SEC staff comment letter", "low", family="CORRESP"),
    "ARS": FormInfo("Annual report to shareholders", "low"),
    "SD": FormInfo("Specialized disclosure (conflict minerals)", "low"),
    "11-K": FormInfo("Employee benefit plan annual report", "low"),
    "13F-HR": FormInfo("Institutional holdings report", "low"),
    "N-PX": FormInfo("Proxy voting record", "low"),
    "3": FormInfo("Initial insider ownership", "low", family="insider"),
    "3/A": FormInfo("Initial insider ownership", "low", family="insider"),
    "4": FormInfo("Insider transaction report", "low", family="insider"),
    "4/A": FormInfo("Insider transaction report", "low", family="insider"),
    "5": FormInfo("Annual insider transaction report", "low", family="insider"),
    "5/A": FormInfo("Annual insider transaction report", "low", family="insider"),
    "144": FormInfo("Notice of proposed insider sale", "low", family="144"),
    "144/A": FormInfo("Notice of proposed insider sale", "low", family="144"),
}

# Families that are always summarized, and the threshold for the rest.
_ALWAYS_COLLAPSE = {"insider", "144", "424B2", "FWP", "DEFA14A", "PX14A6G", "6-K", "13G", "CORRESP"}
_COLLAPSE_AT = 3
_FAMILY_SUMMARY: dict[str, str] = {
    "insider": "insider transaction reports (Forms 3/4/5)",
    "144": "notices of proposed insider sales (Form 144)",
    "424B": "offering prospectuses (424B)",
    "424B2": "pricing supplements for notes/debt programs",
    "FWP": "free-writing prospectuses (offering materials)",
    "DEFA14A": "additional proxy filings",
    "DFAN14A": "dissident proxy filings (proxy fight)",
    "PX14A6G": "shareholder exempt solicitations",
    "6-K": "foreign issuer reports (6-K)",
    "13G": "passive 5%+ holder disclosures (13G)",
    "CORRESP": "SEC comment-letter correspondence",
    "425": "merger communications (425)",
}
# Display form and EDGAR browse filter per collapsed family.
_FAMILY_FORM = {"insider": "4", "13G": "13G", "424B": "424B", "CORRESP": "CORRESP"}
_FAMILY_BROWSE = {"insider": "4", "13G": "SC 13G", "424B": "424B", "6-K": "6-K", "CORRESP": "CORRESP"}
FLOOD_WINDOW_DAYS = 90
_RANK = {"high": 3, "medium": 2, "low": 1}


def _pick_polarity(pols: list[Pol]) -> Pol:
    if "bear" in pols:
        return "bear"
    return "bull" if "bull" in pols else "neutral"


def _settle(evidence: list[tuple[Importance, Pol | None]]) -> tuple[Importance, Pol]:
    """Overall (importance, polarity) of a filing from its (importance, polarity) evidence.

    Importance is the strongest evidence's; polarity is bear > bull > neutral — except that high + bear
    (a red flag downstream) needs evidence that is itself high and bear. A medium dilution prior (Item
    3.02) next to a high but directionless deal or change-in-control signal is a big event, not a red
    flag: AMD's all-stock purchase of World Labs (8-K, 2026-09-28) was raised as one.
    """
    importance: Importance = max((imp for imp, _ in evidence), key=_RANK.__getitem__, default="low")
    polarity = _pick_polarity([pol for _, pol in evidence if pol])
    if importance == "high" and polarity == "bear" and ("high", "bear") not in evidence:
        polarity = "neutral"
    return importance, polarity


def decode_8k(items_field: str) -> tuple[str, list[str], Importance, Pol]:
    """'2.02,9.01' -> ("Results of operations (earnings release)", codes, importance, polarity)."""
    codes = [c.strip() for c in (items_field or "").split(",") if c.strip()]
    known = [(c, ITEMS_8K[c]) for c in codes if c in ITEMS_8K]
    infos = [info for _, info in known]
    meaningful = [info for c, info in known if c != "9.01"] or infos
    if not meaningful:
        return "Current report", codes, "low", "neutral"
    ordered = sorted(meaningful, key=lambda i: -_RANK[i.importance])
    title = "; ".join(dict.fromkeys(i.title for i in ordered))
    return (title, codes, *_settle([(i.importance, i.polarity) for i in meaningful]))


def _fmt_day(d: date) -> str:
    return f"{d:%b} {d.day}"


def filings_from_submissions(
    sub: dict[str, Any], *, limit: int = 20, today: date | None = None, window_days: int = 365
) -> list[Filing]:
    """Pure: EDGAR submissions JSON -> decoded, flood-collapsed, newest-first filings."""
    recent = ((sub or {}).get("filings") or {}).get("recent") or {}
    forms: list[str] = recent.get("form") or []
    cik_int = int(str(sub.get("cik") or "0") or 0)
    today = today or datetime.now(UTC).date()
    since = today - timedelta(days=window_days)

    rows: list[tuple[Filing, str | None]] = []
    for i, form in enumerate(forms):
        try:
            filed = date.fromisoformat(recent["filingDate"][i])
        except (KeyError, IndexError, ValueError):
            continue
        if filed < since:
            continue
        acc = recent["accessionNumber"][i]
        url = INDEX_URL.format(cik=cik_int, acc_nodash=acc.replace("-", ""), acc=acc)
        if form in {"8-K", "8-K/A"}:
            title, codes, importance, polarity = decode_8k(recent.get("items", [""] * len(forms))[i])
            if form == "8-K/A":
                title = f"Amended: {title}"
            rows.append((Filing(form=form, date=filed, title=title, items=codes, url=url,
                                importance=importance, polarity=polarity), None))
            continue
        info = FORMS.get(form) or FORMS.get(form.replace("SCHEDULE ", "SC "))
        if info is None:
            desc = (recent.get("primaryDocDescription") or [""] * len(forms))[i] or form
            desc = desc.capitalize() if desc.isupper() and len(desc) > 6 else desc  # "AMENDED AND …" -> "Amended and …"
            info = FormInfo(desc if desc.upper() != form.upper() else f"Form {form}", "low")
        rows.append((Filing(form=form, date=filed, title=info.title, url=url,
                            importance=info.importance, polarity=info.polarity), info.family))

    family_counts: dict[str, int] = {}
    for _, fam in rows:
        if fam:
            family_counts[fam] = family_counts.get(fam, 0) + 1
    collapse = {f for f, n in family_counts.items() if f in _ALWAYS_COLLAPSE or n >= _COLLAPSE_AT}

    out: list[Filing] = []
    grouped: dict[str, list[Filing]] = {}
    for filing, fam in rows:
        if fam in collapse:
            grouped.setdefault(fam, []).append(filing)  # type: ignore[arg-type]
        else:
            out.append(filing)
    flood_since = today - timedelta(days=FLOOD_WINDOW_DAYS)
    for fam, members in grouped.items():
        members.sort(key=lambda f: f.date, reverse=True)
        # A flood summary is about recent activity; older members just age out.
        members = [m for m in members if m.date >= flood_since] or members[:1]
        n = len(members)
        if n == 1:
            only = members[0]
            out.append(only.model_copy(update={"title": f"{only.title} ({_fmt_day(only.date)})"}))
            continue
        # A routine program (dozens of note supplements) is not a discrete event.
        importance = "low" if n >= 5 else max((m.importance for m in members), key=lambda x: _RANK[x])
        out.append(Filing(
            form=_FAMILY_FORM.get(fam, members[0].form),
            date=members[0].date,
            title=f"{n} {_FAMILY_SUMMARY.get(fam, members[0].title)} since {_fmt_day(members[-1].date)}",
            url=BROWSE_URL.format(cik=cik_int, form=quote(_FAMILY_BROWSE.get(fam, members[0].form))),
            importance=importance,  # type: ignore[arg-type]
            polarity=_pick_polarity([m.polarity for m in members]),
        ))
    out.sort(key=lambda f: (f.date, _RANK[f.importance]), reverse=True)
    return out[:limit]


# --------------------------------------------------------------------------- #
# 8-K narrative excerpts
# --------------------------------------------------------------------------- #
# Items whose text says *what happened* (worth one document fetch). Earnings
# releases (2.02), Reg FD (7.01), votes (5.07) and exhibits (9.01) only point at
# attachments, so their decoded title already says everything.
_NARRATIVE_ITEMS = {"1.01", "1.02", "1.03", "1.05", "2.01", "2.03", "2.04", "2.05", "2.06", "3.01",
                    "3.02", "4.01", "4.02", "5.01", "5.02", "8.01"}
_ITEM_HEADER = re.compile(r"\bItem\s+(\d\.\d\d)\b\.?", re.IGNORECASE)
_DEFINED_TERM = re.compile(r"\s*\((?:[^()]{0,60}?,\s*)?(?:the\s+|collectively\s+)?[“\"][^”\"]{1,40}[”\"]\)")
_BOILERPLATE = re.compile(r"press release|exhibit 99|incorporated (?:herein )?by reference|furnished|"
                          r"forward-looking|shall not be deemed|set forth (?:in|under|above|below)|"
                          r"(?:these|such) statements|current (?:opinions|expectations|beliefs)|"
                          r"is hereby incorporated|see item \d", re.IGNORECASE)
_LEAD_DATE = re.compile(r"^(?:\([a-z]\)\s*)?(?:on|effective)\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4},?\s+",
                        re.UNICODE | re.IGNORECASE)
# "As previously disclosed, on March 4, 2026, …" / "As previously reported … on August 6, 2025, …"
_PREVIOUSLY = re.compile(r"^(?:(?:In addition|Additionally|Further(?:more)?|Also),\s+)?"
                         r"As (?:previously |was )?(?:reported|disclosed|announced)(?:\b.{0,200}?\d{4})?,\s+",
                         re.IGNORECASE)
_SENTENCE_END = re.compile(r"\.\s+(?=[A-Z(“\"])")
_ABBREVIATIONS = {"inc", "corp", "co", "ltd", "no", "mr", "ms", "mrs", "dr", "st", "jr", "sr", "s", "u", "approx",
                  "vs", "n.a", "l.p", "l.l.c", "e.g", "i.e"}


def _split_sentences(text: str) -> list[str]:
    """Sentence split that survives "Hugging Face, Inc. The …" and "Ajay K. Puri"."""
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        prev = text[start:match.start()].rsplit(" ", 1)[-1].lower()
        if prev in _ABBREVIATIONS or (len(prev) == 1 and prev.isalpha()):
            continue
        out.append(text[start:match.start() + 1])
        start = match.end()
    out.append(text[start:])
    return out


_TAGS = re.compile(r"<[^>]+>")


def html_to_text(doc: str) -> str:
    """Filing HTML -> one line of plain text."""
    doc = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", doc)
    text = html.unescape(_TAGS.sub(" ", doc)).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+([.,;:)])", r"\1", text).strip()  # "2026 ." -> "2026."


def summarize_8k(doc: str, items: list[str], max_chars: int = 300) -> str | None:
    """Pure: the first substantive sentence(s) of the most important narrative item.

    Real filing text only (lightly trimmed: defined-term parentheticals and the
    leading "On <date>," clause are removed) — never paraphrased.
    """
    text = html_to_text(doc)
    wanted = sorted((c for c in items if c in _NARRATIVE_ITEMS and c in ITEMS_8K),
                    key=lambda c: -_RANK[ITEMS_8K[c].importance])
    headers = list(_ITEM_HEADER.finditer(text))
    for code in wanted:
        for i, match in enumerate(headers):
            if match.group(1) != code:
                continue
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            section = text[match.end():end]
            # Skip the item's official caption: start at the first "On <date>" / "(a)" narrative,
            # else right after the caption's closing period.
            start = re.search(r"\((?:[a-z])\)\s|\bOn\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\bEffective\s", section[:400])
            if start:
                body = section[start.start():]
            else:
                parts = _split_sentences(section)
                body = " ".join(parts[1:]) if len(parts) > 1 else ""
            excerpt = _excerpt(body, max_chars) if body else None
            if excerpt:
                return excerpt
    return None


_MONEY = re.compile(r"\$\s?\d|\b\d[\d,.]*\s?(?:billion|million|percent)\b|\d%")


def _clean_sentence(sentence: str) -> str:
    sentence = _PREVIOUSLY.sub("", _DEFINED_TERM.sub("", sentence).strip())
    sentence = _LEAD_DATE.sub("", sentence).strip()
    return (sentence[:1].upper() + sentence[1:]).rstrip(".") + "."


def _fit(sentence: str, budget: int) -> str | None:
    """Sentence within `budget` chars, cut at a clause boundary if needed."""
    if len(sentence) <= budget:
        return sentence
    cut = sentence[: budget - 1]
    cut = cut[: cut.rfind(", ")] if ", " in cut[budget // 2:] else cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;") + "…" if len(cut) > 40 else None


def _excerpt(section: str, max_chars: int) -> str | None:
    """Lead sentence + (preferably) the next sentence that carries a number."""
    sentences = [s.strip() for s in _split_sentences(section) if s.strip()]
    # A real narrative sentence names something: skip boilerplate and dangling cross-references.
    narrative = [s for s in sentences[:6] if not _BOILERPLATE.search(s) and len(s) >= 40]
    if not narrative:
        return None
    lead = _fit(_clean_sentence(narrative[0]), max_chars)
    if not lead:
        return None
    rest = [_clean_sentence(s) for s in narrative[1:3]]
    follow = next((s for s in rest if _MONEY.search(s)), rest[0] if rest else None)
    extra = _fit(follow, max_chars - len(lead) - 1) if follow else None
    return f"{lead} {extra}" if extra else lead


# A listing problem that is over: "received notice from Nasdaq that it has regained compliance with the
# minimum bid price requirement" (GPRO 8-K, 2026-09-16). Deficiency letters say "has 180 days to regain
# compliance" / "if the Company regains compliance", so only completed wording counts.
_NOT = r"(?<!not )(?<!n't )(?<!yet )(?<!never )"  # "has not (yet) regained compliance" is still a deficiency
_REGAINED = re.compile(rf"{_NOT}\bregained (?:full )?compliance|{_NOT}\bevidenced compliance\b|"
                       r"\bcompliance has been (?:regained|restored)\b|"
                       r"\b(?:matter|deficiency) (?:is|has been|is now|was) (?:now )?(?:closed|resolved|cured)\b",
                       re.IGNORECASE)
_LISTING_DEFICIENCY = re.compile(r"deficiency (?:letter|notice)|listing qualifications|minimum bid price|"
                                 r"regain compliance|(?:notice|notification) of delisting|delisting determination",
                                 re.IGNORECASE)
# A deal. "to acquire Widget Co." / "to acquire all of the outstanding shares of Target" is one;
# "warrants to acquire 1,131,273 shares" (HHH 8-K, a warrant sale) and "to acquire up to 2,500,000
# shares" (a PIPE) are securities, not a deal.
_SHARE_QTY = (r"(?:up to |an aggregate of |approximately )*(?:[\d,.]+ (?:million )?)?(?:additional )?"
              r"(?:shares|units|ordinary shares|common stock|ADSs?)\b")
_DEAL = re.compile(rf"definitive (?:merger )?agreement|merger agreement|agreement and plan of merger|"
                   rf"(?<!warrants )(?<!warrant )\bto acquire\b(?! {_SHARE_QTY})|tender offer|business combination",
                   re.IGNORECASE)
# Item 3.02 covers any unregistered share issuance: a dilutive cash raise (its bear prior), but also the
# stock an acquirer pays a target with ("Agreement and Plan of Merger to acquire all of the equity
# interests in World Labs … to be paid in shares of the Company's common stock" — AMD 8-K, 2026-09-28),
# which is the deal itself. Shares sold for cash alongside a deal (a SPAC's PIPE) keep the prior.
_STOCK_PAID = re.compile(
    r"\b(?:paid|payable|consideration)\b.{0,60}?\b(?:shares|stock)\b|"
    r"\b(?:shares|stock)\b.{0,80}?\b(?:as|in) (?:partial |full |the )?(?:merger |purchase |acquisition )?"
    r"consideration\b|"
    r"\bin exchange for (?:all|the|their|100%)\b.{0,60}?\b(?:shares|stock|equity|interests|units)\b|"
    r"\bin connection with the (?:acquisition|merger|transaction)\b.{0,60}?\bissu\w*\b.{0,40}?\b(?:shares|stock)\b|"
    r"\bissu\w*\b.{0,80}?\b(?:shares|stock)\b.{0,60}?\bto the (?:former )?(?:stockholders|shareholders|"
    r"equityholders|equity holders|holders|members|owners|sellers)\b",
    re.IGNORECASE | re.DOTALL)
_ACQUIRED = re.compile(rf"\bacqui(?:red|sition of)\b(?! {_SHARE_QTY})", re.IGNORECASE)  # "acquired Azio AI"
_CASH_RAISE = re.compile(r"securities purchase agreement|subscription agreement|\b(?:gross|net|aggregate) proceeds\b|"
                         r"registered direct|\bPIPE\b|\bfor cash\b|per share in cash", re.IGNORECASE)

_EXCERPT_RULES: tuple[tuple[re.Pattern[str], Importance, Pol | None], ...] = (
    (re.compile(r"going concern|material weakness|subpoena|wells notice|investigation by|class action",
                re.IGNORECASE), "high", "bear"),
    (_LISTING_DEFICIENCY, "high", "bear"),  # first: an open deficiency outranks one that was cured
    (_REGAINED, "medium", "bull"),
    (_DEAL, "high", None),
    (re.compile(r"(?:increase|authoriz|approv)\w*.{0,80}(?:share repurchase|stock repurchase|buyback)|"
                r"(?:share repurchase|stock repurchase|buyback).{0,80}(?:increase|authoriz|approv)",
                re.IGNORECASE | re.DOTALL), "medium", "bull"),
)

# Executive changes (Item 5.02 wording varies: "intends to retire as Chief Executive Officer",
# "the Board terminated the employment of X, its CEO", "X, our CFO, resigned"), so the title and the
# verb are matched in either order within one sentence. Compensation boilerplate ("upon a termination
# without cause", "retention award") and "named executive officers" are not changes.
_CHIEF = r"\b(?:chief executive|ceo)\b"
_EXEC_TITLE = r"\b(?:chief executive|chief financial|ceo|cfo)\b"
_ABRUPT = (r"\b(?:resign(?:s|ed|ing|ation)?|terminat\w* (?:of )?(?:the |his |her |its )?employment|"
           r"remov(?:ed|al) (?:as|from)|separation (?:from|agreement)|for cause|effective immediately)\b")
_PLANNED = r"\b(?:retir(?:e|es|ed|ing|ement)|step(?:s|ped|ping)? down|depart(?:s|ed|ing|ure)|transition(?:s|ed|ing)?)\b"
_SUCCESSION = r"\b(?:appoint(?:s|ed|ment)?|succe(?:ed|eds|eded|eding|ssor|ssion)|named(?! executive)|promot(?:ed|ion))\b"
_COMP_CONTEXT = re.compile(r"without cause|good reason|retention|\baward|\bvest|severance|in the event of|"
                           r"change (?:in|of) control|named executive", re.IGNORECASE)


def _near(a: str, b: str, window: int = 120) -> re.Pattern[str]:
    """`a` and `b` within `window` characters, in either order."""
    return re.compile(rf"{a}.{{0,{window}}}{b}|{b}.{{0,{window}}}{a}", re.IGNORECASE | re.DOTALL)


_EXEC_RULES: tuple[tuple[re.Pattern[str], Importance, Pol | None], ...] = (
    (_near(_EXEC_TITLE, _ABRUPT), "high", "bear"),
    (_near(_EXEC_TITLE, _PLANNED), "high", None),
    (_near(_CHIEF, _SUCCESSION), "high", None),
)


def _exec_change(text: str) -> tuple[Importance, Pol | None] | None:
    """(importance, polarity) of a CEO/CFO change named in `text`, judged sentence by sentence."""
    found: tuple[Importance, Pol | None] | None = None
    for sentence in _split_sentences(text):
        if _COMP_CONTEXT.search(sentence):
            continue
        for pattern, imp, pol in _EXEC_RULES:
            if pattern.search(sentence):
                if pol == "bear":
                    return imp, pol
                found = found or (imp, pol)
                break
    return found


def _rule_matches(pattern: re.Pattern[str], text: str) -> bool:
    """Whether a rule fires; the deficiency rule only on sentences that don't report it resolved
    ("regained compliance with the minimum bid price requirement" names the old deficiency)."""
    if pattern is not _LISTING_DEFICIENCY:
        return pattern.search(text) is not None
    return any(pattern.search(s) and not _REGAINED.search(s) for s in _split_sentences(text))


# Item captions that the text can show to be the opposite of their bear prior, and what they then say.
_WAIVED_CAPTION = {"3.01": "Regained compliance with listing rules", "3.02": "Acquisition paid in stock"}


def _waived_items(filing: Filing, excerpt: str, hits: list[tuple[Importance, Pol | None]]) -> set[str]:
    """Bear item priors the filing text explains away (see `_WAIVED_CAPTION`)."""
    waived: set[str] = set()
    if "3.01" in filing.items and _REGAINED.search(excerpt) and not any(pol == "bear" for _, pol in hits):
        waived.add("3.01")  # filed under 3.01, but the notice is that compliance was regained
    if "3.02" in filing.items and (_DEAL.search(excerpt) or _ACQUIRED.search(excerpt)) \
            and _STOCK_PAID.search(excerpt) and not _CASH_RAISE.search(excerpt):
        waived.add("3.02")  # the new shares pay for the company's acquisition: not a dilutive cash raise
    return waived


def reassess_8k(filing: Filing, excerpt: str) -> Filing:
    """Raise importance / set polarity from what the filing text actually says."""
    hits = [(imp, pol) for pattern, imp, pol in _EXCERPT_RULES if _rule_matches(pattern, excerpt)]
    if (change := _exec_change(excerpt)) is not None:
        hits.append(change)
    title, prior = filing.title, (filing.importance, filing.polarity)
    if waived := _waived_items(filing, excerpt, hits):
        _, _, imp, pol = decode_8k(",".join(c for c in filing.items if c not in waived))
        prior = (imp, pol)  # what the remaining items say on their own
        for code in waived:
            title = title.replace(ITEMS_8K[code].title, _WAIVED_CAPTION[code])
    importance, polarity = _settle([prior, *hits])
    return filing.model_copy(update={"title": f"{title}: {excerpt}", "importance": importance,
                                     "polarity": polarity})


def narrative_8k_docs(sub: dict[str, Any], *, today: date, window_days: int = 60, limit: int = 5) -> dict[str, str]:
    """Pure: {filing index URL: primary document URL} for recent narrative 8-Ks."""
    recent = ((sub or {}).get("filings") or {}).get("recent") or {}
    cik_int = int(str(sub.get("cik") or "0") or 0)
    out: dict[str, str] = {}
    for i, form in enumerate(recent.get("form") or []):
        if form not in {"8-K", "8-K/A"}:
            continue
        try:
            filed = date.fromisoformat(recent["filingDate"][i])
        except (KeyError, ValueError):
            continue
        if (today - filed).days > window_days:
            continue
        codes = {c.strip() for c in (recent.get("items") or [""] * len(recent["form"]))[i].split(",")}
        if not codes & _NARRATIVE_ITEMS:
            continue
        acc = recent["accessionNumber"][i]
        nodash = acc.replace("-", "")
        index_url = INDEX_URL.format(cik=cik_int, acc_nodash=nodash, acc=acc)
        out[index_url] = ARCHIVE_URL.format(cik=cik_int, acc_nodash=nodash, doc=recent["primaryDocument"][i])
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
@cached(ttl=86400, none_ttl=120)
async def get_cik_map() -> dict[str, tuple[str, str]]:
    """TICKER -> (10-digit CIK, registrant title), from SEC company_tickers.json."""
    data = await fetch_json(TICKERS_URL, headers=sec_headers(), timeout=20.0)
    return cik_map_from_json(data)


def cik_map_from_json(data: Any) -> dict[str, tuple[str, str]]:
    """Pure: SEC company_tickers.json -> {TICKER: (cik10, title)} (first listing wins)."""
    rows = data.values() if isinstance(data, dict) else (data or [])
    out: dict[str, tuple[str, str]] = {}
    for row in rows:
        try:
            ticker = str(row["ticker"]).upper().replace(".", "-")
            cik = str(int(row["cik_str"])).zfill(10)
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(ticker, (cik, str(row.get("title") or ticker)))
    return out


@cached(ttl=settings.intel_cache_ttl, none_ttl=120)
async def get_submissions(cik: str) -> dict[str, Any] | None:
    """EDGAR submissions JSON for a CIK (recent ~1000 filings / 1 year)."""
    try:
        return await fetch_json(SUBMISSIONS_URL.format(cik=str(cik).zfill(10)), headers=sec_headers(), timeout=15.0)
    except Exception as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            return None
        raise UpstreamError(f"SEC submissions: {type(exc).__name__}") from exc


async def get_filings(company: CompanyRef, limit: int = 20) -> list[Filing]:
    """Recent decoded filings for a US-listed operating company ([] for funds/crypto)."""
    if company.quote_type != "EQUITY":
        return []
    cik = company.cik
    if not cik:
        entry = (await get_cik_map()).get(company.ticker)
        cik = entry[0] if entry else None
    if not cik:
        return []
    sub = await get_submissions(cik)
    if not sub:
        return []
    today = datetime.now(UTC).date()
    filings = filings_from_submissions(sub, limit=limit, today=today)
    docs = narrative_8k_docs(sub, today=today)
    if not docs:
        return filings

    async def excerpt(filing: Filing) -> Filing:
        doc_url = docs.get(filing.url or "")
        if not doc_url:
            return filing
        try:
            text = await _filing_document(doc_url)
        except Exception as exc:  # noqa: BLE001 - excerpts are a bonus; keep the decoded title
            logger.info("8-K document fetch failed %s: %s", doc_url, exc)
            return filing
        summary = summarize_8k(text, filing.items) if text else None
        return reassess_8k(filing, summary) if summary else filing

    return list(await asyncio.gather(*(excerpt(f) for f in filings)))


@cached(ttl=86400, none_ttl=600, maxsize=256)
async def _filing_document(url: str) -> str | None:
    """A filing's primary document (immutable once filed, so cached for a day)."""
    resp = await fetch(url, headers=sec_headers(), timeout=12.0)
    return resp.text[:400_000]


# --------------------------------------------------------------------------- #
# Form 4 insider fallback
# --------------------------------------------------------------------------- #
_CODE_KIND = {"P": "buy", "S": "sell", "A": "award", "M": "exercise", "X": "exercise", "C": "exercise",
              "G": "gift", "F": "other", "D": "other", "J": "other", "W": "other", "I": "other"}
_CODE_TEXT = {"P": "Purchase", "S": "Sale", "A": "Stock Award(Grant)",
              "M": "Conversion of Exercise of derivative security",
              "X": "Exercise of derivative security", "C": "Conversion of derivative security", "G": "Stock Gift",
              "F": "Shares withheld for taxes", "D": "Disposition to issuer"}


def _xml_text(node: Any, path: str) -> str | None:
    el = node.find(path)
    if el is None or el.text is None:
        return None
    text = el.text.strip()
    return text or None


def parse_form4(xml_text: str) -> list[InsiderTxn]:
    """Pure: one Form 4 XML -> non-derivative (common stock) transactions."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    owner = _xml_text(root, "reportingOwner/reportingOwnerId/rptOwnerName") or "Unknown"
    rel = root.find("reportingOwner/reportingOwnerRelationship")
    position = None
    if rel is not None:
        position = _xml_text(rel, "officerTitle")
        if not position:
            flags = [("isDirector", "Director"), ("isTenPercentOwner", "10% Owner"), ("isOfficer", "Officer")]
            position = next((label for tag, label in flags if (_xml_text(rel, tag) or "0") in {"1", "true"}), None)
    planned = (_xml_text(root, "aff10b5One") or "0") in {"1", "true"}
    out: list[InsiderTxn] = []
    for tx_el in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = (_xml_text(tx_el, "transactionCoding/transactionCode") or "").upper()
        when = _xml_text(tx_el, "transactionDate/value")
        try:
            d = date.fromisoformat((when or "")[:10])
        except ValueError:
            continue
        shares = _float(_xml_text(tx_el, "transactionAmounts/transactionShares/value"))
        price = _float(_xml_text(tx_el, "transactionAmounts/transactionPricePerShare/value"))
        value = shares * price if shares and price else None
        indirect = (_xml_text(tx_el, "ownershipNature/directOrIndirectOwnership/value") or "D") == "I"
        text = _CODE_TEXT.get(code, f"Code {code}" if code else "Transaction")
        if price:
            text += f" at price {price:,.2f} per share."
        if planned and code in {"S", "P"}:
            text += " (10b5-1 plan)"
        if indirect:
            text += " (indirect)"
        kind = _CODE_KIND.get(code) or classify_insider(text)
        out.append(InsiderTxn(
            date=d, insider=pretty_insider_name(owner.upper()), position=position,
            kind=kind,  # type: ignore[arg-type]
            shares=shares, value=round(value, 2) if value else None, text=text,
        ))
    return out


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def form4_documents(sub: dict[str, Any], *, since: date, limit: int) -> list[str]:
    """Pure: raw-XML URLs of the newest Form 4 filings since `since`."""
    recent = ((sub or {}).get("filings") or {}).get("recent") or {}
    cik_int = int(str(sub.get("cik") or "0") or 0)
    urls: list[str] = []
    for i, form in enumerate(recent.get("form") or []):
        if form not in {"4", "4/A"}:
            continue
        try:
            if date.fromisoformat(recent["filingDate"][i]) < since:
                continue
        except (KeyError, ValueError):
            continue
        acc = recent["accessionNumber"][i]
        doc = re.sub(r"^xsl[^/]*/", "", recent["primaryDocument"][i])  # raw XML, not the XSLT render
        urls.append(ARCHIVE_URL.format(cik=cik_int, acc_nodash=acc.replace("-", ""), doc=doc))
        if len(urls) >= limit:
            break
    return urls


@cached(ttl=settings.intel_cache_ttl, none_ttl=120)
async def get_form4_insiders(cik: str, window_days: int = 180, max_filings: int = 15) -> InsiderView | None:
    """Insider activity parsed straight from recent Form 4 XML (fallback path)."""
    sub = await get_submissions(cik)
    if not sub:
        return None
    since = datetime.now(UTC).date() - timedelta(days=window_days)
    urls = form4_documents(sub, since=since, limit=max_filings)
    if not urls:
        return None

    async def one(url: str) -> list[InsiderTxn]:
        try:
            return parse_form4((await fetch(url, headers=sec_headers(), timeout=10.0)).text)
        except Exception as exc:  # noqa: BLE001 - one bad document shouldn't sink the rest
            logger.info("Form 4 fetch failed %s: %s", url, exc)
            return []

    rows = [t for batch in await asyncio.gather(*(one(u) for u in urls)) for t in batch]
    return insider_view(rows, since=since, window_days=window_days)
