"""SEC EDGAR: CIK map, decoded filings with flood collapsing, Form 4 fallback."""
from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from app.intel import sec
from app.intel.sec import (
    cik_map_from_json,
    decode_8k,
    filings_from_submissions,
    form4_documents,
    parse_form4,
    sec_headers,
)
from app.sources.base import CompanyRef
from tests.intel.helpers import load_json, load_text

TODAY = date(2026, 10, 4)


def test_sec_user_agent_has_contact_and_no_url() -> None:
    ua = sec_headers()["User-Agent"]
    assert "@" in ua and "http" not in ua and "github" not in ua  # EDGAR's WAF rejects URLs in the UA


@pytest.mark.parametrize(
    ("items", "title", "importance", "polarity"),
    [
        ("2.02,9.01", "Results of operations (earnings release)", "medium", "neutral"),
        ("4.02", "Non-reliance on prior financials (restatement)", "high", "bear"),
        ("1.03,9.01", "Bankruptcy or receivership", "high", "bear"),
        ("3.01", "Delisting notice / listing-rule failure", "high", "bear"),
        ("9.01", "Financial statements and exhibits", "low", "neutral"),
        ("", "Current report", "low", "neutral"),
    ],
)
def test_decode_8k(items: str, title: str, importance: str, polarity: str) -> None:
    got_title, codes, got_imp, got_pol = decode_8k(items)
    assert (got_title, got_imp, got_pol) == (title, importance, polarity)
    assert codes == [c for c in items.split(",") if c]


def test_decode_8k_orders_by_importance() -> None:
    title, _, importance, polarity = decode_8k("5.02,2.06,9.01")
    assert title.startswith("Material impairment") and importance == "high" and polarity == "bear"


def test_cik_map() -> None:
    cik_map = cik_map_from_json(load_json("sec/company_tickers_sample.json"))
    assert cik_map["NVDA"] == ("0001045810", "NVIDIA CORP")
    assert cik_map["BRK-B"][0] == "0001067983"
    assert cik_map_from_json([{"ticker": "x"}]) == {}


def test_filings_nvda_collapse_insider_floods() -> None:
    filings = filings_from_submissions(load_json("sec/submissions_NVDA.json"), today=TODAY)
    assert 0 < len(filings) <= 20
    assert [f.date for f in filings] == sorted((f.date for f in filings), reverse=True)
    forms = [f.form for f in filings]
    assert forms.count("4") == 1 and forms.count("144") <= 1  # summarized, not 40 rows
    insider = next(f for f in filings if f.form == "4")
    assert "insider transaction reports" in insider.title and "since" in insider.title
    assert insider.url and "browse-edgar" in insider.url
    earnings = [f for f in filings if f.items[:1] == ["2.02"]]
    assert earnings and earnings[0].title.startswith("Results of operations")
    assert all(f.url and f.url.startswith("https://www.sec.gov/") for f in filings)
    assert all((TODAY - f.date).days <= 365 for f in filings)


def test_filings_bank_note_program_is_low_importance() -> None:
    filings = filings_from_submissions(load_json("sec/submissions_JPM.json"), today=TODAY)
    notes = [f for f in filings if f.form == "424B2"]
    assert len(notes) == 1 and notes[0].importance == "low"
    assert notes[0].title.split()[0].isdigit()  # "494 pricing supplements … since Sep 29"


def _sub(rows: list[tuple[str, str, str]]) -> dict:
    return {
        "cik": "0000000001",
        "filings": {"recent": {
            "accessionNumber": [f"0000000001-26-{i:06d}" for i in range(len(rows))],
            "filingDate": [r[1] for r in rows],
            "form": [r[0] for r in rows],
            "items": [r[2] for r in rows],
            "primaryDocument": ["doc.htm"] * len(rows),
            "primaryDocDescription": [""] * len(rows),
        }},
    }


def test_filings_red_flags_and_windows() -> None:
    sub = _sub([
        ("8-K", "2026-09-30", "4.02,9.01"),
        ("NT 10-Q", "2026-09-15", ""),
        ("SC 13D", "2026-09-10", ""),
        ("4", "2026-09-09", ""),
        ("8-K", "2025-06-01", "1.03"),  # older than a year: dropped
        ("XYZ-NEW", "2026-08-01", ""),  # unknown form: kept, low
    ])
    filings = filings_from_submissions(sub, today=TODAY)
    by_form = {f.form: f for f in filings}
    assert by_form["8-K"].importance == "high" and by_form["8-K"].polarity == "bear"
    assert by_form["NT 10-Q"].polarity == "bear"
    assert by_form["SC 13D"].importance == "high"
    assert by_form["4"].title.startswith("Insider transaction report (Sep 9)")
    assert by_form["XYZ-NEW"].importance == "low"
    assert all(f.date.year == 2026 for f in filings)


def test_parse_form4_planned_sale_and_tax_withholding() -> None:
    sale = parse_form4(load_text("sec/form4_SOFI_0.xml"))
    assert len(sale) == 1
    tx = sale[0]
    assert tx.kind == "sell" and tx.insider == "Kelli Keough" and tx.position
    assert tx.value == pytest.approx(10203 * 16.9862, rel=1e-4)
    assert "10b5-1" in (tx.text or "")
    vesting = parse_form4(load_text("sec/form4_SOFI_2.xml"))
    kinds = {t.kind for t in vesting}
    assert "exercise" in kinds and "other" in kinds and "sell" not in kinds  # F = tax withholding, not a sale
    assert parse_form4("<not-xml") == []


def test_form4_documents_point_at_raw_xml() -> None:
    sub = load_json("sec/submissions_SOFI.json")
    urls = form4_documents(sub, since=date(2026, 9, 1), limit=3)
    assert len(urls) == 3
    assert all(u.endswith(".xml") and "/xsl" not in u for u in urls)


async def test_get_filings_uses_cik_and_skips_funds() -> None:
    sub = load_json("sec/submissions_NVDA.json")
    company = CompanyRef(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia", cik="0001045810")
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get("https://data.sec.gov/submissions/CIK0001045810.json").mock(return_value=httpx.Response(200, json=sub))
        filings = await sec.get_filings(company)
        assert filings and route.called
        assert "@" in route.calls[0].request.headers["User-Agent"]
    etf = CompanyRef(ticker="SPY", name="SPDR", short_name="S&P 500", quote_type="ETF", cik="0000884394")
    assert await sec.get_filings(etf) == []


async def test_form4_fallback_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = load_json("sec/submissions_SOFI.json")
    since_doc = form4_documents(sub, since=date(2000, 1, 1), limit=3)
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://data.sec.gov/submissions/CIK0001818874.json").mock(return_value=httpx.Response(200, json=sub))
        for i, url in enumerate(since_doc):
            mock.get(url).mock(return_value=httpx.Response(200, text=load_text(f"sec/form4_SOFI_{i}.xml")))
        mock.get(url__regex=r"https://www\.sec\.gov/Archives/.*").mock(return_value=httpx.Response(404))
        monkeypatch.setattr(sec, "datetime", _FrozenDatetime)
        view = await sec.get_form4_insiders("0001818874", max_filings=3)
    assert view is not None and view.sells == 2 and view.buys == 0
    assert view.sell_value == pytest.approx(173310.2 + 325920.0, rel=1e-3)


class _FrozenDatetime(sec.datetime):  # type: ignore[misc, valid-type]
    @classmethod
    def now(cls, tz=None):  # noqa: ANN001, ANN206
        return sec.datetime(2026, 10, 4, 22, 0, tzinfo=tz)
