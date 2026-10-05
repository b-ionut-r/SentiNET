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
from app.schemas import Filing
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
        ("3.02", "Unregistered sale of equity (dilution)", "medium", "bear"),
        # A red flag needs a high *and* bear item: a change in control paid in new shares is a big event,
        # not one (Katapult 8-K, 2026); a restructuring stays one whatever accompanies it.
        ("5.01,3.02", "Change in control; Unregistered sale of equity (dilution)", "high", "neutral"),
        ("2.05,3.02", "Exit/restructuring costs (layoffs, closures); Unregistered sale of equity (dilution)",
         "high", "bear"),
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
    def now(cls, tz=None):
        return sec.datetime(2026, 10, 4, 22, 0, tzinfo=tz)


# --------------------------------------------------------------------------- #
# 8-K narrative excerpts (real documents)
# --------------------------------------------------------------------------- #
def test_summarize_8k_acquisition_keeps_the_dollar_figure() -> None:
    text = sec.summarize_8k(load_text("sec/8k_nvda_801.htm"), ["8.01"])
    assert text is not None
    assert text.startswith("NVIDIA Corporation entered into a definitive agreement to acquire Hugging Face, Inc.")
    assert "$11.9 billion" in text and "“" not in text and len(text) <= 300


def test_summarize_8k_officer_change_and_amendment() -> None:
    officer = sec.summarize_8k(load_text("sec/8k_nvda_502.htm"), ["5.02"])
    assert officer and officer.startswith("Ajay K. Puri") and "retire" in officer
    amended = sec.summarize_8k(load_text("sec/8k_aapl_502a.htm"), ["5.02"])
    assert amended and "Chief Executive Officer transition" in amended and " ." not in amended


def test_summarize_8k_skips_pointer_only_items() -> None:
    assert sec.summarize_8k(load_text("sec/8k_nvda_101.htm"), ["7.01", "9.01"]) is None
    partnership = sec.summarize_8k(load_text("sec/8k_nvda_101.htm"), ["1.01", "2.03", "7.01"])
    assert partnership and partnership.startswith("NVIDIA Corporation announced a multi-year partnership with SB Energy")


def test_listing_deficiency_is_a_red_flag() -> None:
    """Real Beyond Meat 8-K (Item 8.01): a Nasdaq deficiency letter must surface as high/bear."""
    excerpt = sec.summarize_8k(load_text("sec/8k_bynd_801_deficiency.htm"), ["8.01"])
    assert excerpt and excerpt.startswith("Beyond Meat, Inc. received a deficiency letter from the Nasdaq")
    base = Filing(form="8-K", date=date(2026, 9, 1), title="Other material event", items=["8.01"])
    flagged = sec.reassess_8k(base, excerpt)
    assert flagged.importance == "high" and flagged.polarity == "bear"


def test_regained_listing_compliance_is_good_news_not_a_red_flag() -> None:
    """Real GoPro 8-K (Item 8.01, 2026-09-16) was shown as the top "Red-flag filing" alert."""
    excerpt = sec.summarize_8k(load_text("sec/8k_gpro_801_regained.htm"), ["8.01", "9.01"])
    assert excerpt and "regained compliance with the minimum bid price requirement" in excerpt
    base = Filing(form="8-K", date=date(2026, 9, 17), title="Other material event", items=["8.01", "9.01"])
    judged = sec.reassess_8k(base, excerpt)
    assert (judged.importance, judged.polarity) == ("medium", "bull")
    # Some issuers file the all-clear under Item 3.01 itself: the text overrides the item prior and its caption.
    listing = Filing(form="8-K", date=date(2026, 9, 17), title="Delisting notice / listing-rule failure",
                     items=["3.01"], importance="high", polarity="bear")
    cleared = sec.reassess_8k(listing, excerpt)
    assert (cleared.importance, cleared.polarity) == ("medium", "bull")
    assert cleared.title.startswith("Regained compliance with listing rules: ")
    # Another bear item filed alongside still counts on its own.
    restated = sec.reassess_8k(listing.model_copy(update={"items": ["3.01", "4.02"]}), excerpt)
    assert (restated.importance, restated.polarity) == ("high", "bear")


@pytest.mark.parametrize(("excerpt", "importance", "polarity"), [
    # Deficiency letters talk about regaining compliance in the future: still red flags.
    ("The Company received a deficiency letter from Nasdaq. The Company has 180 calendar days to regain "
     "compliance with the minimum bid price requirement.", "high", "bear"),
    ("Nasdaq notified the Company that it has not regained compliance with the minimum bid price requirement "
     "and its shares will be suspended.", "high", "bear"),
    # One requirement met, another one failed: the open deficiency wins.
    ("The Company has regained compliance with the minimum bid price requirement. Separately, the Company "
     "received a deficiency letter regarding the minimum stockholders' equity requirement.", "high", "bear"),
])
def test_listing_deficiencies_stay_red_flags(excerpt: str, importance: str, polarity: str) -> None:
    base = Filing(form="8-K", date=TODAY, title="Other material event", items=["8.01"])
    judged = sec.reassess_8k(base, excerpt)
    assert (judged.importance, judged.polarity) == (importance, polarity)
    listing = Filing(form="8-K", date=TODAY, title="Delisting notice / listing-rule failure", items=["3.01"],
                     importance="high", polarity="bear")
    assert (sec.reassess_8k(listing, excerpt).importance, sec.reassess_8k(listing, excerpt).polarity) == \
        ("high", "bear")


def _item_302() -> Filing:
    title, codes, importance, polarity = decode_8k("3.02")
    return Filing(form="8-K", date=date(2026, 9, 28), title=title, items=codes, importance=importance,
                  polarity=polarity)


def test_acquisition_paid_in_stock_is_not_a_red_flag() -> None:
    """Real AMD 8-K (Item 3.02 only, 2026-09-28): the $8.2B all-stock purchase of World Labs was raised as a
    "Red-flag filing" — the deal wording lifted the 3.02 dilution prior to high and kept its bear."""
    excerpt = sec.summarize_8k(load_text("sec/8k_amd_302_acquisition.htm"), ["3.02"])
    assert excerpt and excerpt.startswith("Advanced Micro Devices, Inc. entered into an Agreement and Plan of Merger "
                                          "to acquire all of the equity interests in World Labs")
    judged = sec.reassess_8k(_item_302(), excerpt)
    assert (judged.importance, judged.polarity) == ("high", "neutral")
    assert judged.title.startswith("Acquisition paid in stock: Advanced Micro Devices") and "$8.2 billion" in judged.title


@pytest.mark.parametrize(("excerpt", "importance", "polarity", "caption"), [
    # Stock as the purchase price (Voyager/Astrobotic 8-K wording): the deal, not a dilutive raise.
    ("Voyager Technologies, Inc. entered into an Agreement and Plan of Merger pursuant to which the Company agreed "
     "to acquire 100% of the outstanding capital stock of Astrobotic Technology, Inc. In connection with the "
     "Acquisition, the Company agreed to issue shares of its Class A common stock…",
     "high", "neutral", "Acquisition paid in stock"),
    # Warrants sold to an executive (HHH 8-K wording) are securities, not a deal: a plain dilution watch.
    ("Howard Hughes Holdings Inc. entered into a warrant agreement with Mr. Grandisson, pursuant to which Mr. "
     "Grandisson agreed to purchase warrants to acquire 1,131,273 shares of the Company's common stock.",
     "medium", "bear", "Unregistered sale of equity (dilution)"),
    ("The Company entered into a Securities Purchase Agreement with investors to sell 4,330,866 shares and "
     "warrants to acquire up to 2,500,000 shares of common stock for gross proceeds of $20 million.",
     "medium", "bear", "Unregistered sale of equity (dilution)"),
    # Shares sold for cash next to a deal (a SPAC PIPE) stay dilution; the deal makes the filing big, not bad.
    ("In connection with the Business Combination, PubCo entered into subscription agreements with investors to "
     "purchase 10,000,000 shares for aggregate proceeds of $100 million.",
     "high", "neutral", "Unregistered sale of equity (dilution)"),
    # Real bear evidence in the text still makes a red flag.
    ("The Company issued 2,000,000 shares to the sellers as consideration for the acquisition of Widget Co. "
     "The Company's auditor expressed substantial doubt about its ability to continue as a going concern.",
     "high", "bear", "Acquisition paid in stock"),
])
def test_item_302_share_issuance_reads(excerpt: str, importance: str, polarity: str, caption: str) -> None:
    judged = sec.reassess_8k(_item_302(), excerpt)
    assert (judged.importance, judged.polarity) == (importance, polarity)
    assert judged.title.startswith(f"{caption}: ")


def test_excerpt_drops_previously_reported_preamble() -> None:
    """"As previously reported … on August 6, 2025, John Boken was appointed…" -> the news itself."""
    excerpt = sec.summarize_8k(load_text("sec/8k_bynd_502.htm"), ["5.02"])
    assert excerpt and excerpt.startswith("John Boken was appointed")
    assert "previously reported" not in excerpt.lower()


def test_reassess_8k_from_excerpt() -> None:
    base = sec.Filing(form="8-K", date=TODAY, title="Other material event", items=["8.01"])
    deal = sec.reassess_8k(base, "Acme entered into a definitive agreement to acquire Widget Co.")
    assert deal.importance == "high" and deal.title.startswith("Other material event: Acme entered")
    ceo = sec.reassess_8k(base, "The Chief Executive Officer resigned from all positions with the Company.")
    assert ceo.importance == "high" and ceo.polarity == "bear"
    buyback = sec.reassess_8k(base, "The Board authorized an increase of $10 billion to the share repurchase program.")
    assert buyback.polarity == "bull"
    plain = sec.reassess_8k(base, "The Company relocated its headquarters.")
    assert plain.importance == "low" and plain.polarity == "neutral"


@pytest.mark.parametrize(("excerpt", "importance", "polarity"), [
    # Standard 5.02 wording puts the verb before the title.
    ("John Smith informed the Board of his intention to retire as Chief Executive Officer, effective March 31, "
     "2027.", "high", "neutral"),
    ("Jane Roe will step down as Chief Financial Officer of the Company.", "high", "neutral"),
    ("The Board terminated the employment of John Smith, its Chief Executive Officer, for cause.", "high", "bear"),
    ("Mr. Doe, our CFO, resigned effective immediately.", "high", "bear"),
    ("John Ternus will succeed Mr. Cook as Chief Executive Officer.", "high", "neutral"),
    # Routine appointments and compensation boilerplate are not red flags.
    ("The Board appointed Jane Doe as Chief Financial Officer, effective October 1, 2026. Separately, the Board "
     "approved an amendment to the bylaws.", "medium", "neutral"),
    ("The CEO received a special retention award; the award vests on termination without cause.", "medium", "neutral"),
    ("The committee approved 2027 salaries for the named executive officers, including the CEO.", "medium", "neutral"),
])
def test_executive_changes_in_either_word_order(excerpt: str, importance: str, polarity: str) -> None:
    base = sec.Filing(form="8-K", date=TODAY, title="Director/officer departure or appointment", items=["5.02"],
                      importance="medium", polarity="neutral")
    judged = sec.reassess_8k(base, excerpt)
    assert (judged.importance, judged.polarity) == (importance, polarity)


def test_narrative_docs_selection() -> None:
    sub = _sub([("8-K", "2026-09-30", "8.01"), ("8-K", "2026-09-29", "2.02,9.01"), ("8-K", "2026-06-01", "5.02")])
    docs = sec.narrative_8k_docs(sub, today=TODAY)
    assert len(docs) == 1  # earnings release skipped; June filing outside 60 days
    (index_url, doc_url), = docs.items()
    assert index_url.endswith("-index.htm") and doc_url.endswith("/doc.htm")


async def test_get_filings_enriches_recent_8k(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = _sub([("8-K", "2026-09-30", "8.01")])
    company = CompanyRef(ticker="ACME", name="Acme", short_name="Acme", cik="0000000001")
    monkeypatch.setattr(sec, "datetime", _FrozenDatetime)
    with respx.mock as mock:
        mock.get("https://data.sec.gov/submissions/CIK0000000001.json").mock(return_value=httpx.Response(200, json=sub))
        mock.get(url__regex=r".*/doc\.htm$").mock(return_value=httpx.Response(200, text=load_text("sec/8k_nvda_801.htm")))
        filings = await sec.get_filings(company)
    assert filings[0].importance == "high" and "Hugging Face" in filings[0].title


async def test_get_filings_all_stock_acquisition_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = _sub([("8-K", "2026-09-28", "3.02")])
    company = CompanyRef(ticker="AMD", name="Advanced Micro Devices, Inc.", short_name="AMD", cik="0000002488")
    monkeypatch.setattr(sec, "datetime", _FrozenDatetime)
    with respx.mock as mock:
        mock.get("https://data.sec.gov/submissions/CIK0000002488.json").mock(return_value=httpx.Response(200, json=sub))
        mock.get(url__regex=r".*/doc\.htm$").mock(
            return_value=httpx.Response(200, text=load_text("sec/8k_amd_302_acquisition.htm")))
        filings = await sec.get_filings(company)
    assert (filings[0].importance, filings[0].polarity) == ("high", "neutral")
    assert filings[0].title.startswith("Acquisition paid in stock: ")


async def test_get_filings_keeps_title_when_document_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = _sub([("8-K", "2026-09-30", "8.01")])
    company = CompanyRef(ticker="ACME", name="Acme", short_name="Acme", cik="0000000001")
    monkeypatch.setattr(sec, "datetime", _FrozenDatetime)
    with respx.mock as mock:
        mock.get("https://data.sec.gov/submissions/CIK0000000001.json").mock(return_value=httpx.Response(200, json=sub))
        mock.get(url__regex=r".*/doc\.htm$").mock(return_value=httpx.Response(404))
        filings = await sec.get_filings(company)
    assert filings[0].title == "Other material event"
