"""Brand-name cleaning: drives search precision for every source."""
from __future__ import annotations

import pytest

from app.resolve.names import (
    clean_company_name,
    clean_crypto_name,
    clean_fund_name,
    derive_names,
    fix_case,
    is_common_word_name,
)
from tests.intel.helpers import load_json


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # SEC registry titles (all caps)
        ("NVIDIA CORP", "Nvidia"),
        ("AMAZON COM INC", "Amazon"),
        ("MICROSOFT CORP", "Microsoft"),
        ("TESLA, INC.", "Tesla"),
        ("LOWE'S COMPANIES INC", "Lowe's"),
        ("BERKSHIRE HATHAWAY INC", "Berkshire Hathaway"),
        ("APPLIED MATERIALS INC", "Applied Materials"),
        ("CBRE Group, Inc.", "CBRE"),
        # Yahoo long names
        ("Apple Inc.", "Apple"),
        ("Alphabet Inc. Class A", "Alphabet"),
        ("Palantir Technologies Inc.", "Palantir"),
        ("Coinbase Global, Inc.", "Coinbase"),
        ("Palo Alto Networks, Inc.", "Palo Alto Networks"),  # head is a place: keep descriptor
        ("Bloom Energy Corporation", "Bloom Energy"),  # common-word head: keep
        ("Beam Therapeutics Inc.", "Beam Therapeutics"),
        ("Merck & Co., Inc.", "Merck"),
        ("Wells Fargo & Company", "Wells Fargo"),
        ("Eli Lilly and Company", "Eli Lilly"),
        ("Johnson & Johnson", "Johnson & Johnson"),
        ("AT&T Inc.", "AT&T"),
        ("Procter & Gamble Company (The)", "Procter & Gamble"),
        ("The Walt Disney Company", "Walt Disney"),
        ("Charles Schwab Corporation (The", "Charles Schwab"),  # truncated by Yahoo
        ("ASML Holding N.V. - New York Re", "ASML"),
        ("Booking Holdings Inc. Common St", "Booking"),
        ("BP p.l.c.", "BP"),
        ("Shell PLC", "Shell"),
        ("Capital One Financial Corporation", "Capital One"),
        ("Philip Morris International Inc.", "Philip Morris"),
        ("Charles River Laboratories International, Inc.", "Charles River Laboratories"),
        ("American International Group, Inc.", "American International"),  # never just "American"
        ("Northern Trust Corporation", "Northern Trust"),
        ("NIKE, Inc.", "Nike"),
        ("QUALCOMM Incorporated", "Qualcomm"),
        ("lululemon athletica inc.", "Lululemon"),
        ("Amazon.com, Inc.", "Amazon"),
        ("JD.com, Inc.", "JD.com"),
        ("3M Company", "3M"),
        ("e.l.f. Beauty, Inc.", "e.l.f. Beauty"),
        ("Alibaba Group Holding Limited", "Alibaba"),
        ("Hilton Worldwide Holdings Inc.", "Hilton"),
        ("Costco Wholesale Corporation", "Costco"),
        # SEC registry quirks
        ("MTN GROUP LTD/ADR", "MTN"),
        ("UNITED STATES STEEL CORP /DE/", "United States Steel"),
        ("RENTOKIL INITIAL PLC /FI", "Rentokil Initial"),
        ("CAPITAL ONE FINANCIAL CORP", "Capital One"),
        ("NEWS CORP", "News Corp"),  # a generic head keeps its suffix
        ("Inflection Point Acquisition Corp. VIII", "Inflection Point Acquisition Corp. VIII"),
        ("DEUTSCHE BANK AKTIENGESELLSCHAFT", "Deutsche Bank"),
    ],
)
def test_clean_company_name(raw: str, expected: str) -> None:
    assert clean_company_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NVIDIA", "Nvidia"),
        ("IBM", "IBM"),
        ("CVS Health", "CVS Health"),
        ("ASML", "ASML"),
        ("COCA-COLA", "Coca-Cola"),
        ("SoFi", "SoFi"),
        ("eBay", "eBay"),
        ("BANK OF AMERICA CORP", "Bank of America Corp"),
    ],
)
def test_fix_case(raw: str, expected: str) -> None:
    assert fix_case(raw) == expected


def test_fund_and_crypto_names() -> None:
    assert clean_fund_name("iShares Russell 2000 ETF") == "Russell 2000"
    assert clean_fund_name("Global X Uranium ETF") == "Uranium"
    assert clean_fund_name("VanEck Gold Miners ETF") == "Gold Miners"
    assert clean_crypto_name("Bitcoin USD") == "Bitcoin"
    assert clean_crypto_name("XRP USD") == "XRP"


@pytest.mark.parametrize(
    ("ticker", "short", "alias"),
    [
        ("GOOGL", "Alphabet", "Google"),
        ("META", "Meta", "Facebook"),
        ("XYZ", "Block", "Cash App"),
        ("TSM", "TSMC", "Taiwan Semiconductor"),
        ("SPY", "S&P 500", None),
        ("JPM", "JPMorgan", "JPMorgan Chase"),
    ],
)
def test_curated_brands(ticker: str, short: str, alias: str | None) -> None:
    names = derive_names(ticker, "EQUITY", long_name="whatever Inc.")
    assert names.short_name == short
    if alias:
        assert alias in names.aliases


def test_real_yahoo_name_survey_has_no_legal_noise() -> None:
    """Every name Yahoo returned for ~150 popular tickers cleans to a usable brand."""
    survey = load_json("yahoo/name_survey.json")
    for ticker, row in survey.items():
        if not (row.get("longName") or row.get("shortName")):
            continue  # delisted symbol in the survey
        names = derive_names(ticker, row.get("quoteType") or "EQUITY", display_name=row.get("displayName"),
                             long_name=row.get("longName"), short_name=row.get("shortName"))
        short = names.short_name
        assert short and short.strip() == short, ticker
        tail = short.split()[-1].lower().strip(".,")
        assert tail not in {"inc", "corp", "corporation", "co", "company", "ltd", "plc", "the", "&", "and"}, (ticker, short)
        assert not short.lower().startswith("the ") or ticker == "TTD", (ticker, short)
        assert short.upper() != short or len(short.replace("&", "")) <= 5, (ticker, short)  # no shouting


def test_survey_spot_checks() -> None:
    survey = load_json("yahoo/name_survey.json")

    def short(t: str) -> str:
        row = survey[t]
        return derive_names(t, row.get("quoteType") or "EQUITY", display_name=row.get("displayName"),
                            long_name=row.get("longName"), short_name=row.get("shortName")).short_name

    assert short("GS") == "Goldman Sachs"  # displayName "The Goldman Sachs"
    assert short("MRK") == "Merck"  # displayName "Merck &" is truncated -> long name used
    assert short("HD") == "Home Depot"
    assert short("COIN") == "Coinbase"
    assert short("CRWD") == "CrowdStrike"
    assert short("DECK") == "Deckers"
    assert short("SNAP" if "SNAP" in survey else "PINS") in {"Snap", "Pinterest"}


def test_aliases_include_full_legal_form_and_ascii_fold() -> None:
    names = derive_names("PLTR_X", "EQUITY", display_name="Palantir", long_name="Palantir Technologies Inc.")
    assert names.short_name == "Palantir"
    assert "Palantir Technologies" in names.aliases
    folded = derive_names("EL", "EQUITY", long_name="The Estée Lauder Companies Inc.")
    assert folded.short_name == "Estée Lauder" and "Estee Lauder" in folded.aliases


def test_common_word_detection() -> None:
    assert is_common_word_name("Target")
    assert is_common_word_name("Apple")
    assert is_common_word_name("AMD")  # short acronyms are ambiguous for full-text search
    assert not is_common_word_name("Nvidia")
    assert not is_common_word_name("Palo Alto Networks")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Everyday-word heads keep their descriptor: "Structure" alone matches everything.
        ("Structure Therapeutics Inc.", "Structure Therapeutics"),
        ("Viking Therapeutics, Inc.", "Viking Therapeutics"),
        ("Align Technology, Inc.", "Align Technology"),
        ("CRISPR Therapeutics AG", "Crispr Therapeutics"),
        ("Exact Sciences Corporation", "Exact Sciences"),
        ("Duke Energy Corporation", "Duke Energy"),
        ("Boston Scientific Corporation", "Boston Scientific"),  # place names are not brands
        ("Archer Aviation Inc.", "Archer Aviation"),
        ("Yum! Brands, Inc.", "Yum! Brands"),
        # Coined heads drop industry words the press leaves out.
        ("Gilead Sciences, Inc.", "Gilead"),
        ("Toyota Motor Corporation", "Toyota"),
        ("Tyson Foods, Inc.", "Tyson"),
        ("Elevance Health, Inc.", "Elevance"),
        ("SoundHound AI, Inc.", "SoundHound"),
        ("D-Wave Quantum Inc.", "D-Wave"),
        ("Rigetti Computing, Inc.", "Rigetti"),
        ("Axon Enterprise, Inc.", "Axon"),
        ("Clover Health Investments, Corp.", "Clover Health"),
        # Tiny heads are no brand on their own.
        ("On Holding AG", "On Holding"),
        ("Nu Holdings Ltd.", "Nu Holdings"),
        # All-caps registry words that are words, not acronyms.
        ("OLD DOMINION FREIGHT LINE, INC.", "Old Dominion Freight Line"),
        ("BLUE OWL CAPITAL INC.", "Blue Owl Capital"),
    ],
)
def test_clean_company_name_precision_rules(raw: str, expected: str) -> None:
    assert clean_company_name(raw) == expected


def test_everyday_word_brands_get_their_legal_form_as_alias() -> None:
    assert derive_names("SE", "EQUITY", long_name="Sea Limited").aliases == ["Sea Limited"]
    assert derive_names("POOL", "EQUITY", long_name="Pool Corporation").aliases == ["Pool Corporation"]
    assert derive_names("LMND", "EQUITY", long_name="Lemonade, Inc.").aliases == ["Lemonade Inc"]
    assert derive_names("ONON", "EQUITY", long_name="On Holding AG").aliases == []  # never the bare "On"


def test_futures_indices_and_foreign_brands() -> None:
    from app.resolve.names import clean_future_name, registry_display_name

    assert clean_future_name("Crude Oil Nov 26") == "Crude Oil"
    assert derive_names("KC=F", "FUTURE", short_name="Coffee Dec 26").short_name == "Coffee"
    gold = derive_names("GC=F", "FUTURE", short_name="Gold Dec 26")
    assert gold.short_name == "Gold" and "gold prices" in gold.aliases
    assert derive_names("^VIX", "INDEX", long_name="CBOE Volatility Index").short_name == "VIX"
    assert derive_names("PBR", "EQUITY", long_name="Petróleo Brasileiro S.A. - Petrobras").short_name == "Petrobras"
    assert registry_display_name("Bank of Montreal /CAN/") == "Bank of Montreal"
    assert registry_display_name("UNITED STATES STEEL CORP /DE/") == "United States Steel Corp"


def test_fund_names_drop_issuers_and_wrappers() -> None:
    assert clean_fund_name("KraneShares CSI China Internet ETF") == "CSI China Internet"
    assert clean_fund_name("iShares iBoxx $ High Yield Corporate Bond ETF") == "High Yield Corporate Bond"
    assert clean_fund_name("iPath Series B S&P 500 VIX Short-Term Futures ETN") == "S&P 500 VIX Short-Term Futures"
