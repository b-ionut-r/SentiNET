"""Text hygiene: cleaning, publisher suffixes, junk detection, dedup keys, tokens."""
from __future__ import annotations

import pytest

from app.nlp.text import (
    clean_text,
    fold,
    is_meaningful,
    is_mostly_upper,
    is_title_case,
    normalize_for_dedup,
    normalize_money,
    stem,
    strip_publisher_suffix,
    tokenize,
)
from tests.conftest import load_json_fixture


class TestCleanText:
    def test_strips_tags_entities_urls_and_invisible_chars(self):
        raw = ('<a href="https://x.com">Nvidia</a>&amp;#39;s &quot;record&quot;\u200b buyback '
               "https://t.co/abc ￼  done")
        assert clean_text(raw) == "Nvidia's \"record\" buyback done"

    def test_double_escaped_stocktwits_apostrophe(self):
        assert clean_text("robin&#39;s hood") == "robin's hood"

    def test_keeps_emojis_and_cashtags(self):
        assert clean_text("$NVDA to the moon 🚀🚀") == "$NVDA to the moon 🚀🚀"

    @pytest.mark.parametrize("value", [None, "", "   ", "<br/>"])
    def test_empty_inputs(self, value):
        assert clean_text(value) == ""

    def test_fold_maps_typographic_punctuation(self):
        assert fold("Nvidia’s “top” pick — again…") == "Nvidia's \"top\" pick - again..."


class TestPublisherSuffix:
    @pytest.mark.parametrize(("title", "publisher", "expected"), [
        ("Nvidia Adds Record $150 Billion to Stock Buyback - WSJ", "WSJ",
         "Nvidia Adds Record $150 Billion to Stock Buyback"),
        ("Apple (NASDAQ:AAPL) Stock Jumps 1% - What's Next? - MarketBeat", "MarketBeat",
         "Apple (NASDAQ:AAPL) Stock Jumps 1% - What's Next?"),
        ("Watch Nvidia Up on Share-Repurchase Program; Mining Stocks Slide | Stock Movers - Bloomberg.com",
         "Bloomberg.com", "Watch Nvidia Up on Share-Repurchase Program; Mining Stocks Slide | Stock Movers"),
        ("Aces' Justine Pissott shares experience amid Cornell investigation - Las Vegas Review-Journal",
         "Las Vegas Review-Journal", "Aces' Justine Pissott shares experience amid Cornell investigation"),
        ("Stock-Market Outlook - BusinessMirror", "BusinessMirror", "Stock-Market Outlook"),
        ("Morgan Stanley lowers Apple stock price target on limited upside By Investing.com - Investing.com Nigeria",
         None, "Morgan Stanley lowers Apple stock price target on limited upside By Investing.com"),
        ("Apple stock: China's iPhone 18 Pro surge has a catch investors should see - techi.com", None,
         "Apple stock: China's iPhone 18 Pro surge has a catch investors should see"),
    ])
    def test_real_google_news_titles(self, title, publisher, expected):
        assert strip_publisher_suffix(title, publisher) == expected

    @pytest.mark.parametrize("title", ["Stock Jumps 1% - What's Next?", "Q3 Results - Nvidia Beats on Revenue Again?"])
    def test_does_not_eat_real_title_segments(self, title):
        assert strip_publisher_suffix(title) == title

    @pytest.mark.parametrize("ticker", ["nvda", "aapl", "meta", "tgt", "xyz"])
    def test_every_captured_google_title_loses_its_outlet(self, ticker):
        items = load_json_fixture(f"nlp/headlines_{ticker}.json")["items"]
        for item in items:
            stripped = strip_publisher_suffix(item["title"], item["publisher"])
            assert not stripped.endswith(" - " + item["publisher"]), item["title"]
            assert item["title"].startswith(stripped)


class TestMeaningful:
    @pytest.mark.parametrize("text", [
        "$GNS $MU $NKE $NVDA $SPCX",
        "Toast, Inc. (TOST) Stock Price, News, Quote & History",
        "2342 Forecast — Price Target — Prediction for 2027",
        "$Apple (AAPL.US)$",
        "🚀🚀🚀",
        "New 2026 Ford F-150 LARIAT for sale in Henderson, NV - 1FTFW5L89TFB18220",
        "",
    ])
    def test_rejects_junk(self, text):
        assert not is_meaningful(text)

    @pytest.mark.parametrize("text", [
        "$NVDA puts are the only play",
        "Nvidia Adds Record $150 Billion to Stock Buyback",
        "HSBC upgraded Target to Buy",
    ])
    def test_accepts_content(self, text):
        assert is_meaningful(text)


class TestDedupNormalization:
    def test_money_variants_collapse(self):
        assert normalize_for_dedup("Nvidia adds $150 billion to buyback") == \
            normalize_for_dedup("NVIDIA adds $150B to buyback!")

    def test_attribution_label_and_exchange_tickers_removed(self):
        a = normalize_for_dedup("Update: Morgan Stanley lowers Apple stock price target By Investing.com")
        b = normalize_for_dedup("Morgan Stanley lowers Apple stock price target")
        assert a == b
        assert normalize_for_dedup("Nvidia unveils $150B buyback (NVDA:NASDAQ)") == \
            normalize_for_dedup("Nvidia unveils $150B buyback")

    def test_distinct_amounts_stay_distinct(self):
        assert normalize_for_dedup("Eisen sells $1.36m in stock") != normalize_for_dedup("Eisen sells $1.33m in stock")


class TestTokens:
    def test_tokenize_money_percent_cashtag_possessive(self):
        tokens = tokenize("Nvidia's $150 Billion buyback, $1.05B suit; shares +5.5% to $1,070 $NVDA")
        assert tokens == ["nvidia", "$150b", "buyback", "$1.05b", "suit", "shares", "5.5%", "to", "$1070", "$nvda"]

    @pytest.mark.parametrize(("raw", "expected"), [
        ("$150 billion", "$150b"), ("$150bn", "$150b"), ("$5.70 trillion", "$5.7t"), ("$1,070", "$1070"),
        ("$8", "$8"),
    ])
    def test_normalize_money(self, raw, expected):
        assert normalize_money(raw) == expected

    @pytest.mark.parametrize("group", [
        ["raise", "raises", "raised", "raising"],
        ["rise", "rises", "rising", "rose"],
        ["buyback", "buybacks"],
        ["launch", "launches", "launched", "launching"],
        ["earnings", "earning"],
        ["rally", "rallies", "rallied"],
        ["cut", "cutting"],
        ["price", "pricing"],
    ])
    def test_stem_groups_inflections(self, group):
        assert len({stem(w) for w in group}) == 1, {w: stem(w) for w in group}

    def test_stem_keeps_special_words(self):
        assert stem("news") == "news"
        assert stem("analysis") == "analysis"
        assert stem("$150b") == "$150b"

    def test_case_shape_helpers(self):
        assert is_title_case("Nvidia Adds Record $150 Billion to Stock Buyback")
        assert not is_title_case("Nvidia adds record $150 billion to stock buyback")
        assert is_mostly_upper("NVDA TO THE MOON BOYS")
        assert not is_mostly_upper("NVDA to the moon")


@pytest.mark.parametrize("text", [
    "SOFI Nov 2026 19.000 put (SOFI261106P00019000) Interactive Stock Chart",
    "Visa stock after-hours at EUR 324.13: plus 0.30 percent versus prior close",
    "Visa stock pre-market at EUR 318.35: plus 0.27 percent",
])
def test_quote_ticks_and_option_pages_are_not_meaningful(text):
    assert not is_meaningful(text)


@pytest.mark.parametrize(("text", "kept"), [
    # real live Bluesky posts: "<" and ">" are comparisons, "->" a link arrow
    ("Weekly: bond rout blinked (10Y <5.25%) - AI admitted a financing problem: Amazon offloads ~$8bn of NVDA "
     "chips. h -> more", "Amazon offloads ~$8bn of NVDA chips"),
    ("RSI 42.67 <65, MACD positive, price above Ichimoku cloud -> Zacks", "MACD positive"),
    ("$TSLA <300 is a gift, >400 by EOY", "<300 is a gift, >400 by EOY"),
    ("Revenue growth <5% while margins >20% beat", "<5% while margins >20% beat"),
    ("&lt;5% growth", "<5% growth"),
])
def test_comparison_signs_are_not_markup(text, kept):
    assert kept in clean_text(text)


def test_markup_and_comments_are_still_stripped_and_emoji_sequences_survive():
    assert clean_text("a<br/>b <!-- tracking --> <p class='x'>c</p>") == "a b c"
    assert clean_text("&lt;b&gt;Nvidia&lt;/b&gt;'s chips") == "Nvidia's chips"
    assert "‍" in clean_text("dev \U0001F468‍\U0001F4BB rocks")  # zero-width joiner keeps the emoji whole


@pytest.mark.parametrize(("title", "publisher"), [
    # live non-Google titles whose last segment is content, not the outlet
    ("Jack Dorsey Says 'Turn It on' as Block's Cash App Introduces 'Bitcoin Bonus' Feature - What You Should Know",
     "Benzinga"),
    ("Elon Musk's Big Nvidia Bet Takes Center Stage After SpaceX Earnings - NVDA Gains While SPCX Falls", "Stocktwits"),
    ("Notable ETF Inflow Detected - XLP, MO, TGT, CL", "BNK Invest"),
    ("Stocktwits Wall Street Wrap: Stocks Mixed - Liquidia And FICO Face Downgrades", None),
])
def test_content_tails_are_not_publishers(title, publisher):
    assert strip_publisher_suffix(title, publisher) == title


@pytest.mark.parametrize(("title", "publisher"), [
    ("Nvidia stock rises - Reuters", "Reuters"),
    ("Nvidia stock rises - Key Context by Tae Kim", None),
    ("Nvidia stock rises - WGAU Radio", None),
    ("Nvidia stock rises - finance.yahoo.com", "Yahoo Finance"),
])
def test_outlet_tails_are_stripped(title, publisher):
    assert strip_publisher_suffix(title, publisher) == "Nvidia stock rises"


@pytest.mark.parametrize(("text", "meaningful"), [
    ("Intel puts Altera stake up for auction, sources say", True),  # real divestiture news
    ("Boeing's Wichita plant for sale in bid to raise cash", True),
    ("Target Corporation", False),  # live Google News stub: a bare legal name
    ("Target Corporation lowers prices as Target stock trades EUR 138.70 versus EUR 138.70", False),
    ("2021 Ford F-150 XLT for sale in Dallas, TX", False),
    ("Used Tesla Model 3 for sale in Austin - 32,000 miles", False),
])
def test_listing_and_stub_detection(text, meaningful):
    assert is_meaningful(text) is meaningful
