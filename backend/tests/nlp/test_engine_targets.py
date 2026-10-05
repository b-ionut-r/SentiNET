"""Subject attribution: a peer's move in the analysed company's headline is context, not its news.

Headlines are modelled on live Google News / Bing items from the engine review (2026-10-04);
company names are kept because attribution is about names. Hand-written variants, no benchmark text.
"""
from __future__ import annotations

import pytest

from app.nlp.engine import SentinelEngine
from app.nlp.rules import OFF_TARGET_FACTOR

SOFI = ["SOFI", "SoFi", "SoFi Technologies"]
NVDA = ["NVDA", "Nvidia", "NVIDIA Corporation"]
META = ["META", "Meta", "Meta Platforms", "Facebook"]
AAPL = ["AAPL", "Apple", "Apple Inc."]
TSLA = ["TSLA", "Tesla"]
TGT = ["TGT", "Target", "Target Corporation"]


@pytest.fixture(scope="module")
def engine() -> SentinelEngine:
    return SentinelEngine()


@pytest.mark.parametrize(("text", "target", "kind"), [
    ("Nu Holdings Jumps 3% After Ruling Out Monzo Deal; SoFi and Robinhood Sit Out the Rally", SOFI, "news"),
    ("Cerebras stock hits post-IPO low, tumbling 20% on Nvidia pressure", NVDA, "news"),
    ("MongoDB stock tanks after Meta poaches CEO", META, "news"),
    ("Qualcomm Is Losing Apple and Adding Amazon", AAPL, "news"),
    ("Tesla rival BYD's sales surge 30%", TSLA, "news"),
    ("AMD shares fall as Nvidia unveils new chip", NVDA, "news"),
])
def test_peer_moves_count_less(engine: SentinelEngine, text: str, target: list[str], kind: str) -> None:
    plain = engine.analyze(text, kind)
    aimed = engine.analyze(text, kind, target)
    assert abs(aimed.score) < abs(plain.score) * 0.8, (plain.drivers, aimed.drivers)


def test_target_move_dominates_a_roundup_of_peers(engine: SentinelEngine) -> None:
    text = "SoFi Falls 3% as Rising Yields Pressure Fintech; Affirm Drops 4%, Robinhood Slips 2%"
    aimed = engine.analyze(text, "news", SOFI)
    assert aimed.label == "bearish"
    assert aimed.drivers[0][0] == "Falls 3%"  # SoFi's own move leads, the peers' moves trail
    assert abs(aimed.score) < abs(engine.analyze(text).score)


@pytest.mark.parametrize(("text", "target"), [
    ("Nvidia shares jump after Microsoft boosts AI spending", NVDA),  # the target's own move
    ("Morgan Stanley raises price target on Nvidia to $250 from $220", NVDA),  # actor + object
    ("Jensen Huang says Nvidia demand is surging", NVDA),  # a speaker is not a subject
    ("Apple and Google beat estimates", AAPL),  # coordinated subjects
    ("Nvidia, AMD shares fall", NVDA),
    ("Nvidia Blackwell sales surge", NVDA),  # the target's product
    ("Apple Stock Rises As iPhone Sales Surge", AAPL),
    ("Target (TGT) Stock Dips While Market Gains", TGT),
    ("Shares surge 12% after record quarter", NVDA),  # target not named: nothing to attribute
    ("Nvidia unveils massive $150B increase to share buyback program (NVDA:NASDAQ)", NVDA),  # "B" is a unit
    ("Ford, GM and Stellantis stock prices slide for the week ending Oct. 2", ["F", "Ford", "Ford Motor"]),
    ("Intel (INTC) Data Center Comeback: Real Recovery or Just a Supply Squeeze?", ["INTC", "Intel"]),
    ("Bitcoin Is Headed For Its Best Quarter In Nearly 2 Years - But Traders Are Using Less Leverage",
     ["BTC-USD", "Bitcoin", "BTC"]),
])
def test_own_news_keeps_full_weight(engine: SentinelEngine, text: str, target: list[str]) -> None:
    assert engine.analyze(text, "news", target).score == pytest.approx(engine.analyze(text).score)


def test_cashtag_posts(engine: SentinelEngine) -> None:
    aimed = engine.analyze("$TSLA up 3%, $NIO down 5%", "social", TSLA)
    assert aimed.label == "bullish"
    assert engine.analyze("$NIO down 5%, $TSLA up 3%", "social", ["NIO", "Nio"]).label == "bearish"


def test_batch_api_accepts_partial_targets(engine: SentinelEngine) -> None:
    texts = ["Affirm Drops 4% as SoFi holds steady", "Affirm Drops 4%"]
    out = engine.score(texts, ["news", "news"], [SOFI])  # second text: no target given
    assert out[1].score == pytest.approx(engine.analyze(texts[1]).score)
    weights = [h.weight for h in engine.evidence(texts[0], "news", SOFI).hits if h.source == "move"]
    assert weights and min(weights) <= OFF_TARGET_FACTOR + 1e-9


def test_short_tickers_match_only_in_capitals(engine: SentinelEngine) -> None:
    # "on" is a word, ON (onsemi) a ticker: lower-case prose must not count as a target mention
    from app.nlp.rules import _target_positions, normalize, tokenize

    text = "Shares move on news"
    tokens = tokenize(normalize(text).lower())
    assert _target_positions(tokens, normalize(text), ["ON", "onsemi"]) == set()


@pytest.mark.parametrize(("text", "target", "expected"), [
    ("Intel Rises 3% as Chip Contender Caps Off Strong September; NVIDIA Ticks Up, AMD Slips", ["INTC", "Intel"],
     "bullish"),
    ("Premarket movers Boeing falls on software glitch; Kodiak surges", ["BA", "Boeing"], "bearish"),
    ("Why Are VZ, T, TMUS Stocks Rising Overnight?", ["T", "AT&T"], "bullish"),  # T is in the list
])
def test_live_roundups_read_for_the_target(engine: SentinelEngine, text: str, target: list[str],
                                           expected: str) -> None:
    assert engine.analyze(text, "news", target).label == expected


def test_comma_clause_with_its_own_subject_is_not_composed_across() -> None:
    # "Rate Fears, NVIDIA Rises 3%": NVIDIA rises; the fears do not
    ev = SentinelEngine().evidence("Intel Drops 4% on Oil-Driven Rate Fears, NVIDIA Rises 3% on Record Buyback")
    assert not any("fears" in h.term.lower() and "rises" in h.term.lower() for h in ev.hits)


def test_target_terms_and_every_engine_accepts_targets() -> None:
    from app.nlp.engine import VaderEngine, target_terms
    from app.sources.base import CompanyRef

    btc = CompanyRef(ticker="BTC-USD", name="Bitcoin USD", short_name="Bitcoin", aliases=["BTC"],
                     quote_type="CRYPTOCURRENCY")
    assert target_terms(btc) == ["BTC-USD", "BTC", "Bitcoin", "Bitcoin USD"]
    text = "$BTC.X up 3%, $ETH.X down 5%"
    assert VaderEngine().score([text], ["social"], [target_terms(btc)])[0].label in ("bullish", "bearish", "neutral")
    assert SentinelEngine().score([text], ["social"], [target_terms(btc)])[0].label == "bullish"


# --- final review (live Google News items, 2026-10-05): a bystander must not inherit the peer's tone
LULU = ["LULU", "Lululemon", "Lululemon Athletica"]
TWLO = ["TWLO", "Twilio", "Twilio Inc."]


@pytest.mark.parametrize(("text", "target"), [
    ("Nu Holdings Jumps 3% After Ruling Out Monzo Deal; SoFi and Robinhood Sit Out the Rally", SOFI),
    ("Nike Sinks 8% as Weak Outlook and Layoffs Follow Revenue Miss; Lululemon and On Holding Remain Flat", LULU),
    ("Stocks Tumble as Yields Surge; Apple Holds Steady", AAPL),
])
def test_bystander_headlines_are_not_the_peers_news(engine: SentinelEngine, text: str, target: list[str]) -> None:
    from app.nlp.rules import is_bystander

    plain, aimed = engine.analyze(text), engine.analyze(text, "news", target)
    assert aimed.label == "neutral" and aimed.score <= 0.0, aimed.drivers  # "sit out the rally" may lean negative
    assert abs(aimed.score) < abs(plain.score) or plain.label == "neutral"
    assert is_bystander(text, target)


def test_a_clause_after_a_comma_with_its_own_subject_is_that_companys(engine: SentinelEngine) -> None:
    text = "Twilio downgraded, Synopsys upgraded: Wall Street's top analyst calls"
    twilio = engine.analyze(text, "news", TWLO)
    assert twilio.label == "bearish" and twilio.drivers[0][0] == "downgraded", twilio.drivers
    assert "top analyst" not in [t.lower() for t, _ in twilio.drivers]  # a column title, not a beat
    assert engine.analyze(text, "news", ["SNPS", "Synopsys"]).label == "bullish"
    swapped = engine.analyze("Synopsys Upgraded, Twilio Downgraded", "news", TWLO)  # Title Case, target second
    assert swapped.label == "bearish"
    from app.nlp.rules import is_bystander

    assert not is_bystander(text, TWLO)  # Twilio has news of its own here


def test_stated_non_move_mutes_peers_further_than_a_plain_peer_mention(engine: SentinelEngine) -> None:
    from app.nlp.rules import OFF_TARGET_STATED_FACTOR

    ev = engine.evidence("Lululemon flat as Nike sinks 8%", "news", LULU)
    assert ev.hits and all(h.weight <= OFF_TARGET_STATED_FACTOR + 1e-9 for h in ev.hits)
    assert engine.analyze("Lululemon flat as Nike sinks 8%", "news", LULU).label == "neutral"


def test_market_clause_is_context_and_never_overturns_the_targets_move(engine: SentinelEngine) -> None:
    assert engine.analyze("Stocks tumble; Apple rises 2%").label == "bearish"  # read for nobody: the market
    aimed = engine.analyze("Stocks tumble; Apple rises 2%", "news", AAPL)
    assert aimed.score >= 0.0 and aimed.drivers[0][0] == "rises 2%"
    roundup = engine.analyze("Microsoft slips 1%, Apple rallies 3%", "news", AAPL)
    assert roundup.label == "bullish"


def test_title_case_common_words_do_not_open_a_company_clause(engine: SentinelEngine) -> None:
    # "Real Recovery" after the colon is a noun phrase, not a company called Real
    text = "Intel (INTC) Data Center Comeback: Real Recovery or Just a Supply Squeeze?"
    assert all(h.weight == pytest.approx(h2.weight) for h, h2 in zip(
        engine.evidence(text, "news", ["INTC", "Intel"]).hits, engine.evidence(text).hits, strict=True))


def test_bystander_needs_a_named_target_and_no_news_of_its_own() -> None:
    from app.nlp.rules import is_bystander

    assert not is_bystander("Nike sinks 8% on weak outlook", LULU)  # not named at all
    assert not is_bystander("Lululemon cuts guidance; Nike sinks 8%", LULU)
    assert not is_bystander("Lululemon lags the rally as Nike surges 6%", LULU)  # Lululemon is the subject


# --- final QA (live BTC-USD / QQQ / META items, 2026-10-05): peers joined by "as", crypto assets, index changes
BTC = ["BTC-USD", "BTC", "Bitcoin"]  # target_terms() of the live CompanyRef
ETH = ["ETH-USD", "ETH", "Ethereum", "Ether"]
QQQ = ["QQQ", "Nasdaq 100", "Invesco QQQ Trust", "Nasdaq-100", "Invesco QQQ"]
AMD = ["AMD", "Advanced Micro Devices"]


@pytest.mark.parametrize(("text", "target"), [
    ("Crypto Weekly: ZEC Plunges as Bitcoin and Ether Hold Key Support Levels", BTC),
    ("Crypto Weekly: ZEC Plunges as Bitcoin and Ether Hold Key Support Levels", ETH),
    ("Ethereum liquidity drops below 50% of Bitcoin's level", BTC),
    ("Ethereum ETF outflows surge as Bitcoin holds steady", BTC),
    ("LINK Hits 2026 High: Chainlink Rally Outpaces Bitcoin, Ethereum And XRP", ETH),
])
def test_another_crypto_assets_move_is_not_the_targets(engine: SentinelEngine, text: str,
                                                       target: list[str]) -> None:
    plain, aimed = engine.analyze(text), engine.analyze(text, "news", target)
    assert plain.label != "neutral"  # read for nobody, the move is there ...
    assert aimed.label == "neutral", aimed.drivers  # ... but it is ZEC's / Ethereum's / Chainlink's


def test_the_named_assets_own_move_still_counts(engine: SentinelEngine) -> None:
    assert engine.analyze("Ethereum liquidity drops below 50% of Bitcoin's level", "news", ETH).label == "bearish"
    rises = engine.analyze("Bitcoin rises 3% as Ether jumps 8%", "news", BTC)
    assert rises.label == "bullish" and rises.drivers[0][0] == "Bitcoin rises 3%"
    ether = [h for h in engine.evidence("Bitcoin rises 3% as Ether jumps 8%", "news", BTC).hits if "Ether" in h.term]
    assert ether and all(h.weight <= OFF_TARGET_FACTOR + 1e-9 for h in ether)
    slides = engine.analyze("Ether drops 5% as Bitcoin slides", "news", BTC)
    assert slides.label == "bearish" and slides.drivers[0][0] == "Bitcoin slides"
    # a fund's own underlying is not "another asset"
    ibit = ["IBIT", "iShares Bitcoin Trust ETF"]
    text = "IBIT outflows hit a record as Bitcoin slides 5%"
    assert engine.analyze(text, "news", ibit).score == pytest.approx(engine.analyze(text).score)


@pytest.mark.parametrize(("text", "target", "lead"), [
    ("AMD Jumps 5% As Nvidia Slips", NVDA, "Slips"),
    ("Stocks fall as Nvidia slides 3%", NVDA, "slides 3%"),
    ("Mark Zuckerberg Loses Nearly $10 Billion as Meta Shares Slide", META, "Shares Slide"),
    ("ZEC plunges as Bitcoin slides", BTC, "Bitcoin slides"),
])
def test_targets_own_as_clause_after_someone_elses_news_is_its_main_news(engine: SentinelEngine, text: str,
                                                                       target: list[str], lead: str) -> None:
    # "X does A as <target> does B": for the target, B is the news - not background to X's move
    aimed = engine.analyze(text, "news", target)
    assert aimed.label == "bearish" and aimed.drivers[0][0] == lead, aimed.drivers
    hit = next(h for h in engine.evidence(text, "news", target).hits if h.display == lead or h.term == lead)
    assert hit.weight == pytest.approx(1.0)


def test_an_as_clause_is_split_only_with_a_subject_of_its_own(engine: SentinelEngine) -> None:
    for text, target in (("Tesla jumps as deliveries beat estimates", TSLA),
                         ("Apple shares rise as investors cheer buyback", AAPL),
                         ("Nvidia rises as well as AMD", NVDA),
                         # "China sales" are Toyota's: a region is no owner of a metric
                         ("Toyota stock falls as China sales drop on fuel price surge", ["7203.T", "Toyota"])):
        assert engine.analyze(text, "news", target).score == pytest.approx(engine.analyze(text).score), text
    assert engine.analyze("AMD Jumps 5% As Nvidia Slips", "news", AMD).label == "bullish"


@pytest.mark.parametrize(("text", "target"), [
    ("Moderna to Join Nasdaq-100, Replacing Warner Bros. Discovery", QQQ),
    ("Moderna, Inc. to Join the Nasdaq-100 Index Beginning October 9, 2026", QQQ),
    ("Coinbase added to S&P 500", ["SPY", "S&P 500", "SPDR S&P 500 ETF Trust"]),
])
def test_a_constituent_change_says_nothing_about_the_index_fund(engine: SentinelEngine, text: str,
                                                                target: list[str]) -> None:
    assert engine.analyze(text).label == "bullish"  # read for the company joining
    fund = engine.analyze(text, "news", target)
    assert fund.score == 0.0 and not fund.drivers
    assert fund.confidence >= 0.7  # recognized as neutral news, not "found nothing"


def test_an_index_change_belongs_to_the_company_joining(engine: SentinelEngine) -> None:
    text = "Moderna to Join Nasdaq-100, Replacing Warner Bros. Discovery"
    assert engine.analyze(text, "news", ["MRNA", "Moderna"]).label == "bullish"
    # the replaced company: often moving up to a bigger index, so no direction either way
    assert engine.analyze(text, "news", ["WBD", "Warner Bros. Discovery"]).label == "neutral"
    assert engine.analyze("FormFactor to Join S&P MidCap 400, Replacing Twilio", "news", TWLO).label == "neutral"
    assert engine.analyze("Coinbase added to S&P 500", "news", ["COIN", "Coinbase"]).label == "bullish"


def test_peer_led_crypto_headline_is_a_bystander_for_the_target() -> None:
    from app.nlp.rules import is_bystander

    assert is_bystander("Crypto Weekly: ZEC Plunges as Bitcoin and Ether Hold Key Support Levels", BTC)
    assert is_bystander("Affirm Drops 4% as SoFi holds steady", SOFI)
    assert not is_bystander("ZEC plunges as Bitcoin slides", BTC)  # Bitcoin has news of its own


def test_capitalized_level_words_are_not_companies(engine: SentinelEngine) -> None:
    # trade-plan posts capitalize "Next", "First", "Key": none of them opens a peer's clause
    text = "$ACME KEY LEVELS — Next Upside Target 740 — First Bullish Trigger 730 — Key Support 725"
    aimed = engine.evidence(text, "social", ["ACME", "Acme"])
    assert [h.weight for h in aimed.hits] == pytest.approx([h.weight for h in engine.evidence(text, "social").hits])
