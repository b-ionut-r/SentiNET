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
