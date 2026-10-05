"""The SentiNET composite: six evidence components -> one calibrated 0..100 score.

Each component maps its evidence to a signed strength x in [-1, 1]
(score = 50 + 50·x) with a confidence in [0, 1]. Calibration choices:

* news/social text: x = tanh((s − b)·n/(n+6) / 0.35) — the weighted mean tone s
  is measured against the typical tone b (headlines skew positive: medians across
  a live sample of 11 tickers on 2026-10-04 were +0.05 for news and +0.06 for
  social text; b = +0.04 / +0.05 sits just below) and shrunk toward neutral for
  small samples (6 pseudo-items).
* StockTwits tags (one vote per account when available) are judged against
  their structural baseline (62% bullish) on a wide scale (0.35) and shrink
  with sample size; beyond the crowding thresholds (>= 85% / <= 35%) the
  signal folds back (contrarian: a unanimous crowd reads like the norm) and the
  whole social component is capped at the threshold's strength, so a one-sided
  crowd never adds points in its own direction (the crowding insight carries
  the risk). WSB sentiment is judged against 0.
* analysts: ratings are judged against the typical consensus (mean 2.4 on the
  1..5 scale), target upside against the typical +10%, revisions relative to
  coverage size.
* insiders: only open-market trades; buying by several insiders is strong,
  selling is scaled by market cap and mild (it is routine).
* momentum: GDELT 7d-vs-30d tone change + 90d percentile, and the last 48 h of
  headlines vs. the prior days. The headline shift is short-window evidence:
  the part of it that merely returns toward the typical tone (news-cycle decay
  after an event day: +0.30 → +0.14 when +0.04 is typical) counts half, it is
  damped unless it is ~2 standard errors, and it is shrunk for small samples
  (k/(k+15), k = items in the thinner window) — so on its own it reads at most
  as a mild lean unless headlines turn decisively. Without GDELT, a shift is
  named in the headline only when decisive (|x| >= 0.15), and one that only
  returns toward the typical tone reads "Headline tone normalizing", never
  "cooling"/"deteriorating".
* technicals: x = tanh(0.35 · mean z) where each z is a return or DMA
  distance, net of the typical market drift (+0.8%/month — an ordinary uptrend
  is the baseline, as +0.04 is for news tone), in units of its
  30d-volatility-implied spread (clipped at ±3σ): a 1σ broad trend reads ~67,
  a 2σ trend ~80 — price confirms sentiment, it does not outshout it.
  Dampened at RSI extremes (contrarian).

Composite = Σ w_eff·score / Σ w_eff with w_eff = nominal weight ×
(0.35 + 0.65·confidence), renormalized over available components, then pulled
toward 50 when little of the nominal weight is available:
50 + (raw − 50)·(0.35 + 0.65·min(1, available weight / 0.6)). When the
sentiment engine failed (texts unscored), the distance from 50 is capped at 11
points: structured data alone reads at most "Leaning".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from app.analytics.aggregate import Summary
from app.analytics.crowd import Tally
from app.analytics.util import cap_share, clamp, count, join_and, money, ordinal, pct, signed, squash, to_100
from app.schemas import AnalystView, Component, CrowdView, InsiderView, Technicals, ToneTrend

ComponentKey = Literal["news", "social", "analysts", "insiders", "momentum", "technicals"]

WEIGHTS: dict[ComponentKey, float] = {
    "news": 0.30, "social": 0.15, "analysts": 0.20, "insiders": 0.10, "momentum": 0.10, "technicals": 0.15,
}
LABELS: dict[ComponentKey, str] = {
    "news": "News", "social": "Social", "analysts": "Analysts", "insiders": "Insiders",
    "momentum": "Momentum", "technicals": "Technicals",
}

TEXT_PRIOR = 6.0  # pseudo-items of neutral prior for text components
TEXT_SCALE = 0.35
NEWS_BASELINE = 0.04  # typical headline tone (see module docstring)
SHIFT_SCALE = 0.5  # 48h-vs-prior headline tone shift giving x = tanh(1)
SHIFT_PRIOR = 15.0  # pseudo-items shrinking a headline shift (k = items in the thinner window)
REVERSION_CREDIT = 0.5  # share of a move back toward the typical tone that counts as momentum
SOCIAL_BASELINE = 0.05  # typical social-post tone
STOCKTWITS_BASELINE = 0.62
STOCKTWITS_MIN_TAGGED = 5
STOCKTWITS_SCALE = 0.35
CROWDED_LONG = 0.85  # bull share beyond which retail positioning is one-sided (contrarian)
CROWDED_SHORT = 0.35
CROWDED_FOLD = 1.5  # slope of the contrarian fold-back above CROWDED_LONG
CROWDING_MIN_TAGGED = 15
TECH_Z_SCALE = 0.35
TECH_Z_CLIP = 3.0
DRIFT_MONTH = 0.8  # typical equity drift, % per month (~10%/yr): a normal uptrend is the baseline
STRONG_TREND = 0.45  # |x| of a strong (not merely positive) price trend
DEGRADED_MAX_DISTANCE = 11.0  # engine failure: at most "Leaning" (39..61)
RATING_BASELINE = 2.4  # typical consensus mean (1 strong buy … 5 strong sell)
UPSIDE_BASELINE = 10.0  # typical upside to the mean target, percent
FULL_COVERAGE = 0.6  # nominal weight available for an unshrunk composite
MIN_PULL = 0.35  # with almost no evidence, only 35% of the raw distance from 50 survives
PHRASE_MARGIN = 0.15  # |x| of a clear signal (named as a driver in the headline)
MILD_MARGIN = 0.05  # |x| of a mild lean (named only as a counterweight)

ASSET_NAMES = {"CRYPTOCURRENCY": "crypto", "ETF": "ETFs", "INDEX": "indices", "MUTUALFUND": "funds"}

CONSENSUS_NAMES = {
    "strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold", "sell": "Sell", "strong_sell": "Strong Sell",
}


@dataclass
class Part:
    """One component with its evidence (becomes a schema `Component`)."""

    key: ComponentKey
    score: float | None = None
    confidence: float = 0.0
    detail: str = "no data"
    reason: str | None = None  # evidence line for the verdict's reasons
    phrase: str | None = None  # short clause for the headline (only when directional)
    strong: bool = False  # the phrase describes a clear signal, not a mild lean
    facts: dict[str, Any] = field(default_factory=dict)  # numbers other modules reuse

    @property
    def available(self) -> bool:
        return self.score is not None

    @property
    def x(self) -> float:
        """Signed strength in [-1, 1] (0 when unavailable)."""
        return 0.0 if self.score is None else (self.score - 50.0) / 50.0

    def component(self) -> Component:
        return Component(
            key=self.key, label=LABELS[self.key],
            score=round(self.score, 1) if self.score is not None else None,
            weight=WEIGHTS[self.key], available=self.available, detail=self.detail,
            confidence=round(self.confidence, 3),
        )


@dataclass
class Sub:
    """A sub-signal inside a component: strength x, reliability r, nominal weight w."""

    x: float
    r: float
    w: float


def _blend(subs: list[Sub]) -> tuple[float, float] | None:
    """(x, confidence) of reliability-weighted sub-signals; None when empty."""
    subs = [s for s in subs if s.w > 0]
    if not subs:
        return None
    eff = sum(s.w * max(s.r, 0.05) for s in subs)
    x = sum(s.w * max(s.r, 0.05) * s.x for s in subs) / eff
    conf = sum(s.w * s.r for s in subs) / sum(s.w for s in subs)
    return clamp(x, -1, 1), clamp(conf)


def text_strength(mean: float, n: float, baseline: float = 0.0) -> float:
    """Shrunk, saturating strength of a weighted mean tone vs its typical level.

    `n` is the effective number of items (Kish n_eff of the weights), so a
    sample dominated by a few heavy items shrinks like the small sample it is."""
    return squash((mean - baseline) * n / (n + TEXT_PRIOR), TEXT_SCALE)


def _pick(x: float, bull: str, bear: str, flat: str, margin: float = 0.1) -> str:
    return bull if x >= margin else bear if x <= -margin else flat


def _phrase(x: float, bull: str, bear: str, mild_bull: str | None = None,
            mild_bear: str | None = None) -> tuple[str | None, bool]:
    """(headline clause, strong?): strong for |x| >= PHRASE_MARGIN, mild down to MILD_MARGIN."""
    if x >= PHRASE_MARGIN:
        return bull, True
    if x <= -PHRASE_MARGIN:
        return bear, True
    if x >= MILD_MARGIN:
        return mild_bull or bull, False
    if x <= -MILD_MARGIN:
        return mild_bear or bear, False
    return None, False


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
def news_part(s: Summary, av_sentiment: float | None = None, av_articles: int | None = None) -> Part:
    """Published-media tone (+ Alpha Vantage's own ticker sentiment when keyed)."""
    part = Part("news", detail="no relevant news")
    has_av = av_sentiment is not None and (av_articles or 0) >= 5
    if s.n == 0 and not has_av:
        return part
    subs = []
    if s.n and s.mean is not None:
        subs.append(Sub(text_strength(s.mean, s.n_eff, NEWS_BASELINE), s.confidence, 0.8))
    if has_av:
        assert av_sentiment is not None and av_articles is not None
        subs.append(Sub(text_strength(av_sentiment, av_articles), av_articles / (av_articles + 10), 0.2))
    x, conf = _blend(subs) or (0.0, 0.0)
    part.score, part.confidence = to_100(x), conf
    shown = s.shrunk
    bits = []
    articles = count(s.n, "article")  # unique articles: the base of the bullish/bearish counts
    copies = s.coverage - s.n if s.coverage > s.n else 0
    if s.n:
        bits.append(f"{signed(shown)} across {articles} ({s.bullish} bullish / {s.bearish} bearish)")
    if has_av:
        bits.append(f"Alpha Vantage {signed(av_sentiment or 0.0)} ({av_articles} articles)")
    part.detail = " · ".join(bits)
    part.facts.update(tone=shown, n=s.n, outlets=s.outlets, articles=s.n, coverage=s.coverage or s.n)
    if s.n:
        soft = abs(shown) < 0.1
        lead = _pick(x, "News flow positive" if not soft else "News flow warmer than usual",
                     "News flow negative" if not soft else "News flow softer than usual", "News flow mixed")
        typical = f"; typical is {signed(NEWS_BASELINE)}" if soft and abs(x) >= 0.1 else ""
        syndicated = f" (+{count(copies, 'syndicated copy', 'syndicated copies')})" if copies else ""
        part.reason = (f"{lead}: {signed(shown)} average tone across {articles}{syndicated} from "
                       f"{count(s.outlets, 'outlet')} ({s.bullish} bullish vs {s.bearish} bearish{typical})")
        size = f"{signed(shown)} across {articles}"
        part.phrase, part.strong = _phrase(x, f"{'upbeat' if shown >= 0.15 else 'positive'} news ({size})",
                                           f"{'negative' if shown <= -0.1 else 'soft'} news ({size})")
    return part


# --------------------------------------------------------------------------- #
# Social
# --------------------------------------------------------------------------- #
def stocktwits_strength(ratio: float, n: int) -> float:
    """Signed strength of a StockTwits bull share vs its 62% norm, shrunk for small n.

    Beyond the crowding thresholds the effective share folds back (contrarian):
    above 85% at 1.5× slope (95% reads like 70%, a unanimous 100% like the 62.5%
    norm — extreme one-sidedness is a positioning risk, not more conviction);
    below 35% at half slope (0% reads like 52.5%, still mildly bearish)."""
    r = ratio
    if r > CROWDED_LONG:
        r = CROWDED_LONG - CROWDED_FOLD * (r - CROWDED_LONG)
    elif r < CROWDED_SHORT:
        r = CROWDED_SHORT + 0.5 * (CROWDED_SHORT - r)
    return squash((r - STOCKTWITS_BASELINE) * n / (n + 10), STOCKTWITS_SCALE)


def crowded(tally: Tally | None) -> int:
    """+1 crowded long / -1 crowded short (>= 15 tags beyond 85% / 35%), else 0."""
    if tally is None or tally.ratio is None or tally.n < CROWDING_MIN_TAGGED:
        return 0
    return 1 if tally.ratio >= CROWDED_LONG else -1 if tally.ratio <= CROWDED_SHORT else 0


def social_part(s: Summary, crowd: CrowdView | None, tally: Tally | None = None) -> Part:
    """Retail tone: social text + StockTwits tags (vs baseline, crowding-tapered) + WSB sentiment."""
    part = Part("social", detail="no social data")
    subs: list[Sub] = []
    bits: list[str] = []
    reason_bits: list[str] = []
    if s.n and s.mean is not None:
        subs.append(Sub(text_strength(s.mean, s.n_eff, SOCIAL_BASELINE), s.confidence, 0.45))
        bits.append(f"posts {signed(s.shrunk)} ({s.n})")
        reason_bits.append(f"social posts average {signed(s.shrunk)} across {s.n}")
    ratio = tally.ratio if tally is not None else None
    tagged = tally.n if tally is not None else 0
    if tally is not None and ratio is not None and tagged >= STOCKTWITS_MIN_TAGGED:
        subs.append(Sub(stocktwits_strength(ratio, tagged), tagged / (tagged + 15), 0.40))
        bits.insert(0, f"StockTwits {ratio:.0%} bullish ({tally.sample})")
        side = crowded(tally)
        crowd_note = (f"; past {CROWDED_LONG:.0%} it is crowding, which adds no further conviction" if side > 0
                      else f"; under {CROWDED_SHORT:.0%} it is capitulation, which adds no further weight"
                      if side < 0 else "")
        reason_bits.insert(0, f"{ratio:.0%} of {tally.described} are bullish "
                              f"({STOCKTWITS_BASELINE:.0%} is typical{crowd_note})")
    else:
        ratio = None
    if crowd is not None and crowd.wsb_sentiment is not None:
        comments = crowd.wsb_comments or 0
        subs.append(Sub(squash(crowd.wsb_sentiment, TEXT_SCALE) * comments / (comments + 20),
                        comments / (comments + 30), 0.15))
        bits.append(f"WSB {signed(crowd.wsb_sentiment)}")
        reason_bits.append(f"WallStreetBets {crowd.wsb_label or 'sentiment'} ({signed(crowd.wsb_sentiment)}, "
                           f"{count(comments, 'comment')})")
    blended = _blend(subs)
    if blended is None:
        return part
    x, conf = blended
    side = crowded(tally) if ratio is not None else 0
    if side:  # one-sided positioning: more euphoria (or despair) in the posts is not more evidence
        limit = abs(stocktwits_strength(CROWDED_LONG if side > 0 else CROWDED_SHORT, tagged))
        x = min(x, limit) if side > 0 else max(x, -limit)
    part.score, part.confidence = to_100(x), conf
    part.detail = " · ".join(bits)
    part.facts.update(ratio=ratio, tagged=tagged, tone=s.shrunk if s.n else None, n=s.n)
    lead = _pick(x, "Retail leaning bullish", "Retail leaning bearish", "Retail sentiment mixed")
    part.reason = f"{lead}: " + "; ".join(reason_bits)
    if ratio is not None and tally is not None:
        side = crowded(tally)
        base = f"{ratio:.0%} bullish of {tally.sample}"
        bull = f"{'crowded-long' if side > 0 else 'bullish'} retail ({base})"
        bear = f"{'capitulating' if side < 0 else 'bearish'} retail ({base})"
        part.phrase, part.strong = _phrase(x, bull, bear, f"mildly bullish retail ({base})",
                                           f"mildly bearish retail ({base})")
        if side:  # a crowded book is a caveat, never a headline driver of the read
            part.strong = False
    elif s.n:
        part.phrase, part.strong = _phrase(x, f"upbeat social chatter ({signed(s.shrunk)})",
                                           f"bearish social chatter ({signed(s.shrunk)})")
    return part


# --------------------------------------------------------------------------- #
# Analysts
# --------------------------------------------------------------------------- #
@dataclass
class Revisions:
    """Rating/target changes in the recent window (from the action list)."""

    upgrades_30d: int = 0
    downgrades_30d: int = 0
    raises_30d: int = 0
    cuts_30d: int = 0
    raise_firms: list[str] = field(default_factory=list)
    cut_firms: list[str] = field(default_factory=list)
    upgrade_firms: list[str] = field(default_factory=list)
    downgrade_firms: list[str] = field(default_factory=list)


def revisions(view: AnalystView, now: datetime) -> Revisions:
    out = Revisions()
    since = now - timedelta(days=30)
    for a in view.actions:
        if a.date < since:
            continue
        if a.action == "up":
            out.upgrades_30d += 1
            out.upgrade_firms.append(a.firm)
        elif a.action == "down":
            out.downgrades_30d += 1
            out.downgrade_firms.append(a.firm)
        if a.action != "init" and a.price_target and a.prior_target and a.prior_target > 0:
            change = a.price_target / a.prior_target - 1
            if change > 0.001:
                out.raises_30d += 1
                out.raise_firms.append(a.firm)
            elif change < -0.001:
                out.cuts_30d += 1
                out.cut_firms.append(a.firm)
    # The provider's counts may see actions beyond the 25 listed; keep the larger.
    out.raises_30d = max(out.raises_30d, view.pt_raises_30d)
    out.cuts_30d = max(out.cuts_30d, view.pt_cuts_30d)
    return out


def consensus_name(view: AnalystView) -> str | None:
    return CONSENSUS_NAMES.get(view.consensus or "")


def not_applicable(asset: str) -> str | None:
    """'n/a for crypto' for assets without analyst ratings / insider filings (None for equities)."""
    name = ASSET_NAMES.get(asset.upper())
    return f"n/a for {name}" if name else None


@dataclass(frozen=True)
class Upside:
    """Upside to the analyst targets, robust to a skewed target distribution.

    The mean target is used when the median agrees with it; when they differ
    by more than TARGET_SPLIT points the median (robust to one outlier target)
    is used; when they point opposite ways the targets sit at about the price
    (upside 0): AAPL with a mean 1.7% below and a median 1.9% above the price
    is not "targets below the price"."""

    mean: float | None
    median: float | None

    @property
    def split(self) -> bool:
        return self.mean is not None and self.median is not None and self.mean * self.median < 0

    @property
    def skewed(self) -> bool:
        return (self.mean is not None and self.median is not None and not self.split
                and abs(self.mean - self.median) > TARGET_SPLIT)

    @property
    def value(self) -> float | None:
        if self.split:
            return 0.0
        return self.median if self.skewed else self.mean


TARGET_SPLIT = 3.0  # points between mean and median upside beyond which the mean is skewed by outliers


def upside(view: AnalystView) -> Upside:
    """Mean and median target upside in percent (the median needs the price implied by the mean)."""
    median = None
    if view.upside_pct is not None and view.target_mean and view.target_median and view.upside_pct > -100:
        price = view.target_mean / (1 + view.upside_pct / 100)
        if price > 0:
            median = (view.target_median / price - 1) * 100
    return Upside(view.upside_pct, median)


def target_clause(view: AnalystView, up: Upside, currency: str | None = "USD") -> str | None:
    """'mean target $327.70 is 40% above the price' — or the median / 'about the price' when the mean misleads."""
    if up.mean is None or view.target_mean is None:
        return None
    mean = money(view.target_mean, price=True, currency=currency)
    if up.split and view.target_median is not None:
        return (f"targets sit at about the price (mean {mean}, {pct(up.mean)}; median "
                f"{money(view.target_median, price=True, currency=currency)}, {pct(up.median or 0.0)})")
    if up.skewed and view.target_median is not None and up.median is not None:
        side = "above" if up.median >= 0 else "below"
        return (f"median target {money(view.target_median, price=True, currency=currency)} is "
                f"{pct(abs(up.median), sign=False)} {side} the price (mean {mean}, {pct(up.mean)})")
    side = "above" if up.mean >= 0 else "below"
    return f"mean target {mean} is {pct(abs(up.mean), sign=False)} {side} the price"


def analysts_part(view: AnalystView | None, now: datetime, asset: str = "EQUITY",
                  currency: str | None = "USD") -> Part:
    """Consensus rating vs typical, upside to the targets (see `Upside`), revision momentum."""
    part = Part("analysts", detail="no analyst coverage")
    if view is None:
        part.detail = not_applicable(asset) or part.detail
        return part
    subs: list[Sub] = []
    total = view.total
    rating_x = upside_x = None
    target = upside(view)
    up = target.value
    if view.mean_rating is not None:
        shrink = total / (total + 3) if total else 0.5
        rating_x = clamp(0.56 * (RATING_BASELINE - view.mean_rating), -1, 1) * shrink
        subs.append(Sub(rating_x, (total / (total + 5)) if total else 0.3, 0.45))
    if up is not None:
        upside_x = squash(up - UPSIDE_BASELINE, 30.0)
        subs.append(Sub(upside_x, 0.8 if total >= 5 else 0.5, 0.30))
    rev = revisions(view, now)
    net = (view.upgrades_90d - view.downgrades_90d) + 0.5 * (rev.raises_30d - rev.cuts_30d)
    recent = any(now - a.date <= timedelta(days=90) for a in view.actions)
    # "No revisions" is information when the stock is rated; stale actions alone are not.
    if view.actions and (subs or recent):
        subs.append(Sub(squash(net, 2 + 0.5 * math.sqrt(max(total, 1))), 0.7, 0.25))
    blended = _blend(subs)
    if blended is None:
        return part
    x, conf = blended
    part.score, part.confidence = to_100(x), conf
    name = consensus_name(view)

    head = []
    if name:
        head.append(f"{name}" + (f" ({view.mean_rating:.2f})" if view.mean_rating is not None else ""))
    if total:
        head.append(count(total, "analyst"))
    if target.split:
        head.append("target ≈ price")
    elif up is not None:
        head.append(f"{'median ' if target.skewed else ''}target {pct(up)}")
    rev_bits = []
    if rev.raises_30d or rev.cuts_30d:
        rev_bits.append(f"30d PT: {rev.raises_30d} up / {rev.cuts_30d} down")
    if view.upgrades_90d or view.downgrades_90d:
        rev_bits.append(f"90d: {view.upgrades_90d} upgrades / {view.downgrades_90d} downgrades")
    part.detail = " · ".join(head + rev_bits) or "coverage without ratings"
    part.facts.update(name=name, net=net, revisions=rev, upside=target)

    clauses = []
    if name:
        rated = f"{name} consensus" + (
            f" (mean {view.mean_rating:.2f}" + (f" from {count(total, 'analyst')})" if total else ")")
            if view.mean_rating is not None else "")
        clauses.append(rated)
    said = target_clause(view, target, currency)
    if said:
        clauses.append(said)
    if rev_bits:
        clauses.append("; ".join(rev_bits))
    # The rating and the targets argue opposite ways: a Buy consensus with no upside left, or a
    # Hold/Sell with targets well above the price.
    rated_side = 1 if name in ("Buy", "Strong Buy") else -1 if name in ("Sell", "Strong Sell") else 0
    target_side = 0 if up is None else 1 if up >= UPSIDE_BASELINE else -1 if up <= 0 else 0
    mixed = (rating_x is not None and upside_x is not None and rating_x * upside_x < 0
             and min(abs(rating_x), abs(upside_x)) >= 0.08) or (rated_side * target_side < 0)
    if mixed and name and said and len(clauses) >= 2:  # "Buy consensus (…), but mean target … is 11% below the price"
        clauses[:2] = [f"{clauses[0]}, but {clauses[1]}"]
    lead = "Analysts mixed" if mixed else _pick(x, "Analysts bullish", "Analysts cautious", "Analysts neutral")
    part.reason = f"{lead}: " + "; ".join(clauses) if clauses else None
    to_target = ("targets ≈ the price" if target.split
                 else f"{pct(up)} to {'median ' if target.skewed else ''}target" if up is not None else None)
    if name and to_target:
        bull = f"a {name} consensus ({to_target})"
    elif to_target:
        bull = f"bullish analysts ({to_target})"
    else:
        bull = f"a {name} consensus" if name else "bullish analyst revisions"
    if up is not None and up < 0:
        which = "median target" if target.skewed else "targets"
        bear = (f"a {name} consensus with {which} {pct(abs(up), sign=False)} below the price" if name
                else f"analyst {which} {pct(abs(up), sign=False)} below the price")
    elif up is not None and up < UPSIDE_BASELINE / 2:
        bear = f"limited analyst upside ({to_target}" + (f", {name})" if name else ")")
    elif rev.cuts_30d + view.downgrades_90d > rev.raises_30d + view.upgrades_90d:
        bear = f"analyst downgrades ({view.downgrades_90d} in 90d, {rev.cuts_30d} PT cuts in 30d)"
    else:
        bear = f"a {name or 'cautious'} analyst consensus"
    inner = ", ".join(b for b in (name, to_target) if b)
    part.phrase, part.strong = _phrase(x, bull, bear, f"supportive analysts ({inner})" if inner else None,
                                       f"cautious analysts ({inner})" if inner else None)
    return part


# --------------------------------------------------------------------------- #
# Insiders
# --------------------------------------------------------------------------- #
def insiders_part(view: InsiderView | None, market_cap: float | None, now: datetime,
                  asset: str = "EQUITY", currency: str | None = "USD") -> Part:
    """Open-market insider flow: clustered buying is strong, selling is routine-scaled.

    `currency` is that of the trade values (None: unknown, shown without a symbol)."""
    part = Part("insiders", detail="no open-market insider trades")
    if view is None or (view.buys == 0 and view.sells == 0):
        if view is None:
            part.detail = not_applicable(asset) or "insider data unavailable"
        else:
            part.detail = f"no open-market insider trades in {view.window_days}d"
        return part
    today = now.date()
    buys = [t for t in view.transactions if t.kind == "buy"]
    latest_by_buyer: dict[str, int] = {}
    for t in buys:
        age = max((today - t.date).days, 0)
        latest_by_buyer[t.insider] = min(age, latest_by_buyer.get(t.insider, age))
    if latest_by_buyer:
        buyer_signal = sum(0.5 ** (age / 60.0) for age in latest_by_buyer.values())
    else:
        buyer_signal = 0.5 * view.buys
    buy_signal = buyer_signal * (1 + 0.5 * math.log10(1 + view.buy_value / 250_000)) if view.buys else 0.0
    sell_bps = view.sell_value / market_cap * 1e4 if market_cap and market_cap > 0 else None
    sell_signal = sell_bps / 5.0 if sell_bps is not None else min(3.0, view.sells / 8.0)
    x = 0.7 * math.tanh(buy_signal / 2.5) - 0.4 * math.tanh(sell_signal / 2.0)
    trades = view.buys + view.sells
    part.score = to_100(x)
    part.confidence = clamp(trades / (trades + 4) * (1.0 if view.buys else 0.7))
    window = f"{view.window_days}d"
    bought, sold = money(view.buy_value, currency=currency), money(view.sell_value, currency=currency)
    part.detail = (f"{view.buys} buys ({bought}) / {view.sells} sells ({sold}) · "
                   f"{window}")
    buyers = len(latest_by_buyer) or None
    part.facts.update(buyers=buyers, sell_bps=sell_bps, buy_signal=buy_signal)
    who = f" by {count(buyers, 'insider')}" if buyers else ""
    share = f", {cap_share(sell_bps)} of market cap" if sell_bps is not None else ""
    days = f"{view.window_days} days"
    if view.buys:
        lead = "Insider buying" if x >= 0 else "Net insider selling"
        sells = f" vs {count(view.sells, 'sale')} ({sold}{share})" if view.sells else ", no sales"
        part.reason = (f"{lead}: {count(view.buys, 'open-market purchase')} ({bought}){who} "
                       f"in {days}{sells}")
        bear = f"net insider selling ({sold} sold vs {bought} bought)"
    else:
        routine = " — routine-sized for its market cap" if sell_bps is not None and sell_bps < 5 else ""
        part.reason = (f"Insider selling only: {count(view.sells, 'open-market sale')} ({sold}{share}) "
                       f"and no purchases in {days}{routine}")
        bear = f"insider selling ({sold}{share})"
    part.phrase, part.strong = _phrase(x, f"insider buying ({bought}{who})", bear)
    return part


# --------------------------------------------------------------------------- #
# Momentum (of sentiment)
# --------------------------------------------------------------------------- #
def momentum_part(tone: ToneTrend | None, recent: Summary, older: Summary) -> Part:
    """Is sentiment improving? GDELT tone trend + last-48h headlines vs the prior days."""
    part = Part("momentum", detail="no tone history")
    subs: list[Sub] = []
    bits: list[str] = []
    reason_bits: list[str] = []
    if tone is not None and tone.change_7d_vs_30d is not None:
        ch, p = tone.change_7d_vs_30d, tone.percentile_7d
        g = math.tanh(ch / 0.8)
        g = 0.6 * g + 0.4 * (2 * p - 1) if p is not None else g
        subs.append(Sub(g, 0.8 if len(tone.series) >= 30 else 0.5, 0.6))
        pctl = f", {ordinal(round(p * 100))} pct of 90d" if p is not None else ""
        t7 = f"{signed(tone.tone_7d)} " if tone.tone_7d is not None else ""
        t30 = f" vs 30d {signed(tone.tone_30d)}" if tone.tone_30d is not None else ""
        bits.append(f"GDELT 7d {t7}({signed(ch)}{t30}){pctl}")
        reason_bits.append(f"global news tone (GDELT) 7d {t7}vs 30d"
                           f"{' ' + signed(tone.tone_30d) if tone.tone_30d is not None else ''}"
                           f" ({signed(ch)}){pctl}")
    shift = headline_shift(recent, older)
    if shift is not None:
        subs.append(Sub(shift.strength, 0.6 * shift.k / (shift.k + SHIFT_PRIOR), 0.4))
        bits.append(f"headlines 48h {signed(shift.recent)} vs {signed(shift.older)} before")
        reason_bits.append(f"last-48h headlines average {signed(shift.recent)} ({recent.n}) vs "
                           f"{signed(shift.older)} in the prior days ({older.n})")
    blended = _blend(subs)
    if blended is None:
        return part
    x, conf = blended
    part.score, part.confidence = to_100(x), conf
    part.detail = " · ".join(bits)
    gdelt = tone is not None and tone.change_7d_vs_30d is not None
    # Without GDELT, a 48h shift that only drifts back toward the typical tone is the news cycle
    # settling after an event day (+0.27 -> +0.13 when +0.04 is typical), not a change of mood.
    normalizing = not gdelt and shift is not None and shift.normalizing and abs(x) >= 0.1
    level = tone.tone_7d if tone is not None and tone.tone_7d is not None else recent.mean
    if normalizing:
        lead = "Headline tone normalizing"
        typical = f" (typical is {signed(NEWS_BASELINE)})"
    elif x >= 0.1:
        lead = "Sentiment turning positive" if level is not None and level > 0 and _was_negative(tone, older) \
            else "Sentiment improving"
        typical = ""
    elif x <= -0.1:
        lead = "Sentiment cooling" if level is not None and level > 0 else "Sentiment deteriorating"
        typical = ""
    else:
        lead, typical = "Sentiment trend flat", ""
    part.reason = f"{lead}: " + "; ".join(reason_bits) + typical
    verb_up, verb_down = "improving", ("cooling" if level is not None and level > 0 else "deteriorating")
    if gdelt:
        assert tone is not None and tone.change_7d_vs_30d is not None
        part.phrase, part.strong = _phrase(
            x, f"{verb_up} global news tone (GDELT {signed(tone.change_7d_vs_30d)} vs 30d)",
            f"{verb_down} global news tone (GDELT {signed(tone.change_7d_vs_30d)} vs 30d)")
    elif shift is not None and abs(x) >= PHRASE_MARGIN:
        # A lone 48h shift is short-window evidence: it is named in the headline only when it is
        # decisive on its own (a mild lean from it is never a "main drag").
        r, o = signed(shift.recent), signed(shift.older)
        if normalizing:
            up, down = (f"easing headline pessimism ({r} in 48h vs {o} before)",
                        f"fading headline optimism ({r} in 48h vs {o} before)")
        else:
            up, down = (f"{verb_up} headlines ({r} in 48h vs {o} before)",
                        f"{'cooling' if shift.recent > NEWS_BASELINE else 'deteriorating'} headlines "
                        f"({r} in 48h vs {o} before)")
        part.phrase, part.strong = _phrase(x, up, down)
    part.facts.update(tone_change=tone.change_7d_vs_30d if tone else None, shift=shift, gdelt=gdelt,
                      normalizing=normalizing)
    return part


@dataclass(frozen=True)
class Shift:
    """Headline tone of the last 48 h vs the prior days (see the module docstring)."""

    recent: float
    older: float
    z: float  # (recent − older) / its standard error
    effective: float  # recent − older, a return toward the typical tone counted at REVERSION_CREDIT
    k: int  # items in the thinner window

    @property
    def change(self) -> float:
        return self.recent - self.older

    @property
    def normalizing(self) -> bool:
        """The recent window only moved back toward the typical tone (without crossing it)."""
        before, after = self.older - NEWS_BASELINE, self.recent - NEWS_BASELINE
        return before * after >= 0 and abs(after) < abs(before)

    @property
    def strength(self) -> float:
        """Signed strength in (-1, 1): decay-discounted, significance-damped, sample-shrunk."""
        return (math.tanh(self.effective / SHIFT_SCALE) * min(1.0, abs(self.z) / 2.0)
                * self.k / (self.k + SHIFT_PRIOR))


def reverting_change(before: float, after: float, typical: float) -> float:
    """after − before, with the part that merely returns toward `typical` (without crossing
    it) counted at REVERSION_CREDIT: after an event day, tone drifting back to normal is
    the news cycle, not a change of sentiment. Moves away from typical count in full."""
    d = after - before
    gap = before - typical
    if gap * d >= 0:
        return d
    toward = min(abs(d), abs(gap))
    return math.copysign(REVERSION_CREDIT * toward + (abs(d) - toward), d)


def headline_shift(recent: Summary, older: Summary) -> Shift | None:
    """The last 48 h of headlines vs the prior days (>= 5 items each), or None."""
    if recent.n < 5 or older.n < 5 or recent.mean is None or older.mean is None:
        return None
    d = recent.mean - older.mean
    se = math.sqrt(recent.spread ** 2 / max(recent.n_eff, 1.0) + older.spread ** 2 / max(older.n_eff, 1.0))
    return Shift(recent=recent.mean, older=older.mean, z=d / max(se, 0.02),
                 effective=reverting_change(older.mean, recent.mean, NEWS_BASELINE), k=min(recent.n, older.n))


def _was_negative(tone: ToneTrend | None, older: Summary) -> bool:
    if tone is not None and tone.tone_30d is not None:
        return tone.tone_30d < 0
    return older.mean is not None and older.mean < 0


# --------------------------------------------------------------------------- #
# Technicals
# --------------------------------------------------------------------------- #
def technicals_part(t: Technicals | None) -> Part:
    """Price-implied sentiment: volatility-scaled returns and DMA distances, RSI-dampened."""
    part = Part("technicals", detail="no price history")
    if t is None:
        return part
    sigma = max((t.volatility_30d or 0.0) / math.sqrt(12), 1.5) if t.volatility_30d else 8.0  # monthly %
    # (value, spread, weight, typical level under normal drift). A price drifting up at the
    # market's usual pace sits ~(N−1)/2 days of drift above its N-day average.
    pieces: list[tuple[float | None, float, float, float]] = [
        (t.return_1m, sigma, 0.25, DRIFT_MONTH),
        (t.return_3m, sigma * math.sqrt(3), 0.30, 3 * DRIFT_MONTH),
        (t.vs_50dma_pct, sigma, 0.20, DRIFT_MONTH * 24.5 / 21),  # distance from an N-day mean spreads like
        (t.vs_200dma_pct, sigma * math.sqrt(3), 0.25, DRIFT_MONTH * 99.5 / 21),  # ~sqrt(N/3) days of moves
    ]
    avail = [(v - drift, sd, w) for v, sd, w, drift in pieces if v is not None]
    if not avail:
        return part
    z = sum(w * clamp(v / sd, -TECH_Z_CLIP, TECH_Z_CLIP) for v, sd, w in avail) / sum(w for _, _, w in avail)
    x = math.tanh(TECH_Z_SCALE * z)
    rsi = t.rsi_14
    if rsi is not None and rsi > 75 and x > 0:
        x *= 1 - min(0.5, (rsi - 75) / 30)
    elif rsi is not None and rsi < 25 and x < 0:
        x *= 1 - min(0.5, (25 - rsi) / 30)
    part.score = to_100(x)
    part.confidence = clamp(0.9 * sum(w for _, _, w in avail))
    bits: list[str] = [t.trend] if t.trend else []
    if t.return_3m is not None:
        bits.append(f"3M {pct(t.return_3m)}")
    elif t.return_1m is not None:
        bits.append(f"1M {pct(t.return_1m)}")
    if t.vs_200dma_pct is not None:
        bits.append(f"{pct(t.vs_200dma_pct)} vs 200-DMA")
    elif t.vs_50dma_pct is not None:
        bits.append(f"{pct(t.vs_50dma_pct)} vs 50-DMA")
    if rsi is not None:
        bits.append(f"RSI {rsi:.0f}")
    part.detail = " · ".join(bits)

    clauses = []
    if t.return_3m is not None:
        clauses.append(f"{pct(t.return_3m)} over 3 months")
    if t.return_1m is not None:
        clauses.append(f"{pct(t.return_1m)} over 1 month")
    if t.vs_200dma_pct is not None:
        side = "above" if t.vs_200dma_pct >= 0 else "below"
        clauses.append(f"{pct(abs(t.vs_200dma_pct), sign=False)} {side} the 200-day average")
    if rsi is not None:
        state = " (overbought)" if rsi >= 70 else " (oversold)" if rsi <= 30 else ""
        clauses.append(f"RSI {rsi:.0f}{state}")
    lead = _pick(x, "Price trend supportive", "Price trend weak", "Price trend flat")
    part.reason = f"{lead}: " + join_and(clauses)
    moves = [(abs(v) / s, f"{pct(v)} in {label}") for v, s, label in (
        (t.return_1m, sigma, "1M"), (t.return_3m, sigma * math.sqrt(3), "3M")) if v is not None]
    horizon = max(moves)[1] if moves else None  # the more striking move, volatility-adjusted
    if horizon:
        up = "a strong price trend" if x >= STRONG_TREND else "positive price action"
        down = "a steep downtrend" if x <= -STRONG_TREND else "weak price action"
        part.phrase, part.strong = _phrase(x, f"{up} ({horizon})", f"{down} ({horizon})",
                                           f"a firm tape ({horizon})", f"a soft tape ({horizon})")
    part.facts.update(sigma_month=sigma, z=z)
    return part


# --------------------------------------------------------------------------- #
# Pending acquisition
# --------------------------------------------------------------------------- #
DEAL_DAMPING = 0.5  # strength and confidence kept by price-anchored components while a deal is pending


def deal_anchored(part: Part) -> Part:
    """Discount a component whose evidence a pending acquisition has made stale.

    Once the company has agreed to be acquired its price tracks the deal terms:
    analyst targets set before the agreement and the price trend that jumped on
    it no longer measure sentiment, so both keep half their strength and
    confidence and say why (the deal itself leads the verdict; see deals.py)."""
    if part.score is None:
        return part
    part.score = 50.0 + (part.score - 50.0) * DEAL_DAMPING
    part.confidence *= DEAL_DAMPING
    part.detail += " · discounted: deal pending"
    if part.reason:
        part.reason += " (discounted: a pending acquisition anchors the price to the deal terms)"
    part.strong = False
    part.facts["deal_anchored"] = True
    return part


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
@dataclass
class Composite:
    score: int
    raw: float  # before the coverage pull toward 50
    coverage: float  # nominal weight of available components (0..1)
    parts: dict[ComponentKey, Part]
    effective: dict[ComponentKey, float]  # w_eff of available components (normalized)
    contributions: dict[ComponentKey, float]  # points vs 50; sums to score - 50 (before rounding)

    def available(self) -> list[Part]:
        return [p for p in self.parts.values() if p.available]


def compose(parts: list[Part], max_distance: float | None = None) -> Composite:
    """Renormalized, confidence-weighted composite with a small-coverage pull to 50.

    `max_distance` caps |score − 50| (degraded runs; see module docstring)."""
    by_key = {p.key: p for p in parts}
    avail = [p for p in parts if p.available]
    if not avail:
        return Composite(score=50, raw=50.0, coverage=0.0, parts=by_key, effective={}, contributions={})
    eff = {p.key: WEIGHTS[p.key] * (0.35 + 0.65 * p.confidence) for p in avail}
    total = sum(eff.values())
    raw = sum(eff[p.key] * (p.score or 50.0) for p in avail) / total
    coverage = sum(WEIGHTS[p.key] for p in avail)
    pull = MIN_PULL + (1 - MIN_PULL) * min(1.0, coverage / FULL_COVERAGE)
    if max_distance is not None and abs(raw - 50.0) * pull > max_distance:
        pull = max_distance / abs(raw - 50.0)
    final = 50.0 + (raw - 50.0) * pull
    contributions = {p.key: eff[p.key] / total * ((p.score or 50.0) - 50.0) * pull for p in avail}
    return Composite(
        score=int(round(clamp(final, 0, 100))), raw=raw, coverage=coverage, parts=by_key,
        effective={k: v / total for k, v in eff.items()}, contributions=contributions,
    )

