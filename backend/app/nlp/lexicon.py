"""Finance sentiment lexicon for the Sentinel engine.

Hand-authored, Loughran-McDonald-inspired vocabulary for market text (headlines,
filings, analyst notes) plus the social register (StockTwits/Reddit slang and
emoji). Pure data: the pattern logic that uses it lives in `app.nlp.rules`.

Valence scale: 1.0 is one clear, unambiguous market signal ("downgrade",
"misses estimates"); 0.3-0.5 is a weak tilt; 1.5+ is extreme ("bankruptcy").

The vocabulary is split by *role*, because finance sentiment is compositional:

* ``POSITIVE`` / ``NEGATIVE`` / ``LITIGIOUS`` / ``SOCIAL`` - terms with their own valence.
* ``DIRECTIONS`` x ``METRICS`` - movement words take their sign from what moved:
  "profit rose" is bullish but "costs rose" / "loss widened" are bearish and
  "loss narrowed" / "eases concerns" are bullish.
* ``RATINGS`` - analyst rating vocabulary, ranked so "to Neutral from Buy" reads as a cut.
* Modifiers: ``NEGATORS``, ``HEDGES`` (uncertainty), ``INTENSIFIERS``, contrast markers.
* ``NEUTRALIZERS`` - phrases whose words look like sentiment but are not
  ("shares outstanding", "Best Buy", "in line with", "record date").

Keys are lowercase; hyphens are treated as spaces ("all-time" == "all time").
"""
from __future__ import annotations

from typing import NamedTuple


# --------------------------------------------------------------------------- #
# Inflection helpers (keep the tables readable: list a verb once)
# --------------------------------------------------------------------------- #
def _forms(base: str, *, double: bool = False, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Regular English verb forms: base, 3rd person, past, gerund (+ irregulars)."""
    if base.endswith(("s", "sh", "ch", "x", "z", "o")):
        third = base + "es"
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        third = base[:-1] + "ies"
    else:
        third = base + "s"
    if double:
        past, ing = base + base[-1] + "ed", base + base[-1] + "ing"
    elif base.endswith("e"):
        past = base + "d"
        ing = base[:-2] + "ying" if base.endswith("ie") else (base + "ing" if base.endswith("ee") else base[:-1] + "ing")
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        past, ing = base[:-1] + "ied", base + "ing"
    else:
        past, ing = base + "ed", base + "ing"
    return (base, third, past, ing, *extra)


def _verbs(table: dict[str, float], **flags: tuple[str, ...]) -> dict[str, float]:
    """Expand ``{"plunge": -1.4}`` into all inflections. ``double=("drop",)`` doubles
    the final consonant; ``irregular_<base>=("fell",)`` style extras via ``extra``."""
    doubled = set(flags.get("double", ()))
    out: dict[str, float] = {}
    for base, value in table.items():
        for form in _forms(base, double=base in doubled):
            out.setdefault(form, value)
    return out


def _plural(table: dict[str, float]) -> dict[str, float]:
    """Add simple plurals for nouns (``lawsuit`` -> ``lawsuits``)."""
    out = dict(table)
    for word, value in table.items():
        if " " in word or word.endswith("s"):
            continue
        plural = word[:-1] + "ies" if word.endswith("y") and word[-2] not in "aeiou" else word + "s"
        out.setdefault(plural, value)
    return out


# --------------------------------------------------------------------------- #
# Positive vocabulary
# --------------------------------------------------------------------------- #
POSITIVE: dict[str, float] = {
    # analyst stance
    **_verbs({"upgrade": 1.1, "outperform": 0.8, "outshine": 0.7, "outpace": 0.6, "outsell": 0.5}),
    "outperformance": 0.7, "overweight": 0.6, "top pick": 1.0, "top picks": 0.6, "strong buy": 1.1,
    "conviction buy": 1.1, "buy rating": 0.8, "rated buy": 0.8, "a buy": 0.6, "buy signal": 0.7,
    "buying opportunity": 0.8, "worth buying": 0.7, "bullish": 1.1, "more bullish": 1.2, "bull case": 0.6,
    "bull market": 0.6, "bull run": 0.9, "upside": 0.7, "upside potential": 0.8, "undervalued": 0.8,
    "underpriced": 0.7, "underappreciated": 0.6, "bargain": 0.6, "cheap": 0.3, "attractive": 0.5,
    "compelling": 0.6, "favorable": 0.6, "favourable": 0.6, "crazy about": 0.6, "warming up to": 0.6,
    "likes": 0.3, "loving": 0.4, "love": 0.3, "loves": 0.3, "upbeat": 0.9, "optimistic": 0.8, "confident": 0.6, "constructive": 0.5,
    "raises the bar": 0.6, "well positioned": 0.6, "on track": 0.4, "ahead of schedule": 0.6,
    "ahead of plan": 0.5, "positive surprise": 0.9, "upside surprise": 0.9,
    # results quality
    "beat": 0.8, "beats": 0.8, "beat and raise": 1.3, "blowout": 0.8, "blowout quarter": 1.2, "stellar": 1.1, "impressive": 0.9,
    "excellent": 0.9, "outstanding": 0.7, "robust": 0.8, "solid": 0.6, "strong": 0.7, "stronger": 0.8,
    "strongest": 0.9, "strength": 0.6, "healthy": 0.5, "resilient": 0.6, "resilience": 0.5,
    "positive": 0.6, "positively": 0.5, "encouraging": 0.7, "promising": 0.6, "bright": 0.4,
    "brighter": 0.6, "boom": 0.8, "booming": 1.0, "thriving": 0.9, "thrive": 0.7, "thrives": 0.7,
    "flourish": 0.7, "flourishing": 0.8, "success": 0.7, "successful": 0.7, "successfully": 0.5,
    "milestone": 0.5, "breakthrough": 0.9, "record breaking": 0.9, "profitable": 0.6, "accretive": 0.7,
    "tailwind": 0.7, "tailwinds": 0.7, "momentum": 0.3, "turnaround": 0.6, "upturn": 0.8,
    "better": 0.35, "best": 0.3, "best day": 1.0, "best week": 0.9, "best month": 0.9, "best quarter": 0.9,
    "best year": 0.8, "best performer": 0.8, "best performing": 0.7, "best ever": 0.9, "good news": 0.7,
    "good": 0.25, "great": 0.4, "nice": 0.2, "superb": 0.9, "terrific": 0.9, "glowing": 0.6, "rosy": 0.6,
    "stable": 0.3, "stability": 0.3, "steady": 0.2, "supportive": 0.4, "support": 0.3,
    "finding support": 0.6, "breaks resistance": 0.7, "broke resistance": 0.7, "breaks above": 0.7,
    "broke above": 0.7, "golden cross": 0.8, "oversold": 0.4, "bottomed": 0.5, "bottoming": 0.4,
    "breakout": 0.8, "breaks out": 0.8, "broke out": 0.7,
    # profitability turns
    "turns profitable": 1.0, "turned profitable": 1.0, "return to profit": 1.0, "returns to profit": 1.0,
    "returned to profit": 1.0, "returned to profitability": 1.0, "return to profitability": 1.0,
    "back to profit": 0.9, "swung to a profit": 1.0, "swings to a profit": 1.0, "swung to profit": 1.0,
    "swings to profit": 1.0, "turned to a profit": 1.0, "turned to profit": 1.0, "back in the black": 0.9,
    "in the black": 0.6, "in the green": 0.6, "into the green": 0.6, "positive territory": 0.6,
    "pares losses": 0.4, "pared losses": 0.4, "erases losses": 0.7, "erased losses": 0.7,
    "recoups losses": 0.6, "recouped losses": 0.6, "trims losses": 0.4, "snaps losing streak": 0.6,
    "ends losing streak": 0.6, "winning streak": 0.7,
    # business wins
    "win": 0.5, "wins": 0.5, "won": 0.45, "winner": 0.4, "winners": 0.35, "winning": 0.35, "awarded": 0.5,
    "secures": 0.6, "secured": 0.5, "clinches": 0.7, "clinched": 0.7, "lands": 0.3, "landed": 0.3,
    "approval": 0.7, "approvals": 0.6, "approved": 0.6, "approves": 0.6, "approve": 0.4,
    "clearance": 0.6, "cleared": 0.5, "green light": 0.6, "greenlight": 0.6, "greenlights": 0.6,
    "fda approval": 1.1, "wins approval": 1.0, "receives approval": 0.9, "gets approval": 0.9,
    "gains approval": 0.9, "fast track": 0.6, "breakthrough designation": 0.8,
    "breakthrough therapy": 0.6, "positive data": 0.9, "positive results": 0.9, "positive topline": 0.9,
    "met primary endpoint": 1.0, "meets primary endpoint": 1.0, "met its primary endpoint": 1.0,
    "meet primary endpoint": 0.9, "promising results": 0.9, "partnership": 0.3, "partners with": 0.3,
    "collaboration": 0.3, "strikes deal": 0.4, "strike a deal": 0.4, "signs deal": 0.3, "inks deal": 0.4,
    "teams up": 0.3, "wins contract": 1.0, "contract win": 0.9, "awarded contract": 0.9, "new contract": 0.5,
    "launches": 0.2, "unveils": 0.2, "innovative": 0.4, "innovation": 0.3, "expansion plan": 0.4,
    "buyback": 0.8, "buybacks": 0.7, "share buyback": 0.8, "stock buyback": 0.8, "share repurchase": 0.8,
    "stock repurchase": 0.8, "repurchase program": 0.8, "repurchase plan": 0.8, "repurchase": 0.5,
    "special dividend": 0.6, "dividend increase": 0.9, "dividend hike": 0.9, "raises dividend": 0.9,
    "opportunity": 0.3, "opportunities": 0.3, "benefit": 0.4, "benefits": 0.4, "benefited": 0.5,
    "benefiting": 0.4, "benefit from": 0.6, "boon": 0.8, "windfall": 0.8, "lucrative": 0.7,
    "cheer": 0.5, "cheers": 0.6, "cheered": 0.6, "applauds": 0.6, "praised": 0.5, "praises": 0.5,
    "snapping up": 0.7, "snaps up": 0.6, "scooping up": 0.6, "piling into": 0.6, "loading up": 0.6,
    "accumulating": 0.4, "accumulate": 0.4, "new stake": 0.3, "takes stake": 0.3, "insider buying": 0.8,
    "insider purchase": 0.7, "insider buys": 0.7, "buys shares": 0.4, "bought shares": 0.4,
    "short squeeze": 0.8, "rewarded": 0.4, "rewarding": 0.4, "dip buying": 0.5,
    "rejects allegations": 0.2, "rejects allegation": 0.2, "denies allegations": 0.2, "refutes allegations": 0.2,
    "rejects claims": 0.2, "denies wrongdoing": 0.2,
    "estimate beating": 1.0, "estimates beating": 1.0, "premarket gainers": 0.6, "top gainers": 0.5,
    "biggest gainers": 0.6, "gainers": 0.5, "red to green": 0.7, "bull of the day": 0.8, "new orders": 0.5,
    "cheapest": 0.4, "comeback": 0.6, "roaring back": 0.9, "upsides": 0.6, "bulls": 0.35,
    "regaining steam": 0.6, "gaining steam": 0.5, "gathers steam": 0.5, "gathering steam": 0.5,
    "stabilizing": 0.4, "stabilize": 0.4, "stabilizes": 0.4, "stabilized": 0.4, "stabilising": 0.4,
    "unusual calls": 0.6, "unusual call activity": 0.6, "call buying": 0.5, "call buyers": 0.4,
    "bullish options": 0.6, "bullish bets": 0.7, "bullish bet": 0.6, "to shame": 0.5, "discovery": 0.4,
    "discovers": 0.4, "restarts": 0.3, "resumes": 0.3, "resumed": 0.3, "resumption": 0.3,
    "right direction": 0.5, "step in the right direction": 0.6, "primed": 0.4, "breakout to the upside": 1.0,
    "to the upside": 0.5, "green shoots": 0.7, "time to buy": 0.6, "went as expected": 0.4,
    "go as expected": 0.4, "goes as expected": 0.4, "going as expected": 0.4, "sold out": 0.4,
    "meets estimates": 0.25, "salvage": 0.4, "impress": 0.6, "impresses": 0.7, "impressed": 0.6,
    "double upgrade": 1.4, "breaks through": 0.6, "broke through": 0.6, "buy back": 0.7, "buys back": 0.7,
    "bought back": 0.7, "buying back": 0.7, "up day": 0.6, "reprieve": 0.4, "good investment": 0.6,
    "live up to": 0.4, "lives up to": 0.5, "lived up to": 0.5, "maintained as buy": 0.6,
    # legal relief
    "acquitted": 0.8, "acquittal": 0.8, "exonerated": 0.8, "vindicated": 0.7, "wins lawsuit": 0.8,
    "wins case": 0.7, "lawsuit dismissed": 0.7, "case dismissed": 0.6, "charges dropped": 0.7,
    "drops charges": 0.6, "wins appeal": 0.7, "favorable ruling": 0.8, "settles lawsuit": 0.3,
    # macro
    "easing": 0.3, "stimulus": 0.5, "rate cut": 0.4, "rate cuts": 0.4, "dovish": 0.6, "soft landing": 0.7,
    "goldilocks": 0.6, "trade deal": 0.6, "truce": 0.5, "ceasefire": 0.4, "deal reached": 0.5,
    "agreement reached": 0.4, "reopening": 0.4, "reopens": 0.3, "hiring spree": 0.6,
    "better than feared": 0.6, "not as bad as feared": 0.6, "consumer confidence": 0.1,
}

# --------------------------------------------------------------------------- #
# Negative vocabulary
# --------------------------------------------------------------------------- #
NEGATIVE: dict[str, float] = {
    # analyst stance
    **_verbs({"downgrade": -1.1, "underperform": -0.8}),
    "underperformance": -0.7, "underweight": -0.6, "sell rating": -0.8, "rated sell": -0.8,
    "strong sell": -1.1, "bearish": -1.1, "more bearish": -1.2, "bear case": -0.6, "bear market": -0.8,
    "downside": -0.6, "downside risk": -0.7, "overvalued": -0.8, "overpriced": -0.7, "overbought": -0.4,
    "expensive": -0.3, "pricey": -0.4, "frothy": -0.6, "bubble": -0.7, "unsustainable": -0.6,
    "negative surprise": -0.9, "downside surprise": -0.9, "cautious": -0.4, "caution": -0.3,
    "skeptical": -0.5, "sceptical": -0.5, "death cross": -0.8,
    # results quality
    "miss": -0.8, "misses": -0.8, "missed": -0.8, "disappoint": -0.9, "disappoints": -0.9,
    "disappointed": -0.8, "disappointing": -0.9, "disappointment": -0.9, "letdown": -0.7,
    "dismal": -1.0, "dire": -0.9, "grim": -0.9, "bleak": -0.9, "gloomy": -0.8, "gloom": -0.8,
    "pessimistic": -0.8, "downbeat": -0.9, "lackluster": -0.7, "lacklustre": -0.7, "tepid": -0.5,
    "anemic": -0.7, "sluggish": -0.7, "subdued": -0.4, "muted": -0.3, "weak": -0.7, "weaker": -0.8,
    "weakest": -1.0, "weakness": -0.7, "soft": -0.3, "softer": -0.4, "softness": -0.5, "poor": -0.8,
    "poorly": -0.7, "bad": -0.7, "bad news": -0.8, "worse": -0.7, "worst": -1.0, "terrible": -1.0,
    "awful": -1.0, "horrible": -1.0, "ugly": -0.7, "brutal": -0.8, "painful": -0.6, "pain": -0.5,
    "trouble": -0.7, "troubled": -0.8, "troubles": -0.7, "troubling": -0.6, "struggle": -0.7,
    "struggles": -0.7, "struggling": -0.8, "struggled": -0.7, "grappling": -0.6, "problem": -0.4,
    "problems": -0.5, "woes": -0.8, "headwind": -0.7, "headwinds": -0.7, "pressured": -0.6,
    "under pressure": -0.8, "selling pressure": -0.7, "margin pressure": -0.8, "challenging": -0.4,
    "challenges": -0.3, "difficult": -0.4, "difficulties": -0.6, "tough": -0.4,
    **_verbs({"hurt": -0.7, "dent": -0.6, "damage": -0.6, "threaten": -0.7, "warn": -0.8, "drag": -0.5,
              "jeopardize": -0.8, "undermine": -0.7, "derail": -0.8, "plague": -0.7, "haunt": -0.6},
             double=("drag",)),
    "hurts": -0.7, "hit by": -0.7, "hit hard": -0.8, "hard hit": -0.8, "weigh on": -0.7, "weighs on": -0.7,
    "weighed on": -0.7, "weighing on": -0.7, "weighed down": -0.8, "drag on": -0.6, "threat": -0.6,
    "threats": -0.6, "warning": -0.8, "warnings": -0.8, "profit warning": -1.3, "red flag": -0.9,
    "red flags": -0.9, "alarm": -0.6, "alarming": -0.7, "worried": -0.6, "worrying": -0.6,
    "fearful": -0.6, "panic": -1.0, "panicked": -0.9, "jitters": -0.7, "jittery": -0.7, "nervous": -0.5,
    "anxiety": -0.6, "anxious": -0.5, "turmoil": -0.9, "chaos": -0.9, "crisis": -1.0, "meltdown": -1.3,
    "bloodbath": -1.3, "carnage": -1.2, "rout": -1.2, "selloff": -1.0, "sell off": -1.0, "sell offs": -0.9,
    "capitulation": -0.9, "correction": -0.6, "correction territory": -0.8, "bear territory": -0.9,
    "freefall": -1.3, "free fall": -1.3, "downward spiral": -1.1, "spiral": -0.5, "vicious": -0.5,
    "losing streak": -0.8, "worst day": -1.1, "worst week": -1.0, "worst month": -1.0, "worst year": -1.0,
    "worst quarter": -1.0, "worst performer": -0.9, "worst performing": -0.8, "losers": -0.5,
    "loser": -0.5, "laggard": -0.6, "laggards": -0.5, "lag": -0.4, "lags": -0.4, "lagging": -0.5,
    "lagged": -0.4, "underwater": -0.6, "wipe out": -1.0, "wiped out": -1.0, "wipes out": -1.0,
    "erases gains": -0.8, "erased gains": -0.8, "pares gains": -0.4, "pared gains": -0.4,
    "gives up gains": -0.6, "gave up gains": -0.6, "in the red": -0.7, "into the red": -0.7,
    "negative territory": -0.6, "profit taking": -0.3, "swung to a loss": -1.0, "swings to a loss": -1.0,
    "swung to loss": -1.0, "swings to loss": -1.0, "turned to a loss": -1.0, "slipped into the red": -0.9,
    "fell into the red": -0.9, "swung into the red": -1.0, "posted a loss": -0.6, "reported a loss": -0.6, "lost half": -1.0, "off the cliff": -1.0,
    "hard pressed": -0.6, "riskiest": -0.7, "irrational": -0.3, "negative": -0.6, "negatively": -0.5,
    "legal action": -0.7, "insolvencies": -0.8, "stagnate": -0.6, "stagnates": -0.6, "stagnated": -0.6,
    "stagnating": -0.6, "hit to": -0.7, "agrees to pay": -0.4, "agreed to pay": -0.4, "going nowhere": -0.5,
    "fizzle": -0.7, "fizzles": -0.7, "fizzled": -0.7, "unusual puts": -0.6, "put buying": -0.5,
    "put buyers": -0.4, "bearish options": -0.6, "bearish bets": -0.7, "bearish bet": -0.6,
    **_verbs({"criticize": -0.5, "criticise": -0.5, "suffer": -0.6, "admonish": -0.6, "accuse": -0.6,
              "hinder": -0.6, "hamper": -0.6, "slash": -0.4}),
    "obliterated": -1.2, "hammered": -1.0, "pummeled": -1.0, "pummelled": -1.0, "battered": -0.9,
    "clobbered": -1.0, "routed": -1.0, "decimated": -1.2, "makes no sense": -0.5, "should not buy": -0.6,
    "shouldnt buy": -0.6, "avoid the stock": -0.6, "time to sell": -0.6, "sell signal": -0.7,
    "take a hit": -0.7, "takes a hit": -0.7, "took a hit": -0.7, "taking a hit": -0.7,
    "under the microscope": -0.6, "implicated": -0.7, "money laundering": -1.0, "breaches": -0.7,
    "to the downside": -0.5, "wrong direction": -0.5, "drop the ball": -0.5, "dropped the ball": -0.5,
    "losing steam": -0.6, "loses steam": -0.6, "lost steam": -0.6, "green to red": -0.7,
    "bear of the day": -0.8, "explosion": -0.6, "revoke": -0.7, "revokes": -0.7, "revoked": -0.7,
    "dithers": -0.3, "snarled": -0.5, "halts trial": -1.0, "trial halted": -1.0, "double downgrade": -1.4,
    "casts shadow": -0.6, "cast shadow": -0.6, "casts a shadow": -0.6, "comes in light": -0.6,
    "came in light": -0.6, "on the light side": -0.6, "apocalypse": -1.0, "ousts": -0.6, "oust": -0.5,
    "clash": -0.5, "clashes": -0.5, "idled": -0.6, "down day": -0.6, "misstep": -0.6, "missteps": -0.6,
    "black swan": -0.8, "headache": -0.5, "headaches": -0.5,
    # macro
    "recession": -0.9, "recessions": -0.8, "depression": -1.0, "stagflation": -0.9, "downturn": -0.9,
    "deflation": -0.5, "hyperinflation": -0.8, "hawkish": -0.6, "rate hike": -0.5, "rate hikes": -0.5,
    "tightening": -0.4, "hard landing": -0.8, "yield curve inversion": -0.7, "inverted yield curve": -0.7,
    "trade war": -0.9, "trade tensions": -0.7, "tensions": -0.5, "tariff": -0.4, "war": -0.6,
    "conflict": -0.5, "attack": -0.6, "attacks": -0.6, "pandemic": -0.4, "outbreak": -0.6,
    "lockdown": -0.6, "lockdowns": -0.6, "shortage": -0.6, "shortages": -0.6, "glut": -0.6,
    "oversupply": -0.6, "price war": -0.7, "stall": -0.6, "stalls": -0.6, "stalled": -0.6,
    "stalling": -0.6, "stagnant": -0.5, "stagnation": -0.6, "waver": -0.4, "wavers": -0.4,
    "falters": -0.7, "faltering": -0.7, "faltered": -0.7, "uncertain": -0.4, "volatile": -0.2,
    "risky": -0.5, "danger": -0.7, "dangerous": -0.7, "vulnerable": -0.6, "fragile": -0.6,
    # balance sheet / corporate distress
    "default": -1.1, "defaults": -1.0, "defaulted": -1.2, "insolvent": -1.5, "insolvency": -1.5,
    "bankrupt": -1.7, "bankruptcy": -1.7, "bankruptcies": -1.0, "chapter 11": -1.7, "chapter 7": -1.7,
    "going concern": -1.3, "going concern doubt": -1.5, "liquidation": -1.3, "receivership": -1.4,
    "into administration": -1.4, "wind down": -0.8, "winding down": -0.8, "shut down": -0.8,
    "shuts down": -0.9, "shutdown": -0.8, "shutters": -0.9, "closure": -0.6, "closures": -0.6,
    "store closures": -0.8, "closing stores": -0.7, "delist": -1.2, "delisted": -1.3, "delisting": -1.2,
    "delisting notice": -1.3, "deficiency notice": -1.0, "noncompliance": -0.8, "non compliance": -0.8,
    "dilution": -0.8, "dilutive": -0.8, "dilute": -0.6, "reverse split": -0.7, "reverse stock split": -0.7,
    "writedown": -0.8, "writedowns": -0.8, "write down": -0.8, "write downs": -0.8, "write off": -0.8,
    "write offs": -0.8, "writeoff": -0.8, "restatement": -1.0, "restate": -0.9, "restates": -0.9,
    "accounting irregularities": -1.3, "irregularities": -0.9, "material weakness": -1.0,
    "non reliance": -1.0, "overleveraged": -0.8, "debt laden": -0.7, "distressed": -0.9,
    "distress": -0.9, "bailout": -0.7, "cash burn": -0.7, "burning cash": -0.8,
    "running out of cash": -1.2, "liquidity crunch": -1.0, "credit crunch": -1.0, "cash crunch": -1.0,
    "shortfall": -0.8, "overhang": -0.5,
    # labor
    "layoff": -0.9, "layoffs": -0.9, "lay off": -0.9, "lays off": -0.9, "laid off": -0.9,
    "laying off": -0.9, "job cuts": -0.9, "job losses": -0.9, "furlough": -0.8, "furloughs": -0.8,
    "furloughed": -0.8, "redundancies": -0.8, "restructuring": -0.3, "on strike": -0.7,
    "walkout": -0.6, "work stoppage": -0.7, "labor trouble": -0.8, "strike": -0.3, "strikes": -0.3,
    # operations
    "halt": -0.7, "halts": -0.7, "halted": -0.8, "trading halt": -0.8, "suspend": -0.6, "suspends": -0.7,
    "suspended": -0.7, "suspension": -0.7, "delay": -0.6, "delays": -0.6, "delayed": -0.6,
    "postpone": -0.5, "postponed": -0.5, "postpones": -0.5, "cancel": -0.6, "cancels": -0.6,
    "canceled": -0.6, "cancelled": -0.6, "cancellation": -0.6, "cancellations": -0.6,
    **_verbs({"abandon": -0.6, "terminate": -0.6, "discontinue": -0.6, "scrap": -0.5, "reject": -0.5},
             double=("scrap",)),
    "termination": -0.5, "rejection": -0.7, "fail": -0.8, "fails": -0.8, "failed": -0.8, "failing": -0.8,
    "failure": -0.9, "failures": -0.8, "flop": -0.8, "flops": -0.8, "setback": -0.8, "setbacks": -0.8,
    "blow": -0.5, "dealt a blow": -0.8, "beaten down": -0.6, "beat down": -0.5, "beaten up": -0.5, "complete response letter": -1.1, "clinical hold": -1.1,
    "recall": -0.9, "recalls": -0.9, "recalled": -0.9, "defect": -0.7, "defects": -0.7,
    "defective": -0.8, "faulty": -0.7, "contamination": -0.8, "contaminated": -0.8, "explosion": -0.8,
    "accident": -0.6, "fatal": -0.7, "spill": -0.8, "disaster": -1.0, "tragedy": -0.8,
    "breach": -0.8, "data breach": -1.0, "hack": -0.8, "hacked": -0.9, "cyberattack": -0.9,
    "ransomware": -0.9, "outage": -0.8, "outages": -0.8, "glitch": -0.6, "disruption": -0.6,
    "disruptions": -0.6, "under fire": -0.8, "fire sale": -0.7, "backlash": -0.8, "boycott": -0.8,
    "controversy": -0.6, "controversial": -0.4, "criticism": -0.5, "criticized": -0.5,
    "slammed": -0.8, "slams": -0.6, "scrutiny": -0.5, "crackdown": -0.8, "crack down": -0.8,
    "exodus": -0.8, "flee": -0.7, "fleeing": -0.7, "jeopardy": -0.8, "vanish": -0.8, "vanishes": -0.8,
    "vanished": -0.8, "evaporate": -0.8, "evaporated": -0.8, "steps down": -0.4, "step down": -0.4,
    "resigns": -0.5, "resigned": -0.5, "resignation": -0.5, "ousted": -0.8, "ouster": -0.7, "fired": -0.3,
    "exits": -0.2, "departure": -0.3,
    # flows & positioning
    "dump": -0.8, "dumps": -0.8, "dumping": -0.8, "dumped": -0.8, "unload": -0.5, "unloads": -0.5,
    "unloading": -0.5, "sells stake": -0.3, "exits stake": -0.4, "insider selling": -0.6,
    "insider sale": -0.4, "insider sells": -0.5, "short seller": -0.8, "short sellers": -0.6,
    "short report": -1.0, "short attack": -1.0, "short position": -0.4, "shorting": -0.5,
    "bet against": -0.7, "bets against": -0.7, "betting against": -0.7, "net outflows": -0.5,
    "bull trap": -0.7, "falling knife": -0.8, "dead cat bounce": -0.8,
    "breaks support": -0.7, "broke support": -0.7, "breaks below": -0.7, "broke below": -0.7,
    "gapping down": -0.7, "gaps down": -0.7, "gapped down": -0.7, "gap down": -0.7,
}

# Litigation / enforcement (Loughran-McDonald "litigious"): valence + category.
LITIGIOUS: dict[str, float] = {
    **_plural({"lawsuit": -0.8, "class action": -0.9, "probe": -0.8, "investigation": -0.7,
               "inquiry": -0.6, "subpoena": -0.9, "indictment": -1.1, "allegation": -0.5,
               "complaint": -0.3, "penalty": -0.7, "fine": 0.0, "sanction": -0.7, "violation": -0.8,
               "settlement": -0.1, "antitrust": -0.5, "injunction": -0.4, "litigation": -0.5,
               "ruling": 0.0, "verdict": 0.0, "appeal": 0.0, "court": 0.0, "regulator": 0.0}),
    **_verbs({"sue": -0.8, "probe": -0.8, "investigate": -0.6, "subpoena": -0.9, "allege": -0.5,
              "indict": -1.2, "sanction": -0.8, "ban": -0.7, "blacklist": -0.9,
              "convict": -1.0, "violate": -0.8, "settle": 0.0}, double=("ban",)),
    "in favor of": 0.3, "rules against": -0.5, "ruled against": -0.5,
    "fines": -0.7, "fined": -0.8, "suit": -0.5, "suits": -0.4, "files suit": -0.7, "filed suit": -0.7,
    "charged": -0.6, "criminal charges": -1.0, "fraud": -1.3, "fraudulent": -1.3, "scandal": -1.1,
    "scam": -0.9, "ponzi": -1.3, "misconduct": -0.9, "wrongdoing": -0.8, "bribery": -1.0,
    "corruption": -1.0, "embezzlement": -1.2, "manipulation": -0.8, "insider trading": -0.8,
    "guilty": -0.9, "sentenced": -0.7, "jail": -0.7, "prison": -0.6, "entity list": -0.7,
    "allegedly": -0.3, "under investigation": -0.9, "sec probe": -1.0, "doj probe": -1.0,
    "wells notice": -1.1, "cease and desist": -0.8, "price fixing": -0.9, "monopoly": -0.2,
    "plaintiff": 0.0, "plaintiffs": 0.0, "defendant": 0.0, "jury": 0.0, "judge": 0.0,
    "attorney": 0.0, "prosecutor": 0.0, "prosecutors": 0.0, "arbitration": 0.0, "regulatory": 0.0,
    "hindenburg": -0.8, "muddy waters": -0.8, "citron": -0.4, "spruce point": -0.8,
}

# --------------------------------------------------------------------------- #
# Social register: slang, trader shorthand, emoji
# --------------------------------------------------------------------------- #
SOCIAL: dict[str, float] = {
    "to the moon": 1.2, "mooning": 1.1, "moon": 0.7, "mooned": 0.9, "moonshot": 0.6, "lambo": 0.6,
    "tendies": 0.8, "stonks": 0.3, "hodl": 0.6, "hodling": 0.6, "lfg": 0.7, "wagmi": 0.6, "ngmi": -0.6,
    "fomo": 0.3, "fud": -0.2, "rekt": -1.1, "wrecked": -0.8, "bagholder": -0.9, "bagholders": -0.8,
    "bag holder": -0.9, "bag holders": -0.8, "bag holding": -0.8, "holding the bag": -0.8,
    "rug pull": -1.4, "rugpull": -1.4, "rugged": -1.0, "pump and dump": -1.2, "pump n dump": -1.2,
    "gamma squeeze": 0.8, "shorts trapped": 0.7, "ripping": 0.9, "rips": 0.6,
    "free money": 0.6, "loaded up": 0.6, "load up": 0.5, "back up the truck": 0.9,
    "btfd": 0.7, "buy the dip": 0.8, "bought the dip": 0.8, "buying the dip": 0.8,
    "yolo": 0.4, "ath": 0.8, "new ath": 1.0, "green day": 0.6, "red day": -0.6, "sea of red": -0.8,
    "bleeding": -0.7, "bleed": -0.6, "bleeds": -0.6, "dumpster fire": -0.9, "garbage": -0.8,
    "trash": -0.7, "worthless": -1.0, "overhyped": -0.7, "short it": -0.7,
    "going long": 0.6, "going short": -0.6, "gap up": 0.7, "gapped up": 0.7, "gapping up": 0.7,
    "broke down": -0.5, "bottom is in": 0.8, "top is in": -0.8, "topped out": -0.6,
    "blow off top": -0.7, "parabolic": 0.5, "diamond hands": 0.8, "paper hands": -0.4, "guh": -0.8,
    "killing it": 0.8, "crushing it": 0.8, "no brainer": 0.7,
    # emoji (variation selectors are stripped by the tokenizer)
    "🚀": 1.0, "🌕": 0.7, "🌙": 0.5, "📈": 0.8, "📉": -0.8, "🐂": 0.6, "🐻": -0.6, "💎": 0.4, "🙌": 0.3,
    "💎 🙌": 0.8, "🔥": 0.4, "💰": 0.4, "🤑": 0.6, "😭": -0.5, "😢": -0.5, "😱": -0.5, "🩸": -0.7,
    "💀": -0.4, "🤡": -0.4, "🟢": 0.5, "🔴": -0.5, "✅": 0.2, "❌": -0.3, "⚠": -0.3, "👎": -0.6,
    "👍": 0.5, "💩": -0.7, "🗑": -0.6, "🆘": -0.6, "😍": 0.5, "🤩": 0.6, "🥳": 0.6, "🎉": 0.5,
    "💪": 0.5, "😬": -0.3, "🤦": -0.4, "😡": -0.6, "🤬": -0.7, "💯": 0.4, "🦍": 0.3, "🍗": 0.4,
    "🧻": -0.3, "🤮": -0.7,
}

# Ambiguous in news ("calls for", "puts pressure on", "short-term") - only scored
# for the social register, where they are trader positions.
SOCIAL_ONLY: dict[str, float] = {
    "calls": 0.6, "puts": -0.6, "long": 0.5, "short": -0.5, "shorts": -0.3, "buying": 0.4, "bought": 0.4,
    "buy": 0.4, "adding": 0.4, "added": 0.3, "add": 0.3, "add more": 0.6, "selling": -0.4, "sold": -0.3,
    "sell": -0.4, "trimmed": -0.2, "green": 0.4, "red": -0.4, "bull": 0.5, "bear": -0.5, "bears": -0.3,
    "bulls": 0.3, "pump": 0.2, "loaded": 0.4, "holding": 0.2, "hold": 0.1, "printing": 0.5,
    "all in": 0.5, "pumped": 0.4, "breakdown": -0.6, "send it": 0.7, "sending it": 0.7, "lets go": 0.5,
    "let's go": 0.5, "ape": 0.2, "apes": 0.2, "squeeze": 0.5, "squeezed": 0.5, "getting squeezed": 0.6,
    "squeezing": 0.5,
}

# --------------------------------------------------------------------------- #
# Composition: direction words x metrics
# --------------------------------------------------------------------------- #
class Direction(NamedTuple):
    """A movement word. ``sign`` is +1 (up) / -1 (down); ``strength`` its vividness.

    ``pos``: "v" verb, "n" noun ("increase in"), "a" adjective/adverb ("higher"),
    "p" particle ("up"/"down": needs numeric/market context), "l" level ("high"/"low").
    ``fixed`` words are evaluative ("improve", "worsen") and keep their own sign
    regardless of the metric. ``default`` scales the valence when no metric is
    attached (bare "increase" is weaker evidence than bare "plunge").
    """

    sign: int
    strength: float
    pos: str
    fixed: bool = False
    default: float = 1.0


def _dir(sign: int, strength: float, pos: str, *words: str, fixed: bool = False,
         default: float = 1.0) -> dict[str, Direction]:
    return {w: Direction(sign, strength, pos, fixed, default) for w in words}


def _dir_verbs(sign: int, strength: float, *bases: str, fixed: bool = False, default: float = 1.0,
               double: tuple[str, ...] = (), extra: tuple[str, ...] = ()) -> dict[str, Direction]:
    out: dict[str, Direction] = {}
    for base in bases:
        for form in _forms(base, double=base in double):
            out[form] = Direction(sign, strength, "v", fixed, default)
    for form in extra:
        out[form] = Direction(sign, strength, "v", fixed, default)
    return out


def _phrasal(sign: int, verbs: tuple[str, ...], particles: tuple[str, ...]) -> dict[str, Direction]:
    """"edges higher", "ticked up", "pulling back" ... (every verb form x particle)."""
    out: dict[str, Direction] = {}
    for base in verbs:
        for form in _forms(base, double=base in ("slip", "drop", "step", "trim")):
            for part in particles:
                out[f"{form} {part}"] = Direction(sign, 0.5, "v", False, 1.6)
    return out


DIRECTIONS: dict[str, Direction] = {
    # ---- up: neutral-ish movement verbs (sign comes from the metric) ----
    **_dir_verbs(1, 0.7, "rise", extra=("rose", "risen")),
    **_dir_verbs(1, 0.6, "increase", "grow", "expand", "widen", "lengthen",
                 extra=("grew", "grown"), default=0.6),
    **_dir_verbs(1, 0.6, "add", double=("add",), default=0.3),
    **_dir_verbs(1, 0.6, "extend", default=0.0),  # only with a trend noun: "extends gains"
    **_dir_verbs(1, 0.6, "raise", "hike", "boost", "lift", "up", "bump", "accelerate", "swell",
                 double=("up", "bump"), default=0.5),
    **_dir_verbs(1, 0.6, "create", "hire", "slap", "impose", "build", "spur", "fuel", default=0.0,
                 extra=("built",)),
    **_dir_verbs(1, 0.8, "climb", "gain", "rally"),
    **_dir_verbs(1, 0.8, "advance", default=0.6, extra=()),
    **_dir_verbs(1, 1.0, "jump", "pop", "leap", "spike", "balloon", "zoom", "double", double=("pop",),
                 extra=("leapt",)),
    **_dir_verbs(1, 1.2, "surge", "soar", "triple", "quadruple"),
    **_dir_verbs(1, 1.4, "skyrocket", "rocket", "explode"),
    **_dir_verbs(1, 0.4, "edge", "inch", "tick", "creep", default=0.8),
    # ---- up: evaluative (always good) ----
    **_dir_verbs(1, 0.8, "improve", "recover", "rebound", "strengthen", "bounce", "outpace", fixed=True),
    "improvement": Direction(1, 0.7, "n", True), "improvements": Direction(1, 0.6, "n", True),
    "recovery": Direction(1, 0.7, "n", True), "rebound": Direction(1, 0.8, "n", True),
    "upswing": Direction(1, 0.8, "n", True), "uptick": Direction(1, 0.5, "n"),
    "upturn": Direction(1, 0.8, "n", True),
    # ---- up: nouns / adjectives ----
    **_dir(1, 0.6, "n", "increase", "increases", "rise", "growth", "expansion", "acceleration", "hike",
           "hikes", "boost", default=0.6),
    **_dir(1, 0.8, "n", "gain", "gains", "jump", "rally", "climb", "advance", "surge", "spike", "pop"),
    "explosion": Direction(1, 1.0, "n", False, 0.0),  # "market explosion" (else: accident, see NEGATIVE)
    **_dir(1, 0.6, "a", "higher", default=0.6),
    **_dir(1, 0.6, "a", "bigger", "larger", "wider", "greater", "faster", "more", default=0.0),
    **_dir(1, 0.9, "a", "higher than expected", "bigger than expected", "larger than expected",
           "wider than expected", "faster than expected", "greater than expected", "more than expected",
           "higher than anticipated", "higher than forecast", default=0.7),
    **_dir(1, 0.9, "a", "record", "records", "doubled", "tripled"),
    **_dir(1, 0.9, "a", "soaring", "surging", "rising", "growing", "increasing", "climbing", "jumping",
           default=0.5),
    "up": Direction(1, 0.7, "p"), "upward": Direction(1, 0.6, "a", False, 0.4),
    "upwards": Direction(1, 0.6, "a", False, 0.4), "up sharply": Direction(1, 1.1, "p"),
    "high": Direction(1, 0.5, "l"), "highs": Direction(1, 0.8, "l"), "highest": Direction(1, 0.8, "l"),
    "new high": Direction(1, 1.0, "l"), "new highs": Direction(1, 1.0, "l"),
    **_phrasal(1, ("tick", "edge", "inch", "move", "go", "head", "point", "trade", "drift", "open", "close",
                   "end", "settle", "finish", "push", "break", "climb", "trend", "creep"), ("up", "higher")),
    **_phrasal(1, ("pick", "ramp", "speed", "heat", "perk", "firm"), ("up",)),
    "went up": Direction(1, 0.5, "v", False, 1.6), "went higher": Direction(1, 0.5, "v", False, 1.6),
    # ---- down: neutral-ish movement verbs ----
    **_dir_verbs(-1, 0.8, "fall", "drop", "decline", "slide", "retreat", "sag", "shed", "lose",
                 double=("drop", "sag"), extra=("fell", "fallen", "slid", "lost")),
    **_dir_verbs(-1, 0.6, "decrease", "reduce", "lower", "cut", "trim", "shrink", "narrow",
                 "ease", "cool", "soften", "slow", "decelerate", "moderate", "halve", "dip", "slip",
                 "subside", "fade", "omit", "diminish", "dwindle", double=("cut", "trim", "dip", "slip", "omit"),
                 extra=("shrank", "shrunk", "cuts", "contracted", "contracting"), default=0.6),
    **_dir_verbs(-1, 0.6, "shorten", "erase", "dash", "remove", "eliminate", default=0.3),
    # only meaningful with an object/subject metric: "allays fears", "knocks shares", "rally halted"
    **_dir_verbs(-1, 0.8, "allay", "assuage", "calm", "soothe", "offset", "outweigh", "knock", "halt",
                 "squeeze", "contain", "stem", "dent", "pare", "stop", "idle", double=("knock", "stop"),
                 default=0.0),
    **_phrasal(-1, ("back", "pull"), ("off",)), **_phrasal(-1, ("give",), ("back",)),
    **{f"{v} {cmp} than {exp}": Direction(sign, 0.9, "v", False, 0.7)
       for v in ("rose", "rise", "rises", "grew", "grow", "grows", "increased", "increases", "gained", "gains",
                 "climbed", "climbs", "jumped", "jumps", "improved")
       for cmp, sign in (("less", -1), ("more", 1), ("slower", -1), ("faster", 1))
       for exp in ("expected", "forecast", "anticipated", "estimated")},
    **{f"{v} {cmp} than {exp}": Direction(sign, 0.9, "v", False, 0.7)
       for v in ("fell", "falls", "fall", "declined", "declines", "dropped", "drops", "slipped", "slid",
                 "decreased", "decreases", "shrank")
       for cmp, sign in (("less", 1), ("more", -1))
       for exp in ("expected", "forecast", "anticipated", "estimated")},
    "contracts": Direction(-1, 0.6, "v", False, 0.0), "contract": Direction(-1, 0.6, "v", False, 0.0),
    **_dir_verbs(-1, 1.0, "sink", "tumble", "slump", "skid", "dive", "slash", "wipe",
                 double=("skid",), extra=("sank", "sunk", "dove")),
    **_dir_verbs(-1, 1.4, "plunge", "plummet", "tank", "crater", "nosedive", "crash", "collapse", "implode",
                 "evaporate", "crumble"),
    # ---- down: evaluative (always bad) ----
    **_dir_verbs(-1, 0.8, "worsen", "deteriorate", "weaken", "erode", "falter", "stumble", "struggle",
                 fixed=True),
    **_dir_verbs(-1, 0.7, "hinder", "hamper", "crimp", "impair", "jeopardize", "threaten", "undermine",
                 fixed=True, default=0.0),
    "deterioration": Direction(-1, 0.9, "n", True), "erosion": Direction(-1, 0.7, "n", True),
    "downswing": Direction(-1, 0.8, "n", True), "downtick": Direction(-1, 0.5, "n"),
    # ---- down: nouns / adjectives ----
    **_dir(-1, 0.6, "n", "decrease", "decreases", "reduction", "reductions", "cut", "cuts", "contraction",
           "slowdown", "deceleration", "fall", default=0.6),
    **_dir(-1, 0.8, "n", "decline", "declines", "drop", "drops", "slide", "dip", "slump", "plunge",
           "tumble", "pullback", "retreat", "selloff"),
    **_dir(-1, 0.6, "a", "lower", default=0.6),
    **_dir(-1, 0.6, "a", "smaller", "narrower", "slower", "fewer", "less", default=0.0),
    **_dir(-1, 0.9, "a", "lower than expected", "smaller than expected", "narrower than expected",
           "slower than expected", "less than expected", "fewer than expected", "lower than anticipated",
           "lower than forecast", default=0.7),
    **_dir(-1, 0.9, "a", "halved"),
    **_dir(-1, 0.9, "a", "falling", "declining", "slumping", "plunging", "sinking", "shrinking",
           "dwindling", "tumbling", "slowing", "diminishing", default=0.5),
    "down": Direction(-1, 0.7, "p"), "downward": Direction(-1, 0.6, "a", False, 0.4),
    "downwards": Direction(-1, 0.6, "a", False, 0.4), "down sharply": Direction(-1, 1.1, "p"),
    "low": Direction(-1, 0.5, "l"), "lows": Direction(-1, 0.8, "l"), "lowest": Direction(-1, 0.8, "l"),
    "new low": Direction(-1, 1.0, "l"), "new lows": Direction(-1, 1.0, "l"),
    **_phrasal(-1, ("tick", "edge", "inch", "move", "go", "head", "point", "trade", "drift", "open", "close",
                    "end", "settle", "finish", "push", "trend", "skid", "slip", "sink", "dip"), ("down", "lower")),
    **_phrasal(-1, ("slow", "pull"), ("down", "back")),
    "went down": Direction(-1, 0.5, "v", False, 1.6), "went lower": Direction(-1, 0.5, "v", False, 1.6),
}

# Verbs that can take the metric as their *object* ("boosts its dividend",
# "cuts costs", "lost market share", "allays fears").
TRANSITIVE: frozenset[str] = frozenset(
    f for base in ("increase", "grow", "expand", "add", "widen", "raise", "hike", "boost", "lift", "up",
                   "bump", "accelerate", "extend", "double", "triple", "decrease", "reduce", "lower", "cut",
                   "trim", "shrink", "narrow", "halve", "slash", "shed", "lose", "erase", "dash", "remove",
                   "eliminate", "improve", "strengthen", "weaken", "erode", "wipe", "shorten", "omit", "create",
                   "hire", "slap", "impose", "build", "spur", "fuel", "allay", "assuage", "calm", "soothe",
                   "offset", "outweigh", "knock", "halt", "squeeze", "contain", "stem", "dent", "hinder", "pare",
                   "stop", "idle",
                   "hamper", "crimp", "impair", "jeopardize", "threaten", "undermine", "diminish")
    for f in _forms(base, double=base in ("up", "bump", "cut", "trim", "add", "omit", "knock", "stop"))
) | {"lost", "cuts", "shrank", "built"}

# Verbs that only make sense with a *trend* object ("extends gains", "extends losses").
TREND_ONLY: frozenset[str] = frozenset(_forms("extend"))
TREND_METRICS: frozenset[str] = frozenset({
    "gains", "gain", "rally", "losses", "loss", "decline", "declines", "slide", "slump", "selloff", "rebound",
    "recovery", "advance", "winning streak", "losing streak", "lead", "momentum", "growth", "shutdown",
    "shutdowns", "closures", "lockdown", "lockdowns", "strike", "delays", "declines", "losses", "slump"})

# Physical footprint verbs: only "opens 500 stores" / "shuts 34 stores", never "closes deal".
FOOTPRINT_VERBS: dict[str, int] = {
    **{f: -1 for f in _forms("close") + _forms("shut", double=True)},
    **{f: 1 for f in _forms("open")},
}
FOOTPRINT_METRICS: frozenset[str] = frozenset({
    "stores", "store", "restaurants", "plants", "plant", "factories", "factory", "locations", "branches",
    "outlets", "mines", "mine", "facilities", "facility", "sites", "offices", "mills", "theaters", "hotels",
    "clinics", "shops", "units", "warehouses", "fulfillment center", "distribution center"})

# Barriers: "lifts tariffs" / "lifted the ban" means *removing* them.
LIFTABLE: frozenset[str] = frozenset({
    "tariffs", "tariff", "sanctions", "ban", "restrictions", "lockdown", "lockdowns", "embargo", "curbs",
    "suspension", "freeze", "moratorium", "halt", "quarantine", "export ban", "price cap"})


class Metric(NamedTuple):
    """What moved. ``polarity`` +1 = good when up (sales), -1 = bad when up (costs).
    ``intrinsic`` is the valence when nothing moves it ("posts net loss")."""

    polarity: float
    intrinsic: float = 0.0


def _metrics(polarity: float, *names: str, intrinsic: float = 0.0) -> dict[str, Metric]:
    return {n: Metric(polarity, intrinsic) for n in names}


METRICS: dict[str, Metric] = {
    # ---- good when up ----
    **_metrics(1.0, "sales", "net sales", "revenue", "revenues", "turnover", "income", "net income",
               "operating income", "profit", "profits", "net profit", "operating profit", "pretax profit",
               "pre tax profit", "gross profit", "earnings", "eps", "earnings per share", "ebit", "ebitda",
               "margin", "margins", "operating margin", "gross margin", "profitability", "orders",
               "order intake", "order book", "backlog", "bookings", "deliveries", "shipments", "production",
               "output", "volume", "volumes", "demand", "market share", "share price", "stock price",
               "share prices", "stock prices", "shares", "stock", "stocks", "equities", "price target",
               "target price", "target", "targets", "guidance", "outlook", "forecast", "forecasts",
               "estimates", "estimate", "dividend", "dividends", "payout", "buyback", "cash flow",
               "free cash flow", "operating cash flow", "same store sales", "comparable sales",
               "comparable store sales", "comp sales", "comps", "users", "subscribers", "customers",
               "traffic", "passengers", "occupancy", "load factor", "visitors", "valuation", "market cap",
               "market value", "value", "stake", "rating", "ratings", "credit rating", "assets",
               "assets under management", "aum", "deposits", "loans", "jobs", "employment", "payrolls",
               "nonfarm payrolls", "hiring", "headcount", "workforce", "staff", "workers", "employees",
               "wages", "gdp", "growth", "economy", "consumer confidence", "confidence", "sentiment",
               "consumer sentiment", "pmi", "retail sales", "industrial production", "manufacturing",
               "exports", "housing starts", "home sales", "investment", "investments", "capacity",
               "efficiency", "productivity", "returns", "return", "performance", "result", "results",
               "operating result", "net result", "cash", "reserves", "index", "indexes", "indices", "dow",
               "nasdaq", "s&p", "s&p 500", "futures", "market", "markets", "wall street", "ftse", "dax",
               "nikkei", "stoxx", "hang seng", "sensex", "russell", "tsx", "asx", "kospi", "cac", "bitcoin",
               "ether", "crypto", "gold", "oil", "crude", "copper", "commodities", "inflows", "net inflows",
               "premium", "bonus", "rally", "gains", "momentum", "recovery", "appetite", "interest",
               "approval", "approvals", "contract", "contracts", "deal", "deals", "dividend payout",
               "distribution", "distributions", "net interest income", "book value", "net asset value",
               "nav", "net worth", "wealth", "lead", "winning streak", "oil prices", "crude prices",
               "gold prices", "copper prices", "home prices", "house prices", "housing prices",
               "technology stocks", "tech stocks", "energy stocks", "bank stocks", "financials", "utilities",
               "industrials", "retailers", "miners", "airlines", "chipmakers", "semis", "small caps", "big tech",
               "treasuries", "dollar", "yen", "yuan", "rupee", "ruble", "lira", "peso", "loonie", "sterling",
               "greenback", "forint", "zloty", "etf", "etfs", "reits"),
    **_metrics(0.6, "price", "prices"),
    **_metrics(1.0, *sorted(FOOTPRINT_METRICS)),
    **_metrics(1.0, "optimism", "hopes", intrinsic=0.5),
    **_metrics(1.0, "record profit", "record revenue", "record sales", "record earnings", intrinsic=0.9),
    # ---- bad when up ----
    **_metrics(-1.0, "loss", "losses", intrinsic=-0.5),
    **_metrics(-1.0, "net loss", "operating loss", "pretax loss", "pre tax loss", "net losses",
               "operating losses", "loan losses", "credit losses", intrinsic=-0.7),
    **_metrics(-0.3, "inventory", "inventories"),
    **_metrics(-1.0, "costs", "cost", "expenses", "expense", "opex", "debt", "net debt", "debts",
               "liabilities", "deficit", "deficits", "unemployment", "unemployment rate", "jobless claims",
               "jobless rate", "initial claims", "layoffs", "job cuts", "inflation", "interest rates",
               "rates", "yields", "bond yields", "mortgage rates", "short interest", "impairment", "impairments", "writedowns", "charges", "provisions",
               "bad loans", "non performing loans", "npls", "delinquencies", "defaults", "default rate",
               "churn", "cash burn", "burn rate", "volatility", "vix", "tariffs", "taxes", "headwinds",
               "dilution", "leverage", "spreads", "credit spreads", "outflows", "net outflows",
               "short positions", "supply", "stockpiles", "competition", "prices paid", "unit labor costs",
               "borrowing costs", "risk premium", "claims", "complaints", "stress index", "fear index",
               "fear gauge", "misery index", "cases", "infections", "deaths", "hospitalizations",
               "consumer prices", "producer prices", "import prices", "food prices", "gas prices",
               "gasoline prices", "fuel prices", "energy prices", "insolvencies", "bankruptcies",
               "foreclosures", "embargo", "curbs", "lockdown", "lockdowns", "quarantine", "freeze",
               "moratorium", "price cap", "export ban", "shutdown", "shutdowns", "closures"),
    **_metrics(-1.0, "risk", "risks", "uncertainty", "stress", intrinsic=-0.3),
    **_metrics(-1.0, "pressure", "pressures", intrinsic=-0.4),
    **_metrics(-1.0, "concerns", "concern", "fears", "fear", "worries", "worry", "pessimism", "anxiety",
               intrinsic=-0.6),
    **_metrics(-1.0, "recession", "crisis", "tensions", "pandemic", "outbreak", "virus", "shortage",
               "shortages", "glut", "turmoil", "recession risk", "recession odds", "recession probability",
               "default risk", "recession fears", "inflation fears"),
    **_metrics(-1.0, "losing streak", intrinsic=-0.8),
    **_metrics(-1.0, "probe", "probes", "investigation", "investigations", "lawsuit", "lawsuits",
               "penalty", "penalties", "fines", "sanctions", "tariff", "ban", "restrictions",
               intrinsic=-0.8),
}

# Words that are metrics when moved by another direction word but are also
# directions themselves ("growth slowed", "gains accelerated", "rally halted").
METRIC_DIRECTIONS: frozenset[str] = frozenset({"growth", "gains", "rally", "recovery", "momentum"})

# --------------------------------------------------------------------------- #
# Analyst ratings (rank: +2 strong buy ... -2 strong sell)
# --------------------------------------------------------------------------- #
RATINGS: dict[str, int] = {
    "strong buy": 2, "conviction buy": 2, "top pick": 2, "buy": 1, "outperform": 1, "market outperform": 1,
    "sector outperform": 1, "overweight": 1, "accumulate": 1, "add": 1, "positive": 1, "speculative buy": 1,
    "moderate buy": 1, "long term buy": 1, "outperformer": 1, "hold": 0, "neutral": 0, "equal weight": 0,
    "market perform": 0, "sector perform": 0, "peer perform": 0, "in line": 0, "sector weight": 0,
    "market weight": 0, "mixed": 0, "perform": 0, "equal weighted": 0, "underperform": -1,
    "market underperform": -1, "sector underperform": -1, "underweight": -1, "reduce": -1, "sell": -1,
    "negative": -1, "cautious": -1, "underperformer": -1, "junk": -1, "strong sell": -2,
}

# --------------------------------------------------------------------------- #
# Modifiers
# --------------------------------------------------------------------------- #
NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "without", "nor", "neither", "none", "nothing", "hardly", "cannot", "lack",
    "lacks", "lacking", "absence", "unlikely", "avoid", "avoids", "avoided", "avoiding", "avert", "averts",
    "averted", "prevent", "prevents", "prevented", "denies", "denied", "deny", "dismiss", "dismisses",
    "dismissed", "rules out", "ruled out", "rule out", "fails to", "failed to", "fail to", "unable to",
    "no longer", "stave off", "staves off", "staved off", "ward off", "wards off", "free of",
    "no sign of", "no signs of", "far from", "isnt", "wasnt", "dont", "doesnt", "didnt", "wont", "cant",
    "couldnt", "wouldnt", "shouldnt", "arent", "werent", "hasnt", "havent", "hadnt", "aint",
})
# "not only", "no doubt", "never been this bullish" ... are not negations.
NEGATION_EXCEPTIONS: frozenset[str] = frozenset({
    "not only", "not just", "no doubt", "no wonder", "nothing but", "not least", "no matter",
    "never been", "never before", "no less", "not to mention", "never seen",
})
# A negator whose own meaning is negative when nothing follows ("fails to meet").
NEGATOR_FALLBACK: dict[str, float] = {"fails to": -0.7, "failed to": -0.7, "fail to": -0.6, "unable to": -0.6}

# Uncertainty / hedging (Loughran-McDonald "uncertainty"): factor applied to nearby evidence.
HEDGES: dict[str, float] = {
    "may": 0.7, "might": 0.65, "could": 0.7, "would": 0.85, "possibly": 0.65, "possible": 0.75,
    "perhaps": 0.65, "maybe": 0.65, "potential": 0.8, "potentially": 0.7, "reportedly": 0.8,
    "rumor": 0.7, "rumors": 0.7, "rumour": 0.7, "rumours": 0.7, "rumored": 0.7, "rumoured": 0.7,
    "speculation": 0.7, "speculative": 0.8, "unconfirmed": 0.7, "considering": 0.75, "considers": 0.75,
    "weighs": 0.75, "weighing": 0.75, "mulls": 0.75, "mulling": 0.75, "explores": 0.8, "exploring": 0.8,
    "eyes": 0.8, "eyeing": 0.8, "in talks": 0.8, "said to": 0.8, "sources say": 0.8, "sources said": 0.8,
    "people familiar": 0.8, "expected to": 0.85, "likely": 0.85, "probably": 0.75, "plans to": 0.85,
    "aims to": 0.85, "seeks to": 0.85, "hopes to": 0.85, "if": 0.8, "whether": 0.75, "predicts": 0.85,
    "projected": 0.9, "projects": 0.9, "sees": 0.9, "expects": 0.9, "estimated": 0.95, "proposed": 0.85,
}
# Global (whole-text) hedges: second-hand reporting.
TEXT_HEDGES: frozenset[str] = frozenset({"reportedly", "rumor", "rumors", "rumour", "rumours", "rumored",
                                          "speculation", "unconfirmed", "sources say", "people familiar"})

INTENSIFIERS: dict[str, float] = {
    "sharply": 1.35, "sharp": 1.3, "steep": 1.3, "steeply": 1.3, "significantly": 1.25, "significant": 1.2,
    "substantially": 1.25, "substantial": 1.2, "dramatically": 1.4, "dramatic": 1.35, "massive": 1.35,
    "massively": 1.35, "huge": 1.3, "hugely": 1.3, "big": 1.2, "biggest": 1.35, "major": 1.15,
    "strongly": 1.25, "heavily": 1.25, "heavy": 1.2, "deep": 1.25, "deeper": 1.25, "deepest": 1.3,
    "severe": 1.3, "severely": 1.3, "sizable": 1.15, "sizeable": 1.15, "considerably": 1.2,
    "considerable": 1.15, "markedly": 1.2, "extremely": 1.3, "very": 1.15, "highly": 1.15,
    "historic": 1.2, "unprecedented": 1.25, "largest": 1.25, "further": 1.1, "whopping": 1.3,
    "staggering": 1.3, "explosive": 1.3, "double digit": 1.25, "triple digit": 1.4, "sustained": 1.1,
    "solidly": 1.15, "nearly doubled": 1.3, "most": 1.1, "fastest": 1.2, "totally": 1.15,
    # diminishers
    "slightly": 0.6, "slight": 0.6, "modestly": 0.7, "modest": 0.7, "marginally": 0.55, "marginal": 0.6,
    "somewhat": 0.7, "mildly": 0.7, "mild": 0.7, "small": 0.75, "partially": 0.7, "partly": 0.7,
    "a bit": 0.7, "a little": 0.65, "barely": 0.5, "minor": 0.6, "limited": 0.75, "fractionally": 0.5,
}

# Level qualifiers that turn "high"/"low" into a market extreme ("52-week low").
LEVEL_QUALIFIERS: frozenset[str] = frozenset({
    "record", "records", "all time", "alltime", "new", "fresh", "historic", "lifetime", "multi year",
    "multiyear", "decade", "session", "intraday", "year to date", "ytd", "week", "month", "year",
    "weeks", "months", "years", "since", "52 week", "highest", "lowest", "time",
})

CONTRAST_SHIFT: frozenset[str] = frozenset({"but", "however", "yet", "nevertheless", "nonetheless",
                                            "still", "though"})
CONTRAST_CONCESSIVE: frozenset[str] = frozenset({"despite", "in spite of", "notwithstanding", "even as",
                                                 "even though", "although", "regardless of", "albeit"})

# Phrases that look like sentiment but are not (claimed first, valence 0).
NEUTRALIZERS: frozenset[str] = frozenset({
    "shares outstanding", "outstanding shares", "outstanding debt", "outstanding notes", "outstanding loans",
    "outstanding balance", "outstanding common", "outstanding stock", "outstanding options",
    "best buy", "rite aid", "under armour", "advance auto", "advance auto parts", "dollar general",
    "plus therapeutics", "fortune brands", "bright health", "rocket lab", "rocket companies",
    "rocket pharmaceuticals", "rocket mortgage", "gain therapeutics", "gain capital", "smart global",
    "super micro", "super micro computer", "united rentals", "first solar", "great southern",
    "great wall", "royal gold", "golden ocean", "golden entertainment", "triumph group", "lucky strike",
    "in line", "inline", "in line with", "line with", "little changed", "mixed", "unchanged", "flat",
    "record date", "on record", "track record", "for the record", "high yield", "high school", "high court",
    "high end", "low end", "high speed", "low carbon", "low income", "high frequency", "high tech",
    "high profile", "low key", "lower house", "high street", "low cost", "high quality", "higher education",
    "highly anticipated", "high net worth", "high growth", "short term", "long term", "near term",
    "medium term", "short sale", "short sales", "short form", "long form", "strike price", "fair value",
    "price range", "price data", "market cap", "market value", "price action", "upside down",
    "top line", "bottom line", "top 10", "top 5", "top ten", "top five", "loss of life", "weight loss",
    "hearing loss", "loss prevention", "risk management", "risk factors", "free cash", "net zero",
    "zero emission", "climate change", "cancer drug", "best practices", "best of", "upgrade cycle",
    "software upgrade", "upgrades its", "share class", "class a", "class b", "growth stock",
    "growth stocks", "growth fund", "value stock", "value stocks", "dividend stocks", "dividend stock",
    "dividend etf", "income fund", "high dividend", "beat the market", "great recession",
    "new low cost", "highs and lows", "ups and downs", "ups and down", "up to", "dead heat",
    "up front", "upfront", "supply chain", "supply chains", "in advance", "advance notice",
    "this fall", "next fall", "last fall", "fall season", "3d printing", "high level", "low level",
    "pop up", "pop ups", "flare up", "flare ups", "make the cut", "makes the cut", "secured notes",
    "senior secured", "secured debt", "secured loan", "secured credit", "secured term", "rallies the troops",
    "rally the troops", "up for grabs", "up for sale", "up for election", "up for debate", "up for renewal",
    "up for auction", "up and running", "set up", "sets up", "setting up", "sign up", "signs up", "line up",
    "back up", "follow up", "shake up", "shakeup", "close to", "closed end", "closed-end", "open ended",
    "open end", "open interest", "open market", "open source", "advanced micro devices", "advanced micro",
    "cuts both ways", "bull and bear", "bulls and bears", "bears and bulls", "price war games",
    "ended up", "end up", "ends up", "ending up", "wound up", "wind up", "winds up", "heart failure",
    "kidney failure", "liver failure", "organ failure", "respiratory failure",
    "canopy growth", "eagle growth", "growth and income", "pare down",
    *NEGATION_EXCEPTIONS,
})

# Rule triggers for questions / listicles / roundups.
QUESTION_STARTERS: frozenset[str] = frozenset({
    "is", "are", "should", "will", "can", "does", "do", "has", "have", "could", "would", "was", "were",
    "what", "why", "how", "which", "who", "when", "where",
})

# VADER words that mislead on finance text: sense differs ("gross margin",
# "crude", "shares", "interest", "credit", "outstanding", "liability") or the
# word is handled compositionally here ("growth", "loss", "cut", "demand").
VADER_NEUTRALIZE: frozenset[str] = frozenset({
    "gross", "crude", "vice", "liability", "liabilities", "interest", "interests", "credit", "credits",
    "share", "shares", "trust", "trusts", "united", "matter", "pretty", "dear", "fine", "outstanding",
    "free", "care", "like", "likes", "agreement", "agreements", "demand", "demands", "no", "growth",
    "profit", "profits", "loss", "losses", "gain", "gains", "drop", "cut", "cuts", "lost", "low", "top",
    "tops", "best", "block", "blocks", "limited", "super", "certain", "surprise", "surprising",
    "surprised", "cancer", "rich", "wealth", "wealthy", "save", "savings", "support", "supports",
    "supported", "debt", "risk", "risks", "crash", "pressure", "high", "highs", "advantage", "please",
    "yes", "ok", "okay", "want", "help", "helps", "fund", "funds", "capital", "charge", "charges",
    "charged", "strike", "strikes", "kill", "killed", "kills", "killing", "hard", "challenge",
    "challenges", "challenging", "smart", "liberty", "pure", "unity", "hope", "hopes", "hoping",
    "hopeful", "honor", "dividend", "beat", "beats", "miss", "misses", "missed", "fall", "falls",
    "fell", "rise", "rises", "rose", "great", "grand", "special", "big", "huge", "giant", "bull",
    "bear", "bears", "bulls", "dead", "split", "splits",
})

# Words / emoji the tokenizer should keep as one token even with hyphens.
CASHTAG_PREFIXES: tuple[str, ...] = ("$",)


def phrase_keys(table: dict[str, object] | frozenset[str]) -> list[str]:
    """Normalize lexicon keys: lowercase, hyphens -> spaces, collapse whitespace."""
    return [" ".join(k.lower().replace("-", " ").split()) for k in table]


def lexicon_size() -> int:
    """Number of distinct scored entries (for docs / tests)."""
    keys = set(phrase_keys(POSITIVE)) | set(phrase_keys(NEGATIVE)) | set(phrase_keys(LITIGIOUS))
    keys |= set(phrase_keys(SOCIAL)) | set(phrase_keys(SOCIAL_ONLY)) | set(phrase_keys(DIRECTIONS))
    keys |= set(phrase_keys(METRICS)) | set(phrase_keys(RATINGS))
    return len(keys)
