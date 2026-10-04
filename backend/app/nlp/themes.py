"""What is a text about? `classify_themes(text)` -> theme keys, strongest first.

Themes are coarse, stable buckets for the "what's driving sentiment" view
(ThemeStat). Each theme is a handful of phrase patterns; a text can carry
several themes and is ranked by how many distinct patterns hit. Patterns are
deliberately phrase-level ("price target", "class action", "data center") so a
single generic word ("deal", "case", "target") never decides a theme alone.
"""
from __future__ import annotations

import re

from app.nlp.text import fold

THEMES: dict[str, str] = {
    "earnings": "Earnings",
    "guidance": "Guidance & outlook",
    "analyst": "Analyst ratings",
    "product": "Products & launches",
    "ai": "AI",
    "legal": "Legal",
    "regulatory": "Regulation & policy",
    "deals": "M&A & partnerships",
    "management": "Management",
    "capital_return": "Buybacks & dividends",
    "macro": "Macro & markets",
    "supply_chain": "Supply chain & production",
    "competition": "Competition",
    "labor": "Labor & layoffs",
    "trading": "Trading & flows",
    "valuation": "Valuation",
    "insider": "Insider activity",
}

_PATTERNS: dict[str, tuple[str, ...]] = {
    "earnings": (
        r"\bearnings\b(?! (?:per share )?yield)", r"\beps\b", (r"\b(?:quarterly|annual|q[1-4]|fiscal) (?:results|report|"
        r"numbers|revenue|sales|profit)"), r"\b(?:first|second|third|fourth)[- ]quarter\b", r"\bq[1-4]\b",
        r"\brevenues?\b", r"\bnet (?:income|loss)\b", r"\bprofits?\b", (r"\b(?:beats?|miss(?:es)?|tops?|topped) "
        r"(?:\w+ ){0,3}?(?:estimates|expectations|consensus)"), r"\b(?:earnings|eps|revenue|profit) (?:beat|miss)\b",
        r"\bearnings call\b", r"\b(?:gross|operating|net) margins?\b", r"\b(?:same-store|comparable) sales\b",
        r"\bcomps\b", r"\bdeliveries\b", r"\bbookings\b", r"\btop[- ]line\b", r"\bbottom[- ]line\b",
    ),
    "guidance": (
        r"\bguidance\b", r"\boutlook\b(?! to (?:positive|negative|stable))", r"\bforecasts?\b(?!\s+(?:and|&) price)",
        r"\bguides? (?:up|down|higher|lower|above|below)\b", r"\bguided\b", r"\bprojects? (?:revenue|sales|growth)\b",
        r"\b(?:full[- ]year|fy\d*|annual) (?:target|view|forecast)\b", r"\bprofit warning\b", r"\bwarns? (?:on|of)\b",
        r"\breaffirms?\b",
    ),
    "analyst": (
        r"\bprice[- ]targets?\b", r"\btarget price\b", r"\bpt\b", r"\bupgrad(?:e|es|ed|ing)\b", r"\bdowngrad(?:e|es|ed|ing)\b",
        r"\binitiat(?:es|ed) (?:coverage|at)\b", r"\b(?:over|under)weight\b", r"\b(?:out|under)perform\b",
        r"\b(?:buy|sell|hold|neutral) rating\b", r"\bequal[- ]weight\b", r"\banalysts?\b", r"\btop pick\b",
        r"\bstreet-high\b", r"\bconsensus (?:rating|target)\b", r"\breiterates?\b", r"\bcoverage\b",
        (r"\b(?:morgan stanley|goldman sachs|jpmorgan|bofa|bank of america|citi(?:group)?|wells fargo|barclays|ubs|"
        r"jefferies|bernstein|mizuho|evercore|piper sandler|wedbush|needham|oppenheimer|raymond james|keybanc|"
        r"truist|td cowen|rbc|bmo|hsbc|deutsche bank|baird|stifel|cantor|loop capital|rosenblatt) (?:says|sees|"
        r"expects|thinks|calls|names|picks|likes|warns|is)\b"),
    ),
    "product": (
        r"\blaunch(?:es|ed|ing)?\b(?! coverage)", r"\bunveil(?:s|ed|ing)?\b", (r"\bnew (?:product|model|chip|phone|"
        r"device|app|feature|service|platform|vehicle|car|drug|lineup|version)s?\b"), r"\brolls? out\b", r"\bdebut",
        r"\brelease[sd]?\b(?! (?:date|of (?:earnings|results)))", r"\biphones?\b", r"\bipads?\b", r"\bmacbooks?\b",
        r"\bgpus?\b", r"\bchips?\b", r"\bsmartphones?\b", r"\bdevices?\b", r"\bsoftware update\b", r"\bfeatures?\b",
        r"\bpre-?orders?\b", r"\bproduct (?:line|lineup|roadmap|event|cycle)\b", r"\brecalls?\b", r"\bsmart glasses\b",
        r"\bvehicles?\b", r"\brobotaxis?\b", r"\bsubscription\b", r"\bkeynote\b", r"\bevent\b(?= (?:on|in|next))",
        r"\bfda (?:approv|clear)", r"\bpipeline\b(?=.{0,30}\b(?:drug|trial|candidate))",
    ),
    "ai": (
        r"\bai\b", r"\ba\.i\.", r"\bartificial intelligence\b", r"\bgenerative\b", r"\bgen ?ai\b", r"\bllms?\b",
        r"\blarge language models?\b", r"\bchatbots?\b", r"\bmachine learning\b", r"\bdata cent(?:er|re)s?\b",
        r"\bai agents?\b", r"\bagentic\b", r"\bopenai\b", r"\bchatgpt\b", r"\banthropic\b", r"\bclaude\b",
        r"\bgemini\b", r"\bcopilot\b", r"\binference\b", r"\bhyperscalers?\b", r"\bgpus?\b", (r"\bai (?:chips?|"
        r"infrastructure|capex|spending|models?|race|bubble|boom|trade)\b"), r"\bsuperintelligence\b",
    ),
    "legal": (
        r"\blawsuits?\b", r"\bsue[sd]?\b", r"\bsuing\b", r"\bclass[- ]actions?\b", r"\blitigation\b", r"\bcourt\b",
        r"\bjudge\b", r"\bjury\b", r"\bverdict\b", r"\btrial\b(?!.{0,20}\b(?:phase|clinical|drug|data))",
        r"\bpatents?\b", r"\bsettle(?:s|d|ment)?\b", r"\bplaintiffs?\b", r"\bdamages\b", r"\binjunction\b",
        r"\bappeal(?:s|ed)?\b", r"\bfraud\b", r"\bindict(?:ed|ment)\b", r"\bshareholder alert\b",
        r"\binvestors? (?:who lost|with losses)\b", r"\blead plaintiff\b", r"\blaw firm\b", r"\ballegations?\b",
        r"\bsecurities (?:fraud|claims?)\b", r"\bbankruptcy\b", r"\bchapter 11\b",
    ),
    "regulatory": (
        r"\bregulat(?:or|ors|ory|ion|ions|ed)\b", r"\bsec\b(?! filing)", r"\bftc\b", r"\bdoj\b",
        r"\bjustice department\b", r"\bantitrust\b", r"\beuropean commission\b", r"\bbrussels\b", r"\beu\b",
        r"\bfda\b", r"\bfcc\b", r"\bnhtsa\b", r"\bcma\b", r"\bprobes?\b", r"\binvestigation\b", r"\bfined?\b",
        r"\bpenalt(?:y|ies)\b", r"\bsanctions?\b", r"\bexport (?:controls?|curbs?|restrictions?|ban|license)s?\b",
        r"\bban(?:s|ned)?\b", r"\bnational security\b", r"\bcongress\b", r"\bsenate\b", r"\blawmakers\b",
        r"\blegislation\b", r"\bbill\b(?= (?:to|would|that|passes))", r"\bcompliance\b", r"\bprivacy\b",
        r"\bdigital markets act\b", r"\bgdpr\b", r"\bwhite house\b", r"\btrump administration\b", r"\btariffs?\b",
        r"\bapproval\b", r"\bdelist", r"\bnasdaq notice\b",
    ),
    "deals": (
        r"\bacqui(?:re|res|red|ring|sition|sitions)\b(?! cost)", r"\bmergers?\b", r"\bmerge\b", r"\btakeover\b",
        r"\bbuyout\b", r"\btender offer\b", r"\bspin-?offs?\b", r"\bdivest", r"\bpartner(?:s|ship|ships|ed)?\b",
        r"\bteams? up\b", r"\bcollaborat", r"\balliance\b", r"\bjoint venture\b", r"\bstake in\b", r"\binvests? in\b",
        r"\b(?:inks|signs|strikes|struck) (?:a )?(?:\w+ )?(?:deal|pact|agreement)\b", (r"\bcontracts?\b(?! (?:talks|"
        r"extension))"), r"\bdeal (?:to|with|for)\b", r"\b(?:supply|licensing|cloud|chip) (?:deal|agreement)\b",
        r"\bbid\b", r"\bgo(?:es|ing)? private\b", r"\bfinancing\b",
    ),
    "management": (
        r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcto\b", r"\bchief executive\b", (r"\bchief (?:financial|operating|"
        r"technology|executive|ai|enterprise) officer\b"), r"\bchair(?:man|woman)?\b", r"\bpresident\b",
        r"\bfounder\b", r"\bsteps? down\b", r"\bresign", r"\bappoint", r"\bnames? (?:new )?(?:ceo|cfo|chief)\b",
        r"\bhires?\b", r"\btaps?\b", r"\bsuccession\b", r"\bleadership\b", r"\bboard (?:of directors|seat|member)s?\b",
        r"\bmanagement\b(?! (?:fee|software|services))", r"\brestructur", r"\bexecutives?\b", r"\breorgani[sz]",
        r"\boverhaul\b",
    ),
    "capital_return": (
        r"\bbuybacks?\b", r"\bbuy-backs?\b", r"\b(?:share|stock) repurchases?\b", r"\brepurchas", r"\bbuy back\b",
        r"\bdividends?\b(?! tax)", r"\bpayouts?\b", r"\bspecial dividend\b", r"\bshareholder returns?\b",
        r"\breturn(?:s|ed|ing)? capital\b", r"\bcapital return\b", r"\byield(?:s|ing)? \d",
    ),
    "macro": (
        r"\bfed\b", r"\bfederal reserve\b", r"\binterest rates?\b", r"\brate (?:cut|cuts|hike|hikes|decision)\b",
        r"\binflation\b", r"\bcpi\b", r"\bpce\b", r"\bjobs report\b", r"\bpayrolls?\b", r"\bunemployment\b",
        r"\bgdp\b", r"\brecession\b", r"\btreasury yields?\b", r"\bbond yields?\b", (r"\byields? (?:rise|fall|jump|"
        r"climb|surge)"), r"\btrade war\b", r"\beconom(?:y|ic|ies)\b", (r"\bconsumer (?:confidence|sentiment|"
        r"spending)\b"), r"\boil prices?\b", r"\bcrude\b", r"\bdollar\b", r"\bs&p 500\b", r"\bnasdaq composite\b",
        r"\bdow jones\b", r"\bstock market\b", (r"\bwall street\b(?! (?:analysts?|estimates|expects|sees|"
        r"consensus))"), r"\bgovernment shutdown\b", r"\belection\b", r"\bgeopolitic", r"\bwar\b(?! chest)",
        r"\bcentral bank\b", r"\bmarket (?:selloff|sell-off|rally|rout|crash)\b", r"\brisk[- ]off\b",
        r"\bvolatility\b",
    ),
    "supply_chain": (
        r"\bsupply chains?\b", r"\bsuppliers?\b", r"\bshortages?\b", r"\bproduction\b", r"\bfactor(?:y|ies)\b",
        r"\bplants?\b(?! based)", r"\bmanufactur", r"\bfoundr(?:y|ies)\b", r"\btsmc\b", r"\bfoxconn\b",
        r"\bcapacity\b", r"\binventor(?:y|ies)\b", r"\blogistics\b", r"\bshipments?\b", r"\bcomponents?\b",
        r"\brare earths?\b", r"\bmemory (?:prices?|pricing|costs?|chips?)\b", r"\bdram\b", r"\bhbm\b",
        r"\bwafers?\b", r"\bsupply (?:deal|constraints?|issues?)\b", r"\bbottlenecks?\b", r"\bfabs?\b",
        r"\bcosts? (?:rise|rising|pressure|inflation)\b", r"\boutput\b",
    ),
    "competition": (
        r"\bcompetit(?:ion|ors?|ive)\b", r"\bcompet(?:e|es|ing)\b", r"\brivals?\b", r"\bmarket share\b",
        r"\btakes? on\b", r"\b(?:takes?|taking|gains?|gaining|steals?|stealing|grabs?|cedes?|ceding) (?:market )?share\b", r"\bthreat(?:s|en|ens)?\b", r"\bchallenger?s?\b", r"\bdethrone\b", r"\bprice war\b",
        r"\bvalue war\b", r"\blos(?:es|ing) (?:share|ground|customers)\b", r"\bdisrupt", r"\bvs\.? ",
        r"\bversus\b", r"\bovertake\b", r"\bcatch(?:es|ing)? up\b",
    ),
    "labor": (
        r"\blayoffs?\b", r"\blay(?:s|ing)? off\b", r"\blaid off\b", r"\bjob cuts?\b", (r"\b(?:cut(?:s|ting)?|eliminat\w+|slash\w*|shed\w*) "
        r"(?:[\w,.]+ ){0,3}(?:jobs|roles|positions)\b"), r"\bworkforce\b", r"\bheadcount\b", r"\bunions?\b", r"\bstrikes?\b(?! (?:a |the )?deal)",
        r"\bwalkout\b", r"\bworkers\b", r"\bemployees\b", r"\bhiring\b", r"\buaw\b", r"\bteamsters\b",
        r"\blabor (?:costs?|shortage|dispute|market)\b", r"\bredundanc",
    ),
    "trading": (
        (r"\boptions? (?:volume|activity|traders?|market|flows?|bets?|trades?|trading|expir\w*|chain|data|"
        r"positioning|premiums?|open interest|are pricing|imply|implies|signal|show)\b"),
        r"\b(?:bullish|bearish|weekly|0dte|unusual|options) (?:options?|bets?|trades?)\b", r"\bcalls?\b(?= (?:and|&|or) puts\b)", r"\bput options?\b",
        r"\bcall options?\b", r"\bshort interest\b", r"\bshort squeeze\b", r"\bsqueeze\b", r"\bshort[- ]sell",
        r"\bshort report\b", r"\bunusual (?:options )?activity\b", r"\bflows?\b", r"\bdark pool\b",
        r"\bblock trades?\b", r"\b(?:trading )?volume\b", r"\bgamma\b", r"\bmeme stocks?\b",
        r"\bretail (?:traders|investors|trading|frenzy|crowd)\b", r"\b13f\b", r"\bhedge funds?\b",
        r"\b(?:stake|position) (?:cut|raised|lifted|trimmed|boosted|reduced|increased) by\b",
        r"\b(?:stock|shares) (?:acquired|bought|sold|purchased) by\b", r"\betf (?:inflows|outflows)\b",
        r"\btechnicals?\b", r"\bdeath cross\b", r"\bgolden cross\b", r"\bbreakout\b", r"\bsupport level\b",
        r"\bresistance\b", r"\bmoving average\b", r"\brsi\b", r"\boverbought\b", r"\boversold\b",
        r"\bpremarket\b", r"\bafter-hours\b", r"\bstock movers\b", r"\bmost active\b", r"\bbuy point\b",
        r"\bchart(?:s)?\b", r"\bsell-?off\b", r"\bprofit[- ]taking\b", r"\bstock split\b", r"\bs&p 500 inclusion\b",
    ),
    "valuation": (
        r"\bvaluations?\b", r"\bvalued at\b", r"\bundervalued\b", r"\bovervalued\b", r"\bcheap\b", r"\bexpensive\b",
        r"\bfair value\b", r"\bp/?e\b", r"\bprice-to-(?:earnings|sales|book)\b", r"\bmultiples?\b",
        r"\bintrinsic value\b", r"\bdiscount\b", r"\bpremium\b(?! (?:iphone|devices?|brand|products?|tier))",
        r"\b(?:could|would) be worth\b", r"\bupside\b", r"\bdownside\b", r"\bmarket (?:cap|value|valuation)\b",
        r"\bbubble\b", r"\btrillion\b", r"\bbuy now\b", r"\b(?:is it|still) a buy\b", (r"\btoo (?:cheap|expensive|"
        r"pricey)\b"), r"\bbargain\b", r"\bworth (?:buying|it)\b",
    ),
    "insider": (
        r"\binsiders?\b(?! (?:monkey|trading ban|threat))", (r"\b(?:director|ceo|cfo|officer|chairman|founder|"
        r"svp|evp|president) (?:[\w.'-]+ ){0,4}?(?:sells?|sold|buys?|bought|purchases?|acquires?|unloads?|"
        r"dumps?|offloads?)\b"), r"\bform (?:4|144)\b", r"\b10b5-1\b", r"\btrading plan\b",
        r"\bproposes? (?:selling|to sell)\b",
    ),
}

_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    theme: tuple(re.compile(p, re.IGNORECASE) for p in patterns) for theme, patterns in _PATTERNS.items()
}
_ORDER = {key: i for i, key in enumerate(THEMES)}


def theme_scores(text: str) -> dict[str, int]:
    """Number of distinct patterns that hit per theme (0 themes omitted)."""
    if not text:
        return {}
    t = fold(text)
    scores: dict[str, int] = {}
    for theme, patterns in _COMPILED.items():
        hits = sum(1 for p in patterns if p.search(t))
        if hits:
            scores[theme] = hits
    return scores


def classify_themes(text: str, max_themes: int = 4) -> list[str]:
    """Theme keys for `text`, strongest first (ties in THEMES order)."""
    scores = theme_scores(text)
    ranked = sorted(scores, key=lambda k: (-scores[k], _ORDER[k]))
    return ranked[:max_themes]
