"""What is a text about? `classify_themes(text)` -> theme keys, strongest first.

Themes are coarse, stable buckets for the "what's driving sentiment" view
(ThemeStat). Each theme is a handful of phrase patterns; a text can carry
several themes and is ranked by how many distinct patterns hit. Patterns are
deliberately phrase-level ("price target", "class action", "data center"), and
words with an everyday sense only count in their market sense: "settle" needs
a legal noun ("stablecoin settlement", "shares settle lower" are not legal),
"trial" a court ("Phase 3 trial", "free trial" are not), "fine" an amount or
regulator ("fine-tune" is not), "strike" a union ("missile strikes", "strikes
a deal" are not), "flow" a market ("free cash flow" is not), "dollar" the
currency ("trillion-dollar", "Dollar General" are not).
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
        r"\brevenues?\b", r"\bnet (?:income|loss)\b", r"\bprofits?\b(?![- ]taking)", r"\bmargin squeeze\b", r"\b(?:free )?cash flows?\b",
        (r"\b(?:beats?|miss(?:es)?|tops?|topped) "
        r"(?:\w+ ){0,3}?(?:estimates|expectations|consensus)"), r"\b(?:earnings|eps|revenue|profit) (?:beat|miss)\b",
        r"\bearnings call\b", r"\b(?:gross|operating|net) margins?\b", r"\b(?:same-store|comparable) sales\b",
        r"\bcomps\b", r"\bdeliveries\b", r"\bbookings\b", r"\btop[- ]line\b", r"\bbottom[- ]line\b",
        r"\bregistrations\b", r"\b(?:unit|car|vehicle|ev|iphone|device) sales\b",
        (r"\b(?:better|worse|more|less|fewer|stronger|weaker)(?:\s+\w+){0,2}\s+than (?:wall street |analysts |"
         r"the street )?expected\b"),
    ),
    "guidance": (
        r"\bguidance\b", r"\boutlook\b(?! to (?:positive|negative|stable))",
        (r"(?<!price )(?<!stock )(?<!weather )(?<!market )(?<!analyst )(?<!analysts' )(?<!beat )(?<!beats )(?<!top )"
         r"(?<!tops )(?<!topped )(?<!miss )(?<!missed )(?<!misses )(?<!exceed )(?<!exceeds )(?<!exceeded )"
         r"\bforecasts?\b(?!\s+(?:and|&) price)"),
        r"\bguides? (?:up|down|higher|lower|above|below)\b", r"\bguided\b", r"\bprojects? (?:revenue|sales|growth)\b",
        r"\b(?:full[- ]year|fy\d*|annual) (?:target|view|forecast)\b", r"\bprofit warning\b", r"\bwarns? (?:on|of)\b",
        r"\breaffirms?\b",
    ),
    "analyst": (
        r"\bprice[- ]targets?\b", r"\btarget price\b", r"\bpt\b", r"\bupgrad(?:e|es|ed|ing)\b", r"\bdowngrad(?:e|es|ed|ing)\b",
        r"\binitiat(?:es|ed) (?:coverage|at)\b", r"\b(?:over|under)weight\b", r"\b(?:out|under)perform\b",
        r"\b(?:buy|sell|hold|neutral) rating\b", r"\bequal[- ]weight\b", r"\banalysts?\b", r"\btop pick\b",
        r"\bstreet-high\b", r"\bconsensus (?:rating|target)\b", r"\breiterates?\b",
        r"\b(?:initiat\w*|starts?|started|begins?|resum\w*|assum\w*|launch\w*|picks? up|reinstat\w*)\s+(?:\w+\s+){0,2}?coverage\b",
        r"\b(?:conviction|focus|top picks?|best ideas) list\b", r"\bdirector'?s cut\b",
        (r"\b(?:morgan stanley|goldman sachs|jpmorgan|bofa|bank of america|citi(?:group)?|wells fargo|barclays|ubs|"
        r"jefferies|bernstein|mizuho|evercore|piper sandler|wedbush|needham|oppenheimer|raymond james|keybanc|"
        r"truist|td cowen|rbc|bmo|hsbc|deutsche bank|baird|stifel|cantor|loop capital|rosenblatt) (?:says|sees|"
        r"expects|thinks|calls|names|picks|likes|warns|is)\b"),
        (r"\b(?:says|according to|per)\s+(?:morgan stanley|goldman sachs|jpmorgan|bofa|bank of america|citi|wells "
         r"fargo|barclays|ubs|jefferies|bernstein|mizuho|evercore|piper sandler|wedbush|needham|oppenheimer|"
         r"raymond james|keybanc|truist|td cowen|rbc|bmo|hsbc|deutsche bank|baird|stifel|cantor|counterpoint|"
         r"analysts?)\b"),
    ),
    "product": (
        r"\blaunch(?:es|ed|ing)?\b(?! coverage)", r"\bunveil(?:s|ed|ing)?\b", (r"\bnew (?:product|model|chip|phone|"
        r"device|app|feature|service|platform|vehicle|car|drug|lineup|version)s?\b"), r"\brolls? out\b", r"\bdebut",
        r"(?<!press )(?<!earnings )(?<!data )(?<!jobs )(?<!news )\brelease[sd]?\b"
        r"(?! (?:date|of (?:earnings|results)|from (?:prison|jail|custody)))", r"\biphones?\b", r"\bipads?\b", r"\bmacbooks?\b",
        r"\bsmartphones?\b", r"\bsoftware update\b", r"\bnew features?\b",
        r"\bpre-?orders?\b", r"\bproduct (?:line|lineup|roadmap|event|cycle)\b", r"\brecalls?\b", r"\bsmart glasses\b",
        r"\brobotaxis?\b", r"\bsubscription\b", r"\bkeynote\b", r"\bevent\b(?= (?:on|in|next))",
        r"\b(?:downloads|weekly users|monthly users|daily active users|active users|paying subscribers)\b",
        r"\b(?:event|launch|reveal|release|unveiling|rollout)\s+(?:\w+\s+){0,2}?(?:delayed|postponed|pushed back)\b",
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
        r"\bjudge\b", r"\bjury\b", r"\bverdict\b",
        (r"\b(?:jury|criminal|civil|antitrust|patent|fraud|murder|bench|mis)\s?trial\b|\b(?:on|to|stand|at|in) trial\b|"
         r"\btrial (?:begins|starts|opens|date|judge|court|verdict|over|against|testimony|lawyers?)\b"),
        r"\bpatents?\b", r"\bplaintiffs?\b", r"\binjunction\b",
        (r"\bsettle(?:s|d|ments?)?\b(?=[^.;!?]{0,40}\b(?:lawsuits?|suits?|cases?|claims|charges|probes?|class[- ]actions?|"
         r"litigation|disputes?|allegations|investigations?|sec|ftc|doj|plaintiffs|regulators?)\b)|"
         r"\b(?:lawsuits?|suits?|cases?|claims|charges|class[- ]actions?|litigation|disputes?)\b[^.;!?]{0,30}\bsettle|"
         r"(?:\$|€|£)[\d.,]+\s?(?:million|billion|m|bn|b)?\s+(?:[\w-]+\s+){0,3}?settlement\b"),
        r"\b(?:\$[\d.,]+\w*|million|billion|punitive|compensatory|pay|paid|awards?|awarded|seeks?|seeking|in) damages\b",
        (r"\bappeals? court\b|\bcourt of appeals\b|\b(?:wins?|won|loses?|lost|files?|filed|plans? to|will|to)\s+(?:an?\s+)?"
         r"appeal\b|\bappeal(?:s|ed|ing)?\b(?=[^.;!?]{0,40}\b(?:court|ruling|verdict|judge|decision|conviction|sentence|"
         r"lawsuit|case|fine|penalty)\b)"),
        r"\bfraud\b", r"\bindict(?:ed|ment)\b", r"\bshareholder alert\b",
        r"\binvestors? (?:who lost|with losses)\b", r"\blead plaintiff\b", r"\blaw firm\b", r"\ballegations?\b",
        r"\bsecurities (?:fraud|claims?)\b", r"\bbankruptcy\b", r"\bchapter 11\b",
    ),
    "regulatory": (
        r"\bregulat(?:or|ors|ory|ion|ions|ed)\b", r"\bsec\b(?! filing)", r"\bftc\b", r"\bdoj\b",
        r"\bjustice department\b", r"\bantitrust\b", r"\beuropean commission\b", r"\bbrussels\b", r"\beu\b",
        r"\bfda\b", r"\bfcc\b", r"\bnhtsa\b", r"\bcma\b", r"\bprobes?\b", r"\binvestigation\b",
        (r"\bfined\b|\bfines?\s+(?:of\s+)?(?:up to\s+)?(?:\$|€|£|\d)|(?:\$|€|£)[\d.,]+\s?(?:million|billion|m|bn|b)?\s+fines?\b|"
         r"\b(?:eu|sec|ftc|doj|regulators?|commission|court|judge|watchdog|authority|cma|fca|finra|cfpb)\s+fines?\b|"
         r"\bfines?\s+(?:\w+\s+)?(?:for|over)\b"),
        r"\bpenalt(?:y|ies)\b", r"\bsanctions?\b", r"\bexport (?:controls?|curbs?|restrictions?|ban|license)s?\b",
        r"\bban(?:s|ned)?\b", r"\bnational security\b", r"\bcongress\b", r"\bsenate\b", r"\blawmakers\b",
        r"\blegislation\b", r"\bbill\b(?= (?:to|would|that|passes))", r"\bcompliance\b", r"\bprivacy\b",
        r"\bdigital markets act\b", r"\bgdpr\b", r"\bwhite house\b", r"\btrump administration\b", r"\btariffs?\b",
        (r"\b(?:fda|regulatory|antitrust|government|eu|ema|fcc|ftc|doj|cfius|samr|china|chinese|beijing|state|federal|"
         r"court|chmp)\s+(?:\w+\s+)?"
         r"(?:approval|clearance)\b|\bapproval (?:from|by) (?:the )?(?:fda|regulators?|ema|eu|government|court)\b"),
        r"\bdelist", r"\bnasdaq notice\b",
    ),
    "deals": (
        (r"\bacqui(?:re|res|red|ring|sition|sitions)\b(?! cost)(?!\s+[\d,.]+\s+(?:[\w-]+\s+){0,2}(?:shares|units|rsus?)\b)"
         r"(?!\s+by\s+(?:[\w&.,'-]+\s+){0,4}?(?:llc|lp|l\.p|inc|ltd|"
         r"ag|a/?s|a\s?s|s\.a|plc|management|capital|advisors?|partners|group|trust|bank|investments?|wealth|asset|financial|"
         r"securities|co)\b)"), r"\bmergers?\b", r"\bmerge\b", r"\btakeover\b",
        r"\bbuyout\b", r"\btender offer\b", r"\bspin-?offs?\b", r"\bdivest", r"\bpartner(?:s|ship|ships|ed)?\b",
        r"\bteams? up\b", r"\bcollaborat", r"\balliance\b", r"\bjoint venture\b", r"\bstake in\b", r"\binvests? in\b",
        r"\b(?:inks?|signs?|signed|strikes?|struck|reach(?:es|ed)?)\s+(?:an?\s+)?(?:[\w-]+\s+){0,4}?(?:deal|pact|agreement)\b", (r"\bcontracts?\b(?! (?:talks|"
        r"extension))"), r"\bdeal (?:to|with|for)\b", r"\b(?:supply|licensing|cloud|chip) (?:deal|agreement)\b",
        r"\bbid\b", r"\bgo(?:es|ing)? private\b", r"\bfinancing\b",
    ),
    "management": (
        r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcto\b", r"\bchief executive\b", (r"\bchief (?:financial|operating|"
        r"technology|executive|ai|enterprise) officer\b"), r"\bchair(?:man|woman)?\b",
        (r"\bpresident (?:and|&) (?:ceo|chief|coo)\b|\b(?:ceo|chief executive|chairman) (?:and|&) president\b|"
         r"\b(?:company|firm|bank|group|division|unit|its|their)\s+president\b"),
        r"\bfounder\b", r"\bsteps? down\b", r"\bresign", r"\bappoint", r"\bnames? (?:new )?(?:ceo|cfo|chief)\b",
        r"\b(?:hires?|taps?)\s+(?:[\w.'&-]+\s+){0,3}?(?:ceo|cfo|coo|cto|chief|president|executive|exec|head)\b",
        r"\bsuccession\b",
        (r"\bleadership (?:change|changes|transition|shake-?up|shuffle|team|turmoil|overhaul|crisis|reshuffle|exodus|"
         r"departures?|vacuum)\b|\bnew leadership\b"),
        r"\bboard (?:of directors|seat|member)s?\b",
        r"(?<!capital )(?<!asset )(?<!investment )(?<!wealth )(?<!portfolio )(?<!fund )(?<!investment )"
        r"\bmanagement\b(?! (?:fee|software|services|company|co|llc|inc|corp|group|lp|ltd))",
        r"\brestructur", r"\bexecutives?\b", r"\breorgani[sz]",
    ),
    "capital_return": (
        r"\bbuybacks?\b", r"\bbuy-backs?\b", r"\b(?:share|stock) repurchases?\b", r"\brepurchas", r"\bbuy back\b",
        r"\bdividends?\b(?! (?:tax|stocks?|kings?|aristocrats?|etfs?|investors?|payers?|growth stocks?|"
        r"portfolio|income|yield))", r"(?<!stock )(?<!bonus )(?<!severance )(?<!pay )(?<!ceo )\bpayouts?\b", r"\bspecial dividend\b", r"\bshareholder returns?\b",
        r"\breturn(?:s|ed|ing)? capital\b", r"\bcapital return\b", r"\byield(?:s|ing)? \d",
    ),
    "macro": (
        r"\bfed\b", r"\bfederal reserve\b", r"\binterest rates?\b", r"\brate (?:cut|cuts|hike|hikes|decision)\b",
        r"\binflation\b", r"\bcpi\b", r"\bpce\b", r"\bjobs report\b",
        r"\b(?:non-?farm |private )?payrolls\b|\bpayroll (?:data|report|numbers|growth|gains)\b", r"\bunemployment\b",
        r"\bgdp\b", r"\brecession\b", r"\btreasury yields?\b", r"\bbond yields?\b", (r"\byields? (?:rise|fall|jump|"
        r"climb|surge)"), r"\btrade war\b", r"\beconom(?:y|ic|ies)\b", (r"\bconsumer (?:confidence|sentiment|"
        r"spending)\b"), r"\boil (?:prices?|markets?|futures)\b", r"\bcrude\b",
        (r"\b(?:u\.?s\.? |the |strong |stronger |weak |weaker |strengthening |weakening )dollar\b(?!-)"
         r"(?! (?:general|tree|store|stores|stocks?))|\bdollar (?:index|strength|weakness|rally|slump|rises|falls|gains|"
         r"weakens|strengthens|surges|slides)\b|\bdxy\b|\bgreenback\b"),
        r"\bs&p 500\b", r"\bnasdaq composite\b",
        r"\bdow jones\b", r"\bstock market\b(?! today)", r"\bwall street (?:stocks|rally|rallies|selloff|"
        r"sell-off|slump|closes|ends|opens|futures)\b", r"\bgovernment shutdown\b", r"\belection\b", r"\bgeopolitic",
        (r"(?<!price )(?<!value )(?<!console )(?<!talent )(?<!bidding )(?<!streaming )(?<!chip )(?<!cola )(?<!fare )"
         r"(?<!subsidy )(?<!tariff )(?<!trade )(?<!format )(?<!browser )(?<!patent )(?<!legal )\bwar\b(?! chest)"),
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
        r"\btakes? on\b", r"\b(?:takes?|taking|gains?|gaining|steals?|stealing|grabs?|cedes?|ceding) (?:market )?share\b", r"\bthreat(?:s|en|ens)?\b", r"\bchallengers?\b|\bchallenges?\s+(?:from|to)\b|\bfac(?:es|ing|e) (?:a |new |growing |stiff |tough )?challenges?\b", r"\bdethrone\b", r"\bprice war\b",
        r"\bvalue war\b", r"\blos(?:es|ing) (?:share|ground|customers)\b", r"\bdisrupt", r"\bvs\.? ",
        r"\bversus\b", r"\bovertake\b", r"\bcatch(?:es|ing)? up\b",
    ),
    "labor": (
        r"\blayoffs?\b", r"\blay(?:s|ing)? off\b", r"\blaid off\b", r"\bjob cuts?\b", (r"\b(?:cut(?:s|ting)?|eliminat\w+|slash\w*|shed\w*) "
        r"(?:[\w,.]+ ){0,3}(?:jobs|roles|positions)\b"), r"\bworkforce\b", r"\bheadcount\b",
        r"(?<!european )(?<!soviet )(?<!credit )\bunions?\b(?! (?:pacific|bank|square|station|budget|address))",
        (r"\b(?:on|end|ends|ended|threaten\w*|avert\w*|union|workers?|labor|machinists?|uaw|teamsters|walkout|planned|"
         r"possible|potential|looming|nationwide|dockworkers?|pilots?|nurses?|actors?|writers?)\s+(?:\w+\s+)?strikes?\b|"
         r"\bstrikes?\s+(?:at|against|by|ends?|vote|deadline|looms?|threat|action|pay)\b|\bgo(?:es|ing)? on strike\b"),
        r"\bwalkout\b", r"\bworkers\b", r"\bemployees\b", r"\bhiring\b", r"\buaw\b", r"\bteamsters\b",
        r"\blabor (?:costs?|shortage|dispute|market)\b", r"\bredundanc",
    ),
    "trading": (
        (r"\boptions? (?:volume|activity|traders?|market|flows?|bets?|trades?|trading|expir\w*|chain|data|"
        r"positioning|premiums?|open interest|are pricing|imply|implies|signal|show)\b"),
        r"\b(?:bullish|bearish|weekly|0dte|unusual|options) (?:options?|bets?|trades?)\b", r"\bcalls?\b(?= (?:and|&|or) puts\b)", r"\bput options?\b",
        r"\bcall options?\b", r"\bshort interest\b", r"\bshort squeeze\b",
        r"(?<!margin )(?<!cost )(?<!profit )(?<!price )(?<!budget )(?<!credit )(?<!liquidity )\bsqueeze\b",
        r"\bshort[- ]sell", r"\bshort report\b", r"\bunusual (?:options )?activity\b",
        r"\boptions? (?:alerts?|sweeps?)\b|\b(?:bullish|bearish|unusual) (?:call|put)s?\b",
        (r"\b(?:options?|order|fund|etf|money|capital|retail|institutional|dark pool|whale|investor)\s+(?:\w+\s+)?flows?\b|"
         r"\b(?:in|out)flows?\b|\bflows? (?:into|out of|data)\b"),
        r"\bdark pool\b",
        r"\bblock trades?\b", r"\b(?:trading )?volume\b", r"\bgamma\b", r"\bmeme stocks?\b",
        r"\bretail (?:traders|investors|trading|frenzy|crowd)\b", r"\b13f\b", r"\bhedge funds?\b",
        r"\b(?:stake|position) (?:cut|raised|lifted|trimmed|boosted|reduced|increased) by\b",
        r"\b(?:stock|shares) (?:acquired|bought|sold|purchased) by\b", r"\betf (?:inflows|outflows)\b",
        r"\b(?:shares|stake|stock|position)\b[^?!]{0,50}\b(?:acquired|bought|sold|purchased|trimmed|raised|lowered|"
        r"cut|boosted|lifted|reduced)\s+by\b", r"\b(?:purchases|acquires|buys|sells|trims|boosts)\s+[\d,]+\s+shares of\b",
        r"\btechnicals?\b", r"\bdeath cross\b", r"\bgolden cross\b", r"\bbreakout\b",
        r"\b(?:resistance|support) (?:level|line|zone)s?\b|\b(?:breaks?|broke|above|below|through|at|near|tests?|testing) "
        r"(?:\w+\s+)?resistance\b", r"\bmoving average\b", r"\brsi\b", r"\boverbought\b", r"\boversold\b",
        r"\bpremarket\b", r"\bafter-hours\b", r"\bstock movers\b", r"\bmost active\b", r"\bbuy point\b",
        r"\bchart(?:s)?\b", r"\bsell-?off\b", r"\bprofit[- ]taking\b", r"\bstock split\b", r"\bs&p 500 inclusion\b",
    ),
    "valuation": (
        r"\bvaluations?\b", r"\bvalued at\b", r"\bundervalued\b", r"\bovervalued\b", r"\bcheap\b", r"\bexpensive\b",
        r"\bfair value\b", r"\bp/?e\b", r"\b\d+\s?(?:x|times)\s+(?:forward\s+|trailing\s+)?(?:earnings|sales|revenue|ebitda|book)\b", r"\bprice-to-(?:earnings|sales|book)\b",
        (r"\b(?:valuation|earnings|revenue|sales|ebitda|forward|trailing|p/e|pe) multiples?\b|"
         r"\bmultiple (?:expansion|compression|contraction)\b"),
        r"\bintrinsic value\b",
        (r"\b(?:trades?|trading|at|steep|deep|big|huge) (?:an? )?(?:\w+\s+)?(?:discount|premium)\b(?! (?:retailer|store|"
         r"brand|prices?|products?|devices?|iphone|tier|pricing))|\b(?:discount|premium) to\b|\bdiscounted cash flow\b|"
         r"\bdcf\b|\bpremium (?:valuation|multiple)\b"),
        r"\b(?:could|would) be worth\b", r"\bupside\b", r"\bdownside\b", r"\bmarket (?:cap|value|valuation)\b",
        r"\bbubble\b",
        (r"\btrillion[- ](?:dollar|market|valuation|club|milestone|company)|\$[\d.]+\s?(?:trillion|t)\b"
         r"(?= (?:market|valuation|club|milestone|company|mark|stock))"), r"\bbuy now\b", r"\b(?:is it|still) a buy\b", (r"\btoo (?:cheap|expensive|"
        r"pricey)\b"), r"\bbargain\b", r"\bworth (?:buying|it)\b",
    ),
    "insider": (
        r"\binsiders?\b(?! (?:monkey|trading ban|threat))", (r"\b(?:director|ceo|cfo|coo|cto|officer|chairman|founder|"
        r"svp|evp|president|executive|exec) (?:[\w.'-]+ ){0,4}?(?:sells?|sold|buys?|bought|purchases?|acquires?|"
        r"unloads?|dumps?|offloads?)\b"), r"\bform (?:4|144)\b", r"\b10b5-1\b", r"\btrading plan\b",
        r"\bproposes? (?:selling|to sell)\b",
    ),
}

_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    theme: tuple(re.compile(p, re.IGNORECASE) for p in patterns) for theme, patterns in _PATTERNS.items()
}
_ORDER = {key: i for i, key in enumerate(THEMES)}


def _heads(items: list) -> frozenset[str] | None:
    """Literal strings one of which every match must start with (lower-case),
    or None when the pattern has no literal head (e.g. it opens with a class)."""
    from re import _constants as c  # parser internals; any surprise just disables the shortcut

    prefix = ""
    for op, av in items:
        if op in (c.AT, c.ASSERT, c.ASSERT_NOT) and not prefix:
            continue
        if op is c.LITERAL:
            prefix += chr(av).lower()
            continue
        if op is c.SUBPATTERN:
            inner = _heads(list(av[3]))
            if inner is None:
                break
            return frozenset(prefix + h for h in inner)
        if op is c.BRANCH:
            alternatives = [_heads(list(alt)) for alt in av[1]]
            if any(a is None for a in alternatives):
                break
            return frozenset(prefix + h for a in alternatives for h in a)  # type: ignore[union-attr]
        break
    return frozenset({prefix}) if prefix else None


def _literal_heads(pattern: str) -> frozenset[str] | None:
    try:
        from re import _parser

        return _heads(list(_parser.parse(pattern, re.IGNORECASE)))
    except Exception:  # noqa: BLE001 - private API; fall back to always searching
        return None


# A pattern can only match where one of its literal heads occurs, so each text
# runs just the patterns whose heads' first two or three letters appear in it
# (most of the ~380 patterns cannot match a given headline: 500 texts ~0.1 s).
_FLAT: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (theme, compiled) for theme, patterns in _COMPILED.items() for compiled in patterns
)
_ALWAYS: list[int] = []
_BY_GRAM: dict[str, list[int]] = {}
for _i, (_theme, _compiled) in enumerate(_FLAT):
    _found = _literal_heads(_compiled.pattern)
    if _found is None or any(len(h) < 2 for h in _found):
        _ALWAYS.append(_i)
        continue
    for _gram in {h[:3] for h in _found}:
        _BY_GRAM.setdefault(_gram, []).append(_i)


def theme_scores(text: str) -> dict[str, int]:
    """Number of distinct patterns that hit per theme (0 themes omitted)."""
    if not text:
        return {}
    t = fold(text)
    low = t.lower()
    candidates = set(_ALWAYS)
    for gram in {low[i:i + n] for n in (2, 3) for i in range(len(low) - n + 1)}:
        candidates.update(_BY_GRAM.get(gram, ()))
    scores: dict[str, int] = {}
    for i in candidates:
        theme, pattern = _FLAT[i]
        if pattern.search(t):
            scores[theme] = scores.get(theme, 0) + 1
    return scores


def classify_themes(text: str, max_themes: int = 4) -> list[str]:
    """Theme keys for `text`, strongest first (ties in THEMES order)."""
    scores = theme_scores(text)
    ranked = sorted(scores, key=lambda k: (-scores[k], _ORDER[k]))
    return ranked[:max_themes]
