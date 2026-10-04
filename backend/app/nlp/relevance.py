"""How clearly is a text about *this* company?  `relevance(text, company)` -> 0..1.

Broad keyword searches return plenty of noise: "price target" headlines about
other stocks for Target, "Democrats block bill" for Block, "apple cider" for
Apple, "meta-analysis" for Meta, five-ticker StockTwits spam. The scorer
weighs evidence per mention instead of string-matching:

    1.00  cashtag ($NVDA, $BTC.X)
    0.95  exchange-qualified / parenthesized ticker ("(NASDAQ: NVDA)", "(TGT)")
    0.90  bare ticker token (never for word-like tickers: ALL, IT, ON, NOW, A, T, F …)
    0.80  company name used as a company (+0.10 when it is the headline subject)
    0.55  ambiguous common-word name with finance/industry context but no
          decisive cue; 0.25 without context; ~0 when every mention is a
          non-company sense (collocations like "price target", "block party")
    ×0.75 when another company is the subject ("Cerebras stock … on Nvidia pressure")
    ≤0.40 roundups/listicles (≥ 4 cashtags, "3 AI Chip Stocks To Watch", "X, Y, Z and More")

Common-word names (Apple, Target, Meta, Block, Snap, Visa, Shell, Amazon,
Oracle, Ford …) are matched case-sensitively and need positive context
(possessive, "Target (TGT)", "Block stock", subject + verb, analyst verb before
it, product/executive cues) to score high; curated negative collocations knock
out known false friends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.nlp.text import fold, is_mostly_upper, wordset
from app.sources.base import CompanyRef

# --------------------------------------------------------------------------- #
# Ticker ambiguity
# --------------------------------------------------------------------------- #
# Tickers that are English words or ubiquitous acronyms: a bare token is never
# trusted (only $TICKER, "(TICKER)", "NYSE: TICKER", "TICKER.US").
WORD_TICKERS: frozenset[str] = wordset("""
A I O U X Y AI ALL AM AN ANY ARE AT BE BIG BEST BUY CAN CAR CASH CEO CFO COO CPI DD DO EAT EPS ETF EU EV FAST FDA FED
FOR FUN GDP GO GOOD HAS HE HIGH HOME IPO IR IT JOB KEY LIFE LOVE LOW MAN ME MY NEW NICE NOW OK ON ONE OPEN OR OUT PM
PR REAL RUN SAFE SAVE SEC SEE SO TEAM TECH TOP TRUE TV UK UP USA US WELL WIN YOLO ATH IMO FOMO HODL GAIN PLAY MOVE
PEAK RACE SITE TRIP HUGE BIG OPEN WOOF EDIT POST JOBS PAY ROI OPEN EARN GROW LEAD SAVE DRUG BOOM BULL BEAR MOON PUMP
""")
# Word-like tickers trusted as bare tokens only when written upper-case in
# normal (not shouted) text: "META Stock Holds Up", "CAT earnings".
SOFT_WORD_TICKERS: frozenset[str] = wordset("""
META CAT DOCS SNOW NET PATH COIN HOOD SNAP BALL ROCK BOX DASH ZETA PINS SHOP GAP PLUG BIRD LULU NOVA RIDE RENT
TOST CART CHWY ROOT LMND MATCH MTCH OPEN PEP KO GE GM HD MA MS BA C F T V K D O ED SO DE
""")

# Brand names that are also common words, surnames or places.
AMBIGUOUS_NAMES: frozenset[str] = wordset("""
apple target meta block snap visa shell amazon oracle ford gap delta united alphabet unity square zoom match toast affirm
chime circle strategy bullish figure gemini intel arm ball carnival discovery progressive southern sun crown fox news
sea grab lucid root compass upstart lemonade wish ring box slack monster energy pool mosaic ally dover crane corning
dominion paramount charter dish lumen frontier host sands simon duke hilton marathon edison principal prudential
citizens regions key equity realty global general american national first public digital genius bumble coach express
guess sonic jack keurig dollar tree five below best buy home depot shopify domino spotify waste management chewy
xerox micron rocket opendoor robinhood vertex abbott baxter edwards zimmer stryker gilead lilly merck mercury apollo
blackstone carlyle ares kkr nu sofi marvell coherent qualcomm entegris onto amkor wolfspeed lumentum ciena juniper
cadence synopsys autodesk adobe salesforce workday datadog elastic confluent palantir snowflake twilio okta
""") - wordset("""
qualcomm entegris amkor wolfspeed lumentum ciena juniper synopsys autodesk adobe salesforce workday datadog confluent
palantir snowflake twilio okta marvell sofi kkr blackstone carlyle gilead chewy shopify spotify xerox robinhood opendoor
""")

# Corporate suffixes stripped to get the brand; their presence confirms a company.
_CORP_SUFFIX = (r"(?:,?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|Companies|Ltd|Limited|plc|PLC|LLC|L\.P|LP|"
                r"Holdings?|Group|N\.V|NV|S\.A|SA|SE|AG|ASA|AB|Oyj|A/S)\b\.?)")
_SUFFIX_STRIP_RE = re.compile(_CORP_SUFFIX + r"+\s*$|\s+(?:Class|Series)\s+[A-C]\b.*$|\s+\(The\)$|^The\s+", re.IGNORECASE)
_FUND_ISSUERS_RE = re.compile(r"^(?:SPDR|iShares|Vanguard|Invesco|ProShares|Direxion|Global X|Schwab|Fidelity|VanEck|"
                              r"WisdomTree|First Trust|Select Sector SPDR)\s+", re.IGNORECASE)
_FUND_WORDS_RE = re.compile(r"\s+(?:ETF|Trust|Fund|Index Fund|Shares|ETF Trust)\b.*$", re.IGNORECASE)

DETERMINERS = wordset("""the a an its their his her our your this that these those my own new every each any no some
same price stock key main primary easy prime next latest fresh lost moving soft hard upper lower""")

# Words that, right after a name, mean "the company".
_COMPANY_FOLLOWERS = (
    r"stock|stocks|shares|share price|stake|investors|shareholders|stockholders|earnings|results|revenue|sales|profit|"
    r"guidance|outlook|ceo|cfo|coo|cto|chief|executives?|execs?|board|directors?|insiders?|management|employees|workers|"
    r"staff|stores?|app|apps|spokes(?:man|woman|person)|founder|chair(?:man|woman)?|president|unit|division|subsidiary|"
    r"logo|brand|products?|devices?|users|customers|analysts?|bulls|bears|market cap|valuation|ipo|bonds?|"
    r"options|calls|puts|q[1-4]|fiscal|quarter|quarterly|deliveries|layoffs|lawsuit|headquarters|hq|retail|website|"
    r"patents?|verdict|trial|case|suit|antitrust|probe|settlement|fine|ruling|appeal|partners?|suppliers?|rivals?|"
    r"competitors?|event|keynote|ai|chips?|software|hardware|ecosystem|services|ads|advertising|cloud|campus|"
    r"smartphones?|phones?|earnings call|growth|margins?|buyback|dividend|turnaround|strategy|leadership"
)
_FOLLOWER_RE = re.compile(rf"^\s+(?:{_COMPANY_FOLLOWERS})\b(?!-)", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"^'s\b")
_PAREN_TICKER_RE = re.compile(r"^\s*\((?:[A-Z]{2,12}\s?:\s?)?([A-Z][A-Z0-9.\-]{0,9})(?:\.[A-Z]{1,3})?\)")
_HYPHEN_OK = wordset("backed owned led based made branded related linked focused funded parent maker rival supplier "
                       "partner like style designed built powered approved listed")
# Analyst/transaction verbs immediately before a name: "HSBC upgrades Target".
_STRONG_BEFORE_RE = re.compile(
    r"(?:upgrades?|downgrades?|upgraded|downgraded|initiates?(?: coverage)?(?: on| of)?|coverage (?:on|of)|"
    r"(?:rating|stance|view) on|reiterates? \w+ on|maintains? \w+ on|buy|buying|sell|selling|short|shorting|long|"
    r"own|owning|bought|sold|vs\.?|versus|sues?|sued|suing|acquires?|acquiring|backs|likes|prefers|favors|"
    r"names|picks|loves|hates|dumps|trims|adds|boosts stake in|cuts stake in|praises|beats|joins|"
    r"(?:invest(?:ed|ing|s)?|stake|position|bet|bets|betting|exposure) in|shares of|stock of|bullish on|bearish on)\s*$",
    re.IGNORECASE,
)
# A verb right after a sentence-initial name: "Target Slashes Prices", "Meta taps".
_SUBJECT_VERB_RE = re.compile(
    r"^\s+(?:is|was|to|will|has|had|have|can|could|may|might|should|would|must|does|did|says|said|just|now|also|"
    r"still|again|reportedly|officially|finally|quietly|hit|cut|set|put|won|lost|got|made|took|sold|bought|beat|"
    r"led|paid|rose|fell|sank|grew|ran|[a-z]+(?:s|ed))\b(?!-)",
    re.IGNORECASE,
)
# Mid-sentence, only an unmistakable 3rd-person verb makes the name a subject ("says Apple owes").
_MID_VERB_RE = re.compile(
    r"^\s+(?:is|was|has|will|says|said|owes|faces|wins|loses|gets|sees|makes|takes|plans|wants|unveils|launches|"
    r"announces|reports|posts|agrees|hires|taps|cuts|raises|slashes|lowers|boosts|warns|expects|delivers|"
    r"rolls|ships|sells|buys|acquires|beats|misses|tops|sues|settles|denies|confirms|reveals|introduces|"
    r"opens|closes|signs|joins|loses|drops|falls|rises|jumps|soars|slides|sinks|climbs|gains|trims|adds|"
    r"brings|pushes|bets|targets|prepares|readies|weighs|considers|explores|eyes|enters|exits|halts|pauses|"
    r"resumes|files|wins|won|hit|lost|got|made|took|sold|bought|led|paid|rose|fell|sank)\b(?!-)",
    re.IGNORECASE,
)
_CLAUSE_START_RE = re.compile(
    r"(?:^|[:;.!?|]\s*|\s-\s)(?:(?:why|how|what|when|where|will|can|could|should|would|is|does|did|has|"
    r"here's|as|after|while|because|if|but|and|so)\s+){0,2}$",
    re.IGNORECASE,
)

_FINANCE_CONTEXT_RE = re.compile(
    r"\b(?:stocks?|shares?|shareholders?|investors?|earnings|revenue|profit|quarter(?:ly)?|q[1-4]|guidance|outlook|"
    r"analysts?|upgrades?|downgrades?|rating|ceo|cfo|executives?|dividend|buyback|market (?:cap|value)|valuation|"
    r"ipo|acquisition|merger|nyse|nasdaq|wall street|traders?|bullish|bearish|rall(?:y|ies)|plunges?|surges?|"
    r"premarket|after-hours|sec filing|10-[kq]|8-k|eps|margin|forecast|fiscal|retailer|company|firm|brand|market|"
    r"customers?|consumers?|shoppers?|sales|stores?|deal|layoffs|lawsuit|regulators?|antitrust|chips?|ai)\b",
    re.IGNORECASE,
)
_LISTICLE_RE = re.compile(
    r"\b(?:[2-9]|1[0-9]|20|two|three|four|five|six|seven|eight|nine|ten|dozen|several|these)\s+"
    r"(?:(?!hours?|days?|weeks?|months?|years?)[\w&'-]+\s+){0,3}?"
    r"(?:stocks|names|picks|companies|equities|tickers|chipmakers|plays|buys|etfs)\b"
    r"|\b(?:[1-9]|1[0-9]|20)\s+(?:[\w&'-]+\s+){0,3}?stock\s+(?:to|that|you|we|i)\b"
    r"|\b(?:stocks? to (?:buy|watch|sell|avoid|own)|stocks? that explain|stock movers|biggest (?:movers|moves)|"
    r"(?:midday|premarket|after-hours) movers|top (?:gainers|losers)|most active|trending stocks|"
    r"top midday stories|morning squawk|weekly review|week ahead|market wrap|and more:|and more stocks|"
    r"stocks making the biggest moves|earnings to watch|what to watch)\b",
    re.IGNORECASE,
)
_COMMA_LIST_RE = re.compile(r"(?:\b[A-Z][\w&.'-]*(?:\s[A-Z][\w&.'-]*){0,2},\s+){3,}")
_TICKER_LIST_RE = re.compile(r"(?:\b[A-Z]{2,5},\s*){3,}[A-Z]{2,5}\b")
_CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z]{1,3})?)\b")
_QUALIFIED_RE_T = (r"(?:\b(?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|AMEX|OTC|OTCMKTS|TSX|TSXV|LSE|ASX|CBOE|BATS)\s?:\s?{t}\b"
                   r"|\b{t}\s?:\s?(?:NASDAQ|NYSE|US|CA)\b|\({t}(?:\.[A-Z]{{1,3}})?\)|\b{t}\.US\b)")
# Capitalized words before "Stock" that do not name a company ("Better AI Stock").
_GENERIC_SUBJECT_WORDS = wordset("""better best top my our this that the a an ai artificial intelligence dividend growth
value tech technology chip chips semiconductor bank banking energy oil retail consumer defense quantum crypto bitcoin
blue chip cheap hot buy sell hold one two three four five 1 2 3 4 5 penny meme small mid large cap mega magnificent seven
why how what is are should could would will can high yield income utility healthcare biotech reit index etf""")
_OTHER_SUBJECT_RE = re.compile(
    r"^(?:[\w&.'-]+\s){0,2}?([A-Z][\w&.'-]+(?:\s[A-Z][\w&.'-]+){0,2})(?:'s)?\s+(?:stock|shares|Stock|Shares)\b"
)


@dataclass(frozen=True)
class NameRule:
    """Curated knowledge for one ambiguous brand name."""

    negative: str = ""  # regex (case-insensitive); spans that are NOT the company
    cues: str = ""  # regex (case-insensitive); words that confirm the company anywhere in the text
    products: tuple[str, ...] = ()  # capitalized words that may follow the name ("Apple Watch")


# Brokers that precede "target" in price-target headlines ("JPMorgan Target Cut").
_BROKERS = (r"jpmorgan|jp morgan|morgan stanley|goldman(?: sachs)?|citi(?:group)?|bofa|bank of america|wells fargo|"
            r"barclays|ubs|deutsche bank|hsbc|jefferies|mizuho|nomura|macquarie|rbc|bmo|td cowen|cowen|truist|"
            r"piper sandler|raymond james|stifel|baird|keybanc|wedbush|oppenheimer|needham|bernstein|evercore|"
            r"guggenheim|loop capital|rosenblatt|cantor(?: fitzgerald)?|benchmark|b\. ?riley|btig|d\.?a\.? davidson|"
            r"canaccord|susquehanna|melius|redburn|daiwa|clsa|argus|morningstar|cfra|wolfe|seaport|william blair|"
            r"stephens|kbw|telsey|bnp paribas|societe generale|berenberg|kepler|scotiabank|cibc|td securities")
_FIRMS_TARGET_VERBS = (r"raise[sd]?|raising|lift(?:s|ed)?|boost(?:s|ed)?|hike[sd]?|cut[s]?|cutting|lower(?:s|ed)?|"
                       r"trim(?:s|med)?|slash(?:es|ed)?|set[s]?|maintain(?:s|ed)?|reiterate[sd]?|keep[s]?|kept|"
                       r"up(?:s|ped)?|double[sd]?|triple[sd]?|reset[s]?|revamp[s]?|has|have|gets?|got|draws?|sees?|"
                       r"tops?|topped|hits?|meets?|missed|misses|miss|beats?|exceeds?|surpass(?:es)?|reach(?:es)?|"
                       r"above|below|near|under|over|toward|towards|of|on|her|their|its")
NAME_RULES: dict[str, NameRule] = {
    "target": NameRule(
        negative=(
            rf"\b(?:price|stock|analysts?'?|street'?s?|consensus|average|mean|median|base|bull|bear|new|fresh|"
            rf"higher|lower|street-high|street-low|monthly|annual|upper|inflation|revenue|sales|growth|earnings|"
            rf"margin|profit|emissions?|climate|production|delivery|output|fed|boj|ecb|rbi|bank'?s|official|"
            rf"{_BROKERS}|"
            rf"\d[\d.,]*%?|\$[\d.,]+[kmb]?|(?:usd|eur|gbp|cad|sek|nok|dkk|c\$|a\$)\s?[\d.,]+|{_FIRMS_TARGET_VERBS})"
            rf"\s+target\b"
            r"|[$\u20ac\u00a3][\d.,]+[kmb]?\s+target\b"
            r"|\btarget[- ](?:price|prices|date|dates|range|audience|market|rate|zone|level|levels|list|weight|"
            r"allocation|fund|funds|retirement|practice|acquisition|company|of|for|says|implies|suggests)\b"
            r"|\btarget[- ](?:\$|\d|to (?:\$|usd|eur|c\$))"
            r"|\btarget (?:cut|cuts|raised|lowered|hike|hikes|raise|increase|increases|boost|trimmed|slashed)"
            r"(?= (?:to|by|of|at|after|from)\b|[,.;:]|$)"
            r"|\b(?:stocks?|names?|picks?|ones?|plays?|sectors?|areas?) to target\b|\bon target\b|"
            r"\btarget this week\b|\bmissile\b|\bhit (?:a |the |its )?target\b|\btarget hospitality\b|"
            r"\btarget(?:ed|ing|s)\b"
        ),
        cues=r"\b(?:TGT|walmart|costco|kroger|cornell|fiddelke|retailer|shoppers?|good & gather|cat & jack|"
             r"target circle|bullseye|in-store|same-store|comparable sales)\b",
    ),
    "apple": NameRule(
        negative=r"\bbig apple\b|\bapple (?:pie|pies|cider|ciders|orchard|orchards|picking|harvest|juice|tree|trees|"
                 r"variet(?:y|ies)|festival|fest|crisp|butter|sauce|cinnamon|maturity|growers?|farm|farms|crop|"
                 r"hospitality|bank|valley|blossom|vinegar|fritters?|turnover|cake|slices?|nachos|day)\b",
        cues=r"\b(?:AAPL|iphone|ipad|macbook|mac|ios|siri|tim cook|cook|ternus|app store|vision pro|airpods|"
             r"apple (?:watch|tv|music|pay|intelligence|card|arcade|store|silicon)|cupertino|foxconn)\b",
        products=("Watch", "TV", "Music", "Pay", "Intelligence", "Card", "Arcade", "Store", "Silicon", "Vision",
                  "Maps", "News", "Books", "ID", "Park", "Newsroom", "Support"),
    ),
    "meta": NameRule(
        negative=r"\bmeta[- ](?:analysis|analyses|analytic|data|description|tags?|review|regression|learning|"
                 r"narrative|commentary|game|strategy|joke|humor|level|cognition)\b|\bmeta materials\b|"
                 r"\bmeta financial\b|\b(?:very|so|too|pretty|kinda) meta\b",
        cues=r"\b(?:facebook|instagram|whatsapp|threads|zuckerberg|reality labs|llama|oculus|quest|ray-ban|"
             r"menlo park|meta ai)\b",
        products=("AI", "Platforms", "Quest", "Connect", "Superintelligence"),
    ),
    "block": NameRule(
        negative=r"\b(?:h&r|h & r|ken|priscilla|city|cell|road|chopping|starting|building|mental|writer'?s|"
                 r"on the|stock the|new kid on the|one|each|whole|entire|neighborhood|apartment|"
                 r"democrats?|republicans?|senate|senators|house|gop|congress|lawmakers|court|courts|judge|judges|"
                 r"regulators?|government|police|ftc|doj|eu|china|trump|biden|governor|council|union|voters|"
                 r"moves? to|seeks? to|tries to|tried to|vote to|votes to|bid to|effort to|attempt to|could|would|"
                 r"will|may|might)\s+block\b"
                 r"|\bblock (?:party|parties|deal|deals|trade|trades|trading window|admission|sale|sales|order|orders|"
                 r"buy|club|chain|grant|grants|vote|votes|bill|bills|ban|bans|law|laws|legislation|measure|"
                 r"proposal|nomination|merger|acquisition|takeover|access|imports?|exports?|funding|release|move|"
                 r"plan|plans|efforts?|rule|rules|tariffs?|of shares|of stock|of the|letters?|schedule|"
                 r"heater|island|association|captain|watch|by block|height|reward|explorer|size|time|production)\b"
                 r"|\bblock(?:ed|ing|s)\b|\bthe block\b|\bstock the block\b",
        cues=r"\b(?:XYZ|square|cash app|afterpay|dorsey|tidal|bitkey|proto|spiral|tbd)\b",
        products=("Inc",),
    ),
    "snap": NameRule(
        negative=r"\bsnap (?:election|elections|poll|polls|judgment|judgement|decision|back|shot|chat|benefits?|"
                 r"recipients?|program|cuts?|work requirements|peas|pea)\b|\b(?:oh|cold|ginger) snap\b|"
                 r"\bsnap(?:s|ped|ping)\b|\bSNAP\b(?!\s?\))",
        cues=r"\b(?:snapchat|spiegel|snap inc|spectacles|bitmoji)\b",
    ),
    "visa": NameRule(
        negative=r"\b(?:h-1b|h1b|student|work|travel|tourist|golden|investor|spousal|transit|entry|e-|digital nomad|"
                 r"business) visas?\b|\bvisa(?:s|-free)\b|\bvisa (?:application|applications|holders?|rules?|program|"
                 r"programs|fee|fees|ban|bans|waiver|requirements?|policy|restrictions?|overstay|interview|"
                 r"processing|lottery|backlog|status|sponsorship|regime|scheme)\b",
        cues=r"\b(?:mastercard|payments?|card(?:s|holders)?|cross-border|payment volume|mcinerney|interchange)\b",
    ),
    "shell": NameRule(
        negative=r"\bshell (?:company|companies|corporation|game|games|shock|shocked|script|command|scripts|"
                 r"egg|eggs|fish|casing|beach)\b|\bshell-shocked\b|\b(?:sea|egg|turtle|hard|soft|empty|outer) shell\b|"
                 r"\bshells?\b(?! (?:plc|stock|shares))",
        cues=r"\b(?:SHEL|oil|gas|lng|crude|refin\w+|bp|exxon|chevron|totalenergies|sawan|upstream|downstream)\b",
    ),
    "amazon": NameRule(
        negative=r"\b(?:the|brazilian|peruvian) amazon\b|\bamazon (?:rainforest|river|basin|region|jungle|"
                 r"deforestation|fires?|forest|tribes?|indigenous)\b",
        cues=r"\b(?:AMZN|aws|jassy|bezos|prime|alexa|kuiper|whole foods|e-commerce|amazon web services)\b",
        products=("Web", "Prime", "Pharmacy", "Music", "Fresh", "Go", "Air", "Kuiper", "Q"),
    ),
    "oracle": NameRule(
        negative=r"\boracle of (?:omaha|delphi)\b|\bthe oracle\b|\boracles?\b(?= (?:network|protocol|price feed|"
                 r"problem|data))|\bchainlink\b",
        cues=r"\b(?:ORCL|ellison|catz|oci|cloud infrastructure|stargate|cerner|database|netsuite|magouyrk|sicilia)\b",
        products=("Cloud", "Health", "Database"),
    ),
    "ford": NameRule(
        negative=r"\b(?:harrison|gerald|betty|tom|doug|henry ford (?:museum|hospital|health)|tennessee ernie|"
                 r"rob|john|aaron|colin|wendell|lana) ford\b|\bford (?:foundation|nation|model agency)\b",
        cues=r"\b(?:F-150|f150|mustang|bronco|farley|lightning|maverick|model e|ford pro|dearborn|uaw|"
             r"automaker|vehicles?|ev|evs|trucks?|recall)\b",
    ),
    "gap": NameRule(
        negative=r"\b(?:the|a|an|wage|pay|gender|wealth|trade|funding|valuation|price|supply|skills|generation|"
                 r"growth|output|credibility|widening|narrowing|big|huge|large|wide|yawning|fill|filled|filling|"
                 r"mind|bridge|bridging|close|closing|closes|closed|enthusiasm|performance)\s+gaps?\b|"
                 r"\bgap (?:up|down|higher|lower|fill|between|in|of|year|filler)\b|\bgaps\b|\bgapp(?:ed|ing)\b",
        cues=r"\b(?:old navy|banana republic|athleta|dickson|retailer|apparel)\b",
    ),
    "delta": NameRule(
        negative=r"\bdelta (?:variant|wave|neutral|hedg\w+|exposure|force|state|blues|region|smelt|faucet|"
                 r"one|t|v)\b|\b(?:mississippi|river|nile|mekong|sacramento|niger|pearl river|options?|greek) delta\b",
        cues=r"\b(?:DAL|airlines?|air lines|flights?|bastian|skymiles|carriers?|airfare|passengers?)\b",
        products=("Air",),
    ),
    "united": NameRule(
        negative=r"\bunited (?:states|nations|kingdom|arab|auto workers|steelworkers|way|front|church|"
                 r"healthcare|health group|rentals|parcel|therapeutics|natural|fire|airlines? holdings? stock)\b|"
                 r"\b(?:manchester|leeds|newcastle|west ham|sheffield|dundee|atlanta|dc|minnesota|new england|"
                 r"red bull|hearts? and) united\b",
        cues=r"\b(?:UAL|airlines?|flights?|kirby|mileageplus|carriers?|passengers?|airfare)\b",
        products=("Airlines",),
    ),
    "alphabet": NameRule(
        negative=r"\balphabet (?:soup|letters?|book|song)\b|\b(?:the|english|latin|greek|phonetic|cyrillic) alphabet\b",
        cues=r"\b(?:GOOGL?|google|youtube|waymo|pichai|gemini|deepmind|android|chrome|search giant)\b",
    ),
    "square": NameRule(
        negative=r"\b(?:times|town|city|red|tiananmen|union|trafalgar|washington|madison|public|village|market|"
                 r"main|fair and|back to) square\b|\bsquare (?:feet|foot|meters?|metres|miles?|kilometers?|km|one|"
                 r"off|root|enix|dance|deal|peg|meal)\b|\bsquare(?:s|d)\b",
        cues=r"\b(?:block|XYZ|SQ|dorsey|cash app|sellers?|merchants?|point of sale|pos)\b",
    ),
    "strategy": NameRule(
        negative=r"\b(?:growth|investment|investing|trading|business|pricing|ai|marketing|exit|corporate|national|"
                 r"defense|military|new|long-term|winning|bold|key|our|their|its|the|a|this|that|options|dividend|"
                 r"income|retirement|portfolio|tax|game|content|digital|go-to-market|turnaround|hedging|barbell|"
                 r"core|smart|simple|best|top|\d+)\s+strateg(?:y|ies)\b|\bstrateg(?:y|ies) (?:for|to|of|behind|that|"
                 r"is|game|session|meeting|document|update|shift|change)\b|\bstrategies\b",
        cues=r"\b(?:MSTR|STRK|STRF|saylor|microstrategy|bitcoin|btc|treasury company|phong le)\b",
    ),
    "circle": NameRule(
        negative=r"\b(?:the|a|full|inner|vicious|virtuous|arctic|winner'?s|family|social|trading|close|tight|"
                 r"circle k|target)\s+circles?\b|\bcircles?\b(?= (?:back|around|the|of))|\bcircle (?:k|back|of)\b",
        cues=r"\b(?:CRCL|usdc|stablecoins?|allaire|eurc|arc blockchain)\b",
    ),
    "bullish": NameRule(
        negative=r"\b(?:turns?|turned|remain|remains|stays?|stayed|is|are|was|were|gets?|very|more|most|less|"
                 r"so|too|still|super|ultra|extremely|incredibly|getting|looks?|looking|sounds?|seems?|feel|feels|"
                 r"not|stay|be|being|been|very|quite|cautiously|increasingly|wildly)\s+bullish\b|"
                 r"\bbullish (?:on|about|case|signal|signals|sentiment|pattern|trend|momentum|outlook|view|call|"
                 r"calls|bets?|traders?|investors|engulfing|divergence|setup|flag|reversal|crossover|bias|stance|"
                 r"thesis|tone|move|run|price action|breakout|options|flow|fresh pick|catalyst)\b",
        cues=r"\b(?:BLSH|bullish exchange|bullish inc|coindesk|tom farley)\b",
    ),
    "intel": NameRule(
        negative=r"\b(?:the|some|good|bad|military|russian|israeli|ukrainian|us|u\.s\.|actionable|fresh|new|"
                 r"latest|insider|market|threat|open-source|human)\s+intel\b|\bintel (?:on|about|from|suggests|"
                 r"report|reports|sharing|officials?|agencies|community|chief|services?)\b",
        cues=r"\b(?:INTC|chips?|chipmaker|foundry|foundries|lip-bu tan|gelsinger|x86|cpus?|pc|18a|14a|fabs?)\b",
    ),
    "arm": NameRule(
        negative=r"\b(?:the|an|his|her|their|its|left|right|robotic|robot|investment|venture|lending|financing|"
                 r"armed|strong|twist|broken|military|political|enforcement|research|retail|consumer|asset "
                 r"management|wealth|brokerage|trading|insurance|services|banking|credit|marketing|sales|security)\s+arms?\b|"
                 r"\barms? (?:race|deal|deals|sales|sale|control|length|embargo|dealers?|exports?|shipments?|"
                 r"trade|makers?|industry|manufacturers?|stockpile|cache|supply|supplies)\b|\barms\b|\barmed\b",
        cues=r"\b(?:softbank|rene haas|haas|chip designs?|royalt(?:y|ies)|architecture|licens\w+|cpus?)\b",
        products=("Holdings",),
    ),
    "unity": NameRule(
        negative=r"\b(?:national|party|christian|european|global|in|of|with|show of|call for|calls for|sense of|"
                 r"government of|family|community|racial|social)\s+unity\b|\bunity (?:government|day|rally|"
                 r"party|candle|march|in|among|between|of)\b",
        cues=r"\b(?:game engine|unity software|bromberg|ironsource|vector|grow|developers)\b",
        products=("Software",),
    ),
    "zoom": NameRule(
        negative=r"\bzoom (?:in|out|into|lens|meeting|meetings|call|calls)\b|\b(?:on|via|over|a) zoom\b|"
                 r"\bzoom(?:s|ed|ing)\b",
        cues=r"\b(?:zoom communications|zoom video|eric yuan|yuan|workvivo)\b",
    ),
    "carnival": NameRule(
        negative=r"\b(?:a|the|street|school|church|notting hill|rio|brazil|venice|winter|summer|fall|town|"
                 r"county|annual)\s+carnivals?\b|\bcarnival (?:rides?|games?|barker|atmosphere|season|parade|"
                 r"food|of)\b|\bcarnivals\b",
        cues=r"\b(?:CCL|CUK|cruise|cruises|cruise line|weinstein|royal caribbean|norwegian cruise|bookings|"
             r"net yields)\b",
    ),
    "progressive": NameRule(
        negative=r"\b(?:a|the|more|most|very|so|too|left-wing|liberal|democratic|young|house|senate|"
                 r"self-described)\s+progressives?\b|\bprogressive (?:caucus|democrats?|politics|policies|"
                 r"policy|left|agenda|movement|candidates?|activists?|groups?|values|voters|lawmakers|rock|tax|"
                 r"disease|supranuclear|overload|web app|jpeg|lenses|era)\b|\bprogressives\b|\bprogressively\b",
        cues=r"\b(?:PGR|insurer|auto insurance|insurance|premiums|policies in force|combined ratio|tricia griffith)\b",
    ),
}

# Industry -> context words that confirm an ambiguous name in a neutral mention.
_INDUSTRY_CUES: tuple[tuple[re.Pattern[str], str], ...] = tuple((re.compile(k, re.IGNORECASE), v) for k, v in (
    (r"retail|store|apparel|department|grocery|discount",
     r"retail(?:er|ers)?|shoppers?|stores?|holiday|prices?|same-store|comparable sales|consumers?|merchandise"),
    (r"semiconductor", r"chips?|semiconductors?|gpus?|ai|data cent(?:er|re)s?|foundry|wafers?"),
    (r"software|internet|interactive media|information technology|computer",
     r"ai|apps?|users|cloud|software|platform|subscribers?|advertis\w+|ads"),
    (r"consumer electronics", r"devices?|smartphones?|hardware|iphone|ai"),
    (r"bank|credit services|capital markets|financial|insurance|asset management",
     r"bank|banking|lending|loans?|deposits?|payments?|fintech|credit|bitcoin|crypto|merchants?"),
    (r"auto", r"vehicles?|cars?|evs?|electric vehicles?|deliveries|autonomous|robotaxi"),
    (r"oil|gas|energy", r"oil|gas|crude|barrels?|refin\w+|lng|drilling|output"),
    (r"drug|biotech|pharma|medical|health", r"drugs?|fda|trials?|patients|therap\w+|vaccines?|approval"),
    (r"airline", r"flights?|airlines?|passengers?|travel|fares?|aircraft"),
    (r"travel|leisure|lodging|resort|cruise", r"cruises?|travel|bookings?|guests|hotels?|ships?"),
))

# Minimal brand cues (products/executives) for frequently analysed tickers whose
# names are not ambiguous but whose news often omits the name.
_TICKER_CUES: dict[str, str] = {
    "AAPL": r"\b(?:iphones?|ipads?|macbooks?|tim cook|ternus|vision pro|airpods|app store|apple watch|"
            r"apple intelligence)\b",
    "META": r"\b(?:zuckerberg|instagram|whatsapp|reality labs|llama \d|oculus|ray-ban meta)\b",
    "XYZ": r"\b(?:cash app|afterpay|square (?:pos|terminal|sellers?))\b",
    "TGT": r"\b(?:brian cornell|fiddelke|target circle)\b",
    "ORCL": r"\b(?:larry ellison|safra catz|oracle cloud)\b",
    "NVDA": r"\b(?:jensen huang|geforce|cuda|blackwell|rubin|hopper|h100|h200|b200|gb200|dgx)\b",
    "TSLA": r"\b(?:elon musk|musk|cybertruck|model [3ysx]|robotaxi|fsd|full self-driving|optimus|gigafactory)\b",
    "MSFT": r"\b(?:satya nadella|nadella|azure|copilot|xbox|windows|openai)\b",
    "GOOGL": r"\b(?:sundar pichai|pichai|youtube|waymo|gemini|deepmind|android)\b",
    "GOOG": r"\b(?:sundar pichai|pichai|youtube|waymo|gemini|deepmind|android)\b",
    "AMZN": r"\b(?:andy jassy|jassy|aws|bezos|kuiper|alexa)\b",
}


# --------------------------------------------------------------------------- #
# Compiled per-company matcher
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _NameVariant:
    text: str
    pattern: re.Pattern[str]
    ambiguous: bool
    strong: bool  # full legal name ("Target Corporation"): always the company
    rule: NameRule | None


@dataclass(frozen=True)
class _Matcher:
    ticker: str
    symbols: tuple[str, ...]  # base + share-class variants
    cashtag: re.Pattern[str]
    qualified: re.Pattern[str]
    bare_upper: re.Pattern[str] | None
    bare_any: re.Pattern[str] | None
    soft_ticker: bool
    names: tuple[_NameVariant, ...]
    cues: re.Pattern[str] | None
    brand_cues: re.Pattern[str] | None
    industry_cues: re.Pattern[str] | None
    own_words: frozenset[str]  # lower-case words that belong to the company names


@dataclass
class RelevanceResult:
    """Score plus the evidence behind it (for debugging and tests)."""

    score: float
    evidence: list[str] = field(default_factory=list)
    roundup: bool = False
    secondary: bool = False


def _clean_name(name: str) -> str:
    out = fold(name).strip()
    for _ in range(3):
        stripped = _SUFFIX_STRIP_RE.sub("", out).strip(" ,.")
        if stripped == out:
            break
        out = stripped
    return out


def _name_pattern(text: str, case_sensitive: bool) -> re.Pattern[str]:
    escaped = re.escape(text).replace(r"\ ", r"\s+")
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![\w&$@#])(?:{escaped})(?![\w&]|\.\w)", flags)


def _is_ambiguous(name: str) -> bool:
    low = name.lower()
    if low in AMBIGUOUS_NAMES or low in NAME_RULES:
        return True
    return len(low.split()) == 1 and len(low) <= 3  # "Arm", "Box", "Sea"…


def _symbol_variants(company: CompanyRef) -> tuple[str, ...]:
    base = fold(company.base_symbol).upper()
    out = {base}
    for sep in ("-", ".", "/"):
        if "-" in base or "." in base:
            out.add(re.sub(r"[-./]", sep, base))
    return tuple(sorted(out, key=len, reverse=True))


@lru_cache(maxsize=256)
def _matcher_for(ticker: str, name: str, short_name: str, aliases: tuple[str, ...], quote_type: str,
                 industry: str, sector: str, base_symbol: str) -> _Matcher:
    company = CompanyRef(ticker=ticker, name=name, short_name=short_name, aliases=list(aliases),
                         quote_type=quote_type, industry=industry or None, sector=sector or None)
    symbols = _symbol_variants(company)
    sym_alt = "|".join(re.escape(s) for s in symbols)
    crypto_suffix = r"(?:\.X|-USD)?" if company.is_crypto else ""
    cashtag = re.compile(rf"(?<![\w$])\$(?:{sym_alt}){crypto_suffix}(?![\w])", re.IGNORECASE)
    qualified = re.compile(_QUALIFIED_RE_T.format(t=f"(?:{sym_alt})"))
    word_like = base_symbol in WORD_TICKERS or len(base_symbol) <= 2
    soft = base_symbol in SOFT_WORD_TICKERS
    bare_upper = None if word_like else re.compile(rf"(?<![\w$#@/.-])(?:{sym_alt})(?![\w]|-\w|\.\w)")
    bare_any = None
    if not word_like and not soft and len(base_symbol) >= 3 and base_symbol.lower() not in AMBIGUOUS_NAMES:
        bare_any = re.compile(rf"(?<![\w$#@/.-])(?:{sym_alt})(?![\w]|-\w|\.\w)", re.IGNORECASE)

    raw_names: list[tuple[str, bool]] = []  # (variant, strong)
    full = fold(name).strip()
    cleaned_full = _clean_name(full)
    if full and full != cleaned_full and re.search(_CORP_SUFFIX, full, re.IGNORECASE):
        raw_names.append((re.sub(r"\.$", "", full), True))  # "Target Corporation"
        short_full = re.sub(r",?\s+(?:Class|Series)\s+[A-C]\b.*$", "", full)
        raw_names.append((re.sub(r"\.$", "", short_full), True))
    for candidate in (short_name, cleaned_full, *aliases):
        cand = _clean_name(candidate or "")
        if not cand:
            continue
        if company.quote_type in {"ETF", "MUTUALFUND"}:
            fund = _FUND_WORDS_RE.sub("", _FUND_ISSUERS_RE.sub("", cand)).strip()
            if fund and (len(fund.split()) >= 2 or re.search(r"\d|[A-Z]{2,}", fund)):
                raw_names.append((fund, False))
        raw_names.append((cand, False))
    if cleaned_full.lower() != (short_name or "").lower() and len(cleaned_full.split()) >= 2:
        raw_names.append((cleaned_full, not _is_ambiguous(cleaned_full)))

    seen: set[str] = set()
    variants: list[_NameVariant] = []
    for variant, strong in sorted(raw_names, key=lambda v: (-len(v[0]), not v[1])):
        key = variant.lower()
        if key in seen or len(variant) < 2 or variant.upper() == base_symbol and word_like:
            continue
        seen.add(key)
        ambiguous = not strong and _is_ambiguous(variant)
        variants.append(_NameVariant(
            text=variant,
            pattern=_name_pattern(variant, case_sensitive=ambiguous),
            ambiguous=ambiguous,
            strong=strong,
            rule=NAME_RULES.get(key) or NAME_RULES.get(_clean_name(variant).lower()),
        ))

    cue_parts = [v.rule.cues for v in variants if v.rule and v.rule.cues]
    if base_symbol in _TICKER_CUES:
        cue_parts.append(_TICKER_CUES[base_symbol])
    cues = re.compile("|".join(f"(?:{c})" for c in cue_parts), re.IGNORECASE) if cue_parts else None
    brand = _TICKER_CUES.get(base_symbol)
    brand_cues = re.compile(brand, re.IGNORECASE) if brand else None
    industry_text = f"{industry} {sector}"
    ind = [v for k, v in _INDUSTRY_CUES if industry_text.strip() and k.search(industry_text)]
    industry_cues = re.compile(rf"\b(?:{'|'.join(ind)})\b", re.IGNORECASE) if ind else None
    own_words = frozenset(w for v in variants for w in re.findall(r"[a-z0-9]+", v.text.lower()))
    return _Matcher(
        ticker=ticker, symbols=symbols, cashtag=cashtag, qualified=qualified, bare_upper=bare_upper,
        bare_any=bare_any, soft_ticker=soft, names=tuple(variants), cues=cues, brand_cues=brand_cues,
        industry_cues=industry_cues,
        own_words=own_words,
    )


def _matcher(company: CompanyRef) -> _Matcher:
    return _matcher_for(
        company.ticker, company.name or "", company.short_name or "", tuple(company.aliases or ()),
        company.quote_type or "EQUITY", company.industry or "", company.sector or "", fold(company.base_symbol).upper(),
    )


# --------------------------------------------------------------------------- #
# Mention classification
# --------------------------------------------------------------------------- #
def _negative_spans(text: str, rule: NameRule | None) -> list[tuple[int, int]]:
    if not rule or not rule.negative:
        return []
    return [m.span() for m in re.finditer(rule.negative, text, re.IGNORECASE)]


def _classify(text: str, start: int, end: int, variant: _NameVariant, matcher: _Matcher,
              neg_spans: list[tuple[int, int]]) -> int:
    """+1 company, -1 not the company, 0 undecided (ambiguous names only)."""
    if any(s <= start < e for s, e in neg_spans):
        return -1  # curated false friend ("H&R Block", "price target", "apple cider")
    if variant.strong:
        return 1
    after = text[end:end + 60]
    before = text[max(0, start - 60):start]
    paren = _PAREN_TICKER_RE.match(after)
    if paren and paren.group(1).upper().rstrip(".") in matcher.symbols:
        return 1
    if re.match(_CORP_SUFFIX, after, re.IGNORECASE):
        return 1
    if not variant.ambiguous:
        return 1
    if after.startswith("-"):
        nxt = re.match(r"-([A-Za-z]+)", after)
        if not nxt or nxt.group(1).lower() not in _HYPHEN_OK:
            return -1
    if start > 0 and text[start - 1] == "-":
        return -1
    next_word = re.match(r"\s+([A-Za-z]+)", after)
    products = variant.rule.products if variant.rule else ()
    if _FOLLOWER_RE.match(after) or (next_word and next_word.group(1) in products):
        return 1
    prev = re.search(r"([A-Za-z$][\w'$.]*)\s+$", before)
    if prev and prev.group(1).lower().removesuffix("'s") in DETERMINERS:
        return -1
    if _POSSESSIVE_RE.match(after) or _STRONG_BEFORE_RE.search(before):
        return 1
    # Subject followed by a verb: "Target Slashes Prices", "jury says Apple owes".
    clause_start = _CLAUSE_START_RE.search(before) is not None
    if (clause_start and _SUBJECT_VERB_RE.match(after)) or _MID_VERB_RE.match(after):
        return 1
    return 0


def _other_subject_first(text: str, first_pos: int, matcher: _Matcher) -> bool:
    """True when another company is the grammatical subject before our first mention."""
    for m in _CASHTAG_RE.finditer(text):
        if m.start() >= first_pos:
            break
        sym = m.group(1).upper().split(".")[0]
        if sym not in matcher.symbols:
            return True
    m = _OTHER_SUBJECT_RE.match(text)
    if m and m.start(1) < first_pos:
        words = set(re.findall(r"[a-z0-9]+", m.group(1).lower()))
        if (words and not words & matcher.own_words and not words & _GENERIC_SUBJECT_WORDS
                and m.group(1).upper() not in matcher.symbols):
            return True
    return False


def _is_roundup(text: str) -> bool:
    tags = {m.group(1).upper() for m in _CASHTAG_RE.finditer(text)}
    if len(tags) >= 4:
        return True
    return bool(_LISTICLE_RE.search(text) or _COMMA_LIST_RE.search(text) or _TICKER_LIST_RE.search(text))


def explain_relevance(text: str, company: CompanyRef) -> RelevanceResult:
    """Relevance score with the evidence that produced it."""
    if not text or not company:
        return RelevanceResult(0.0)
    t = fold(text)
    matcher = _matcher(company)
    shouting = is_mostly_upper(t)
    evidence: list[str] = []
    positions: list[int] = []
    score = 0.0
    mentions = 0

    for m in matcher.cashtag.finditer(t):
        score = max(score, 1.0)
        positions.append(m.start())
        mentions += 1
        evidence.append(f"cashtag {m.group(0)}")
    for m in matcher.qualified.finditer(t):
        score = max(score, 0.95)
        positions.append(m.start())
        mentions += 1
        evidence.append(f"qualified ticker {m.group(0).strip()}")
    if score < 0.9:
        bare = None
        if matcher.bare_upper and not (matcher.soft_ticker and shouting):
            bare = matcher.bare_upper
        if matcher.bare_any and not shouting:
            bare = matcher.bare_any
        if bare:
            for m in bare.finditer(t):
                score = max(score, 0.9)
                positions.append(m.start())
                mentions += 1
                evidence.append(f"ticker {m.group(0)}")

    name_level = 0.0
    any_neutral = False
    taken: list[tuple[int, int]] = []
    for variant in matcher.names:  # longest variants first
        pattern = variant.pattern
        if variant.ambiguous and shouting:
            pattern = re.compile(pattern.pattern, re.IGNORECASE)
        neg_spans = _negative_spans(t, variant.rule)
        for m in pattern.finditer(t):
            if any(s <= m.start() < e for s, e in taken):
                continue
            taken.append(m.span())
            verdict = _classify(t, m.start(), m.end(), variant, matcher, neg_spans)
            if verdict > 0:
                mentions += 1
                positions.append(m.start())
                lead = t[:m.start()]
                primary = len(lead.split()) <= 2 or re.match(r"^[^:]{0,40}:\s*$", lead) is not None
                level = 0.9 if primary else 0.8
                if t[m.end():m.end() + 1] == "-":  # "Nvidia-backed CoreWeave": a modifier, not the subject
                    level, primary = 0.6, False
                name_level = max(name_level, level)
                evidence.append(f"name '{m.group(0)}'" + (" (subject)" if primary else ""))
            elif verdict == 0:
                any_neutral = True
                positions.append(m.start())
                evidence.append(f"ambiguous '{m.group(0)}'")
            else:
                evidence.append(f"not-company '{m.group(0)}'")

    if name_level == 0.0 and any_neutral:
        masked = t
        for variant in matcher.names:
            for s, e in _negative_spans(masked, variant.rule):
                masked = masked[:s] + " " * (e - s) + masked[e:]
        if matcher.cues and matcher.cues.search(t):
            name_level = 0.7
            evidence.append("brand cue")
        elif _FINANCE_CONTEXT_RE.search(masked) or (matcher.industry_cues and matcher.industry_cues.search(masked)):
            name_level = 0.55
            evidence.append("finance context")
        else:
            name_level = 0.25
    elif name_level == 0.0 and score == 0.0 and matcher.brand_cues:
        cue = matcher.brand_cues.search(t)
        if cue:
            name_level = 0.5
            positions.append(cue.start())
            evidence.append(f"cue '{cue.group(0)}'")

    if score and name_level:
        score = min(1.0, max(score, name_level) + 0.05)
    else:
        score = max(score, name_level)
    if mentions >= 2 and score < 0.95:
        score = min(0.95, score + 0.05)

    result = RelevanceResult(score=score, evidence=evidence)
    if score > 0 and positions:
        if _other_subject_first(t, min(positions), matcher):
            result.secondary = True
            score *= 0.75
            evidence.append("another company is the subject")
        if _is_roundup(t):
            result.roundup = True
            score = min(score, 0.4)
            evidence.append("roundup/listicle")
    result.score = round(score, 3)
    return result


def relevance(text: str, company: CompanyRef) -> float:
    """0..1: how clearly `text` is about `company` (see module docstring)."""
    return explain_relevance(text, company).score


@lru_cache(maxsize=256)
def _terms_for(ticker: str, name: str, short_name: str, aliases: tuple[str, ...]) -> frozenset[str]:
    words: set[str] = set()
    base = ticker.split("-")[0].split(".")[0].lower()
    words.update({ticker.lower(), base, f"${base}"})
    for value in (name, short_name, *aliases):
        cleaned = _clean_name(value or "").lower()
        for w in re.findall(r"[a-z0-9&]+", cleaned):
            if len(w) > 1:
                words.add(w)
    return frozenset(words)


def company_terms(company: CompanyRef | None) -> frozenset[str]:
    """Lower-case tokens that name the company (ticker, cashtag, name words,
    aliases). Used to keep the company's own name out of keywords/narrative
    features."""
    if company is None:
        return frozenset()
    return _terms_for(company.ticker, company.name or "", company.short_name or "", tuple(company.aliases or ()))
