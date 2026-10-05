"""How clearly is a text about *this* company?  `relevance(text, company)` -> 0..1.

Broad keyword searches return plenty of noise: "price target" headlines about
other stocks for Target, "Democrats block bill" for Block, "apple cider" for
Apple, "meta-analysis" for Meta, five-ticker StockTwits spam. The scorer
weighs evidence per mention instead of string-matching:

    1.00  cashtag ($NVDA, $BTC.X)
    0.95  exchange-qualified / parenthesized ticker ("(NASDAQ: NVDA)", "(TGT)")
    0.90  bare ticker token (never for word-like tickers: ALL, IT, ON, NOW, A, T, F …,
          except right before "stock"/"shares": "MU stock soars")
    0.85  social tag ("#NVDA", "#Bitcoin", "$NVIDIA")
    0.80  company name used as a company (+0.10 when it is the headline subject)
    0.60  index funds (SPY, QQQ): market-wide news ("Stocks Settle Higher as …")
    0.55  ambiguous common-word name with finance/industry context but no
          decisive cue; 0.25 without context; ~0 when every mention is a
          non-company sense (collocations like "price target", "block party")
    ≤0.45 named only as context for another entity ("Tesla rival Nikola files
          for bankruptcy", "… after delays in AT&T deal")
    ×0.75 when another company is the subject ("Cerebras stock … on Nvidia pressure")
    0     a separately listed sister company sharing the brand ("Toyota Industries",
          "Vodafone Idea", "Meta Materials") — not a mention of the company at all
    ≤0.40 roundups/listicles (≥ 4 cashtags, "3 AI Chip Stocks To Watch", "X, Y, Z and More"),
          unless the company leads the headline ("Bitcoin beats Gold, SPY, Silver, QQQ");
          0.30 when it is one tag in a hashtag soup

Common-word names (Apple, Target, Meta, Block, Snap, Visa, Shell, Amazon,
Oracle, Ford …) are matched case-sensitively and need positive context
(possessive, "Target (TGT)", "Block stock", subject + verb, analyst verb before
it, product/executive cues) to score high; curated negative collocations knock
out known false friends (incl. medical "wet AMD").

Brokerages (JPMorgan, Goldman Sachs, Morgan Stanley …) are mostly named as the
*author* of research on other stocks ("JPMorgan downgrades PepsiCo", "Target
lowered at JPMorgan", "Morgan Stanley warns of AI bubble"); such mentions don't
count as news about the firm — on a real JPM feed this lifts precision from
0.62 to 0.98 at full recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.nlp.events import is_known_firm
from app.nlp.text import COMMON_HEADLINE_WORDS, HEADLINE_VERBS, MOVE_WORDS, STOPWORDS, fold, is_mostly_upper, \
    is_title_case, wordset
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

# Word-like tickers that are also common UPPER-CASE acronyms ("SNAP benefits",
# "ICE raids", "CAT scan"): a bare match needs market vocabulary nearby.
ACRONYM_TICKERS: frozenset[str] = wordset("""
SNAP ICE CAT NET PATH ROOT OPEN BOX RIDE RENT CART MATCH GAP BALL ROCK SHOP PLUG BIRD NOVA DASH NOW AI
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

# Ceiling for a company named only as context for another entity ("Tesla rival
# Nikola files for bankruptcy"): kept as background (above the 0.35 feed cut) but
# below the 0.5 that event insights require.
CONTEXT_ONLY_MAX = 0.45

# Common given names: "Katrina Ford", "Priscilla Block" are people, not companies.
FIRST_NAMES = wordset("""
aaron adam alan albert alex alexander alice amanda amy andrew angela anna anne anthony ashley barbara ben benjamin
betty beth bill billy bob bobby brad brandon brenda brian bruce carl carol caroline catherine charles charlie chris
christina christine christopher cindy claire craig cynthia dan daniel danny david deborah debra dennis diana diane
donald donna doug douglas dylan edward elizabeth ellen emily emma eric erin ethan eugene frank fred gary george
gerald grace greg gregory hannah harold harrison harry heather helen henry jack jacob james jamie jane janet jason
jeff jeffrey jennifer jeremy jerry jessica jill jim jimmy joan joe john johnny jon jonathan jordan jose joseph joshua
joyce judith judy julia julie justin karen kate katherine kathleen kathy katie katrina keith kelly ken kenneth kevin
kim kimberly kyle larry laura lauren lawrence linda lisa lori louis luke lynn margaret maria marie marilyn mark martha
martin mary matt matthew megan melissa michael michelle mike nancy natalie nathan nicholas nicole noah olivia pamela
patricia patrick paul peter philip priscilla rachel ralph randy raymond rebecca richard rick rob robert roger ronald
rose roy russell ruth ryan sam samantha samuel sandra sara sarah scott sean sharon shirley sophia stephanie stephen
steve steven susan teresa terry thomas tim timothy tina todd tom tony tyler victoria vincent virginia walter wayne
william zachary francis troy coppola stallone sylvester harvey leonard lloyd marcus maurice morgan nick nolan oscar
otis owen preston quentin reggie rex rodney roland ross russ sebastian seth shane spencer stanley stuart ted theodore
toby trevor tyson vince warren wendell wesley whitney willie xavier zach zoe chloe ella grace hazel isla ivy jasmine
kayla kylie leah lily lucy maya mia nora paige piper quinn riley ruby sadie stella tessa violet willow amber april
autumn brooke carly dana darcy eden faith gemma hope iris jade june kara lacey leslie mabel nadia opal pearl rae
reese sienna summer tara vera wren yvonne
""")
DETERMINERS = wordset("""the a an its their his her our your this these those my own new every each any no some
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
    r"smartphones?|phones?|earnings call|growth|margins?|buyback|dividend|turnaround|strategy|leadership|"
    r"data cent(?:er|re)s?|glasses|headsets?|"
    r"price targets?|price objectives?|target prices?|pts?|ratings?|short interest"
)
_FOLLOWER_RE = re.compile(rf"^\s+(?:{_COMPANY_FOLLOWERS})\b(?!-)", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"^'s\b")
# Sponsored venues and events: "SoFi Stadium", "Target Center", "Chase Center",
# "Citi Field", "Wells Fargo Championship" are places, not the company.
_VENUE_AFTER_RE = re.compile(r"^\s+(?-i:Stadium|Arena|Center|Centre|Field|Park|Bowl|Amphitheat(?:er|re)|"
                             r"Theat(?:er|re)|Dome|Coliseum|Pavilion|Forum|Garden|Gardens|Ballpark|Speedway|"
                             r"Championship|Invitational)\b")
_PAREN_TICKER_RE = re.compile(r"^\s*\((?:[A-Z]{2,12}\s?:\s?)?([A-Z][A-Z0-9.\-]{0,9})(?:\.[A-Z]{1,3})?\)")
# Separately listed companies that share the brand: "Toyota Industries", "Toyota
# Tsusho", "Vodafone Idea", "Mitsubishi Heavy", "Samsung Biologics". The word
# after the brand names another issuer — unless it is part of the company's own
# name ("Toyota Motor" for Toyota Motor) or its own industry ("Dow Chemical" for
# Dow) — so the mention is not the company.
_SISTER_WORDS = wordset("""industries tsusho boshoku shatai heavy chemical chemicals chem steel estate realty fudosan
hospitality materials biologics mobis glovis hynix innotek uplus healthineers vernova otosan hexacom finserv""")
_BRAND_SISTERS: dict[str, frozenset[str]] = {
    "toyota": wordset("motor motors industries tsusho boshoku"),
    "vodafone": wordset("idea"),
    "mitsubishi": wordset("ufj heavy electric estate chemical motors"),
    "mitsui": wordset("fudosan osk chemicals"),
    "sumitomo": wordset("mitsui chemical metal realty electric"),
    "samsung": wordset("electronics sdi biologics heavy life fire securities sds"),
    "hyundai": wordset("motor mobis steel glovis heavy engineering"),
    "tata": wordset("motors steel power consultancy chemicals elxsi communications"),
    "adani": wordset("enterprises ports power green energy transmission total wilmar"),
    "reliance": wordset("power infrastructure infra capital communications home"),
    "bajaj": wordset("finance finserv auto housing"),
    "siemens": wordset("energy healthineers"),
    "alibaba": wordset("health pictures"),
    "tencent": wordset("music"),
    "airtel": wordset("africa hexacom"),
    "sony": wordset("financial"),
    "daimler": wordset("truck"),
    "nissan": wordset("chemical"),
    "berkshire": wordset("hills"),
}
# Words after a brand that never start another issuer's name ("Toyota ADR", "Alphabet Class A").
_NOT_SISTER = wordset("class series adr adrs ads gdr gdrs ordinary preferred pref common unit units")
# After "<Brand> <Word>": legal suffixes always mark an issuer; "stock"/"shares" do so
# only in sentence case, where a capitalized word is a name ("Vodafone Idea shares").
_SISTER_LEGAL_AFTER_RE = re.compile(r"^\s+(?:Ltd|Limited|plc|PLC|Inc|Corp|Corporation|Co|Company|AG|SA|NV|SE|Bhd|"
                                    r"Tbk|ASA|AB|Oyj|K\.?K)\b")
_SISTER_SHARES_AFTER_RE = re.compile(r"^\s+(?:stock|stocks|shares|share price|shareholders|stockholders|ipo)\b")
_NEXT_NAME_RE = re.compile(r"^\s+([A-Za-z][\w&]*(?:&[A-Za-z]+)?)")
_HYPHEN_OK = wordset("backed owned led based made branded related linked focused funded parent maker rival supplier "
                       "partner like style designed built powered approved listed")
# "Nvidia-backed CoreWeave", "Tesla-like margins": the name qualifies something else.
_HYPHEN_MODIFIER_RE = re.compile(r"^-(?:backed|owned|funded|led|linked|related|supported|partnered|affiliated|like|"
                                 r"style|rival|supplier|partner|sized|fueled|driven|powered|exposed)\b", re.IGNORECASE)
# Appositives: "Tesla rival Nikola", "Super Micro, a key Nvidia partner", "Nvidia's
# server partner Wistron". Up to two descriptive words may sit in between, never
# a conjunction or verb ("Apple and its suppliers" is still about Apple).
_ROLE_AFTER_RE = re.compile(
    r"^(?:'s)?\s+(?:(?!(?:and|or|its|their|his|her|with|to|for|of|is|are|was|were|has|have|says?|said|will|"
    r"shares|stock)\b)[\w-]+\s+){0,2}?(?:rivals?|peers?|competitors?|challengers?|suppliers?|vendors?|partners?|"
    r"licensees?|investees?|backers?|allies|ally|contractors?|portfolio compan(?:y|ies)|spin-?offs?|"
    r"customers?|clients?)\b"
    r"(?!\s+(?:with|on|in|to|for|program|programs|network|summit|day|event|portal|conference)\b)",
    re.IGNORECASE,
)
# "..., a key Nvidia partner" / "..., Nvidia server partner": the appositive
# describes the entity before the comma.
_APPOSITIVE_BEFORE_RE = re.compile(r"[\w)],\s+(?:(?:a|an|the|another|fellow|longtime|key|major|big)\s+){0,2}$",
                                   re.IGNORECASE)
_NAMED_AFTER_RE = re.compile(r"\s+(\$?(?-i:[A-Z])[\w&.'-]*)")
# Background mentions after the main clause: "... after delays in AT&T spectrum
# deal", "... as AT&T deal stalls", "... amid Apple trade talks".
_ADJUNCT_BEFORE_RE = re.compile(r"\b(?:after|amid|despite|following|as|while|in|over|with|before|since)\s+"
                                r"(?:[\w&'.-]+\s+){0,3}$", re.IGNORECASE)
_ADJUNCT_AFTER_RE = re.compile(r"^(?:'s)?\s+(?:[\w-]+\s+)?(?:deal|deals|transaction|merger|takeover|acquisition|"
                               r"spectrum|contract|tie-up|agreement|partnership|order|orders|bid|talks|"
                               r"negotiations|dispute)\b", re.IGNORECASE)
_VERBISH = HEADLINE_VERBS | MOVE_WORDS | COMMON_HEADLINE_WORDS | STOPWORDS


def _mention_role(text: str, start: int, end: int) -> tuple[bool, bool]:
    """(modifier, adjunct) for a company mention at [start, end) — see Mention.

    A role noun after the name ("Tesla rival", "Nvidia partner") marks a
    modifier only when it describes someone else: another name follows
    ("Tesla rival Nikola files …") or it is an appositive after a comma
    ("Super Micro, a key Nvidia partner"). "Apple's suppliers face tariffs"
    and "Tesla rivals struggle" stay about the company."""
    after = text[end:end + 60]
    modifier = bool(_HYPHEN_MODIFIER_RE.match(after))
    role = None if modifier else _ROLE_AFTER_RE.match(after)
    if role:
        if _APPOSITIVE_BEFORE_RE.search(text[max(0, start - 40):start]):
            modifier = True
        else:
            named = _NAMED_AFTER_RE.match(after[role.end():])
            word = named.group(1).lower() if named else ""
            modifier = bool(named) and (not is_title_case(text) or word.lstrip("$") not in _VERBISH)
    adjunct = False
    if not modifier and len(text[:start].split()) >= 3:
        clauses = re.split(r"[.;:!?|]\s|\s-\s", text[:start])
        clause = clauses[-1]
        lead = re.match(r"\s*([A-Z][\w&.'-]*)", clause)
        adjunct = bool(_ADJUNCT_BEFORE_RE.search(clause) and _ADJUNCT_AFTER_RE.match(after) and lead
                       and lead.group(1).lower() not in _GENERIC_LEAD_WORDS)
        # "Dish DBS files for bankruptcy protection; AT&T transaction delayed": the
        # company's deal is the backdrop of another entity's story.
        first = re.match(r"\s*([A-Z][\w&.'-]*)", text)
        adjunct = adjunct or bool(len(clauses) > 1 and not clause.strip() and _ADJUNCT_AFTER_RE.match(after) and first
                                  and first.group(1).lower() not in _GENERIC_LEAD_WORDS
                                  and first.start(1) != start)
    return modifier, adjunct


# Analyst/transaction verbs immediately before a name: "HSBC upgrades Target".
_STRONG_BEFORE_RE = re.compile(
    r"(?:upgrades?|downgrades?|upgraded|downgraded|initiates?(?: coverage)?(?: on| of)?|coverage (?:on|of)|"
    r"(?:rating|stance|view) on|reiterates? \w+ on|maintains? \w+ on|buy|buying|sell|selling|short|shorting|long|"
    r"own|owning|bought|sold|vs\.?|versus|sues?|sued|suing|acquires?|acquiring|backs|likes|prefers|favors|"
    r"names|picks|loves|hates|dumps|trims|adds|boosts stake in|cuts stake in|praises|beats|joins|"
    r"(?:invest(?:ed|ing|s)?|stake|position|bet|bets|betting|exposure) in|shares of|stock of|bullish on|bearish on|"
    r"(?:loading|loads|load|loaded) up on|pil(?:es|ing|ed) into|bets? on|betting on|long on|short on|"
    # legal and regulatory actions taken against the company
    r"fines?|fined|penali[sz]es|penali[sz]ed|charges?|charged|accuses?|accused|probes?|probed|"
    r"investigates?|investigated|blocks?|blocked|bans?|banned|"
    r"(?:penalt(?:y|ies)|damages|fine|settlement|payment|lawsuit|suit|claims?) (?:from|against)|"
    r"(?:jury|judge|court|regulators?) (?:finds|found|rules|ruled|orders|ordered)|catches up with|against)\s*$",
    re.IGNORECASE,
)
# "wants Meta to pay $40B", "orders Apple to open its App Store": the name is the
# object of a demand.
_DEMAND_BEFORE_RE = re.compile(r"\b(?:wants|asks|urges|forces|forced|requires|required|tells|told|orders|ordered|"
                               r"pushes|pushed|presses|pressed|compels|compelled)\s*$", re.IGNORECASE)
# Coordinated subjects: "Ford and JPMorganChase launch", "Ford, JPMorgan Chase and
# Michigan establish $3B initiative".
_COORDINATED_RE = re.compile(r"^\s*(?:,|and|&)\s+(?:(?-i:[A-Z])[\w&.'-]*\s+){1,3}(?:(?:,\s*)?(?:and|&)\s+"
                             r"(?:(?-i:[A-Z])[\w&.'-]*\s+){1,3})?(?=\S)", re.IGNORECASE)
_PLURAL_VERB_RE = re.compile(
    r"^(?:launch|establish|form|sign|team|partner|unveil|announce|agree|join|build|create|open|plan|win|face|settle|"
    r"sue|recall|cut|raise|invest|expand|hike|slash|report|post|beat|miss|strike|ink|back|bet|push|take|make|are|"
    r"have|will|to|say|see|deny|lose|clash|spar|compete|race|merge|unite|split|call|warn|weigh|eye|seek|reach|"
    r"[a-z]+ed)\b",
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
# For a capitalized (case-sensitive) name, any past-tense verb after it marks the
# subject: "Overnight, Meta deleted all her accounts".
_PAST_VERB_RE = re.compile(r"^\s+(?-i:[a-z]{3,}ed)\b(?!-)")
# Clause openings: sentence/clause punctuation, or a reported-speech "that"
# ("Analysts say that Apple will raise prices" — not a relative "the buyback
# that Apple unveiled").
_CLAUSE_START_RE = re.compile(
    r"(?:^|[:;.!?|,]\s*|\s-\s|\b(?:says?|said|warn(?:s|ed)?|believes?|thinks?|shows?|means?|argues?|claims?|"
    r"notes?|told \w+)\s+(?:that\s+)?)(?:(?:why|how|what|when|where|will|can|could|should|would|is|does|did|has|"
    r"here's|as|after|while|because|if|but|and|so)\s+){0,2}$",
    re.IGNORECASE,
)

_FINANCE_CONTEXT_RE = re.compile(
    r"\b(?:stocks?|shares?|shareholders?|investors?|earnings|revenue|profit|quarter(?:ly)?|q[1-4]|guidance|outlook|"
    r"analysts?|upgrades?|downgrades?|rating|ceo|cfo|executives?|dividend|buyback|market (?:cap|value)|valuation|"
    r"ipo|acquisition|merger|nyse|nasdaq|wall street|traders?|bullish|bearish|rall(?:y|ies)|plunges?|surges?|"
    r"premarket|after-hours|sec filing|10-[kq]|8-k|eps|margin|forecast|fiscal|retailer|company|firm|brand|market|"
    r"customers?|consumers?|shoppers?|sales|stores?|deal|layoffs|lawsuit|regulators?|antitrust|chips?|ai|"
    r"financing|debt|bonds?|capex|spending|funding|backlog|contracts?|cloud|data cent(?:er|re)s?)\b",
    re.IGNORECASE,
)
# Market vocabulary that confirms a word-like bare ticker ("SNAP shares", "META earnings").
_TICKER_CONTEXT_RE = re.compile(
    r"\b(?:stocks?|shares?|share price|earnings|eps|revenue|guidance|analysts?|price target|upgrades?|"
    r"downgrades?|calls|puts|options|nyse|nasdaq|market cap|investors?|traders?|rall(?:y|ies)|premarket|"
    r"after-hours|short interest|valuation|dividend|buyback|bullish|bearish|ipo|q[1-4]|quarter)\b",
    re.IGNORECASE,
)
# Nouns that confirm a bare word-like/short ticker: "MU stock soars", "GE shares".
_TICKER_NOUNS = r"(?:stock|stocks|shares|calls|puts|earnings|options|price target|short interest)"
# Index ETFs whose news is the market itself ("Stock futures rise", "Wall Street's AI party").
_BROAD_INDEX_RE = re.compile(r"s&p 500|s&p500|nasdaq[- ]?100|nasdaq composite|dow jones industrial|\bdow\b|"
                             r"russell (?:1000|2000|3000)|total (?:stock|us|u\.s\.) market|s&p total market|"
                             r"msci (?:world|acwi|usa)|wilshire 5000|s&p midcap 400|s&p smallcap 600", re.IGNORECASE)
# The market itself as the subject — not a sector ("Chip Stocks Extend Their Run")
# or a backdrop ("Crushed the Market").
_MARKET_WIDE_RE = re.compile(
    r"\b(?:the )?stock market\b|\bwall street(?!'s? (?:analysts?|estimates?|expectations?|forecasts?|targets?|zen))\b|"
    r"\b(?:stock|equity|index|dow(?: jones)?|nasdaq|s&p(?: 500)?) futures\b|\b(?:dow jones|dow|nasdaq|s&p 500|s&p)\b|"
    r"(?:^|[:;,.!?]\s*|\b(?:as|while|but|after|says?|said)\s+)(?:u\.?s\.?\s+|us\s+|global\s+|world\s+)?"
    r"(?:stocks|equities)\s+(?:\w+ly\s+)?(?:rally|rallies|rise|rises|fall|falls|climb|climbs|slide|slides|settle|"
    r"settles|close|closes|end|ends|open|opens|rebound|rebounds|erase|erases|push|pushed|pressured|mixed|higher|lower|"
    r"edge|edges|extend|extends|surge|surges|tumble|tumbles|sink|sinks|drop|drops|gain|gains|slump|slumps|jump|jumps|"
    r"retreat|retreats|waver|wavers|stall|stalls|fall|steady|flat)\b|"
    r"\bbroader market\b|\b(?:bull|bear) market\b|\bmarket (?:correction|crash|rally|selloff|sell-off|rout)\b",
    re.IGNORECASE,
)
_LISTICLE_RE = re.compile(
    r"\b(?:[2-9]|1[0-9]|20|two|three|four|five|six|seven|eight|nine|ten|dozen|several|these)\s+"
    r"(?:(?!hours?|days?|weeks?|months?|years?)[\w&'-]+\s+){0,3}?"
    r"(?:stocks|names|picks|companies|equities|tickers|chipmakers|plays|buys)\b"
    r"|\b(?:[1-9]|1[0-9]|20)\s+(?:[\w&'-]+\s+){0,3}?stock\s+(?:to|that|you|we|i)\b"
    r"|\b(?:stocks to (?:buy|watch|sell|avoid|own)|stocks? that explain|stock movers|biggest (?:movers|moves)|"
    r"(?:midday|premarket|after-hours) movers|top (?:gainers|losers)|most active|trending stocks|"
    r"top midday stories|morning squawk|weekly review|week ahead|market wrap|and more:|and more stocks|"
    r"stocks making the biggest moves|earnings to watch|what to watch)\b",
    re.IGNORECASE,
)
_COMMA_LIST_RE = re.compile(r"(?:\b[A-Z][\w&.'-]*(?:\s[A-Z][\w&.'-]*){0,2},\s+){3,}")
_TICKER_LIST_RE = re.compile(r"(?:\b[A-Z]{2,5},\s*){3,}[A-Z]{2,5}\b")
_HASHTAG_RE = re.compile(r"(?<![\w#])#(\w{2,})")
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
# Headline openers that are not another company ("Why ...", "Stocks ...").
_GENERIC_LEAD_WORDS = _GENERIC_SUBJECT_WORDS | wordset("""stocks shares stock investors analysts analyst wall street
markets market dow nasdaq futures traders here here's there exclusive breaking update watch report reports
earnings shareholders jury judge court""")


@dataclass(frozen=True)
class NameRule:
    """Curated knowledge for one ambiguous brand name."""

    negative: str = ""  # regex (case-insensitive); spans that are NOT the company
    cues: str = ""  # regex (case-insensitive); words that confirm the company anywhere in the text
    products: tuple[str, ...] = ()  # lower-case words that may follow the name ("Apple Watch")


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
        products=("watch", "tv", "music", "pay", "intelligence", "card", "arcade", "store", "silicon", "vision",
                  "maps", "news", "books", "id", "park", "newsroom", "support"),
    ),
    "meta": NameRule(
        negative=r"\bmeta[- ](?:analysis|analyses|analytic|data(?! cent(?:er|re)s?)|description|tags?|review|regression|learning|"
                 r"narrative|commentary|game|strategy|joke|humor|level|cognition)\b|\bmeta materials\b|"
                 r"\bmeta financial\b|\b(?:very|so|too|pretty|kinda) meta\b",
        cues=r"\b(?:facebook|instagram|whatsapp|threads|zuckerberg|reality labs|llama|oculus|quest|ray-ban|"
             r"menlo park|meta ai)\b",
        products=("ai", "platforms", "quest", "connect", "superintelligence", "glasses", "smart", "ray-ban", "muse"),
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
        products=("inc",),
    ),
    "snap": NameRule(
        negative=r"\bsnap (?:election|elections|poll|polls|judgment|judgement|decision|back|shot|chat|benefits?|"
                 r"recipients?|program|programs|cuts?|changes?|rules?|funding|eligibility|enrollment|cards?|costs?|"
                 r"customers|households|participants|work requirements|error rate|fraud|peas|pea|counts?|"
                 r"skid|streak|drought)\b|\b(?:oh|cold|ginger|on|of|food) snap\b|\bsnap(?:s|ped|ping)\b|"
                 r"\bfood stamps\b|\b(?:medicaid|ebt|usda|wic)\b",
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
        products=("web", "prime", "pharmacy", "music", "fresh", "go", "air", "kuiper", "q"),
    ),
    "oracle": NameRule(
        negative=r"\boracle of (?:omaha|delphi)\b|\bthe oracle\b(?! (?:stock|shares|corp))|"
                 r"\boracles?\b(?= (?:network|protocol|price feeds?))|\bchainlink\b",
        cues=r"\b(?:ORCL|ellison|catz|oci|cloud infrastructure|stargate|cerner|database|netsuite|magouyrk|sicilia)\b",
        products=("cloud", "health", "database"),
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
        products=("air",),
    ),
    "united": NameRule(
        negative=r"\bunited (?:states|nations|kingdom|arab|auto workers|steelworkers|way|front|church|"
                 r"healthcare|health group|rentals|parcel|therapeutics|natural|fire|airlines? holdings? stock)\b|"
                 r"\b(?:manchester|leeds|newcastle|west ham|sheffield|dundee|atlanta|dc|minnesota|new england|"
                 r"red bull|hearts? and) united\b",
        cues=r"\b(?:UAL|airlines?|flights?|kirby|mileageplus|carriers?|passengers?|airfare)\b",
        products=("airlines",),
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
        products=("holdings",),
    ),
    "unity": NameRule(
        negative=r"\b(?:national|party|christian|european|global|in|of|with|show of|call for|calls for|sense of|"
                 r"government of|family|community|racial|social)\s+unity\b|\bunity (?:government|day|rally|"
                 r"party|candle|march|in|among|between|of)\b",
        cues=r"\b(?:game engine|unity software|bromberg|ironsource|vector|grow|developers)\b",
        products=("software",),
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
    "amd": NameRule(  # also age-related macular degeneration in medical news
        negative=r"\b(?:wet|dry|neovascular|exudative|non-?exudative|atrophic|early|intermediate|advanced|late)\s+amd\b|"
                 r"\bamd\s+(?:treatments?|therap(?:y|ies)|risk|patients?|progression|lesions?|eyes?|vision|screening|"
                 r"diagnosis|drugs?|care|prevalence|incidence|genetics|biomarkers?)\b|"
                 r"\bmacular degeneration\b[^.]{0,80}?\bamd\b|\bamd\b[^.]{0,80}?\b(?:macular|retina\w*|ophthalm\w*|"
                 r"anti-vegf|geographic atrophy)\b",
        cues=r"\b(?:lisa su|ryzen|radeon|epyc|instinct|xilinx|chips?|chipmaker|semiconductors?|gpus?|cpus?|"
             r"data cent(?:er|re)s?|nvidia|intel)\b",
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
     (r"bank|banking|lending|loans?|deposits?|payments?|fintech|credit|bitcoin|crypto|merchants?|stablecoins?|"
      r"cards?|tokeniz\w+|wallets?|checkout")),
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
    # bank chiefs: "JPMorgan's Dimon warns ..." is the bank's own news, not research
    "JPM": r"\b(?:jamie dimon|dimon)\b",
    "GS": r"\b(?:david solomon|solomon|john waldron|waldron)\b",
    "MS": r"\b(?:ted pick|(?-i:Pick)(?= says| warns| sees| expects))\b",
    "BAC": r"\b(?:brian moynihan|moynihan)\b",
    "C": r"\b(?:jane fraser|fraser)\b",
    "WFC": r"\b(?:charlie scharf|scharf)\b",
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
    acronym_ticker: bool
    names: tuple[_NameVariant, ...]
    cues: re.Pattern[str] | None
    brand_cues: re.Pattern[str] | None
    industry_cues: re.Pattern[str] | None
    own_words: frozenset[str]  # lower-case words that belong to the company names
    broker: bool = False  # a brokerage: its name also appears as the author of research on others
    hashtag: re.Pattern[str] | None = None  # "#NVDA", "#Bitcoin", "$NVIDIA"
    confirmed_bare: re.Pattern[str] | None = None  # short/word-like ticker + "stock": "MU stock soars"
    index_fund: bool = False  # tracks a broad index (SPY, QQQ): market-wide news is about it
    industry_text: str = ""  # lower-case industry + sector ("auto manufacturers consumer cyclical")


@dataclass(frozen=True)
class Mention:
    """One span of the text that names the company.

    `modifier`: the name only qualifies another entity ("Tesla rival Nikola",
    "a key Nvidia partner", "Nvidia-backed CoreWeave"); `adjunct`: it sits in
    background detail after the main clause ("... files for bankruptcy after
    delays in AT&T spectrum deal"). Either way the text is not *about* it."""

    start: int
    end: int
    modifier: bool = False
    adjunct: bool = False

    @property
    def subject_like(self) -> bool:
        return not (self.modifier or self.adjunct)


@dataclass
class RelevanceResult:
    """Score plus the evidence behind it (for debugging and tests).

    `mentions` are positions in `fold(text)` (same length as the input for
    already-folded text), used by event attribution."""

    score: float
    evidence: list[str] = field(default_factory=list)
    roundup: bool = False
    secondary: bool = False
    mentions: list[Mention] = field(default_factory=list)


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
        bare_any=bare_any, soft_ticker=soft, acronym_ticker=base_symbol in ACRONYM_TICKERS, names=tuple(variants), cues=cues, brand_cues=brand_cues,
        industry_cues=industry_cues,
        own_words=own_words,
        broker=any(is_known_firm(n) for n in (name, short_name, *aliases) if n),
        hashtag=_hashtag_pattern(symbols, variants, base_symbol),
        confirmed_bare=(re.compile(rf"(?<![\w$#@/.-])(?:{sym_alt})(?=\s+{_TICKER_NOUNS}\b)")
                        if word_like and base_symbol not in WORD_TICKERS and len(base_symbol) >= 2 else None),
        index_fund=quote_type in {"ETF", "INDEX", "MUTUALFUND"} and bool(
            _BROAD_INDEX_RE.search(" ".join((name, short_name, *aliases)))),
        industry_text=industry_text.strip().lower(),
    )


def _hashtag_pattern(symbols: tuple[str, ...], variants: list[_NameVariant], base_symbol: str) -> re.Pattern[str] | None:
    """Social tags for the company: "#NVDA", "#Bitcoin", "#BTC", "$NVIDIA" (a
    name written as a cashtag). Word-like tickers ("#AI") and common-word names
    ("#Apple") stay out."""
    alts = [re.escape(sym) for sym in symbols if base_symbol not in WORD_TICKERS and len(sym) >= 2]
    names = [re.escape(v.text.replace(" ", "")) for v in variants if not v.ambiguous and len(v.text) >= 3]
    parts = [f"#(?:{'|'.join(alts)})"] if alts else []
    if names:
        parts.append(f"[#$](?:{'|'.join(names)})")
    if not parts:
        return None
    return re.compile(rf"(?<![\w#$&])(?:{'|'.join(parts)})(?![\w&])", re.IGNORECASE)


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
    after = text[end:end + 60]
    if _VENUE_AFTER_RE.match(after):
        return -1  # "SoFi Stadium"
    if variant.strong:
        return 1
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
    prev = re.search(r"([A-Za-z$][\w'$.]*)\s+$", before)
    if prev and prev.group(1)[:1].isupper() and _is_given_name(prev.group(1), text, start - len(prev.group(0))):
        return -1  # a person: "Katrina Ford", "Francis Ford Coppola", "Ken Block"
    if _FOLLOWER_RE.match(after) or (next_word and next_word.group(1).lower() in products):
        return 1
    if not text[:start].strip() and re.match(r"^:\s+[A-Z0-9\"']", after):
        return 1  # Seeking Alpha style "Meta: Muse Is Nice, But Not Enough"
    if _POSSESSIVE_RE.match(after) or _STRONG_BEFORE_RE.search(before):
        return 1
    if _DEMAND_BEFORE_RE.search(before) and re.match(r"^\s+to\s+[a-z]", after, re.IGNORECASE):
        return 1
    if prev and prev.group(1).lower().removesuffix("'s") in DETERMINERS:
        return -1
    # Subject followed by a verb: "Target Slashes Prices", "jury says Apple owes".
    clause_start = _CLAUSE_START_RE.search(before) is not None
    if clause_start and (_SUBJECT_VERB_RE.match(after) or _PAST_VERB_RE.match(after)):
        return 1
    if _MID_VERB_RE.match(after):
        return 1
    coordinated = _COORDINATED_RE.match(after)
    if clause_start and coordinated and _PLURAL_VERB_RE.match(after[coordinated.end():]):
        return 1  # "Ford and JPMorganChase launch Michigan LIFT"
    return 0


def _sister_issuer(text: str, end: int, variant: _NameVariant, matcher: _Matcher, title_case: bool) -> str | None:
    """The name of a separately listed sister company when the brand at
    [.., end) is the first word of it: "Toyota Industries", "Vodafone Idea
    shares", "Samsung Biologics Co" (see _SISTER_WORDS)."""
    nxt = _NEXT_NAME_RE.match(text[end:end + 40])
    if not nxt:
        return None
    word = nxt.group(1)
    low = word.lower()
    forms = {low, low.removesuffix("s"), low + "s"}
    products = variant.rule.products if variant.rule else ()
    if forms & matcher.own_words or low in products or low in _NOT_SISTER or re.match(_CORP_SUFFIX, nxt.group(0)):
        return None
    name = f"{variant.text} {word}"
    if low in _BRAND_SISTERS.get(variant.text.lower(), ()):
        return name
    if low in _SISTER_WORDS:
        return None if low[:5] in matcher.industry_text else name  # "Dow Chemical" is Dow's own
    if (not word[0].isupper() or low in _VERBISH or low in DETERMINERS or _FOLLOWER_RE.match(nxt.group(0))
            or _VENUE_AFTER_RE.match(nxt.group(0))):
        return None  # an unknown word names an issuer only when capitalized ("Vodafone Idea shares")
    rest = text[end + nxt.end():end + nxt.end() + 30]
    if _SISTER_LEGAL_AFTER_RE.match(rest) or (not title_case and _SISTER_SHARES_AFTER_RE.match(rest)):
        return name  # "Toyota Tsusho Corp", "Vodafone Idea shares slump"
    return None


def _is_given_name(word: str, text: str, pos: int) -> bool:
    """A capitalized word that reads as a person's first name: a known given
    name, or — in sentence case, where capitals carry information — any
    capitalized word that is not the first of its sentence and not a known
    company ("I'm Troy Ford", "says Priscilla Block")."""
    low = word.lower()
    if low in FIRST_NAMES:
        return True
    if is_title_case(text) or is_known_firm(word) or low in _VERBISH or low in AMBIGUOUS_NAMES:
        return False
    return not re.search(r"(?:^|[.!?:;]\s*|[\"(]\s*)$", text[:pos]) and word.isalpha()


# --------------------------------------------------------------------------- #
# Brokerages as authors of research ("JPMorgan downgrades PepsiCo")
# --------------------------------------------------------------------------- #
_DESK = (r"(?:(?:global|us|u\.s\.|equity|equities|market|markets|research|investment|asset management|wealth management|"
         r"private bank|chief|senior|top|lead|head|macro|quant|credit|fixed income|trading|intelligence|"
         r"analysts?|strategists?|economists?|desk|team)\s+){0,4}")
_ANALYST_OBJECT = (r"(?:price targets?|targets?|\bpt\b|ratings?|expectations for|stock price|estimates?|"
                   r"(?:stock|equity|market|s&p 500|s&p|us stock|u\.s\. stock|earnings|gdp|economic|oil|gold|"
                   r"bitcoin|rate|yield)\s+(?:outlook|forecast|target)|forecast|outlook for|view on)")
# After the name: the firm rates, targets or lists something (always research).
_ACTOR_AFTER_RE = re.compile(
    rf"^(?:'s)?\s+{_DESK}(?:analysts?|strategists?|economists?|desk|research team)\b"
    rf"|^(?:'s)?\s+{_DESK}(?:(?:double[- ])?upgrades?|(?:double[- ])?downgrades?|(?:re)?initiates?|reiterates?|"
    r"resumes?|assumes?|starts? coverage|(?:has|sends) (?:a )?(?:strong|bold|clear|big|new)?\s*(?:message|verdict|"
    r"warning))\b"
    rf"|^(?:'s)?\s+{_DESK}(?:has\s+)?(?:raises?|raised|lifts?|boosts?|hikes?|cuts?|lowers?|lowered|trims?|"
    rf"slashes?|sets?|revamps?|adjusts?|tweaks?|maintains?|keeps?)\s+(?:its\s+|the\s+|their\s+)?"
    rf"(?:[\w.&'-]+\s+){{0,4}}?{_ANALYST_OBJECT}"
    r"|^\s+(?:adds?|added|removes?|removed|drops?|dropped)\s+(?:[\w.&'-]+\s+){1,4}?(?:to|from)\s+(?:its\s+)?"
    r"(?:\w+\s+)?(?:focus|conviction|best ideas|top picks?|analyst focus|director'?s cut)\s+list"
    r"|^(?:'s|')?\s+(?:best|top|favou?rite|highest[- ]conviction)\s+(?:stock\s+)?(?:ideas|picks)\b"
    r"|^(?:'s|')?\s+(?:[\w-]+\s+)?(?:conviction|focus|top picks?|best ideas|director'?s cut)\s+list\b"
    r"|^\s+(?:downgrade|upgrade|price target|target (?:increase|cut|hike|raise)s?|note|call)\b"
    r"|^(?:'s)?\s+(?:[\w.&'-]+\s+){0,3}?(?:etfs?|etf trust|icav|ucits|fund|funds)\b",
    re.IGNORECASE,
)
# Opinion verbs: research only when the clause is about others or the market
# ("JPMorgan sees Micron positioned for a beat"), not the firm's own business
# ("Morgan Stanley sees record wealth inflows").
_SOFT_ACTOR_RE = re.compile(
    r"^(?:'s)?\s+(?:says?|said|sees?|saw|expects?|predicts?|warns?|thinks?|likes?|prefers?|picks?|calls?|flags?|"
    r"cautions?|is (?:bullish|bearish|positive|negative)|turns? (?:bullish|bearish)|goes (?:bullish|bearish))\b"
    r"(?P<rest>[^.;:!?]{0,70})",
    re.IGNORECASE,
)
_MARKET_WORDS_RE = re.compile(
    r"\b(?:stocks?|equit(?:y|ies)|markets?|s&p|nasdaq|dow|fed|rates?|economy|recession|inflation|volatility|"
    r"bubble|rally|sell-?off|setup|upside|downside|valuation|investors|tariffs?|yields?|bonds?|oil|gold|bitcoin|"
    r"crypto|dollar|sector|cycles?)\b",
    re.IGNORECASE,
)
_OWN_BUSINESS_RE = re.compile(
    r"\b(?:its|own|our|the bank|the firm|the company)\b|\b(?:inflows?|outflows?|deposits?|net interest|nii|loan growth|"
    r"fees?|trading revenue|investment banking|wealth|clients?|headcount|branches|card|jobs|hiring|hire|layoffs|"
    r"employees|staff|bonus(?:es)?|pay|succession|board|ceo|aum|assets under management|capital|cet1|rotce|roe|"
    r"profitability|efficiency|expenses?|costs?)\b",
    re.IGNORECASE,
)
# "JPMorgan's Kolanovic says stocks will fall" (research) vs "JPMorgan's Dimon warns ...".
_PERSON_SAYS_RE = re.compile(r"^'s\s+(?P<who>(?-i:[A-Z])[\w.'-]+(?:\s+(?-i:[A-Z])[\w.'-]+)?)\s+(?:says?|sees?|warns?|"
                             r"expects?|thinks?)\b(?P<rest>[^.;:!?]{0,70})", re.IGNORECASE)


# Before the name: something was done *by/at/from* the firm, or another firm hires its people.
_ACTOR_BEFORE_RE = re.compile(
    r"\b(?:upgrades?|downgrades?|upgraded|downgraded|initiated|coverage|rating|ratings|targets?|lowered|raised|"
    r"cut|increased|trimmed|boosted|reiterated|maintained|note|call|picked|named|added)\s+"
    r"(?:\w+\s+){0,3}?(?:at|by|from|after|per)\s+(?:the\s+)?$"
    r"|\b(?:after|following|per|says?|according to)\s+(?:an?\s+)?$"
    r"|\b(?:hires?|hired|poach(?:es|ed)?|taps?|tapped|recruits?|lures?)\s+$",
    re.IGNORECASE,
)


def _broker_actor(text: str, start: int, end: int, own_people: re.Pattern[str] | None = None) -> bool:
    """True when a brokerage's name appears as the author of research or a
    market call — "JPMorgan downgrades PepsiCo", "Target Lowered at
    JPMorgan", "...: JPMorgan", "JPMorgan's best stock ideas" — or names its
    funds ("JPMorgan BBSC ETF"), i.e. the text is not about the firm itself."""
    after = re.sub(r"^(?:\s*&\s*co\b|\s+chase(?:\s*&\s*co\b)?|,?\s*inc\b)?\.?", "", text[end:end + 90],
                   flags=re.IGNORECASE)
    before = text[max(0, start - 50):start]
    actor = _ACTOR_AFTER_RE.match(after)
    if actor and not (re.search(r"\bsets?\b", actor.group(0), re.IGNORECASE)
                      and _OWN_BUSINESS_RE.search(actor.group(0))):  # "sets new wealth management target"
        return True
    if _ACTOR_BEFORE_RE.search(before):
        return True
    person = _PERSON_SAYS_RE.match(after)
    if person:
        if own_people and own_people.search(person.group("who")):
            return False  # its own executive: "JPMorgan's Dimon warns of cockroaches"
        rest = person.group("rest")
        return bool((re.search(r"\s(?-i:[A-Z])[\w.&'-]{2,}", rest) or _MARKET_WORDS_RE.search(rest))
                    and not _OWN_BUSINESS_RE.search(rest))
    soft = _SOFT_ACTOR_RE.match(after)
    if soft:
        rest = soft.group("rest")
        other_entity = re.search(r"\s(?-i:[A-Z])[\w.&'-]{2,}", rest)
        if (other_entity or _MARKET_WORDS_RE.search(rest)) and not _OWN_BUSINESS_RE.search(rest):
            return True
    if re.match(r"^\s+[A-Z]{3,5}\b", after) and re.search(r"\betfs?\b", text, re.IGNORECASE):
        return True  # a fund ticker: "State Street SPSM vs. JPMorgan BBSC"
    # Trailing attribution: "Leverage could challenge stocks in Q4: JPMorgan"
    return bool(re.search(r"[:–—-]\s*$", before) and not after.strip(" .\"'"))


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


def _is_roundup(text: str, subject_end: int = -1) -> bool:
    """Multi-stock roundups, listicles and hashtag-stuffed posts. A list that
    follows the company as the headline's subject is about the company
    ("Bitcoin beats Gold, SPY, Silver, QQQ in Iran war"; "Bitcoin: ETF
    Inflows, Fed Hikes, ...")."""
    tags = {m.group(1).upper() for m in _CASHTAG_RE.finditer(text)}
    if len(tags) >= 4 or (subject_end < 0 and len({h.lower() for h in _HASHTAG_RE.findall(text)}) >= 4):
        return True
    if _LISTICLE_RE.search(text):
        return True
    lists = [m for rx in (_COMMA_LIST_RE, _TICKER_LIST_RE) if (m := rx.search(text))]
    return any(not (0 <= subject_end <= m.start()) for m in lists)


def explain_relevance(text: str, company: CompanyRef) -> RelevanceResult:
    """Relevance score with the evidence that produced it."""
    if not text or not company:
        return RelevanceResult(0.0)
    t = fold(text)
    matcher = _matcher(company)
    shouting = is_mostly_upper(t)
    title_case = is_title_case(t)
    evidence: list[str] = []
    positions: list[int] = []
    spans: list[Mention] = []
    score = 0.0
    mentions = 0
    subject_end = -1  # end of a headline-leading mention (the subject), if any

    def record(start: int, end: int) -> Mention:
        modifier, adjunct = _mention_role(t, start, end)
        mention = Mention(start, end, modifier=modifier, adjunct=adjunct)
        spans.append(mention)
        return mention

    for m in matcher.cashtag.finditer(t):
        if subject_end < 0 and len(t[:m.start()].split()) <= 1:
            subject_end = m.end()
        score = max(score, 1.0)
        positions.append(m.start())
        record(*m.span())
        mentions += 1
        evidence.append(f"cashtag {m.group(0)}")
    for m in matcher.qualified.finditer(t):
        score = max(score, 0.95)
        positions.append(m.start())
        record(*m.span())
        mentions += 1
        evidence.append(f"qualified ticker {m.group(0).strip()}")
    tag_hits = 0
    if matcher.hashtag:
        for m in matcher.hashtag.finditer(t):
            tag_hits += 1
            if subject_end < 0 and len(t[:m.start()].split()) <= 4 and not re.match(r"\s*[#$]", t[m.end():]):
                subject_end = m.end()  # "🤖 AI Agent Upgrade: #AAPL is now a BUY" (not "#MU #ACN #LQDA ...")
            score = max(score, 0.85)
            positions.append(m.start())
            record(*m.span())
            mentions += 1
            evidence.append(f"tag {m.group(0)}")
    if matcher.confirmed_bare and score < 0.9 and not shouting:
        for m in matcher.confirmed_bare.finditer(t):
            score = max(score, 0.9)
            positions.append(m.start())
            record(*m.span())
            mentions += 1
            evidence.append(f"ticker {m.group(0)} + stock noun")
    if score < 0.9:
        bare = None
        if matcher.bare_upper and not (matcher.soft_ticker and shouting):
            bare = matcher.bare_upper
        if matcher.bare_any and not shouting:
            bare = matcher.bare_any
        if bare:
            neg_spans = [span for v in matcher.names if v.text.upper() in matcher.symbols
                         for span in _negative_spans(t, v.rule)]
            for m in bare.finditer(t):
                if any(a <= m.start() < b for a, b in neg_spans) or _VENUE_AFTER_RE.match(t[m.end():m.end() + 20]):
                    evidence.append(f"not-ticker {m.group(0)}")
                    continue
                if matcher.acronym_ticker and (neg_spans or not _TICKER_CONTEXT_RE.search(t)):
                    evidence.append(f"unconfirmed ticker {m.group(0)}")
                    continue  # "SNAP benefits": an acronym, not the stock
                score = max(score, 0.9)
                positions.append(m.start())
                record(*m.span())
                mentions += 1
                evidence.append(f"ticker {m.group(0)}")
    if score < 0.8:
        listed = re.search(rf"(?:\b[A-Z]{{1,5}},\s*)+(?:{'|'.join(map(re.escape, matcher.symbols))})\b(?:,\s*[A-Z]{{1,5}}\b)*"
                           rf"|\b(?:{'|'.join(map(re.escape, matcher.symbols))}),\s*[A-Z]{{1,5}}\b", t)
        if listed and not shouting:
            score = max(score, 0.8)
            positions.append(listed.start())
            mentions += 1
            evidence.append("ticker in symbol list")

    name_level = 0.0
    any_neutral = False
    actor_mentions = 0
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
            sister = None if variant.strong else _sister_issuer(t, m.end(), variant, matcher, title_case)
            if sister:
                evidence.append(f"sister company '{sister}'")
                continue
            verdict = _classify(t, m.start(), m.end(), variant, matcher, neg_spans)
            if verdict > 0 and matcher.broker and _broker_actor(t, m.start(), m.end(), matcher.brand_cues):
                actor_mentions += 1
                evidence.append(f"firm as research author '{m.group(0)}'")
                continue
            if verdict > 0:
                mentions += 1
                positions.append(m.start())
                mention = record(*m.span())
                lead = t[:m.start()]
                primary = len(lead.split()) <= 2 or re.match(r"^[^:]{0,40}:\s*$", lead) is not None
                if primary and subject_end < 0 and len(lead.split()) <= 2:
                    subject_end = m.end()
                level = 0.9 if primary else 0.8
                if mention.modifier or t[m.end():m.end() + 1] == "-":
                    # "Tesla rival Nikola", "Nvidia-backed CoreWeave", "Apple-designed": qualifies something else
                    level, primary = 0.6, False
                    evidence.append(f"modifier '{m.group(0)}'")
                name_level = max(name_level, level)
                evidence.append(f"name '{m.group(0)}'" + (" (subject)" if primary else ""))
            elif verdict == 0:
                any_neutral = True
                positions.append(m.start())
                record(*m.span())
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

    spans.sort(key=lambda s: s.start)
    result = RelevanceResult(score=score, evidence=evidence, mentions=spans)
    if score > 0 and positions:
        if spans and not any(s.subject_like for s in spans):
            # Only named as another entity's rival/partner/backer or in background
            # detail: "Tesla rival Nikola files for bankruptcy".
            result.secondary = True
            score = min(score, CONTEXT_ONLY_MAX)
            evidence.append("named only as context for another entity")
        elif _other_subject_first(t, min(positions), matcher):
            result.secondary = True
            score *= 0.75
            evidence.append("another company is the subject")
        if matcher.index_fund and _MARKET_WIDE_RE.search(t):
            pass  # "Dow, S&P 500, Nasdaq Futures Rise ...: NKE, NFLX In Focus" is the index's own news
        elif _is_roundup(t, subject_end):
            result.roundup = True
            only_tags = tag_hits == mentions and name_level == 0.0
            score = min(score, 0.3 if only_tags else 0.4)  # one label in a hashtag soup says little
            evidence.append("hashtag soup" if only_tags else "roundup/listicle")
    if (matcher.index_fund and score < 0.6 and _MARKET_WIDE_RE.search(t) and not _LISTICLE_RE.search(t)
            and not _OTHER_SUBJECT_RE.match(t)):
        score = 0.6
        result.secondary = False
        evidence.append("market-wide news (index fund)")
    result.score = round(score, 3)
    return result


def relevance(text: str, company: CompanyRef) -> float:
    """0..1: how clearly `text` is about `company` (see module docstring)."""
    return explain_relevance(text, company).score


def brand_cue_mentions(text: str, company: CompanyRef) -> list[Mention]:
    """Spans of `fold(text)` naming the company's own products or executives
    ("Zuckerberg", "iPhone", "Cash App") — they stand for the company as a
    party to launches and deals even when its name is absent."""
    if not text or not company:
        return []
    matcher = _matcher(company)
    if matcher.brand_cues is None:
        return []
    return [Mention(m.start(), m.end()) for m in matcher.brand_cues.finditer(fold(text))]


# Name words that are everyday vocabulary: "First Solar" should not hide "solar",
# nor "American Airlines" hide "airlines", from keywords and story features.
_COMMON_NAME_WORDS = wordset("""
american first general national united international global solar energy airlines airline motors motor technologies
technology systems financial holdings bank banks bancorp group health healthcare pharmaceuticals pharmaceutical
therapeutics semiconductor semiconductors devices micro advanced platforms communications networks network software
foods food brands entertainment resources industries materials capital partners realty properties trust insurance
services solutions labs sciences biosciences medical petroleum oil gas power electric water steel mining gold silver
digital data cloud security robotics auto automotive aerospace defense cruise hotels resorts restaurants pharma bio
new home homes depot stores store mobile wireless media interactive enterprises corporation company the of and
""")


@lru_cache(maxsize=256)
def _terms_for(ticker: str, name: str, short_name: str, aliases: tuple[str, ...]) -> frozenset[str]:
    words: set[str] = set()
    base = ticker.split("-")[0].split(".")[0].lower()
    words.update({ticker.lower(), base, f"${base}"})
    for value in (name, short_name, *aliases):
        tokens = [w for w in re.findall(r"[a-z0-9&]+", _clean_name(value or "").lower()) if len(w) > 1]
        if len(tokens) == 1:
            words.update(tokens)  # "Nvidia", "Target", "AMD": the name itself
        else:
            words.update(w for w in tokens if w not in _COMMON_NAME_WORDS and w not in _VERBISH)
    return frozenset(words)


def company_terms(company: CompanyRef | None) -> frozenset[str]:
    """Lower-case tokens that name the company (ticker, cashtag, one-word
    names, the distinctive words of longer names). Used to keep the company's
    own name out of keywords/narrative features; everyday words inside a
    longer name ("solar" in First Solar) stay — `company_phrases` covers the
    full name."""
    if company is None:
        return frozenset()
    return _terms_for(company.ticker, company.name or "", company.short_name or "", tuple(company.aliases or ()))


@lru_cache(maxsize=256)
def _phrases_for(name: str, short_name: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    phrases = {" ".join(w for w in re.findall(r"[a-z0-9&]+", _clean_name(v or "").lower()) if len(w) > 1)
               for v in (name, short_name, *aliases)}
    return tuple(sorted((p for p in phrases if " " in p), key=len, reverse=True))


def company_phrases(company: CompanyRef | None) -> tuple[str, ...]:
    """Lower-case multi-word names of the company, longest first ("first
    solar", "advanced micro devices"), for removal as whole phrases."""
    if company is None:
        return ()
    return _phrases_for(company.name or "", company.short_name or "", tuple(company.aliases or ()))
