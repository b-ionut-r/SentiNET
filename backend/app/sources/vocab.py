"""Word lists and theme tables behind query construction and mention matching.

Pure data, curated from live checks (2026-10-04). Kept apart from `query.py`
so the logic there stays readable. Everything is keyed by upper-case symbol or
lower-case word.
"""
from __future__ import annotations

# Brand names whose everyday meaning dominates general text ("price target", "bash shell",
# "H-1B visa"): searches must pair them with the ticker or legal name.
COMMON_WORD_NAMES = frozenset({
    "target", "block", "visa", "shell", "snap", "gap", "coach", "match", "zoom", "unity", "sea", "ball",
    "box", "ring", "square", "delta", "southern", "progressive", "general", "national", "united",
    "american", "first", "global", "pioneer", "digital", "energy", "root", "rocket", "lemonade", "dollar",
    "crown", "sun", "edge", "monster", "marathon", "riot", "compass", "fortune", "genius", "carnival",
    "progress", "premier", "frontier", "liberty", "pinnacle", "summit",
})

# Names whose lower-case (or other-entity) use is common enough that a text match must
# respect the company's own casing: "Apple" not "apple", "SoFi" not "Sofi Tukker".
CASE_SENSITIVE_NAMES = COMMON_WORD_NAMES | frozenset({
    "apple", "amazon", "meta", "oracle", "ford", "alphabet", "sofi", "snowflake", "lucid", "booking",
    "discover", "affirm", "upstart", "oscar", "plug", "ally", "citizens", "regions", "continental",
    "spirit", "celsius", "constellation", "crocs", "academy", "express", "guess", "columbia", "beam",
    "applied", "analog", "advanced", "bloom", "intuitive", "coherent", "lumen", "public", "service",
    "enterprise", "waste", "republic", "pacific", "atlantic", "freedom", "fidelity", "prudential",
    "principal", "arch", "arm", "strategy", "chewy", "elastic", "confluent", "micron", "intel", "uber",
    "chevron", "caterpillar", "hertz", "carrier", "mosaic", "travelers", "dominion", "paramount", "fox",
    "chime", "toast", "clover", "bumble", "peloton", "capri", "chipotle", "cava", "vertex", "pool", "grab",
    "archer",
    # coins whose names are words
    "ether", "polygon", "avalanche", "cosmos", "stellar", "tron", "optimism", "maker", "stacks", "jupiter",
    "immutable", "cronos", "near", "flow", "gala", "render", "pepe", "sui", "dash",
})

# Symbols that read as words or abbreviations in ordinary text: never searched bare.
WORD_TICKERS = frozenset({
    "A", "AI", "ALL", "AM", "AN", "ANY", "APP", "ARE", "ARM", "AT", "BE", "BEST", "BIG", "BOX", "BUY", "CAN",
    "CAR", "CASH", "CAT", "CEO", "COST", "DD", "DO", "DOG", "DOT", "EAT", "EDIT", "EOD", "EV", "EVER", "EYE",
    "FAST", "FLY", "FOR", "FUN", "GAIN", "GAS", "GO", "GOLD", "GOOD", "GROW", "HAS", "HE", "HOME", "HOPE",
    "HUGE", "IT", "JOB", "KEY", "KIDS", "LAND", "LIFE", "LINK", "LIVE", "LOVE", "LOW", "MAIN", "MAN", "MARK",
    "MIND", "MOVE", "NEAR", "NEW", "NEXT", "NICE", "NOW", "OK", "ON", "ONE", "OPEN", "OUT", "PAY", "PEAK",
    "PLAY", "PLUG", "PUMP", "REAL", "RIDE", "RUN", "SAFE", "SAVE", "SEE", "SHOP", "SKY", "SNOW", "SO", "SOL",
    "SPY", "STAR", "TECH", "TELL", "TEN", "TOP", "TRUE", "TWO", "UK", "UP", "US", "USA", "WELL", "WORK",
    "YOU", "DASH", "BAND", "COMP", "ATOM", "RAY", "WAVE", "AUDIO", "OCEAN", "SUPER", "MASK", "ACE", "ROSE",
    "CAKE", "MEME", "HYPE", "PEOPLE", "MAGIC", "BLUR", "PORTAL", "FLOW", "GALA", "SAND", "MANA", "TRUMP",
    "APE", "RUNE", "JOE", "KEEP", "BONK", "WIF",
})

# Ticker-words people actually type in capitals as words ("YOU", "BUY", "ALL IN"). Crowd
# leaderboards count upper-case tokens, so for these symbols they count the word, not the stock.
# (ARM, APP, CAT, COST, SNOW, SHOP, PLUG, DASH... are mostly the company when capitalised.)
SHOUTED_WORDS = frozenset({
    "A", "I", "AI", "ALL", "AM", "AN", "ANY", "ARE", "AT", "BE", "BEST", "BIG", "BUY", "CAN", "CASH", "CEO",
    "DD", "DO", "EOD", "EV", "EVER", "FOR", "FUN", "GAIN", "GO", "GOLD", "GOOD", "HAS", "HE", "HOME", "HOPE",
    "HUGE", "IT", "JOB", "KEY", "LIFE", "LIVE", "LOVE", "LOW", "MAIN", "MAN", "MIND", "MOVE", "NEW", "NEXT",
    "NICE", "NOW", "OK", "ON", "ONE", "OPEN", "OUT", "PAY", "PEAK", "PLAY", "PUMP", "REAL", "RUN", "SAFE",
    "SAVE", "SEE", "SO", "TELL", "TEN", "TOP", "TRUE", "TWO", "UK", "UP", "US", "USA", "WELL", "WORK", "YOU",
})

# Upper-case tokens Reddit/WSB write that are not about the same-named stock: options and
# trading lingo, macro acronyms, chat shorthand. Crowd boards count these words, not tickers.
REDDIT_ACRONYMS = frozenset({
    "ES", "NQ", "CD", "EU", "TP", "SL", "PM", "IQ", "WTI", "ET", "PT", "FTC", "API", "CC", "HR", "RR", "PR",
    "IP", "CSV", "HYSA", "CAPE", "REIT", "TLDR", "NAN", "FTW", "ASX", "DTE", "GDP", "CPI", "PPI", "FED",
    "FOMC", "SEC", "FDA", "IRS", "ETF", "EPS", "PE", "IV", "OTM", "ITM", "ATM", "ATH", "IPO", "YOLO", "IMO",
    "FOMO", "HODL", "LOL", "OMG", "WTF", "AR", "VR", "OP", "TA", "FA", "RSI", "EMA", "SMA", "DCA", "FIRE",
    "ROTH", "IRA", "HSA", "BOND", "CFO", "CTO", "USD", "EUR", "JPY", "GBP", "OG", "ER", "MOON", "EDGE",
    "BULL", "BEAR", "PUT", "CALL", "RH", "WSB", "SP", "SPX", "NDX", "VIX",
})

# Fund/index/futures/FX symbols -> what their news is about (most specific first).
THEMES: dict[str, tuple[str, ...]] = {
    # Broad US equity
    "SPY": ("S&P 500",), "VOO": ("S&P 500",), "IVV": ("S&P 500",), "SPLG": ("S&P 500",), "^GSPC": ("S&P 500",),
    "^SPX": ("S&P 500",), "SPXL": ("S&P 500",), "UPRO": ("S&P 500",), "SSO": ("S&P 500",),
    "SH": ("S&P 500",), "SDS": ("S&P 500",), "SPXS": ("S&P 500",), "ES=F": ("S&P 500 futures", "S&P 500"),
    "QQQ": ("Nasdaq 100", "Nasdaq"), "QQQM": ("Nasdaq 100", "Nasdaq"), "TQQQ": ("Nasdaq 100", "Nasdaq"),
    "SQQQ": ("Nasdaq 100", "Nasdaq"), "^NDX": ("Nasdaq 100", "Nasdaq"), "^IXIC": ("Nasdaq Composite", "Nasdaq"),
    "NQ=F": ("Nasdaq futures", "Nasdaq 100"),
    "DIA": ("Dow Jones",), "^DJI": ("Dow Jones",), "YM=F": ("Dow futures", "Dow Jones"),
    "IWM": ("Russell 2000", "small caps"), "^RUT": ("Russell 2000", "small caps"), "TNA": ("Russell 2000",),
    "VTI": ("stock market",), "VT": ("global stocks",), "^VIX": ("VIX", "volatility index"),
    "VXX": ("VIX", "volatility index"), "UVXY": ("VIX", "volatility index"),
    "MAGS": ("Magnificent Seven",), "VUG": ("growth stocks",), "VTV": ("value stocks",),
    "SCHD": ("dividend stocks",), "VYM": ("dividend stocks",), "VIG": ("dividend stocks",),
    # Sectors / industries
    "XLK": ("tech stocks",), "VGT": ("tech stocks",), "XLF": ("bank stocks", "financial stocks"),
    "XLE": ("energy stocks", "oil stocks"), "XLV": ("health care stocks", "healthcare stocks"),
    "XLY": ("consumer discretionary stocks",), "XLP": ("consumer staples stocks",),
    "XLI": ("industrial stocks",), "XLB": ("materials stocks",), "XLU": ("utility stocks", "utilities stocks"),
    "XLC": ("communication services stocks",), "XLRE": ("REITs", "real estate stocks"),
    "VNQ": ("REITs", "real estate stocks"), "KRE": ("regional banks", "regional bank stocks"),
    "KBE": ("bank stocks",), "XBI": ("biotech stocks",), "IBB": ("biotech stocks",),
    "ITB": ("homebuilder stocks", "homebuilders"), "XHB": ("homebuilder stocks", "homebuilders"),
    "SMH": ("chip stocks", "semiconductor stocks"), "SOXX": ("chip stocks", "semiconductor stocks"),
    "SOXL": ("chip stocks", "semiconductor stocks"), "SOXS": ("chip stocks", "semiconductor stocks"),
    "GDX": ("gold miners", "gold mining stocks"), "GDXJ": ("gold miners", "gold mining stocks"),
    "URA": ("uranium stocks",), "LIT": ("lithium stocks",), "TAN": ("solar stocks",),
    "ICLN": ("clean energy stocks",), "JETS": ("airline stocks",), "KWEB": ("China internet stocks",),
    "ARKK": ("ARK Innovation", "Cathie Wood"),
    # International
    "EEM": ("emerging markets",), "VWO": ("emerging markets",), "EFA": ("international stocks",),
    "FXI": ("China stocks",), "MCHI": ("China stocks",), "EWJ": ("Japan stocks", "Nikkei"),
    "EWZ": ("Brazil stocks",), "INDA": ("India stocks",), "^N225": ("Nikkei",), "^FTSE": ("FTSE 100",),
    "^GDAXI": ("DAX",), "^HSI": ("Hang Seng",), "^STOXX50E": ("Euro Stoxx 50",),
    # Rates & credit
    "TLT": ("Treasury yields", "Treasuries"), "IEF": ("Treasury yields", "Treasuries"),
    "SHY": ("Treasury yields", "Treasuries"), "^TNX": ("10-year Treasury yield", "Treasury yields"),
    "HYG": ("junk bonds", "high-yield bonds"), "JNK": ("junk bonds", "high-yield bonds"),
    "LQD": ("corporate bonds",), "AGG": ("bond market",), "BND": ("bond market",),
    # Commodities
    "GLD": ("gold price", "gold"), "IAU": ("gold price", "gold"), "GC=F": ("gold price", "gold futures", "gold"),
    "SLV": ("silver price", "silver"), "SI=F": ("silver price", "silver futures", "silver"),
    "USO": ("oil prices", "crude oil"), "CL=F": ("oil prices", "crude oil"), "BZ=F": ("Brent crude", "oil prices"),
    "UNG": ("natural gas prices", "natural gas"), "NG=F": ("natural gas prices", "natural gas"),
    "HG=F": ("copper prices", "copper"), "DBC": ("commodities",),
    "ZC=F": ("corn prices",), "ZW=F": ("wheat prices",), "ZS=F": ("soybean prices",), "KC=F": ("coffee prices",),
    # Currencies
    "UUP": ("US dollar", "dollar index"), "DX-Y.NYB": ("dollar index", "US dollar"),
    "EURUSD=X": ("euro", "EUR/USD"), "JPY=X": ("yen", "USD/JPY"), "GBPUSD=X": ("pound sterling", "GBP/USD"),
    # Crypto funds
    "IBIT": ("Bitcoin ETF", "Bitcoin"), "FBTC": ("Bitcoin ETF", "Bitcoin"), "GBTC": ("Bitcoin ETF", "Bitcoin"),
    "ARKB": ("Bitcoin ETF", "Bitcoin"), "BITB": ("Bitcoin ETF", "Bitcoin"), "BITO": ("Bitcoin futures", "Bitcoin"),
    "ETHA": ("Ether ETF", "Ethereum"), "ETHE": ("Ether ETF", "Ethereum"),
}

# Theme names that are market talk on their own (a post saying "the S&P 500" needs no
# extra finance word). Others ("gold", "euro") need market vocabulary nearby.
SELF_EVIDENT_THEMES = frozenset({
    "dow jones", "nasdaq", "vix", "treasuries", "nikkei", "dax", "hang seng", "ark innovation",
    "magnificent seven", "reits", "junk bonds", "emerging markets", "brent crude", "small caps",
})

# Crypto base symbol -> (names, ambiguous). `ambiguous` = the coin's name is also an everyday
# word/other entity ("Polygon" the gaming site, "Avalanche", "Trump"), so searches need a
# crypto anchor. Unlisted coins are treated as ambiguous (precision first).
CRYPTO_NAMES: dict[str, tuple[tuple[str, ...], bool]] = {
    "BTC": (("Bitcoin",), False), "ETH": (("Ethereum", "Ether"), False), "SOL": (("Solana",), False),
    "XRP": (("XRP", "Ripple"), False), "DOGE": (("Dogecoin",), False), "ADA": (("Cardano",), False),
    "BNB": (("BNB", "Binance Coin"), False), "LTC": (("Litecoin",), False), "LINK": (("Chainlink",), False),
    "DOT": (("Polkadot",), False), "TON": (("Toncoin",), False), "HBAR": (("Hedera",), False),
    "BCH": (("Bitcoin Cash",), False), "UNI": (("Uniswap",), False), "XMR": (("Monero",), False),
    "ARB": (("Arbitrum",), False), "APT": (("Aptos",), False), "FIL": (("Filecoin",), False),
    "AAVE": (("Aave",), False), "ALGO": (("Algorand",), False), "VET": (("VeChain",), False),
    "KAS": (("Kaspa",), False), "INJ": (("Injective",), False), "TAO": (("Bittensor",), False),
    "WLD": (("Worldcoin",), False), "HYPE": (("Hyperliquid",), False), "ONDO": (("Ondo",), False),
    "ENA": (("Ethena",), False), "TIA": (("Celestia",), False), "ETC": (("Ethereum Classic",), False),
    "SHIB": (("Shiba Inu",), True), "AVAX": (("Avalanche",), True), "XLM": (("Stellar",), True),
    "TRX": (("Tron",), True), "ATOM": (("Cosmos",), True), "NEAR": (("Near Protocol",), True),
    "OP": (("Optimism",), True), "POL": (("Polygon",), True), "MATIC": (("Polygon",), True),
    "SUI": (("Sui",), True), "PEPE": (("Pepe",), True), "MKR": (("Maker",), True), "STX": (("Stacks",), True),
    "JUP": (("Jupiter",), True), "IMX": (("Immutable",), True), "CRO": (("Cronos",), True),
    "ICP": (("Internet Computer",), True), "GRT": (("The Graph",), True), "RENDER": (("Render",), True),
    "SAND": (("The Sandbox",), True), "MANA": (("Decentraland",), False), "GALA": (("Gala",), True),
    "APE": (("ApeCoin",), False), "WIF": (("dogwifhat",), False), "BONK": (("Bonk",), True),
    "TRUMP": (("Trump memecoin", "Trump meme coin", "TRUMP token"), False),
}

# Share classes of one issuer: news feeds tag either class (Yahoo tags Alphabet news GOOG only).
SHARE_CLASS_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(group)
    for group in (
        ("GOOG", "GOOGL"), ("BRK-A", "BRK-B"), ("FOX", "FOXA"), ("NWS", "NWSA"), ("UA", "UAA"),
        ("LBRDA", "LBRDK"), ("BF-A", "BF-B"), ("HEI", "HEI-A"), ("LEN", "LEN-B"), ("MOG-A", "MOG-B"),
        ("GEF", "GEF-B"), ("CWEN", "CWEN-A"), ("LILA", "LILAK"), ("FWONA", "FWONK"), ("Z", "ZG"),
        ("BIO", "BIO-B"), ("UHAL", "UHAL-B"), ("RUSHA", "RUSHB"), ("MKC", "MKC-V"), ("STZ", "STZ-B"),
        ("PARA", "PARAA"), ("KELYA", "KELYB"), ("CRD-A", "CRD-B"), ("BATRA", "BATRK"),
    )
)
