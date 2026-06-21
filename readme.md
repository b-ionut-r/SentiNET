# ◆ Sentinel — Real-Time Stock Sentiment

An advanced financial **sentiment analysis platform** that gathers **real,
authentic signals** from **free** social-media and news sources, scores them,
and shows genuine, at-a-glance insight about any company / stock / ticker.

No paid APIs. No fabricated data — when a source is unavailable it is shown as
such, never faked.

## What it does

Type a ticker and Sentinel fans out **concurrently** to multiple free sources,
normalizes everything into one signal shape, scores each item, and aggregates a
**weighted** overall sentiment (trusted news counts for more than anonymous
social volume; recent and high-engagement items count for more).

### Free data sources (all keyless, or optional free auth)

| Source | Kind | Notes |
| --- | --- | --- |
| **Yahoo Finance** (`yfinance`) | news + price | headlines + price history |
| **Google News** (RSS) | news | per-company search feed |
| **Hacker News** (Algolia API) | news | tech-company discussion |
| **Reddit** (public `.json`) | social | r/wallstreetbets, r/stocks, … (optional PRAW creds) |
| **WSB via Tradestie** | social | bullish/bearish + comment volume, no key |
| **ApeWisdom** | social | mention volume / buzz, no key |
| **StockTwits** | social | best-effort public stream |

If any source is rate-limited, blocked or down, it **degrades gracefully** — the
rest of the analysis still returns, and the UI shows that source's status.

### At-a-glance UI

Hero **sentiment gauge** · price + change · signal volume · active sources ·
**source breakdown** (news vs. hype) · **price chart** · **sentiment timeline** ·
**trending keywords** · a **live signal feed** with real outbound links · and an
honest **source-status** bar.

## Architecture

```
Sources → Normalize → Sentiment → Aggregate → Cache → API → UI
```

- **Backend** — FastAPI. Each source implements one `async fetch()` interface and
  is fetched via `asyncio.gather` with per-source timeouts. Sentiment is a
  swappable strategy (VADER default; FinBERT behind the same interface).
  In-process TTL cache (no external infra).
- **Frontend** — React + TypeScript + Tailwind + Recharts, data via TanStack
  Query.

Key files: `backend/app/sources/registry.py` (fan-out),
`backend/app/pipeline/aggregate.py` (weighting),
`backend/app/sentiment/` (engines), `frontend/src/pages/Dashboard.tsx` (UI).

## Run it

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload          # serves API on http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                            # dev server on http://127.0.0.1:5173 (proxies /api)
```

### One-process (production-style)

Build the frontend, then the backend serves it:

```bash
cd frontend && npm run build           # outputs frontend/dist
cd ../backend && source .venv/bin/activate
uvicorn app.main:app                   # open http://127.0.0.1:8000
```

## Configuration (all optional — see `backend/.env.example`)

| Var | Default | Purpose |
| --- | --- | --- |
| `SENTIMENT_ENGINE` | `vader` | `vader` or `finbert` |
| `ANALYZE_CACHE_TTL` | `600` | analysis cache seconds |
| `SOURCE_TIMEOUT` | `8` | per-source fetch timeout |
| `DISABLED_SOURCES` | — | e.g. `stocktwits,hackernews` |
| `REDDIT_CLIENT_ID` / `_SECRET` | — | optional free Reddit app for robustness |

### Enabling FinBERT (optional, more accurate on financial text)

VADER is the default (instant, no download). To use the FinBERT transformer:

```bash
pip install transformers torch        # heavy; model downloads on first use
export SENTIMENT_ENGINE=finbert
```

The engine lazy-loads `ProsusAI/finbert` on first request and falls back to
VADER automatically if the dependencies or model aren't available.

## API

| Endpoint | Description |
| --- | --- |
| `GET /api/analyze/{ticker}` | full sentiment analysis (`?refresh=true` to bypass cache) |
| `GET /api/price/{ticker}?range=1D\|5D\|1M\|6M\|1Y` | price history |
| `GET /api/health` · `GET /api/sources` | status + source list |

## Tests & diagnostics

```bash
cd backend && source .venv/bin/activate
pytest                                 # offline unit + e2e tests (no network)
python -m scripts.check_sources AAPL   # probe which sources your network allows
```

## Network note

Outbound access depends on the environment's network policy. The platform needs
egress to the source endpoints above to pull live data; in a locked-down
network, sources return empty/error (shown honestly in the UI) rather than
crashing. Run `scripts/check_sources.py` to see what your environment permits.
