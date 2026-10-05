/**
 * Shared harness for the e2e scripts: serve the built app with `vite preview`
 * and answer every /api/** call from e2e/fixtures (dev/test data only — never
 * bundled into the app).
 *
 * Fixtures (e2e/fixtures/*.json) are schema-valid samples validated against
 * backend/app/schemas.py. They are seeded from real captured payloads (Google
 * News headlines, StockTwits/Bluesky/HN posts, ApeWisdom, Tradestie, CNN and
 * crypto Fear & Greed, SEC submissions, yfinance prices/analysts/earnings/
 * insiders, Oct 2026); derived fields (scores, narratives, verdicts) and GDELT
 * tone series are illustrative. Verbatim backend output: *.LIVE.json (a real
 * AAPL run, its SSE progress stream and 90-day history; price.LIVE.* are the real
 * AAPL series for that same session), market.json, sources.json, health.json
 * (monitor flag switched on) and the BTC-USD / MSFT price series (captured
 * 2026-10-04). *.SPARSE.json is a real backend run of AAPL with every news/social
 * source forced offline (connection refused) — the genuine degraded output.
 *
 * LIVE and SPARSE payloads carry ticker "AAPL", and the app moves the URL to the
 * canonical ticker. Each browser context therefore remembers which fixture answered
 * for a ticker and keeps serving that fixture's own price/history/snapshots.
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

process.env.PLAYWRIGHT_BROWSERS_PATH ??= "/opt/pw-browsers";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const FIX = join(ROOT, "e2e", "fixtures");
/** Frozen "now" for every page so relative times in fixtures stay meaningful. */
export const NOW = new Date("2026-10-05T00:05:00Z");

export const fixture = (name) => {
  const p = join(FIX, name);
  return existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : null;
};

/** SSE body: progress events (from fixture if present) then the result. */
function sseBody(ticker) {
  const analysis = fixture(`analysis.${ticker}.json`);
  const progress = fixture(`progress.${ticker}.json`) ?? [];
  let body = "";
  for (const p of progress) body += `event: progress\ndata: ${JSON.stringify(p)}\n\n`;
  body += analysis
    ? `event: result\ndata: ${JSON.stringify(analysis)}\n\n`
    : `event: error\ndata: ${JSON.stringify({ detail: `Unknown ticker "${ticker}"`, status: 400 })}\n\n`;
  return body;
}

/** A scan caught mid-flight: the first 38 events of a real captured stream (LIVE), or a synthetic cut. */
function partialSse(ticker) {
  const real = fixture(`progress.${ticker}.json`) ?? [];
  const events = ticker === "LIVE" ? real.slice(0, 38) : real.filter((p) => p.stage === "resolve" || p.stage === "source");
  return events.map((p) => `event: progress\ndata: ${JSON.stringify(p)}\n\n`).join("");
}

const json = (route, data, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(data) });
const never = () => new Promise(() => {});
const SSE_HEADERS = { "content-type": "text/event-stream", "cache-control": "no-cache" };

/** Price series to borrow when a fixture namespace has none of its own (same real ticker, same session). */
const PRICE_FALLBACK = { SPARSE: "LIVE" };

/**
 * Answer one /api request from fixtures. `hold` = tickers whose analysis never
 * completes (scan screenshot). `calls` (optional) records "METHOD /path" for assertions.
 * `state.alias` maps a payload ticker (AAPL) to the fixture namespace that answered for it (LIVE).
 */
export async function handleApi(route, hold = new Set(), calls = null, state = { alias: new Map() }) {
  const req = route.request();
  const url = new URL(req.url());
  const path = url.pathname.replace(/^\/api/, "");
  const method = req.method();
  calls?.push(`${method} ${path}${url.search}`);
  let m;
  const ns = (t) => state.alias.get(t) ?? t;
  const remember = (t) => {
    const a = fixture(`analysis.${t}.json`);
    if (a && a.ticker !== t) state.alias.set(a.ticker, t);
  };

  if ((m = path.match(/^\/analyze\/([^/]+)\/stream$/))) {
    const t = ns(decodeURIComponent(m[1]).toUpperCase());
    if (!hold.has(t)) remember(t);
    return route.fulfill({ status: 200, headers: SSE_HEADERS, body: hold.has(t) ? partialSse(t) : sseBody(t) });
  }
  if ((m = path.match(/^\/analyze\/([^/]+)$/))) {
    const t = ns(decodeURIComponent(m[1]).toUpperCase());
    if (hold.has(t)) return never();
    remember(t);
    const a = fixture(`analysis.${t}.json`);
    return a ? json(route, a) : json(route, { detail: `Unknown ticker "${t}" — try a symbol like AAPL or BTC-USD.` }, 400);
  }
  if ((m = path.match(/^\/price\/([^/]+)$/))) {
    const raw = decodeURIComponent(m[1]).toUpperCase();
    const t = ns(raw);
    const r = url.searchParams.get("range") ?? "3M";
    const p = fixture(`price.${t}.${r}.json`) ?? (PRICE_FALLBACK[t] ? fixture(`price.${PRICE_FALLBACK[t]}.${r}.json`) : null);
    return json(route, p ?? { ticker: raw, range: r, interval: "1d", currency: "USD", candles: [], available: false, error: "Yahoo Finance returned no candles for this range." });
  }
  if ((m = path.match(/^\/history\/([^/]+)$/))) {
    const raw = decodeURIComponent(m[1]).toUpperCase();
    const h = fixture(`history.${ns(raw)}.json`);
    return h ? json(route, h) : json(route, { ticker: raw, days: 90, points: [], lags: [], best_lag: null, interpretation: "", status: { tone: "empty" } });
  }
  if (path === "/market") return json(route, fixture("market.json"));
  if (path === "/search") {
    const q = (url.searchParams.get("q") ?? "").toLowerCase();
    const all = fixture("search.json") ?? [];
    return json(route, all.filter((s) => s.symbol.toLowerCase().startsWith(q) || s.name.toLowerCase().includes(q)).slice(0, 8));
  }
  if (path === "/watchlist" || path.startsWith("/watchlist/")) return json(route, fixture("watchlist.json"));
  if ((m = path.match(/^\/snapshots\/([^/]+)$/))) {
    // Only tickers with a stored-history fixture have looks; everything else is a first look.
    return json(route, fixture(`snapshots.${ns(decodeURIComponent(m[1]).toUpperCase())}.json`) ?? []);
  }
  if (path === "/alerts" && method === "POST") {
    const body = JSON.parse(req.postData() ?? "{}");
    return json(route, { id: 99, enabled: true, created_at: NOW.toISOString(), last_triggered_at: null, threshold: null, ...body }, 201);
  }
  if (path === "/alerts") return json(route, fixture("alerts.json"));
  if (path.startsWith("/alerts/events")) return json(route, fixture("alert-events.json"));
  if (path.startsWith("/alerts/")) return route.fulfill({ status: 204, body: "" });
  if (path === "/sources") return json(route, fixture("sources.json"));
  if (path === "/health") return json(route, fixture("health.json"));
  if (path === "/lab/score") return json(route, fixture("lab-score.json"));
  return json(route, { detail: `No fixture for ${method} ${path}` }, 404);
}

/** Route a browser context: block the outside world, serve /api from fixtures and logos from disk. */
export async function installMocks(ctx, { hold = new Set(), calls = null } = {}) {
  // Playwright matches the most recently registered route first: block the network, then allow fixtures.
  await ctx.route(/^https?:\/\/(?!127\.0\.0\.1)/, (route) => route.abort());
  const state = { alias: new Map() };
  await ctx.route("**/api/**", (route) => handleApi(route, hold, calls, state));
  await ctx.route("https://logos.stocktwits-cdn.com/**", (route) => {
    const file = join(FIX, "logos", new URL(route.request().url()).pathname.slice(1));
    return existsSync(file) ? route.fulfill({ path: file, contentType: "image/png" }) : route.fulfill({ status: 404 });
  });
}

/** Start `vite preview` on `port`; resolves with the child process once it listens. */
export function startPreview(port) {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [join(ROOT, "node_modules", "vite", "bin", "vite.js"), "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"], {
      cwd: ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });
    proc.stdout.on("data", (d) => String(d).includes(String(port)) && resolve(proc));
    proc.stderr.on("data", (d) => process.stderr.write(d));
    proc.on("exit", (code) => reject(new Error(`vite preview exited (${code}) — is port ${port} already in use?`)));
    setTimeout(() => reject(new Error("vite preview did not start")), 20000);
  });
}
