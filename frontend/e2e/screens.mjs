/**
 * Visual QA: serve the built app (vite preview), answer every /api/** call
 * from e2e/fixtures (dev/test data only — never bundled), and screenshot each
 * page at 1440px and 390px in dark and light themes.
 *
 *   npm run build && npm run screens            # everything
 *   node e2e/screens.mjs intel-aapl --w=1440 --t=dark   # one page / width / theme
 *
 * Output: e2e/screens/<page>.<width>.<theme>.png  (--parts also writes full-res slices)
 *
 * Fixtures (e2e/fixtures/*.json) are schema-valid samples validated against
 * backend/app/schemas.py. They are seeded from real captured payloads (Google
 * News headlines, StockTwits/Bluesky/HN posts, ApeWisdom, Tradestie, CNN and
 * crypto Fear & Greed, SEC submissions, yfinance prices/analysts/earnings/
 * insiders, Oct 2026); derived fields (scores, narratives, verdicts) and GDELT
 * tone series are illustrative. *.LIVE.json is verbatim backend output.
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

process.env.PLAYWRIGHT_BROWSERS_PATH ??= "/opt/pw-browsers";
const { chromium } = await import("playwright");

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FIX = join(ROOT, "e2e", "fixtures");
const OUT = join(ROOT, "e2e", "screens");
const PORT = 4173;
const BASE = `http://127.0.0.1:${PORT}`;
const NOW = new Date("2026-10-04T21:45:00Z");

const args = process.argv.slice(2);
const only = args.filter((a) => !a.startsWith("--"));
const opt = (k) => args.find((a) => a.startsWith(`--${k}=`))?.split("=")[1];
const WIDTHS = (opt("w") ?? "1440,390").split(",").map(Number);
const THEMES = (opt("t") ?? "dark,light").split(",");

const fixture = (name) => {
  const p = join(FIX, name);
  return existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : null;
};

/** SSE body: progress events (from fixture if present) then the result. */
function sseBody(ticker, withResult) {
  const analysis = fixture(`analysis.${ticker}.json`);
  const progress = fixture(`progress.${ticker}.json`) ?? [];
  let body = "";
  for (const p of progress) body += `event: progress\ndata: ${JSON.stringify(p)}\n\n`;
  if (withResult) {
    body += analysis
      ? `event: result\ndata: ${JSON.stringify(analysis)}\n\n`
      : `event: error\ndata: ${JSON.stringify({ detail: `Unknown ticker "${ticker}"`, status: 400 })}\n\n`;
  }
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

/** Mock API. `hold` = tickers whose analysis never completes (scan screenshot). */
async function handleApi(route, hold) {
  const req = route.request();
  const url = new URL(req.url());
  const path = url.pathname.replace(/^\/api/, "");
  const method = req.method();
  let m;

  if ((m = path.match(/^\/analyze\/([^/]+)\/stream$/))) {
    const t = decodeURIComponent(m[1]).toUpperCase();
    if (hold.has(t)) {
      return route.fulfill({ status: 200, headers: { "content-type": "text/event-stream", "cache-control": "no-cache" }, body: partialSse(t) });
    }
    return route.fulfill({ status: 200, headers: { "content-type": "text/event-stream", "cache-control": "no-cache" }, body: sseBody(t, true) });
  }
  if ((m = path.match(/^\/analyze\/([^/]+)$/))) {
    const t = decodeURIComponent(m[1]).toUpperCase();
    if (hold.has(t)) return never();
    const a = fixture(`analysis.${t}.json`);
    return a ? json(route, a) : json(route, { detail: `Unknown ticker "${t}" — try a symbol like AAPL or BTC-USD.` }, 400);
  }
  if ((m = path.match(/^\/price\/([^/]+)$/))) {
    const t = decodeURIComponent(m[1]).toUpperCase();
    const r = url.searchParams.get("range") ?? "3M";
    return json(route, fixture(`price.${t}.${r}.json`) ?? { ticker: t, range: r, interval: "1d", currency: "USD", candles: [], available: false, error: "Yahoo Finance returned no candles for this range." });
  }
  if ((m = path.match(/^\/history\/([^/]+)$/))) {
    const t = decodeURIComponent(m[1]).toUpperCase();
    const h = fixture(`history.${t}.json`);
    return h ? json(route, h) : json(route, { ticker: t, days: 90, points: [], lags: [], best_lag: null, interpretation: "", status: { tone: "empty" } });
  }
  if (path === "/market") return json(route, fixture("market.json"));
  if (path === "/search") {
    const q = (url.searchParams.get("q") ?? "").toLowerCase();
    const all = fixture("search.json") ?? [];
    return json(route, all.filter((s) => s.symbol.toLowerCase().startsWith(q) || s.name.toLowerCase().includes(q)).slice(0, 8));
  }
  if (path === "/watchlist") return json(route, fixture("watchlist.json"));
  if (path.startsWith("/watchlist/")) return json(route, fixture("watchlist.json"));
  if (path.startsWith("/snapshots/")) return json(route, fixture("snapshots.AAPL.json"));
  if (path === "/alerts" && method === "POST") {
    const body = JSON.parse(req.postData() ?? "{}");
    return json(route, { id: 99, enabled: true, created_at: NOW.toISOString(), last_triggered_at: null, threshold: null, ...body });
  }
  if (path === "/alerts") return json(route, fixture("alerts.json"));
  if (path.startsWith("/alerts/events")) return json(route, fixture("alert-events.json"));
  if (path.startsWith("/alerts/")) return json(route, { ok: true });
  if (path === "/sources") return json(route, fixture("sources.json"));
  if (path === "/health") return json(route, fixture("health.json"));
  if (path === "/lab/score") return json(route, fixture("lab-score.json"));
  return json(route, { detail: `No fixture for ${method} ${path}` }, 404);
}

const PAGES = [
  { name: "market", path: "/" },
  { name: "intel-aapl", path: "/t/AAPL", wait: "#verdict" },
  { name: "intel-btc", path: "/t/BTC-USD", wait: "#verdict" },
  // Real backend output captured before the analytics module existed: sparse/stub data must not break the page.
  { name: "intel-live", path: "/t/LIVE", wait: "#verdict" },
  { name: "intel-scan", path: "/t/LIVE", hold: ["LIVE"], settle: 900 },
  { name: "intel-error", path: "/t/APPL", settle: 600 },
  { name: "compare", path: "/compare?t=AAPL,NVDA,MSFT", settle: 1500 },
  { name: "watchlist", path: "/watchlist" },
  {
    name: "lab",
    path: "/lab",
    act: async (page) => {
      await page.getByRole("button", { name: /score/i }).first().click();
      await page.waitForTimeout(500);
    },
  },
  {
    name: "palette",
    path: "/",
    viewportOnly: true,
    act: async (page) => {
      await page.keyboard.press("Control+k");
      await page.keyboard.type("ap");
      await page.waitForTimeout(600);
    },
  },
];

function startPreview() {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [join(ROOT, "node_modules", "vite", "bin", "vite.js"), "preview", "--port", String(PORT), "--strictPort", "--host", "127.0.0.1"], {
      cwd: ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const onData = (d) => {
      if (String(d).includes(String(PORT))) resolve(proc);
    };
    proc.stdout.on("data", onData);
    proc.stderr.on("data", (d) => process.stderr.write(d));
    proc.on("exit", (code) => reject(new Error(`vite preview exited (${code}) — is port ${PORT} already in use?`)));
    setTimeout(() => reject(new Error("vite preview did not start")), 20000);
  });
}

const preview = await startPreview();
mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch();
const errors = [];
try {
  for (const pg of PAGES) {
    if (only.length && !only.some((o) => pg.name.includes(o))) continue;
    for (const width of WIDTHS) {
      for (const theme of THEMES) {
        const ctx = await browser.newContext({
          viewport: { width, height: width < 600 ? 844 : 900 },
          deviceScaleFactor: width < 600 ? 2 : 1,
          reducedMotion: "reduce",
          colorScheme: theme,
          locale: "en-US",
          timezoneId: "America/New_York",
        });
        await ctx.addInitScript((t) => {
          try {
            localStorage.setItem("sentinet.theme", t);
            localStorage.setItem("sentinet.recent", JSON.stringify(["AAPL", "NVDA", "BTC-USD"]));
          } catch {}
        }, theme);
        const hold = new Set(pg.hold ?? []);
        // Playwright matches the most recently registered route first: block the network, then allow fixtures.
        await ctx.route(/^https?:\/\/(?!127\.0\.0\.1)/, (route) => route.abort());
        await ctx.route("**/api/**", (route) => handleApi(route, hold));
        await ctx.route("https://logos.stocktwits-cdn.com/**", (route) => {
          const file = join(FIX, "logos", new URL(route.request().url()).pathname.slice(1));
          return existsSync(file) ? route.fulfill({ path: file, contentType: "image/png" }) : route.fulfill({ status: 404 });
        });
        const page = await ctx.newPage();
        await page.clock.setFixedTime(NOW);
        page.on("pageerror", (e) => errors.push(`${pg.name}@${width}/${theme}: ${e.message}`));
        page.on("console", (msg) => msg.type() === "error" && !/Failed to load resource/.test(msg.text()) && errors.push(`${pg.name}@${width}/${theme} console: ${msg.text()}`));
        if (process.env.DEBUG) console.log("goto", pg.path, width, theme);
        await page.goto(BASE + pg.path, { waitUntil: "load", timeout: 20000 });
        if (process.env.DEBUG) console.log("loaded");
        await page.waitForSelector("main", { timeout: 10000 });
        if (pg.wait) await page.waitForSelector(pg.wait, { timeout: 10000 });
        await page.waitForTimeout(250); // let effects (hotkeys, observers) attach
        if (pg.act) await pg.act(page);
        await page.waitForTimeout(pg.settle ?? 700);
        const file = join(OUT, `${pg.name}.${width}.${theme}.png`);
        await page.screenshot({ path: file, fullPage: !pg.viewportOnly });
        console.log("shot", file);
        if (args.includes("--parts") && !pg.viewportOnly) {
          // Review aid: full-resolution slices of tall pages.
          const h = await page.evaluate(() => document.documentElement.scrollHeight);
          const step = width < 600 ? 1100 : 1000;
          mkdirSync(join(OUT, "parts"), { recursive: true });
          for (let y = 0, i = 0; y < h; y += step, i++) {
            await page.screenshot({ path: join(OUT, "parts", `${pg.name}.${width}.${theme}.${i}.png`), fullPage: true, clip: { x: 0, y, width, height: Math.min(step, h - y) } });
          }
        }
        await ctx.close();
      }
    }
  }
} finally {
  await browser.close();
  preview.kill();
}
if (errors.length) {
  console.error(`\n${errors.length} page error(s):\n` + errors.join("\n"));
  process.exitCode = 1;
}
