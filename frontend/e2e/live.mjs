/**
 * Live smoke test: serve the built app proxied to a REAL backend (no mocks),
 * open the main pages, wait for real data, and report page errors.
 *
 *   SENTINET_API=http://127.0.0.1:8000 node e2e/live.mjs [TICKER]
 *
 * Read-only: never mutates the watchlist or alerts. Screenshots go to
 * e2e/screens/live/.
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// This sandbox keeps Chromium in /opt/pw-browsers; elsewhere Playwright's default applies.
if (!process.env.PLAYWRIGHT_BROWSERS_PATH && existsSync("/opt/pw-browsers")) process.env.PLAYWRIGHT_BROWSERS_PATH = "/opt/pw-browsers";
const { chromium } = await import("playwright");

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "e2e", "screens", "live");
const PORT = Number(process.env.E2E_PORT) || 4180;
const BASE = `http://127.0.0.1:${PORT}`;
const TICKER = process.argv[2] ?? "AAPL";

const preview = spawn(process.execPath, [join(ROOT, "node_modules", "vite", "bin", "vite.js"), "preview", "--port", String(PORT), "--strictPort", "--host", "127.0.0.1"], {
  cwd: ROOT,
  stdio: ["ignore", "pipe", "inherit"],
  env: process.env,
});
await new Promise((resolve, reject) => {
  preview.stdout.on("data", (d) => String(d).includes(String(PORT)) && resolve());
  preview.on("exit", (c) => reject(new Error(`vite preview exited (${c})`)));
});

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch();
const errors = [];
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, locale: "en-US" });
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));

const steps = [
  { name: "intel", path: `/t/${TICKER}`, until: "#verdict", timeout: 90_000 },
  {
    // The backend normalises symbols; the page must move to the canonical URL (served from cache).
    name: "canonical",
    path: `/t/$${TICKER.toLowerCase()}`,
    until: "#verdict",
    timeout: 60_000,
    act: async () => {
      await page.waitForURL(new RegExp(`/t/${TICKER}$`), { timeout: 10_000 });
      console.log(`  canonical URL: ${new URL(page.url()).pathname}`);
    },
  },
  { name: "market", path: "/", until: "main section, main [role=alert], main p", timeout: 60_000 },
  {
    name: "compare",
    path: `/compare?t=${TICKER},MSFT,NVDA`,
    until: "text=Key stats",
    timeout: 120_000,
    act: () => page.waitForSelector("text=/Analyzing .*first runs/", { state: "detached", timeout: 120_000 }),
  },
  { name: "watchlist", path: "/watchlist", until: "main section", timeout: 20_000 },
  { name: "lab", path: "/lab", until: "main section", timeout: 20_000, act: async () => {
      await page.getByRole("button", { name: /^score$/i }).click();
      await page.waitForSelector("text=Summary", { timeout: 30_000 });
    } },
];
try {
  for (const s of steps) {
    const t0 = Date.now();
    await page.goto(BASE + s.path, { waitUntil: "load" });
    await page.waitForSelector(s.until, { timeout: s.timeout });
    if (s.act) await s.act();
    await page.waitForTimeout(800);
    await page.screenshot({ path: join(OUT, `${s.name}.png`), fullPage: true });
    console.log(`${s.name}: ok in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  }
} finally {
  await browser.close();
  preview.kill();
}
if (errors.length) {
  console.error(errors.join("\n"));
  process.exitCode = 1;
}
