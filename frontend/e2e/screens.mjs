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
 * Also fails (exit 1) on page errors, console errors and horizontal overflow.
 * Fixtures and the mock API live in e2e/mock.mjs.
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { installMocks, NOW, ROOT, startPreview } from "./mock.mjs";

const { chromium } = await import("playwright");

const OUT = join(ROOT, "e2e", "screens");
const PORT = 4173;
const BASE = `http://127.0.0.1:${PORT}`;

const args = process.argv.slice(2);
const only = args.filter((a) => !a.startsWith("--"));
const opt = (k) => args.find((a) => a.startsWith(`--${k}=`))?.split("=")[1];
const WIDTHS = (opt("w") ?? "1440,390").split(",").map(Number);
const THEMES = (opt("t") ?? "dark,light").split(",");

const PAGES = [
  { name: "market", path: "/" },
  { name: "intel-aapl", path: "/t/AAPL", wait: "#verdict" },
  { name: "intel-btc", path: "/t/BTC-USD", wait: "#verdict" },
  // Verbatim backend output from a real AAPL run: the page must hold up on genuine payloads.
  { name: "intel-live", path: "/t/LIVE", wait: "#verdict" },
  { name: "intel-scan", path: "/t/LIVE", hold: ["LIVE"], settle: 900 },
  // Degraded run (every text source down, no components): honest empty states, no broken layout.
  { name: "intel-sparse", path: "/t/SPARSE", wait: "#verdict" },
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

const preview = await startPreview(PORT);
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
        await installMocks(ctx, { hold: new Set(pg.hold ?? []) });
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
        // Layout check: nothing may push the page wider than the viewport (no sideways scroll on phones).
        const overflow = await page.evaluate(() => {
          const vw = document.documentElement.clientWidth;
          const extra = document.documentElement.scrollWidth - vw;
          if (extra <= 1) return null;
          const culprits = [...document.querySelectorAll("main *")]
            .filter((el) => el.getBoundingClientRect().right > vw + 1 && el.children.length === 0)
            .slice(0, 3)
            .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).split(" ").slice(0, 3).join(".")} "${(el.textContent ?? "").trim().slice(0, 30)}"`);
          return `${extra}px wider than the viewport: ${culprits.join(" | ")}`;
        });
        if (overflow) errors.push(`${pg.name}@${width}/${theme} overflow: ${overflow}`);
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
        await page.goto("about:blank"); // cancel held requests so they never fall through to the dev proxy
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
