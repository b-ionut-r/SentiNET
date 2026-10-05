/**
 * Interaction smoke test against the built app with the fixture API: keyboard
 * shortcuts, command palette, intel page interactions, compare, lab, watchlist
 * and alerts. Exits 1 on any failed check, page error or console error.
 *
 *   npm run build && node e2e/smoke.mjs
 */
import { installMocks, NOW, startPreview } from "./mock.mjs";

const { chromium } = await import("playwright");

const PORT = 4174;
const BASE = `http://127.0.0.1:${PORT}`;
const failures = [];
let passed = 0;

function check(name, ok, detail = "") {
  if (ok) {
    passed++;
    console.log(`  ok  ${name}`);
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

const preview = await startPreview(PORT);
const browser = await chromium.launch();
const calls = [];
const errors = [];
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "en-US", timezoneId: "America/New_York", acceptDownloads: true });
await ctx.addInitScript(() => {
  try {
    localStorage.setItem("sentinet.recent", JSON.stringify(["AAPL", "NVDA"]));
  } catch {}
});
await installMocks(ctx, { calls });
const page = await ctx.newPage();
await page.clock.setFixedTime(NOW);
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => m.type() === "error" && !/Failed to load resource/.test(m.text()) && errors.push(`console: ${m.text()}`));
const since = () => {
  const mark = calls.length;
  return () => calls.slice(mark);
};
const settle = (ms = 300) => page.waitForTimeout(ms);

try {
  console.log("global chrome");
  await page.goto(`${BASE}/`);
  await page.waitForSelector("main section");
  await settle();
  await page.keyboard.press("?");
  check("? opens the shortcut help", await page.getByRole("dialog", { name: "Keyboard shortcuts" }).isVisible());
  await page.keyboard.press("Escape");
  check("Escape closes it", !(await page.getByRole("dialog", { name: "Keyboard shortcuts" }).isVisible()));
  const before = await page.evaluate(() => document.documentElement.dataset.theme);
  await page.keyboard.press("t");
  const after = await page.evaluate(() => document.documentElement.dataset.theme);
  check("t toggles the theme", before !== after, `${before} → ${after}`);
  await page.keyboard.press("t");

  await page.keyboard.press("/");
  const palette = page.getByRole("dialog", { name: "Command palette" });
  check("/ opens the command palette", await palette.isVisible());
  check("palette lists recent tickers", (await palette.getByRole("option").allTextContents()).some((t) => t.includes("NVDA")));
  await page.keyboard.type("nvda");
  await page.waitForTimeout(450); // debounce + search
  const searched = calls.some((c) => c.startsWith("GET /search?q=nvda"));
  check("palette search is debounced to one request", calls.filter((c) => c.startsWith("GET /search")).length === 1 && searched, calls.filter((c) => c.startsWith("GET /search")).join(", "));
  await page.keyboard.press("Enter");
  await page.waitForURL(/\/t\/NVDA$/);
  await page.waitForSelector("#verdict");
  check("Enter analyzes the typed ticker", page.url().endsWith("/t/NVDA"));
  check("document title carries the verdict", /^NVDA \d+ · /.test(await page.title()), await page.title());

  await page.keyboard.press("Control+k");
  await page.keyboard.type("ap");
  await page.waitForTimeout(500);
  const first = (await page.getByRole("dialog", { name: "Command palette" }).getByRole("option").first().textContent()) ?? "";
  check("a partial query puts the best symbol match first", first.includes("AAPL"), first);
  await page.keyboard.press("Escape");

  console.log("intel page");
  await page.goto(`${BASE}/t/AAPL`);
  await page.waitForSelector("#verdict");
  await settle(600);
  check("hero shows the headline", (await page.locator("#verdict h2").textContent())?.length > 20);
  check("six score components", (await page.locator("#verdict ul").last().locator("li").count()) === 6);

  let mark = since();
  await page.keyboard.press("r");
  await page.waitForTimeout(800);
  check("r re-runs the analysis with refresh", mark().some((c) => c.startsWith("GET /analyze/AAPL/stream?refresh=true")), mark().join(", "));

  mark = since();
  await page.keyboard.press("w");
  await settle();
  check("w toggles the watchlist (AAPL is watched → DELETE)", mark().some((c) => c === "DELETE /watchlist/AAPL"), mark().join(", "));

  const firstStory = page.locator("#narratives ol > li").first();
  await firstStory.locator("button").first().click();
  check("a story expands to its member articles", (await firstStory.locator("ul a[href^='http']").count()) > 0);

  const reasonLink = page.locator("#verdict a[href^='#narr-']").first();
  if (await reasonLink.count()) {
    const target = (await reasonLink.getAttribute("href")).slice(1);
    await reasonLink.click();
    await page.waitForTimeout(900);
    check("a reason jumps to and opens its story", (await page.locator(`#${target} > button`).getAttribute("aria-expanded")) === "true");
  }

  const signals = page.locator("#signals");
  const total = await signals.locator("ul > li").count();
  await signals.getByLabel("Filter signals").fill("morgan");
  await settle();
  const titles = await signals.locator("ul > li p").allTextContents();
  check("text filter narrows the signal list", titles.length > 0 && titles.length < total && titles.every((t) => /morgan/i.test(t)), `${titles.length}/${total}`);
  await signals.getByLabel("Filter signals").fill("");
  await signals.getByRole("radio", { name: /Bear/ }).click();
  await settle();
  const chips = await signals.locator("ul > li > div:first-child span").allTextContents();
  check("bearish filter keeps only bearish items", chips.length > 0 && chips.every((c) => c.includes("▼")), chips.slice(0, 4).join(" "));

  mark = since();
  await page.locator("#price").getByRole("radio", { name: "1Y" }).click();
  await page.waitForTimeout(600);
  check("range tabs fetch that range", mark().includes("GET /price/AAPL?range=1Y"), mark().join(", "));

  await page.getByRole("button", { name: /Export/ }).click();
  const hrefs = await page.getByRole("menuitem").evaluateAll((els) => els.map((e) => e.getAttribute("href")));
  check("export menu links CSV and JSON", hrefs.includes("/api/export/AAPL.csv") && hrefs.includes("/api/export/AAPL.json"), hrefs.join(", "));
  await page.keyboard.press("Escape");

  console.log("navigation shortcuts");
  await page.mouse.click(5, 300);
  await page.keyboard.press("g");
  await page.keyboard.press("w");
  await page.waitForURL(/\/watchlist$/);
  check("g w opens the watchlist", page.url().endsWith("/watchlist"));

  console.log("watchlist & alerts");
  await page.waitForSelector("text=Alert rules");
  mark = since();
  await page.getByLabel("Alert ticker").fill("nvda");
  await page.getByLabel("Alert condition").selectOption("score_below");
  await page.getByLabel("Threshold").fill("40");
  await page.getByRole("button", { name: "Add rule" }).click();
  await settle();
  check("creating an alert rule POSTs it", mark().includes("POST /alerts"), mark().join(", "));
  mark = since();
  await page.getByRole("button", { name: /Delete alert for AAPL/ }).first().click();
  await settle();
  check("deleting an alert rule handles 204", mark().some((c) => /^DELETE \/alerts\/\d+$/.test(c)) && errors.length === 0, mark().join(", "));

  console.log("compare");
  await page.goto(`${BASE}/compare?t=AAPL`);
  await page.waitForSelector("text=Component matrix");
  await page.getByLabel("Add ticker to compare").fill("msft");
  await page.keyboard.press("Enter");
  await page.waitForURL(/t=AAPL%2CMSFT|t=AAPL,MSFT/);
  check("adding a ticker updates the shareable URL", /t=AAPL(%2C|,)MSFT/.test(page.url()), page.url());
  await page.waitForSelector("text=Microsoft");
  check("tone lines share one axis with a legend", (await page.locator("figcaption").first().textContent())?.includes("MSFT"));

  await page.goto(`${BASE}/compare?t=AAPL,NVDA,MSFT`);
  await page.waitForSelector("text=Component matrix");
  const swatch = (t) =>
    page.evaluate((sym) => {
      const chip = [...document.querySelectorAll("main span.font-mono")].find((n) => n.textContent === sym)?.parentElement;
      const sw = chip?.querySelector("span[aria-hidden]");
      return sw ? getComputedStyle(sw).backgroundColor : null;
    }, t);
  const swatchesBefore = [await swatch("NVDA"), await swatch("MSFT")];
  await page.getByRole("button", { name: "Remove AAPL" }).click();
  await settle(300);
  const swatchesAfter = [await swatch("NVDA"), await swatch("MSFT")];
  check("removing a ticker never repaints the others", swatchesBefore[0] != null && swatchesBefore.join() === swatchesAfter.join(), `${swatchesBefore.join(" / ")} → ${swatchesAfter.join(" / ")}`);

  console.log("lab");
  await page.goto(`${BASE}/lab`);
  await page.getByRole("button", { name: /^Score$/ }).click();
  await page.waitForSelector("text=Driver terms underlined");
  check("lab scores the pasted lines", (await page.locator("mark").count()) > 0);
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "CSV" }).click()]);
  check("lab exports CSV", download.suggestedFilename() === "sentinet-lab.csv");

  console.log("review fixes");
  // Dialogs take focus and hand it back to their opener.
  await page.goto(`${BASE}/`);
  await page.getByRole("button", { name: "Search tickers and commands" }).click();
  await page.keyboard.press("Escape");
  await settle(150);
  check("closing the palette returns focus to its opener", await page.evaluate(() => document.activeElement?.getAttribute("aria-label") === "Search tickers and commands"));
  await page.keyboard.press("?");
  await settle(150);
  check("help dialog takes focus on open", await page.evaluate(() => !!document.activeElement?.closest("[role=dialog]")));
  await page.keyboard.press("Escape");

  // A failed refresh says so (old data stays, marked stale) instead of looking current.
  await page.goto(`${BASE}/t/AAPL`);
  await page.waitForSelector("#verdict");
  await page.route("**/api/analyze/**", (route) => route.fulfill({ status: 502, contentType: "application/json", body: JSON.stringify({ detail: "Upstream gateway failed" }) }));
  await page.keyboard.press("r");
  await page.waitForSelector("text=Refresh failed", { timeout: 5000 }).catch(() => null);
  check("a failed refresh shows an inline alert", await page.getByRole("alert").filter({ hasText: "Refresh failed" }).isVisible());
  check("the freshness line turns stale", await page.locator("header.intel-head").getByText(/Stale · updated/).isVisible());
  await page.unroute("**/api/analyze/**");

  // The lead/lag chart never calls an uncorrected p < 0.05 "significant".
  await page.evaluate(() => document.getElementById("price")?.scrollIntoView({ block: "start" }));
  await settle(500);
  check("lead/lag copy follows the 7-lag correction", (await page.locator("text=statistically significant").count()) === 0 && (await page.getByText(/survive testing 7 lags|under 20, no lag/).count()) > 0);
  check("section nav tracks the lazy price section", (await page.locator('a[aria-current="location"]').textContent())?.includes("Price"), await page.locator('a[aria-current="location"]').textContent());

  // The URL moves to the backend's canonical ticker (LIVE is a real AAPL payload).
  await page.goto(`${BASE}/t/LIVE`);
  await page.waitForURL(/\/t\/AAPL$/, { timeout: 8000 }).catch(() => null);
  check("intel URL is made canonical", page.url().endsWith("/t/AAPL"), page.url());

  // The catalysts rail pins under the section nav while the longer story column scrolls past.
  await settle(300);
  const rail = await page.evaluate(() => {
    const el = document.querySelector(".lg\\:sticky");
    const narr = document.getElementById("narratives");
    if (!(el instanceof HTMLElement) || !narr) return null;
    const stick = parseFloat(getComputedStyle(el).top);
    const docTop = el.getBoundingClientRect().top + window.scrollY;
    const room = narr.getBoundingClientRect().bottom + window.scrollY - (docTop + el.offsetHeight);
    window.scrollTo(0, docTop - stick + Math.max(0, Math.min(room - 10, 200)));
    return { room, stick, top: el.getBoundingClientRect().top };
  });
  check("catalysts rail is sticky on desktop", rail != null && rail.room > 20 && Math.abs(rail.top - rail.stick) < 2, JSON.stringify(rail));

  // A run with no text items explains itself instead of showing empty filters.
  await page.goto(`${BASE}/t/SPARSE`);
  await page.waitForSelector("#verdict");
  check("zero signals → honest empty state, no filter bar", (await page.getByText("No news or social items were collected this run").isVisible()) && (await page.getByLabel("Filter signals").count()) === 0);

  // Watchlist: a rejected add keeps the input and says why; phones can remove.
  await page.goto(`${BASE}/watchlist`);
  await page.route("**/api/watchlist", (route) =>
    route.request().method() === "POST" ? route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ detail: 'Invalid ticker "ZZZZZZZ"' }) }) : route.fallback(),
  );
  const add = page.getByRole("textbox", { name: "Add ticker" });
  await add.fill("zzzzzzz");
  await add.press("Enter");
  await page.waitForSelector("text=Couldn't add ZZZZZZZ", { timeout: 5000 }).catch(() => null);
  check("a rejected watchlist add shows the reason", await page.getByRole("alert").filter({ hasText: "Invalid ticker" }).isVisible());
  check("…and keeps what was typed", (await add.inputValue()) === "zzzzzzz");
  await page.unroute("**/api/watchlist");
  await page.setViewportSize({ width: 390, height: 844 });
  await settle(200);
  check("phones get a remove button per watched ticker", (await page.getByRole("button", { name: /^Remove / }).filter({ visible: true }).count()) > 0);
  await page.setViewportSize({ width: 1440, height: 900 });

  // Lab: an oversized upload loads what one run can score, and Score works.
  await page.goto(`${BASE}/lab`);
  const big = Array.from({ length: 620 }, (_, i) => `Item ${i + 1}: shares rose after earnings beat`).join("\n");
  await page.locator('input[type="file"]').setInputFiles({ name: "big.txt", mimeType: "text/plain", buffer: Buffer.from(big) });
  await settle(200);
  const loaded = (await page.locator("textarea").inputValue()).split("\n").filter(Boolean).length;
  check("a 620-line upload loads the first 500", loaded === 500, `${loaded} lines`);
  check("…and Score stays enabled", await page.getByRole("button", { name: /^Score$/ }).isEnabled());

  console.log("errors");
  await page.goto(`${BASE}/t/APPL`);
  await page.waitForSelector("text=Did you mean");
  check("unknown ticker suggests close symbols", await page.getByRole("link", { name: /AAPL/ }).first().isVisible());
} catch (err) {
  failures.push(`crashed: ${err.message}`);
} finally {
  await browser.close();
  preview.kill();
}

const real = errors.filter((e) => !/status of 400/.test(e));
if (real.length) failures.push(...real);
console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
}
