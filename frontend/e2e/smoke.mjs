/**
 * Interaction smoke test against the built app with the fixture API: keyboard
 * shortcuts, command palette, intel page interactions, compare, lab, watchlist
 * and alerts. Exits 1 on any failed check, page error or console error.
 *
 *   npm run build && node e2e/smoke.mjs
 */
import { fixture, installMocks, NOW, previewPort, startPreview } from "./mock.mjs";

const { chromium } = await import("playwright");

const PORT = previewPort(4174);
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
const fix = (name) => JSON.parse(JSON.stringify(fixture(name)));

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
  await page.getByRole("textbox", { name: /^Threshold/ }).fill("40");
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

  // Lab: an oversized upload loads in full; one run scores what fits and offers the rest next.
  await page.goto(`${BASE}/lab`);
  const big = Array.from({ length: 620 }, (_, i) => `Item ${i + 1}: shares rose after earnings beat`).join("\n");
  await page.locator('input[type="file"]').setInputFiles({ name: "big.txt", mimeType: "text/plain", buffer: Buffer.from(big) });
  await settle(200);
  const loaded = (await page.locator("textarea").inputValue()).split("\n").filter(Boolean).length;
  check("a 620-line upload loads all 620", loaded === 620, `${loaded} lines`);
  check("…the notice says only the first 500 are scored", await page.getByTestId("lab-limits").getByText("the first 500 of 620 will be scored, 120 left out").isVisible());
  const labBodies = [];
  await page.route("**/api/lab/score", (route) => {
    labBodies.push(JSON.parse(route.request().postData() ?? "{}"));
    return route.fallback();
  });
  await page.getByRole("button", { name: "Score 500" }).click();
  await page.waitForSelector("text=Driver terms underlined");
  check("…and Score sends exactly the first 500", labBodies.at(-1)?.texts?.length === 500 && labBodies.at(-1).texts[499] === "Item 500: shares rose after earnings beat", `${labBodies.at(-1)?.texts?.length}`);
  await page.getByRole("button", { name: "Load the other 120" }).click();
  const rest = (await page.locator("textarea").inputValue()).split("\n");
  check("'Load the other 120' queues the next batch", rest.length === 120 && rest[0] === "Item 501: shares rose after earnings beat", `${rest.length}: ${rest[0]}`);

  // Over-long texts are cut and the 250,000-character budget is respected, said up front.
  const words = (n) => Array.from({ length: n }, (_, i) => `rally${i % 10}`).join(" ");
  const heavy = [words(2000), ...Array.from({ length: 29 }, (_, i) => `${i} ${words(1400)}`)].join("\n"); // first ~14k chars, then 29 × ~9.8k
  await page.locator("textarea").fill(heavy);
  const notes = await page.getByTestId("lab-limits").locator("li").allTextContents();
  check("lab says an over-long text will be cut", notes.some((n) => n.startsWith("1 text is longer than 10,000 characters")), notes.join(" | "));
  check("…and that only the first texts within budget are scored", notes.some((n) => /^Over the 250,000-character budget of one run: the first 25 of 30 items/.test(n)), notes.join(" | "));
  await page.getByRole("button", { name: "Score 25" }).click();
  await page.waitForSelector("text=Driver terms underlined");
  const sent = labBodies.at(-1)?.texts ?? [];
  const sentChars = sent.reduce((n, t) => n + [...t].length, 0);
  check("…the request fits the backend's limits", sent.length === 25 && sent.every((t) => [...t].length <= 10_000) && sentChars <= 250_000, `${sent.length} texts, ${sentChars} chars`);
  await page.unroute("**/api/lab/score");

  // A busy lab (503 + Retry-After) is waited out with a countdown, then retried.
  await page.locator("textarea").fill("Nvidia beats and raises\nTesla recalls 120,000 vehicles");
  let tries = 0;
  await page.route("**/api/lab/score", (route) => {
    tries++;
    return tries === 1
      ? route.fulfill({ status: 503, headers: { "Retry-After": "2" }, contentType: "application/json", body: JSON.stringify({ detail: "The sentiment lab is busy scoring other requests; retry in a few seconds." }) })
      : route.fallback();
  });
  await page.getByRole("button", { name: /^Score$/ }).click();
  const busy = page.getByTestId("lab-busy");
  await busy.waitFor({ timeout: 3000 }).catch(() => null);
  const busyText = (await busy.textContent().catch(() => "")) ?? "";
  check("busy lab shows 'retrying in Ns' with the attempt", /Lab busy — retrying in [12]s… \(retry 1 of 3\)/.test(busyText), busyText);
  check("…not an error", (await page.getByText("Scoring failed").count()) === 0);
  await busy.waitFor({ state: "detached", timeout: 6000 }).catch(() => null);
  await page.waitForSelector("text=Driver terms underlined");
  check("…then retries on its own and scores", tries === 2 && (await busy.count()) === 0, `${tries} tries`);
  await page.unroute("**/api/lab/score");

  // Cancel stops the wait: no error, no further request, previous results stay.
  tries = 0;
  await page.route("**/api/lab/score", (route) => {
    tries++;
    return route.fulfill({ status: 503, headers: { "Retry-After": "5" }, contentType: "application/json", body: JSON.stringify({ detail: "The sentiment lab is busy scoring other requests; retry in a few seconds." }) });
  });
  await page.getByRole("button", { name: /^Score$/ }).click();
  await busy.waitFor({ timeout: 3000 }).catch(() => null);
  await busy.getByRole("button", { name: "Cancel" }).click();
  await page.waitForTimeout(5800);
  check("Cancel stops the retry (one request, no error, Score enabled)", tries === 1 && (await busy.count()) === 0 && (await page.getByText("Scoring failed").count()) === 0 && (await page.getByRole("button", { name: /^Score$/ }).isEnabled()), `${tries} tries`);
  check("…and the last results stay on screen", await page.getByText("Driver terms underlined").isVisible());
  // Still busy after every retry: say so, offer to try again.
  tries = 0;
  await page.unroute("**/api/lab/score");
  await page.route("**/api/lab/score", (route) => {
    tries++;
    return route.fulfill({ status: 503, headers: { "Retry-After": "1" }, contentType: "application/json", body: JSON.stringify({ detail: "The sentiment lab is busy scoring other requests; retry in a few seconds." }) });
  });
  await page.getByRole("button", { name: /^Score$/ }).click();
  await page.waitForSelector("text=Lab still busy", { timeout: 8000 }).catch(() => null);
  check("gives up after 3 retries with 'Lab still busy'", tries === 4 && (await page.getByText("Lab still busy").isVisible()), `${tries} tries`);
  await page.unroute("**/api/lab/score");

  // Watchlist prices carry the snapshot's quote currency.
  await page.goto(`${BASE}/watchlist`);
  await page.waitForSelector("text=Alert rules");
  const usd = page.locator("tbody").getByText("$333.69", { exact: true });
  await usd.waitFor({ timeout: 5000 }).catch(() => null);
  check("watchlist prices show their currency", await usd.isVisible());
  const vod = fix("watchlist.json");
  Object.assign(vod[0], { ticker: "VOD.L", name: "Vodafone Group Plc" });
  vod[0].last = { ...vod[0].last, ticker: "VOD.L", price: 126.8, currency: "GBp" };
  vod[0].previous = { ...vod[0].previous, ticker: "VOD.L", price: 124.1, currency: "GBp" };
  await page.route("**/api/watchlist", (route) => (route.request().method() === "GET" ? route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(vod) }) : route.fallback()));
  await page.reload();
  const pence = page.locator("tbody").getByText("126.80p", { exact: true });
  await pence.waitFor({ timeout: 5000 }).catch(() => null);
  check("a pence listing reads in pence", await pence.isVisible());
  check("…with its move between looks", await page.locator("tbody tr").filter({ hasText: "VOD.L" }).getByText("+2.2%").isVisible());
  await page.unroute("**/api/watchlist");

  console.log("final review fixes");
  const sse = (a) => `event: result\ndata: ${JSON.stringify(a)}\n\n`;

  // Price chart: the axis follows the bars on screen. Switching 5D → a daily range keeps the
  // intraday bars up while the new range loads; they must never be keyed by date. Real clock
  // here: a frozen Date.now() stalls the chart's render loop and hides its errors.
  {
    const live = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "en-US", timezoneId: "America/New_York" });
    await installMocks(live);
    const pg = await live.newPage();
    const chartErrors = [];
    pg.on("pageerror", (e) => chartErrors.push(e.message));
    await pg.goto(`${BASE}/t/NVDA`);
    await pg.waitForSelector("#verdict");
    await pg.evaluate(() => document.getElementById("price")?.scrollIntoView({ block: "start" }));
    await pg.locator("#price").getByRole("radio", { name: "5D" }).click();
    await pg.waitForTimeout(800);
    await pg.route("**/api/price/**", async (route) => {
      await new Promise((r) => setTimeout(r, 1200));
      return route.fallback();
    });
    for (const r of ["1M", "5D", "6M", "5D", "5Y", "1D", "3M"]) {
      await pg.locator("#price").getByRole("radio", { name: r }).click();
      await pg.waitForTimeout(1800);
    }
    check("switching between 5D and daily ranges throws no chart errors", chartErrors.length === 0, `${chartErrors.length}: ${chartErrors.slice(0, 2).join(" | ")}`);
    await live.close();
  }
  await page.goto(`${BASE}/t/NVDA`);
  await page.waitForSelector("#verdict");

  // Bull/bear case lists only what the verdict's reasons above leave out.
  const heroReasons = await page.locator("#verdict ul").first().locator("li").allTextContents();
  const caseText = (await page.locator("#case").textContent()) ?? "";
  check("bull/bear case doesn't repeat the hero's reasons", heroReasons.length > 0 && heroReasons.every((r) => !caseText.includes(r.trim())), heroReasons.join(" | "));

  // Signal pulse: a window of six days or more labels its ends with dates. (A fresh context:
  // this one already maps AAPL to the SPARSE/LIVE payloads that carry that ticker.)
  const desk = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "en-US", timezoneId: "America/New_York" });
  await installMocks(desk);
  const aapl = await desk.newPage();
  await aapl.clock.setFixedTime(NOW);
  aapl.on("pageerror", (e) => errors.push(`pageerror (aapl): ${e.message}`));
  await aapl.goto(`${BASE}/t/AAPL`);
  await aapl.waitForSelector("#narratives ol");
  const axis = await aapl.locator("#narratives").getByText("bullish items above").locator("xpath=..").locator(":scope > span").allTextContents();
  check("pulse axis ends carry distinct dates", axis.length >= 2 && axis[0] !== axis[axis.length - 1] && /[A-Z][a-z]{2} \d/.test(axis[0]), axis.join(" … "));

  // Desktop layout: the case follows the stories directly — no void beside a long rail.
  const gap = await aapl.evaluate(() => {
    const n = document.getElementById("narratives")?.getBoundingClientRect();
    const c = document.getElementById("case")?.getBoundingClientRect();
    const i = document.getElementById("insights")?.getBoundingClientRect();
    return n && c && i ? { gap: Math.round(c.top - n.bottom), topsAligned: Math.abs(n.top - i.top) < 2, caseLeftCol: c.right <= i.left } : null;
  });
  check("bull/bear case sits right under the stories", gap != null && gap.gap <= 20 && gap.topsAligned && gap.caseLeftCol, JSON.stringify(gap));
  // Phones read one column: insights, stories, bull/bear case, then catalysts.
  await aapl.setViewportSize({ width: 390, height: 844 });
  await aapl.waitForTimeout(300);
  const order = await aapl.evaluate(() => {
    const top = (el) => (el ? Math.round(el.getBoundingClientRect().top + scrollY) : null);
    const watch = [...document.querySelectorAll("section.panel h2")].find((h) => h.textContent?.trim() === "Watch next")?.closest("section");
    return [top(document.getElementById("insights")), top(document.getElementById("narratives")), top(document.getElementById("case")), top(watch)];
  });
  check("phones stack insights → stories → case → catalysts", order.every((v, i) => v != null && (i === 0 || v > order[i - 1])), order.join(" < "));
  await desk.close();

  // Score components: shares are effective (n/a inputs carry none), and add to 100.
  await page.goto(`${BASE}/t/BTC-USD`);
  await page.waitForSelector("#verdict");
  const comp = await page.locator("#verdict ul").last().locator("li").evaluateAll((lis) =>
    lis.map((li) => {
      const cells = [...(li.querySelector("[tabindex]")?.children ?? [])].map((e) => e.textContent?.trim() ?? "");
      return { label: cells[0], score: cells[2], share: cells[3] };
    }),
  );
  const pcts = comp.filter((c) => /%$/.test(c.share ?? "")).map((c) => Number(c.share.replace("%", "")));
  check("n/a components show no share", comp.filter((c) => c.score === "n/a").every((c) => c.share === "—"), JSON.stringify(comp));
  check("component shares add to 100%", pcts.length > 0 && pcts.reduce((s, v) => s + v, 0) === 100, pcts.join("+"));

  // A single analyst's target (low == high) gets one label inside the panel, cents precision.
  const one = fix("analysis.AAPL.json");
  Object.assign(one.analysts, { target_low: 0.5, target_high: 0.5, target_mean: 0.5, target_median: 0.5, total: 1, upside_pct: -74 });
  one.quote.price = 1.92;
  await page.route("**/api/analyze/AAPL/stream*", (route) => route.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: sse(one) }));
  await page.goto(`${BASE}/t/AAPL`);
  await page.waitForSelector("#smart-money");
  await page.evaluate(() => document.getElementById("smart-money")?.scrollIntoView({ block: "start" }));
  await settle(300);
  const analysts = page.locator("#smart-money section, #smart-money > *").filter({ hasText: "Price targets" }).first();
  const spill = await analysts.evaluate((panel) => {
    const box = panel.getBoundingClientRect();
    return [...panel.querySelectorAll("div, span")]
      .filter((el) => el.children.length === 0 && (el.textContent ?? "").trim())
      .filter((el) => {
        const r = el.getBoundingClientRect();
        return r.left < box.left - 0.5 || r.right > box.right + 0.5;
      })
      .map((el) => el.textContent);
  });
  const labels = (await analysts.textContent()) ?? "";
  check("single-target labels stay inside the panel", spill.length === 0, spill.join(" | "));
  check("…merged into one label at cents precision", labels.includes("target $0.50 · 1 analyst") && !labels.includes("0.5000"), labels.slice(0, 200));
  await page.unroute("**/api/analyze/AAPL/stream*");

  // Late GDELT tone: the page draws tone from the history meanwhile, then quietly re-reads
  // the analysis and swaps in the version that carries it.
  const late = fix("analysis.NVDA.json");
  late.tone = null;
  // The live wording (backend analytics/insights.py) — the result event carries no progress, so
  // only the insight can trigger the re-read, exactly as for a cached or plain result.
  const LATE_TITLE = "Global news tone not loaded this run";
  late.insights = [{ kind: "quality", severity: "info", polarity: "neutral", title: LATE_TITLE, detail: "GDELT history did not arrive in time for this read; momentum uses the last 48 h of headlines (12) vs the prior days (40) instead." }, ...late.insights];
  await page.route("**/api/analyze/NVDA/stream*", (route) => route.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: sse(late) }));
  // History arrives after the candles: the chart must not narrow under its first fit.
  await page.route("**/api/history/NVDA*", async (route) => {
    await new Promise((r) => setTimeout(r, 1500));
    return route.fallback();
  });
  mark = since();
  await page.goto(`${BASE}/t/NVDA`);
  await page.waitForSelector("#verdict");
  await page.evaluate(() => document.getElementById("price")?.scrollIntoView({ block: "start" }));
  const pricePanel = page.locator("#price section").filter({ hasText: "Price × news tone" }).first();
  await page.locator("#price canvas").first().waitFor();
  const w0 = (await pricePanel.boundingBox())?.width;
  // Hover the candles while the tone history is still on its way: the crosshair is mirrored
  // only onto panes that can place it (lightweight-charts throws "Value is null" otherwise).
  const pbox = await page.locator("#price canvas").first().boundingBox();
  const hoverErrors = errors.length;
  for (let i = 0; i < 12 && pbox; i++) {
    await page.mouse.move(pbox.x + 20 + (i * (pbox.width - 40)) / 12, pbox.y + pbox.height / 2);
    await page.waitForTimeout(60);
  }
  await settle(2000);
  const w1 = (await pricePanel.boundingBox())?.width;
  check("price chart keeps its width when the lead/lag column arrives", w0 != null && w0 === w1, `${w0} → ${w1}`);
  check("hovering the chart while tone history loads throws nothing", errors.length === hoverErrors, errors.slice(hoverErrors).join(" | "));
  await page.mouse.move(5, 5);
  await page.unroute("**/api/history/NVDA*");
  check("tone-less result still draws GDELT tone from the 90-day history", await page.locator("#price").getByText("Daily global news tone (GDELT) in its own pane below").isVisible());
  // The history call brought the tone the result lacked: the server has superseded its cached
  // analysis, so the page re-reads it at once (no 20 s wait) and swaps in the version with tone.
  check("history with tone triggers an immediate re-read", mark().includes("GET /analyze/NVDA"), mark().filter((c) => c.includes("analyze")).join(", "));
  await settle(300);
  check("…and the late tone replaces the 'not loaded this run' result", (await page.locator("#insights").getByText(LATE_TITLE).count()) === 0);

  // History still waiting on GDELT too: the 'not loaded' insight stays, and the page re-reads on its timer.
  const hist = fix("history.NVDA.json");
  hist.status = { ...hist.status, tone: "error: still loading after 22s; continuing in the background (reload to include)" };
  hist.points = hist.points.map((p) => ({ ...p, tone: null, volume: null }));
  await page.route("**/api/history/NVDA*", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(hist) }));
  await page.goto(`${BASE}/`);
  mark = since();
  await page.goto(`${BASE}/t/NVDA`);
  await page.waitForSelector("#verdict");
  await settle(2000);
  check("…without tone anywhere, the insights rail says tone is not loaded yet", await page.locator("#insights").getByText(LATE_TITLE).isVisible());
  check("…and nothing is re-read early", !mark().includes("GET /analyze/NVDA"), mark().filter((c) => c.includes("analyze")).join(", "));
  await page.waitForTimeout(21_000);
  check("the analysis is re-read on its timer without a forced refresh", mark().includes("GET /analyze/NVDA"), mark().filter((c) => c.includes("analyze")).join(", "));
  await settle(300);
  check("…and the late tone replaces the 'not loaded this run' result", (await page.locator("#insights").getByText(LATE_TITLE).count()) === 0);
  await page.unroute("**/api/history/NVDA*");
  await page.unroute("**/api/analyze/NVDA/stream*");

  // Compare: a ticker that fails to analyze says so in every row and can be removed.
  await page.goto(`${BASE}/compare?t=NVDA,XQZQZQ`);
  await page.waitForSelector("text=Component matrix");
  await settle(1200);
  const matrix = page.locator("table").first();
  const col = await matrix.locator("tbody tr").evaluateAll((rows) => rows.map((r) => r.querySelectorAll("td")[2]?.textContent?.trim()));
  check("a failed ticker's matrix column reads '—', never a forever '…'", col.length > 0 && col.every((t) => t === "—"), col.join(" "));
  await page.getByRole("button", { name: "Remove XQZQZQ" }).last().click();
  await page.waitForURL(/t=NVDA$/);
  check("…and its card offers to remove it", page.url().endsWith("t=NVDA"), page.url());

  // Alert form: a real default per condition, range-checked next to the field.
  await page.goto(`${BASE}/watchlist`);
  await page.waitForSelector("text=Alert rules");
  await page.getByLabel("Alert condition").selectOption("score_above");
  const thr = page.getByRole("textbox", { name: /^Threshold/ });
  check("choosing a condition fills its default threshold", (await thr.inputValue()) === "65", await thr.inputValue());
  await page.getByLabel("Alert ticker").fill("nvda");
  check("ticker + default threshold is enough to add", await page.getByRole("button", { name: "Add rule" }).isEnabled());
  await thr.fill("150");
  check("an out-of-range threshold is flagged inline", await page.getByText("Threshold must be between 1 and 99.").isVisible());
  check("…and blocks submit", !(await page.getByRole("button", { name: "Add rule" }).isEnabled()));

  // First-load scan on a phone: every chip (label, count, latency, badge) stays inside the card.
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, locale: "en-US", timezoneId: "America/New_York" });
  await installMocks(phone, { hold: new Set(["LIVE"]) });
  const scan = await phone.newPage();
  scan.on("pageerror", (e) => errors.push(`pageerror (scan): ${e.message}`));
  await scan.goto(`${BASE}/t/LIVE`);
  await scan.waitForSelector("text=Scanning");
  await scan.waitForTimeout(900);
  const scanFit = await scan.evaluate(() => {
    const card = document.querySelector("section.panel")?.getBoundingClientRect();
    const chips = [...document.querySelectorAll("section.panel li")].map((li) => li.getBoundingClientRect());
    return card ? { card: Math.round(card.right), chips: chips.length, worst: Math.round(Math.max(...chips.map((r) => r.right))) } : null;
  });
  check("scan chips fit the card at 390px", scanFit != null && scanFit.chips > 0 && scanFit.worst <= scanFit.card, JSON.stringify(scanFit));
  const badges = await scan.locator("section.panel li").filter({ hasText: /needs .*API key/ }).allTextContents();
  check("…keyless sources say 'key'", badges.length > 0 && badges.every((t) => /key/.test(t)), badges.slice(0, 2).join(" | "));
  await scan.goto("about:blank");
  await phone.close();

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
