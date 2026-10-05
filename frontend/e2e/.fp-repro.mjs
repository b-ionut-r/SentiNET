import { fixture, installMocks, startPreview } from "./mock.mjs";
const { chromium } = await import("playwright");
const OUT = "/tmp/claude-0/-home-user-SentiNET/bb7d6257-211d-5596-8667-847abc67ebae/scratchpad/fe3";
const PORT = 4471;
const preview = await startPreview(PORT);
const browser = await chromium.launch();
const sse = (a) => `event: result\ndata: ${JSON.stringify(a)}\n\n`;
try {
  for (const mode of ["notone", "tone"]) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "en-US", timezoneId: "America/New_York" });
    await installMocks(ctx);
    const page = await ctx.newPage();
    if (mode === "notone") {
      const a = fixture("analysis.NVDA.json");
      a.tone = null;
      await page.route("**/api/analyze/NVDA/stream*", (r) => r.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: sse(a) }));
      await page.route("**/api/history/NVDA*", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ticker: "NVDA", days: 90, points: [], lags: [], best_lag: null, interpretation: "", status: { tone: "error: GDELT is rate-limiting" } }) }));
    }
    await page.goto(`http://127.0.0.1:${PORT}/t/NVDA`);
    await page.waitForSelector("#verdict");
    await page.locator("#price canvas").first().waitFor();
    await page.waitForTimeout(2500);
    await page.locator("#price").screenshot({ path: `${OUT}/fp-${mode}-a.png` });
    await page.evaluate(() => {
      window.__sizes = [];
      for (const lbl of ["Candlestick price chart", "Daily volume", "Daily news tone histogram"]) {
        const el = document.querySelector(`[aria-label="${lbl}"]`);
        new ResizeObserver((es) => es.forEach((e) => window.__sizes.push(`${lbl.split(" ")[1]} ${Math.round(e.contentRect.width)}x${Math.round(e.contentRect.height)} @${Math.round(performance.now())}`))).observe(el);
      }
      window.addEventListener("resize", () => window.__sizes.push(`win ${innerWidth}x${innerHeight} @${Math.round(performance.now())}`));
    });
    await page.waitForTimeout(300);
    const t0 = Date.now();
    await page.screenshot({ path: `${OUT}/fp-${mode}-full.png`, fullPage: true });
    await page.waitForTimeout(800);
    await page.locator("#price").screenshot({ path: `${OUT}/fp-${mode}-b.png` });
    console.log(mode, "done", Date.now() - t0, await page.evaluate(() => window.__sizes.join("\n")));
    await ctx.close();
  }
} finally {
  await browser.close();
  preview.kill();
}
