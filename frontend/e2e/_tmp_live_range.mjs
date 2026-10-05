// Temporary live verification (deleted after use).
import { spawn } from "node:child_process";
process.env.PLAYWRIGHT_BROWSERS_PATH ??= "/opt/pw-browsers";
const { chromium } = await import("playwright");
const OUT = process.env.OUT;
const port = "4465";
const preview = spawn(process.execPath, ["node_modules/vite/bin/vite.js", "preview", "--port", port, "--strictPort", "--host", "127.0.0.1"], { stdio: ["ignore", "pipe", "inherit"], env: process.env });
await new Promise((res) => preview.stdout.on("data", (d) => String(d).includes(port) && res()));
const browser = await chromium.launch();
const tickers = process.argv.slice(2);
try {
  for (const spec of tickers) {
    const [t, w = "1440", theme = "dark"] = spec.split(":");
    const width = Number(w);
    const ctx = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 900 }, deviceScaleFactor: 1, colorScheme: theme, locale: "en-US", timezoneId: "America/New_York" });
    await ctx.addInitScript((th) => { try { localStorage.setItem("sentinet.theme", th); } catch {} }, theme);
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${port}/t/${encodeURIComponent(t)}`);
    await page.waitForSelector("#verdict", { timeout: 150000 });
    await page.waitForTimeout(4000);
    const info = await page.evaluate(() => {
      const txt = (sel) => document.querySelector(sel)?.textContent?.replace(/\s+/g, " ").trim() ?? null;
      const head = txt("header.intel-head");
      const sm = [...document.querySelectorAll("#smart-money section")].map((s) => s.textContent.replace(/\s+/g, " ").slice(0, 400));
      const comps = [...document.querySelectorAll("#verdict ul")].pop();
      const shares = comps ? [...comps.querySelectorAll("li [tabindex]")].map((g) => [...g.children].map((c) => c.textContent.trim()).filter(Boolean).join(" ")) : [];
      const pulse = [...(document.querySelector("#narratives")?.querySelectorAll("div.mt-1.flex.justify-between > span") ?? [])].map((s) => s.textContent);
      const caseText = txt("#case")?.slice(0, 300);
      const n = document.getElementById("narratives")?.getBoundingClientRect();
      const c = document.getElementById("case")?.getBoundingClientRect();
      return { head: head?.slice(0, 220), sm, shares, pulse, caseText, gap: n && c ? Math.round(c.top - n.bottom) : null, scrollW: document.documentElement.scrollWidth, vw: innerWidth };
    });
    console.log(`\n=== ${spec}`, JSON.stringify(info, null, 1), "errors:", errors);
    await page.screenshot({ path: `${OUT}/${t}.${width}.${theme}.png`, fullPage: true });
    const h = await page.evaluate(() => document.documentElement.scrollHeight);
    for (let y = 0, i = 0; y < h && i < 8; y += 1100, i++) {
      await page.screenshot({ path: `${OUT}/${t}.${width}.${theme}.p${i}.png`, fullPage: true, clip: { x: 0, y, width, height: Math.min(1100, h - y) } });
    }
    await ctx.close();
  }
} finally { await browser.close(); preview.kill(); }
