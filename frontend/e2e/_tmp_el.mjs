import { spawn } from "node:child_process";
process.env.PLAYWRIGHT_BROWSERS_PATH ??= "/opt/pw-browsers";
const { chromium } = await import("playwright");
const OUT = process.env.OUT;
const port = "4466";
const preview = spawn(process.execPath, ["node_modules/vite/bin/vite.js", "preview", "--port", port, "--strictPort", "--host", "127.0.0.1"], { stdio: ["ignore", "pipe", "inherit"], env: process.env });
await new Promise((res) => preview.stdout.on("data", (d) => String(d).includes(port) && res()));
const browser = await chromium.launch();
try {
  for (const spec of process.argv.slice(2)) {
    const [path, sel, w = "1440", theme = "dark", name] = spec.split("|");
    const width = Number(w);
    const ctx = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 900 }, deviceScaleFactor: width < 600 ? 2 : 1, colorScheme: theme, locale: "en-US", timezoneId: "America/New_York" });
    await ctx.addInitScript((th) => { try { localStorage.setItem("sentinet.theme", th); } catch {} }, theme);
    const page = await ctx.newPage();
    await page.goto(`http://127.0.0.1:${port}${path}`);
    await page.waitForSelector(sel, { timeout: 150000 });
    await page.waitForTimeout(3500);
    await page.locator(sel).first().scrollIntoViewIfNeeded();
    await page.locator(sel).first().screenshot({ path: `${OUT}/${name}.png` });
    await ctx.close();
  }
} finally { await browser.close(); preview.kill(); }
