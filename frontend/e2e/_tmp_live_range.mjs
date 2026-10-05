// Temporary: reproduce the 5D -> daily range error against the live backend for a given dist.
import { spawn } from "node:child_process";
process.env.PLAYWRIGHT_BROWSERS_PATH ??= "/opt/pw-browsers";
const { chromium } = await import("playwright");
const [outDir, port, ticker = "NVDA"] = process.argv.slice(2);
const preview = spawn(process.execPath, ["node_modules/vite/bin/vite.js", "preview", "--outDir", outDir, "--port", port, "--strictPort", "--host", "127.0.0.1"], { stdio: ["ignore", "pipe", "inherit"], env: process.env });
await new Promise((res) => preview.stdout.on("data", (d) => String(d).includes(port) && res()));
const browser = await chromium.launch();
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, locale: "en-US" });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}/t/${ticker}`);
  await page.waitForSelector("#verdict", { timeout: 120000 });
  await page.evaluate(() => document.getElementById("price")?.scrollIntoView({ block: "start" }));
  await page.locator("#price").getByRole("radio", { name: "5D" }).click();
  await page.waitForTimeout(4000);
  const before = errors.length;
  for (const r of ["1M", "5D", "6M", "5D", "5Y"]) {
    await page.locator("#price").getByRole("radio", { name: r }).click();
    await page.waitForTimeout(4000);
  }
  console.log(outDir, "pageerrors during switches:", errors.length - before, errors.slice(before, before + 2));
} finally { await browser.close(); preview.kill(); }
