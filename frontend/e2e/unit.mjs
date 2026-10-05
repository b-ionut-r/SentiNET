/**
 * Unit checks for the pure chart logic, run against the real fixtures:
 *   - priceRows.buildRows: tone dated before the first session is dropped (the first
 *     bar is that day's tone, never weeks of history), weekend tone only rolls across
 *     small gaps, and the time index is strictly ascending.
 *   - lagRule: the chart's "reliable lag" rule matches the backend's (Bonferroni over
 *     7 lags, |r| >= 0.2, n >= 20), so bars can't contradict the sentence above them.
 *
 *   node e2e/unit.mjs
 */
import { build } from "esbuild";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { tmpdir } from "node:os";
import { mkdtempSync } from "node:fs";

import { FIX, ROOT, fixture } from "./mock.mjs";

/** Bundle a TS module to a temp ESM file (types erased) and import it. */
async function load(rel) {
  const dir = mkdtempSync(join(tmpdir(), "sentinet-unit-"));
  const out = join(dir, "mod.mjs");
  await build({ entryPoints: [join(ROOT, rel)], bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error" });
  return import(pathToFileURL(out).href);
}

let failed = 0;
let passed = 0;
function check(name, ok, detail = "") {
  if (ok) passed++;
  else failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
}

const { buildRows, sessionDay } = await load("src/features/intel/priceRows.ts");
const { reliableLag, criticalR, MIN_R } = await load("src/features/intel/lagRule.ts");

/* ---------------------------------------------------------------- buildRows */
const DAY = 864e5;
const analyses = readdirSync(FIX).filter((f) => /^analysis\..+\.json$/.test(f));
for (const file of analyses) {
  const ns = file.split(".")[1];
  const a = fixture(file);
  const tone = a.tone?.series ?? [];
  if (!tone.length) continue;
  for (const range of ["1M", "3M", "6M", "1Y"]) {
    const p = fixture(`price.${ns}.${range}.json`) ?? (ns === "SPARSE" ? fixture(`price.LIVE.${range}.json`) : null);
    if (!p?.candles?.length) continue;
    const rows = buildRows(p.candles, tone, true);
    const first = rows[0];
    const own = tone.filter((t) => t.date === first.time && t.tone != null);
    const expected = own.length ? own[0].tone : null;
    check(
      `${ns} ${range}: first bar carries only its own day's tone`,
      expected == null ? first.tone == null : Math.abs(first.tone - expected) < 1e-9,
      `bar ${first.time}: ${first.tone?.toFixed(3) ?? "—"} vs day ${expected?.toFixed(3) ?? "—"}`,
    );
    // Every rolled-in tone point sits at most 4 calendar days before its session.
    const days = rows.filter((r) => r.candle).map((r) => r.time);
    let worst = 0;
    for (const t of tone) {
      if (t.tone == null || t.date < days[0]) continue;
      const target = days.find((d) => d >= t.date);
      if (target) worst = Math.max(worst, (Date.parse(target) - Date.parse(t.date)) / DAY);
    }
    check(`${ns} ${range}: tone only rolls across small gaps`, worst <= 4, `largest roll ${worst} days`);
    const keys = rows.map((r) => String(r.time));
    check(`${ns} ${range}: time index strictly ascending`, keys.every((k, i) => i === 0 || k > keys[i - 1]));
  }
}
check("sessionDay maps a 04:00Z New York bar to its own date", sessionDay("2026-10-02T04:00:00Z") === "2026-10-02");
check("sessionDay maps a 15:00Z Tokyo bar to the next calendar day", sessionDay("2026-10-01T15:00:00Z") === "2026-10-02");

/* ------------------------------------------------------------------ lagRule */
// The reviewer's contradiction case: p < 0.05 alone, but not after 7 tests.
check("p = 0.028 at n = 58 is NOT reliable (fails Bonferroni)", !reliableLag({ lag_days: 1, r: -0.29, p_value: 0.028, n: 58 }));
check("p = 0.001, |r| = 0.42, n = 60 is reliable", reliableLag({ lag_days: 1, r: 0.42, p_value: 0.001, n: 60 }));
check("tiny p on n = 15 is not reliable (n < 20)", !reliableLag({ lag_days: 0, r: 0.8, p_value: 0.0001, n: 15 }));
check("|r| = 0.15 is never reliable (< 0.2)", !reliableLag({ lag_days: 0, r: 0.15, p_value: 0.0001, n: 900 }));
const r62 = criticalR(62);
check("critical r at n = 62 is ~0.34 (two-sided alpha 0.05/7)", r62 > 0.32 && r62 < 0.36, r62.toFixed(3));
check("critical r never drops below the 0.2 floor", criticalR(5000) === MIN_R, criticalR(5000).toFixed(3));
const live = fixture("history.LIVE.json");
check(
  "real AAPL history: no lag is drawn as reliable (backend calls it noise)",
  live.lags.every((l) => !reliableLag(l)),
  live.lags.map((l) => `${l.lag_days}:${l.r}/${l.p_value}`).join(" "),
);

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
