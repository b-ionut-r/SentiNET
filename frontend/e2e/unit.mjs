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

/* -------------------------------------------- final review: frontend fixes */
// Pin the zone so local-time labels are deterministic (the review's screenshots are New York).
process.env.TZ = "America/New_York";
const F = await load("src/lib/format.ts");
const { tonePending, TONE_FOLLOW_UP_MS } = await load("src/api/pending.ts");
const { toneOf } = await load("src/features/intel/priceRows.ts");
const { skipReason, SKIP_BADGE } = await load("src/features/intel/scanChip.ts");
const { thresholdProblem, KINDS, needsThreshold } = await load("src/features/watchlist/alertRules.ts");
const { scoreCell100 } = await load("src/lib/sentiment.ts");
const { freshPoints } = await load("src/features/intel/brief.ts");
const { effectiveShares, wholePercents } = await load("src/features/intel/weights.ts");
const { edgeSafe, rangeLabels } = await load("src/components/charts/rangeLabels.ts");

// Currency: Yahoo's minor units are case-sensitive (GBp = pence), never upper-cased into pounds.
check("GBp price reads as pence, not pounds", F.price(126.8, "GBp") === "126.80p", F.price(126.8, "GBp"));
check("GBX price reads as pence", F.price(126.8, "GBX") === "126.80p", F.price(126.8, "GBX"));
check("ZAc price reads as cents", F.price(4512, "ZAc") === "4,512.00c", F.price(4512, "ZAc"));
check("GBP stays pounds", F.price(1.268, "GBP") === "£1.27", F.price(1.268, "GBP"));
check("a pence listing's market cap is in pounds", F.money(29.4e9, "GBp") === "£29.4B", F.money(29.4e9, "GBp"));
check("USD price", F.price(187.2, "USD") === "$187.20", F.price(187.2, "USD"));
check("no currency → no guessed symbol", F.price(187.2, null) === "187.20", F.price(187.2, null));
check("negative minor-unit price keeps the sign first", F.price(-3.5, "GBp") === "−3.50p", F.price(-3.5, "GBp"));
check("negative EPS: sign before the symbol", F.perShare(-0.14, "USD") === "−$0.14", F.perShare(-0.14, "USD"));
check("a $0.50 target is cents precision", F.perShare(0.5, "USD") === "$0.50", F.perShare(0.5, "USD"));
check("sub-dime per-share keeps more digits", F.perShare(0.0123, "USD") === "$0.0123", F.perShare(0.0123, "USD"));
check("sub-dime EPS without extra digits reads in cents", F.perShare(0.04, "USD") === "$0.04" && F.perShare(-0.02, "USD") === "−$0.02", `${F.perShare(0.04, "USD")} ${F.perShare(-0.02, "USD")}`);
check("pence targets", F.perShare(121.82, "GBp") === "121.82p", F.perShare(121.82, "GBp"));
check("majorCurrency maps GBp → GBP", F.majorCurrency("GBp") === "GBP" && F.majorCurrency("usd") === "USD");
check(
  "reporting currency: SHOP.TO (CAD quote) without a stated currency is not labelled CAD",
  F.reportingCurrency({ quote: { currency: "CAD" }, profile: { country: "Canada" } }) === null,
);
check("reporting currency: a US company quoted in USD is USD", F.reportingCurrency({ quote: { currency: "USD" }, profile: { country: "United States" } }) === "USD");
check("reporting currency: an ADR quoted in USD is not assumed USD", F.reportingCurrency({ quote: { currency: "USD" }, profile: { country: "Netherlands" } }) === null);
check(
  "reporting currency: profile.financial_currency wins (SHOP.TO reports USD)",
  F.reportingCurrency({ quote: { currency: "CAD" }, profile: { country: "Canada", financial_currency: "USD" } }) === "USD",
);

// Links: only absolute http(s) URLs from providers become hrefs.
for (const bad of ["javascript:fetch('/api/alerts')", "JAVASCRIPT:alert(1)", " javascript:alert(1)", "java\tscript:alert(1)", "data:text/html,<script>alert(1)</script>", "vbscript:msgbox(1)", "/api/alerts", "//evil.example/x", ""]) {
  check(`safeHref rejects ${JSON.stringify(bad)}`, F.safeHref(bad) === undefined, String(F.safeHref(bad)));
}
check("safeHref keeps https", F.safeHref("https://www.reuters.com/a?b=1") === "https://www.reuters.com/a?b=1");
check("safeHref keeps http", F.safeHref("http://www.bing.com/news/x") === "http://www.bing.com/news/x");
check("safeHref null → undefined", F.safeHref(null) === undefined);

// Late GDELT tone: a result that went out without it is re-read, one with it is not.
const nvda = fixture("analysis.NVDA.json");
const loadingInsight = { kind: "quality", severity: "info", polarity: "neutral", title: "Global news tone still loading", detail: "GDELT history is still being fetched…" };
check("result with tone is never re-read", !tonePending({ tone: nvda.tone, insights: [loadingInsight] }));
check("tone-less result flagged 'still loading' is re-read", tonePending({ tone: null, insights: [loadingInsight] }));
check(
  "tone-less result whose stream said GDELT is still loading is re-read",
  tonePending({ tone: null, insights: [] }, [{ stage: "intel", key: "tone", status: "error", detail: "still loading (rate-limited)" }]),
);
check("tone-less result with no pending signal is not re-read (GDELT simply has nothing)", !tonePending({ tone: null, insights: [] }));
// The live wording since b4c6ad2 (backend/app/analytics/insights.py) — the old regex only knew "still loading".
const liveLoading = {
  kind: "quality",
  severity: "info",
  polarity: "neutral",
  title: "Global news tone not loaded this run",
  detail: "GDELT history did not arrive in time for this read; momentum uses the last 48 h of headlines (12) vs the prior days (40) instead.",
};
check("tone-less result with the live 'not loaded this run' insight is re-read (cached/plain path)", tonePending({ tone: null, insights: [liveLoading] }));
check("…and a cached result whose detail alone says 'did not arrive in time' is re-read", tonePending({ tone: null, insights: [{ ...liveLoading, title: "Global news tone missing" }] }));
check("the live stream detail ('still loading after 22s; …') is re-read", tonePending({ tone: null, insights: [] }, [{ stage: "intel", key: "tone", status: "error", detail: "still loading after 22s; continuing in the background (reload to include)" }]));
check(
  "another feed's gap is not a tone re-read ('Market data not loaded in time')",
  !tonePending({ tone: null, insights: [{ ...liveLoading, title: "Market data not loaded in time", detail: "Analyst ratings did not arrive in time for this read." }] }),
);
check(
  "a hard GDELT failure is not pending ('could not be loaded this run')",
  !tonePending({ tone: null, insights: [{ ...liveLoading, severity: "watch", title: "Some market data unavailable", detail: "GDELT tone could not be loaded this run." }] }),
);
check("a non-quality insight mentioning tone never triggers it", !tonePending({ tone: null, insights: [{ ...liveLoading, kind: "momentum" }] }));
check("a hard stream error is not pending", !tonePending({ tone: null, insights: [] }, [{ stage: "intel", key: "tone", status: "error", detail: "HTTP 429 Too Many Requests" }]));
check("follow-up schedule is bounded and increasing", TONE_FOLLOW_UP_MS.length <= 4 && TONE_FOLLOW_UP_MS.every((v, i) => i === 0 || v > TONE_FOLLOW_UP_MS[i - 1]));

// The chart draws tone from the 90-day history when the analysis went out without it.
const hist = fixture("history.NVDA.json");
const fromHist = toneOf({ tone: null }, hist);
check("no analysis tone → tone series from the history points", fromHist.length > 0 && fromHist.length === hist.points.filter((p) => p.tone != null).length, `${fromHist.length}`);
check("…as TonePoints (date, tone, volume)", fromHist.every((p) => typeof p.date === "string" && typeof p.tone === "number" && "volume" in p));
check("analysis tone wins when present", toneOf(nvda, hist) === nvda.tone.series);
check("neither → empty", toneOf({ tone: null }, undefined).length === 0);

// 5-minute bars rendered on a daily axis (the placeholder frame of a 5D → 1M switch) must
// never produce duplicate dates; intraday bars must stay strictly ascending.
for (const ns of ["NVDA", "AAPL", "LIVE"]) {
  const five = fixture(`price.${ns}.5D.json`);
  if (!five?.candles?.length) continue;
  const asDaily = buildRows(five.candles, [], true).map((r) => String(r.time));
  check(`${ns} 5D candles keyed by day: unique, ascending`, asDaily.every((k, i) => i === 0 || k > asDaily[i - 1]), `${asDaily.length} rows from ${five.candles.length} bars`);
  const shuffled = [...five.candles].reverse().concat(five.candles.slice(0, 3));
  const intraday = buildRows(shuffled, [], false).map((r) => Number(r.time));
  check(`${ns} 5D intraday rows: unique, ascending even from a disordered feed`, intraday.every((k, i) => i === 0 || k > intraday[i - 1]) && intraday.length === new Set(five.candles.map((c) => c.t)).size);
}
const merged = buildRows(
  [
    { t: "2026-10-01T13:30:00Z", o: 10, h: 11, l: 9, c: 10.5, v: 100 },
    { t: "2026-10-01T19:55:00Z", o: 10.5, h: 12, l: 10, c: 11.5, v: 50 },
  ],
  [],
  true,
);
check("same-day bars merge into one OHLC bar", merged.length === 1 && merged[0].candle.o === 10 && merged[0].candle.h === 12 && merged[0].candle.l === 9 && merged[0].candle.c === 11.5 && merged[0].candle.v === 150, JSON.stringify(merged[0]?.candle));

// Signal pulse axis: a 7-day window must not read "Sun 20:00 … Sun 20:00".
const WEEK = 7 * 864e5;
const a0 = F.bucketTime("2026-09-28T00:00:00Z", WEEK, 6 * 36e5);
const a1 = F.bucketTime("2026-10-05T00:00:00Z", WEEK, 6 * 36e5);
check("7-day pulse ends carry distinct dates", a0 !== a1 && /Sep 27/.test(a0) && /Oct 4/.test(a1), `${a0} … ${a1}`);
check("short windows keep weekday + time", F.bucketTime("2026-10-02T18:00:00Z", 2 * 864e5, 6 * 36e5) === "Fri 14:00", F.bucketTime("2026-10-02T18:00:00Z", 2 * 864e5, 6 * 36e5));
check("daily buckets show their UTC calendar day", F.bucketTime("2026-09-28T00:00:00Z", WEEK, 864e5) === "Sep 28", F.bucketTime("2026-09-28T00:00:00Z", WEEK, 864e5));
check("tooltips always carry weekday and date", F.bucketTime("2026-09-28T00:00:00Z", 864e5, 6 * 36e5, true) === "Sun, Sep 27 20:00", F.bucketTime("2026-09-28T00:00:00Z", 864e5, 6 * 36e5, true));
check("midnight renders as 00:00, never 24:00", F.bucketTime("2026-10-02T04:00:00Z", 864e5, 36e5) === "Fri 00:00", F.bucketTime("2026-10-02T04:00:00Z", 864e5, 36e5));

// Scan chips: only a missing API key shows a key.
check("'needs a free API key' → key", skipReason("needs a free API key") === "key");
check("unconfigured keyed source → key", skipReason(null, { requires_key: true, configured: false }) === "key");
check("'n/a for ETFs' → n/a, not key", skipReason("n/a for ETFs") === "na" && SKIP_BADGE.na === "n/a");
check("'disabled in settings' → off", skipReason("disabled in settings") === "off");

// Alert thresholds: a real default per kind, range-checked like the backend.
check("score_change starts filled with its default", KINDS.score_change.initial === "8" && thresholdProblem("score_change", "8") === null);
check("150 for 'rises above' is rejected client-side", thresholdProblem("score_above", "150") === "Must be between 1 and 99");
check("empty threshold explains itself", thresholdProblem("score_below", " ") === "Enter a threshold");
check("non-numeric threshold rejected", thresholdProblem("score_below", "abc") === "Must be a number");
check("kinds without a threshold never block", !needsThreshold("analyst_action") && thresholdProblem("analyst_action", "") === null);

// Compare matrix: what you see is what is coloured.
const c449 = scoreCell100(44.9);
const c452 = scoreCell100(45.2);
check("44.9 and 45.2 render identically (value, mark, tint)", c449.value === 45 && c452.value === 45 && c449.polarity === c452.polarity && c449.tint === c452.tint, `${JSON.stringify(c449)} vs ${JSON.stringify(c452)}`);
check("54.6 rounds to a bullish 55", scoreCell100(54.6).polarity === "bull" && scoreCell100(54.4).polarity === "neutral");

// Bull/bear case doesn't repeat the hero's reasons.
const bull = freshPoints(nvda.brief.bull_points, nvda.verdict.reasons);
check("NVDA bull points already in the hero are dropped", bull.fresh.length === 0 && bull.repeated === nvda.brief.bull_points.length, JSON.stringify(bull));
const extra = freshPoints(["All-time high; market cap $5.7T.", "Insiders bought $2M"], nvda.verdict.reasons);
check("…matching ignores case/trailing punctuation, keeps new evidence", extra.fresh.length === 1 && extra.fresh[0] === "Insiders bought $2M");

// Score components: shares are the backend's effective weights (confidence-scaled, renormalized).
const btc = fixture("analysis.BTC-USD.json");
const shares = effectiveShares(btc.verdict.components);
const sum = [...shares.values()].reduce((s, v) => s + v, 0);
check("n/a components carry no share", !shares.has("analysts") && !shares.has("insiders"));
check("shares sum to 1", Math.abs(sum - 1) < 1e-9, sum.toFixed(6));
const whole = wholePercents(shares);
check("displayed whole percents add to exactly 100", [...whole.values()].reduce((s, v) => s + v, 0) === 100, JSON.stringify([...whole]));
const conf = effectiveShares([
  { key: "news", weight: 0.3, available: true, confidence: 1 },
  { key: "social", weight: 0.3, available: true, confidence: 0 },
]);
check("confidence scales the share (w·(0.35+0.65c))", Math.abs(conf.get("news") - 1 / 1.35) < 1e-9, conf.get("news").toFixed(4));

// Range bar labels stay inside the track; one label when low ≈ high.
check("edgeSafe clamps below 0", edgeSafe(-6).left === "0%" && edgeSafe(-6).transform === "translateX(-0%)");
check("edgeSafe right-aligns at 100", edgeSafe(104).left === "100%" && edgeSafe(104).transform === "translateX(-100%)");
const one = rangeLabels(0.5, 0.5, [{ value: 1.92, kind: "current" }]);
check("single-analyst target (low == high) → one merged label", one.merged, JSON.stringify(one));
const wide = rangeLabels(100, 200, [{ value: 150, kind: "current" }]);
check("a real range keeps separate low/high labels", !wide.merged);

/* ------------------------------------------- last follow-ups: lab limits, busy, watchlist */
const B = await load("src/features/lab/batch.ts");
const { parseRetryAfter, busyRetryAfter, retryWhenBusy, BUSY_RETRIES, MAX_RETRY_AFTER_S } = await load("src/api/busy.ts");
const { priceChange } = await load("src/features/watchlist/priceChange.ts");

// Characters are code points, as the backend's Python len() counts them.
check("an emoji is one character, not two UTF-16 units", B.charCount("🚀🚀 up") === 5, String(B.charCount("🚀🚀 up")));
check("a lone surrogate still counts once", B.charCount("a\ud800b") === 3);
const longWords = Array.from({ length: 2000 }, (_, i) => `word${i % 10}`).join(" "); // 11,999 chars
const clipped = B.clip(longWords);
check("clip keeps at most 10,000 characters", B.charCount(clipped) <= B.MAX_TEXT_CHARS, String(B.charCount(clipped)));
check("…cut at a word boundary, not mid-word", /word\d$/.test(clipped) && longWords.startsWith(clipped + " "), clipped.slice(-12));
const astral = "🚀".repeat(B.MAX_TEXT_CHARS + 5);
const astralCut = B.clip(astral);
check("clip never splits a surrogate pair", B.charCount(astralCut) === B.MAX_TEXT_CHARS && astralCut.length === 2 * B.MAX_TEXT_CHARS);
check("a text under the limit is untouched", B.clip("Apple beats") === "Apple beats");

const fits = B.planBatch(["Apple beats", "Tesla recalls 120,000 vehicles"]);
check("a small batch is sent as is, with no notes", fits.texts.length === 2 && fits.cut === 0 && fits.left === 0 && B.planNotes(fits).length === 0);
const withLong = B.planBatch(["short", "x".repeat(12_011), "y ".repeat(6000).trim(), "also short"]);
check("over-long texts are cut, not rejected", withLong.texts.length === 4 && withLong.cut === 2 && withLong.texts.every((t) => B.charCount(t) <= B.MAX_TEXT_CHARS), JSON.stringify({ n: withLong.texts.length, cut: withLong.cut }));
check("…and the note says how many", B.planNotes(withLong)[0] === "2 texts are longer than 10,000 characters and will be cut to their first 10,000.", B.planNotes(withLong)[0]);
const heavy = Array.from({ length: 30 }, (_, i) => `${i} ` + "Stock rallies on strong earnings ".repeat(300).trim()); // 30 × ~9,900 chars (the live 400 case)
const overBudget = B.planBatch(heavy);
check("over the 250,000-character budget: only the first texts that fit are sent", overBudget.texts.length === 25 && overBudget.chars <= B.MAX_TOTAL_CHARS && overBudget.leftBy === "budget" && overBudget.left === 5, JSON.stringify({ sent: overBudget.texts.length, chars: overBudget.chars }));
check("…in input order", overBudget.texts.every((t, i) => t === heavy[i]));
check("…and the note names the first N", /^Over the 250,000-character budget of one run: the first 25 of 30 items will be scored, 5 left out/.test(B.planNotes(overBudget)[0]), B.planNotes(overBudget)[0]);
const many = B.planBatch(Array.from({ length: 620 }, (_, i) => `Item ${i + 1}: shares rose`));
check("past 500 items: the first 500 are sent", many.texts.length === 500 && many.left === 120 && many.leftBy === "items" && many.texts[499] === "Item 500: shares rose");
check("…and the note says so", /first 500 of 620 will be scored, 120 left out/.test(B.planNotes(many)[0]), B.planNotes(many)[0]);
check("character budget counts what is sent after cutting", B.planBatch(Array.from({ length: 26 }, () => "z".repeat(20_000))).texts.length === 25);
check("blank items are ignored", B.planBatch(["", "  ", "a"]).items === 1);

// Busy lab: 503 + Retry-After is waited out and retried; anything else fails at once.
check("Retry-After seconds", parseRetryAfter("5") === 5);
check("Retry-After as an HTTP date", parseRetryAfter(new Date(Date.parse("2026-10-05T00:00:07Z")).toUTCString(), Date.parse("2026-10-05T00:00:00Z")) === 7);
check("Retry-After is clamped to a sane wait", parseRetryAfter("3600") === MAX_RETRY_AFTER_S && parseRetryAfter("0") === 1);
check("missing or junk Retry-After → null", parseRetryAfter(null) === null && parseRetryAfter("soon") === null);
const busyErr = Object.assign(new Error("The sentiment lab is busy scoring other requests; retry in a few seconds."), { status: 503, retryAfter: 5 });
const outage = Object.assign(new Error("Scoring failed (engine crashed)."), { status: 503, retryAfter: null });
check("503 with Retry-After is busy", busyRetryAfter(busyErr) === 5);
check("503 without Retry-After is an outage, never retried", busyRetryAfter(outage) === null);
{
  const waits = [];
  const slept = [];
  let calls = 0;
  const res = await retryWhenBusy(
    async () => {
      calls++;
      if (calls < 3) throw busyErr;
      return "scored";
    },
    { onWait: (w) => waits.push(w), sleep: async (ms) => void slept.push(ms), now: () => 1000 },
  );
  const pending = waits.filter(Boolean);
  check("busy twice, then scored", res === "scored" && calls === 3, `${calls} calls`);
  check("…waits the server's Retry-After each time", slept.join() === "5000,5000", slept.join());
  check("…and reports each pending retry for the countdown", pending.length === 2 && pending[0].retry === 1 && pending[1].retry === 2 && pending[0].until === 6000 && pending[0].retries === BUSY_RETRIES, JSON.stringify(pending));
  check("…clearing it once the retry is under way", waits[waits.length - 1] === null);
}
{
  let calls = 0;
  const err = await retryWhenBusy(
    async () => {
      calls++;
      throw busyErr;
    },
    { sleep: async () => undefined },
  ).catch((e) => e);
  check(`gives up after ${BUSY_RETRIES} retries with the busy error`, err === busyErr && calls === BUSY_RETRIES + 1, `${calls} calls`);
}
{
  let calls = 0;
  const err = await retryWhenBusy(
    async () => {
      calls++;
      throw outage;
    },
    { sleep: async () => undefined },
  ).catch((e) => e);
  check("an outage fails on the first try", err === outage && calls === 1);
}
{
  const ctrl = new AbortController();
  let calls = 0;
  const run = retryWhenBusy(
    async () => {
      calls++;
      throw busyErr;
    },
    { signal: ctrl.signal, onWait: (w) => w && setTimeout(() => ctrl.abort(), 10) },
  );
  const t0 = Date.now();
  const err = await run.catch((e) => e);
  check("cancel during the wait stops at once (AbortError, no further request)", err?.name === "AbortError" && calls === 1 && Date.now() - t0 < 1000, `${err?.name} after ${Date.now() - t0}ms, ${calls} calls`);
}

// Watchlist: price change only between looks quoted in the same unit.
check("price change between two USD looks", Math.abs(priceChange({ price: 333.69, currency: "USD" }, { price: 330.32, currency: "USD" }) - 1.0202) < 1e-3);
check("a look stored before snapshots carried a currency still compares", priceChange({ price: 110, currency: "USD" }, { price: 100, currency: null }) !== null);
check("GBp vs GBP is not a 99% crash", priceChange({ price: 1.268, currency: "GBP" }, { price: 126.8, currency: "GBp" }) === null);
check("missing price → no change", priceChange({ price: null, currency: "USD" }, { price: 1, currency: "USD" }) === null && priceChange({ price: 1, currency: "USD" }, null) === null);

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
