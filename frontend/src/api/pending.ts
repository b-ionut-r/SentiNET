/**
 * Late data (pure, unit-tested in e2e/unit.mjs). GDELT is rate-limited and often
 * outlives a run's time budget; the backend keeps that call alive, says so (a "quality"
 * insight, and a streamed intel:tone event whose status reads "still loading",
 * analytics/facts.SOFT_PENDING), and supersedes its cached result once the data lands —
 * so the next plain read recomputes with it.
 *
 * The payload carries no machine flag for this yet, so it is read from the wording.
 * Every phrasing the backend has used for the soft gap is accepted (the insight was
 * "Global news tone still loading", then "Global news tone not loaded this run" with
 * "GDELT history did not arrive in time…"), and the insight must be about tone/GDELT:
 * other feeds' gaps ("Market data not loaded in time") are not a reason to re-read.
 * A hard failure ("could not be loaded this run") is not pending either.
 */
import type { Analysis, HistoryResponse, Insight, ProgressEvent } from "./types";

const SOFT_PENDING = /still loading|not loaded this run|did not arrive in time/i;
const ABOUT_TONE = /\b(tone|gdelt)\b/i;

/** Re-read the analysis this long after a result that is still waiting on tone (ms). */
export const TONE_FOLLOW_UP_MS = [20_000, 45_000, 90_000] as const;

/** Re-read a /api/history response whose GDELT call is still running this often, this many times. */
export const HISTORY_TONE_RETRY_MS = 30_000;
export const HISTORY_TONE_RETRIES = 3;

/**
 * /api/history went out while its GDELT call was still running ("error: still loading
 * after 22s; continuing in the background (reload to include)") — a temporary gap the
 * next read fills, unlike a real error (rate limit, HTTP 5xx) or an empty series.
 */
export function historyTonePending(h: Pick<HistoryResponse, "status"> | undefined): boolean {
  const st = h?.status?.tone ?? "";
  return st.startsWith("error") && SOFT_PENDING.test(st);
}

/** A "quality" insight saying this run's GDELT tone timed out but is still on its way. */
export function isTonePendingInsight(i: Pick<Insight, "kind" | "title" | "detail">): boolean {
  return i.kind === "quality" && ABOUT_TONE.test(i.title) && SOFT_PENDING.test(`${i.title} ${i.detail}`);
}

/** This result went out without its GDELT tone because the call was still running. */
export function tonePending(a: Pick<Analysis, "tone" | "insights">, progress: ReadonlyArray<Pick<ProgressEvent, "stage" | "key" | "status" | "detail">> = []): boolean {
  if (a.tone?.series.length) return false;
  return (
    a.insights.some(isTonePendingInsight) ||
    progress.some((p) => p.stage === "intel" && p.key === "tone" && p.status === "error" && SOFT_PENDING.test(p.detail ?? ""))
  );
}

/**
 * The analysis went out without GDELT tone (for any reason: still loading, timed out,
 * failed), but the history endpoint has since obtained it. The server supersedes its
 * cached analysis when that happens (analyzer.intel_landed), so a plain re-read now
 * comes back with tone in the verdict.
 */
export function toneArrivedViaHistory(
  a: Pick<Analysis, "ticker" | "tone"> | undefined,
  h: Pick<HistoryResponse, "ticker" | "status" | "points"> | undefined,
): boolean {
  if (!a || !h || a.ticker !== h.ticker || a.tone?.series.length) return false;
  return h.status?.tone === "ok" && h.points.some((p) => p.tone != null);
}
