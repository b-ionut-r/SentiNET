/**
 * Late data (pure, unit-tested in e2e/unit.mjs). GDELT is rate-limited and often
 * outlives a run's time budget; the backend keeps that call alive, says so ("still
 * loading", analytics/facts.SOFT_PENDING), and supersedes its cached result once the
 * data lands — so the next plain read recomputes with it.
 */
import type { Analysis, ProgressEvent } from "./types";

const SOFT_PENDING = /still loading/i;

/** Re-read the analysis this long after a result that is still waiting on tone (ms). */
export const TONE_FOLLOW_UP_MS = [20_000, 45_000, 90_000] as const;

/** This result went out without its GDELT tone because the call was still running. */
export function tonePending(a: Pick<Analysis, "tone" | "insights">, progress: ReadonlyArray<Pick<ProgressEvent, "stage" | "key" | "status" | "detail">> = []): boolean {
  if (a.tone?.series.length) return false;
  return (
    a.insights.some((i) => i.kind === "quality" && SOFT_PENDING.test(i.title)) ||
    progress.some((p) => p.stage === "intel" && p.key === "tone" && p.status === "error" && SOFT_PENDING.test(p.detail ?? ""))
  );
}
