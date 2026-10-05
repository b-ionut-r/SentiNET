/**
 * Compare's tone lines (pure, unit-tested in e2e/unit.mjs). A run that went out without
 * its GDELT tone (rate-limited, or still loading when the run's budget ran out) is
 * backfilled from /api/history — the same series the Intel page's tone pane falls back
 * to (`toneOf`) — and a ticker that still has none says why, without blaming GDELT for
 * having no data when this run simply could not fetch it.
 */
import { historyTonePending } from "../../api/pending";
import type { Analysis, HistoryResponse, TonePoint } from "../../api/types";
import { toneOf } from "../intel/priceRows";

/** Why a ticker has no tone line: history on its way, GDELT has no series, or not loaded this run. */
export type ToneGap = "loading" | "empty" | "unloaded";

export interface ToneSource {
  series: TonePoint[];
  gap: ToneGap | null;
}

/** The history is only worth a (rate-limited) GDELT call when the run itself has no tone. */
export const needsHistory = (a: Pick<Analysis, "tone"> | null | undefined): boolean => !!a && !a.tone?.series.length;

export function toneSource(a: Pick<Analysis, "tone">, history: Pick<HistoryResponse, "points" | "status"> | undefined, loading: boolean): ToneSource {
  const series = toneOf(a, history);
  if (series.length) return { series, gap: null };
  if (!history) return { series, gap: loading ? "loading" : "unloaded" };
  if (historyTonePending(history)) return { series, gap: "unloaded" };
  return { series, gap: history.status?.tone === "empty" ? "empty" : "unloaded" };
}

/** One line per kind of gap, naming its tickers (none when every ticker has a line). */
export function toneGapNotes(gaps: Array<{ ticker: string; gap: ToneGap | null }>): string[] {
  const of = (g: ToneGap) => gaps.filter((x) => x.gap === g).map((x) => x.ticker);
  const notes: string[] = [];
  const loading = of("loading");
  const unloaded = of("unloaded");
  const empty = of("empty");
  if (loading.length) notes.push(`Loading GDELT tone history for ${loading.join(", ")}…`);
  if (unloaded.length)
    notes.push(`No GDELT tone loaded for ${unloaded.join(", ")} in this run — GDELT allows one request every 5 s, so tone can lag behind; refresh in a minute.`);
  if (empty.length) notes.push(`GDELT has no daily tone series for ${empty.join(", ")}.`);
  return notes;
}
