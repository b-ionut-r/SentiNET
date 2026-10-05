/**
 * Alert rule kinds as the form offers them (pure, unit-tested in e2e/unit.mjs).
 * Ranges mirror the backend's THRESHOLD_RANGES (services/alerts.py) so a bad value is
 * caught next to the field instead of after a round trip.
 */
import type { AlertKind } from "../../api/types";

export interface KindMeta {
  label: string;
  unit?: string;
  /** Pre-filled threshold for kinds that take one — a real value, not a look-alike placeholder. */
  initial?: string;
  /** Inclusive bounds the backend accepts. */
  range?: readonly [number, number];
}

export const KINDS: Record<AlertKind, KindMeta> = {
  score_above: { label: "SentiNET rises above", initial: "65", range: [1, 99] },
  score_below: { label: "SentiNET falls below", initial: "40", range: [1, 99] },
  score_change: { label: "SentiNET moves by at least", unit: "pts", initial: "8", range: [1, 100] },
  attention_spike: { label: "Attention spikes" },
  new_narrative: { label: "A new narrative appears" },
  analyst_action: { label: "An analyst rating/target changes" },
};

export const needsThreshold = (kind: AlertKind): boolean => KINDS[kind].range != null;

/** Why this threshold can't be submitted (shown inline), or null when it is fine. */
export function thresholdProblem(kind: AlertKind, raw: string): string | null {
  const range = KINDS[kind].range;
  if (!range) return null;
  const text = raw.trim();
  if (text === "") return "Enter a threshold";
  const n = Number(text);
  if (!Number.isFinite(n)) return "Must be a number";
  const [lo, hi] = range;
  if (n < lo || n > hi) return `Must be between ${lo} and ${hi}`;
  return null;
}
