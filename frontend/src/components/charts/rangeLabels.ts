/**
 * Geometry for RangeBar's labels (pure, unit-tested in e2e/unit.mjs).
 */

/**
 * Label anchored at `pct` (0–100) of the track that never leaves it: shifting the
 * label left by the same share of its own width puts its left edge on the track's
 * left end at 0%, centres it at 50% and right-aligns it at 100%.
 */
export function edgeSafe(pct: number): { left: string; transform: string } {
  const p = Math.max(0, Math.min(100, pct));
  return { left: `${p}%`, transform: `translateX(-${p}%)` };
}

/** Low and high sit too close to label separately (in % of the track) — one label covers both. */
export const RANGE_MERGE_PCT = 30;

/**
 * Track position (0–100) of a value on a band from `low` to `high` that also fits every
 * marker, with 4% padding (or 4% of the value when everything coincides, e.g. a single
 * analyst's target equal to the price); `merged` when low and high need one shared label.
 */
export function rangeLabels(low: number, high: number, markers: ReadonlyArray<{ value: number }>): { x: (v: number) => number; merged: boolean } {
  const values = [low, high, ...markers.map((m) => m.value)];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo) * 0.04 || Math.abs(hi) * 0.04 || 1;
  const d0 = lo - pad;
  const d1 = hi + pad;
  const x = (v: number) => ((v - d0) / (d1 - d0)) * 100;
  return { x, merged: x(high) - x(low) < RANGE_MERGE_PCT };
}
