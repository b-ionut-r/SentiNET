/** Lead/lag reliability rule shared by the chart and its labels (pure, unit-tested in e2e/unit.mjs). */
import type { LagStat } from "../../api/types";

/**
 * The backend's reliability rule (analytics/stats.py): seven lags are tested,
 * so a link only counts when it survives Bonferroni (p × 7 < 0.05) with
 * |r| ≥ 0.2 on at least 20 paired days. The chart mirrors it exactly so the
 * bars can never contradict the sentence above them.
 */
export const LAGS_TESTED = 7;
export const ALPHA = 0.05;
export const MIN_R = 0.2;
export const MIN_RELIABLE_N = 20;
/** Two-sided normal quantile for α / 7: Φ⁻¹(1 − 0.05 / 14). */
const Z_BONF = 2.6901;

export function reliableLag(l: LagStat): boolean {
  return l.n >= MIN_RELIABLE_N && Math.abs(l.r) >= MIN_R && l.p_value * LAGS_TESTED < ALPHA;
}

/**
 * Smallest |r| that survives the 7-lag correction for n paired points:
 * r = t / sqrt(df + t²), Student-t quantile from the Cornish–Fisher expansion
 * of Z_BONF (matches the exact p within 0.1% for n ≥ 10). Never below MIN_R.
 */
export function criticalR(n: number): number {
  const df = n - 2;
  const z = Z_BONF;
  const t = z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2) + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3);
  return Math.max(MIN_R, t / Math.sqrt(df + t * t));
}

/** p-value at the precision the backend's sentence uses. */
export function fmtP(p: number): string {
  return p < 0.001 ? "<0.001" : p < 0.01 ? p.toFixed(3) : p.toFixed(2);
}

export function lagVerdict(l: LagStat): string {
  if (reliableLag(l)) return `survives correction for ${LAGS_TESTED} tested lags`;
  if (l.p_value < ALPHA) return `p < 0.05 alone, but not after correcting for ${LAGS_TESTED} tested lags — noise`;
  return "not significant — noise";
}
