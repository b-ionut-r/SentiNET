/**
 * Why a scan task was skipped (pure, unit-tested in e2e/unit.mjs). The backend skips
 * for three different reasons — a source with no API key ("needs a free API key"), one
 * switched off ("disabled in settings"), and a task that doesn't apply to the asset
 * ("n/a for ETFs") — and only the first is something a key would unlock.
 */
import type { SourceInfo } from "../../api/types";

export type SkipReason = "key" | "off" | "na";

export const SKIP_BADGE: Record<SkipReason, string> = { key: "key", off: "off", na: "n/a" };

export function skipReason(detail: string | null | undefined, source?: Pick<SourceInfo, "requires_key" | "configured"> | null): SkipReason {
  if (source?.requires_key && !source.configured) return "key";
  const d = detail ?? "";
  if (/\bkey\b/i.test(d)) return "key";
  if (/\bdisabled\b/i.test(d)) return "off";
  return "na";
}
