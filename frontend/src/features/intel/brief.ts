/**
 * Bull/bear points that add to the verdict (pure, unit-tested in e2e/unit.mjs). The
 * brief's case lists are built from the same evidence as the verdict's reasons, so
 * most of them arrive word for word; the hero already shows those lines a scroll
 * above, and the case panel only lists what they leave out.
 */
import type { Reason } from "../../api/types";

/** How many verdict reasons the hero shows (VerdictHero's ReadBlock). */
export const HERO_REASONS = 5;

const norm = (t: string) =>
  t
    .toLowerCase()
    .replace(/[‒-―−]/g, "-")
    .replace(/\s+/g, " ")
    .replace(/[\s.;:,]+$/, "")
    .trim();

export function freshPoints(points: string[], reasons: Pick<Reason, "text">[]): { fresh: string[]; repeated: number } {
  const shown = new Set(reasons.slice(0, HERO_REASONS).map((r) => norm(r.text)));
  const fresh = points.filter((p) => !shown.has(norm(p)));
  return { fresh, repeated: points.length - fresh.length };
}
