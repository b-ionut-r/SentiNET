/**
 * Bull/bear points that add to the verdict (pure, unit-tested in e2e/unit.mjs). The
 * brief's case lists are built from the same evidence as the verdict's reasons, so
 * most of them arrive word for word; the hero already shows those lines a scroll
 * above, and the case panel only lists what they leave out.
 *
 * "Word for word" is loose in practice: the hero labels its stories "Top story:" /
 * "Counter-story:" where the brief says "Story:", and a brief point restating a hero
 * reason may append a clause ("… — the main drag on the Leaning Bullish read (−3.6
 * points)."). Both are the same evidence and count as repeats.
 */
import type { Reason } from "../../api/types";

/** How many verdict reasons the hero shows (VerdictHero's ReadBlock). */
export const HERO_REASONS = 5;

/** Story labels the backend puts in front of a story line (verdict.py: Top/Counter-story; brief.py: Story). */
const STORY_LABEL = /^(?:top story|counter-story|story)\s*:\s*/i;
/** A restatement only counts when the shorter line ends where the longer one starts a new clause. */
const CLAUSE_BREAK = /^(?:$|\s*[-–—;,(.:])/;
/** Below this, a prefix match could be a coincidence ("News flow positive"), not the same evidence. */
const MIN_PREFIX = 24;

const norm = (t: string) =>
  t
    .toLowerCase()
    .replace(/[‒-―−]/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .replace(STORY_LABEL, "")
    .replace(/[\s.;:,]+$/, "")
    .trim();

/** The quoted headline of a story line ("Top story: ‘…’ — 8 articles…"), to match stories whatever their tally. */
function storyHeadline(t: string): string | null {
  if (!STORY_LABEL.test(t.trim())) return null;
  const m = /‘([^’]+)’/.exec(t);
  return m ? norm(m[1]) : null;
}

/** `a` and `b` state the same evidence (one may extend the other with a trailing clause). */
function sameEvidence(a: string, b: string): boolean {
  const na = norm(a);
  const nb = norm(b);
  if (na === nb) return true;
  const ha = storyHeadline(a);
  if (ha && ha === storyHeadline(b)) return true;
  const [short, long] = na.length <= nb.length ? [na, nb] : [nb, na];
  return short.length >= MIN_PREFIX && long.startsWith(short) && CLAUSE_BREAK.test(long.slice(short.length));
}

export function freshPoints(points: string[], reasons: Pick<Reason, "text">[]): { fresh: string[]; repeated: number } {
  const shown = reasons.slice(0, HERO_REASONS).map((r) => r.text);
  const fresh = points.filter((p) => !shown.some((r) => sameEvidence(p, r)));
  return { fresh, repeated: points.length - fresh.length };
}
