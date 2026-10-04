/**
 * Text with the engine's driver terms underlined by polarity (teal up, red
 * down). Drivers can be normalized phrases ("price target raised to $250 from
 * $220") that don't occur verbatim, so when a whole term isn't found its
 * distinctive tokens are matched instead.
 */
import type { Driver } from "../../api/types";
import { cx } from "../../lib/cx";
import { signed } from "../../lib/format";

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
/** Typographic apostrophes match straight ones (same length, so offsets stay aligned). */
const foldQuotes = (s: string) => s.replace(/[\u2018\u2019]/g, "'");
const STOP = new Set([
  "the", "and", "for", "from", "with", "to", "of", "on", "in", "at", "by", "a", "an", "is", "are", "its", "into", "after", "over",
  "may", "might", "could", "will", "would", "be", "been", "has", "have", "this", "that", "than", "more", "but", "not", "was",
]);

function patterns(text: string, drivers: Driver[]): Map<string, number> {
  const lower = foldQuotes(text.toLowerCase());
  const out = new Map<string, number>();
  for (const d of drivers) {
    const term = foldQuotes(d.term.trim().toLowerCase());
    if (!term) continue;
    if (lower.includes(term)) {
      out.set(term, d.impact);
      continue;
    }
    // A normalized phrase ("stock fall", "price target raised to $250"): mark its distinctive words.
    for (const tok of term.split(/\s+/)) {
      const t = tok.replace(/^[^\p{L}\p{N}$%]+|[^\p{L}\p{N}%]+$/gu, "");
      if (t.length >= 3 && !STOP.has(t) && lower.includes(t) && !out.has(t)) out.set(t, d.impact);
    }
  }
  return out;
}

/** Case-insensitive whole-word alternation; engines without lookbehind fall back to plain matching. */
function wordRegex(alternation: string): RegExp {
  try {
    return new RegExp(`(?<![\\p{L}\\p{N}])(${alternation})(?![\\p{L}\\p{N}])`, "giu");
  } catch {
    return new RegExp(`(${alternation})`, "giu");
  }
}

export function DriverText({ text, drivers }: { text: string; drivers: Driver[] }) {
  const impact = patterns(text, drivers);
  if (!impact.size) return <>{text}</>;
  // Whole words only ("fall" must not light up "Fallout"); longest terms first.
  const alts = [...impact.keys()].sort((a, b) => b.length - a.length).map((k) => escapeRe(k).replace(/'/g, "['\u2018\u2019]"));
  const re = wordRegex(alts.join("|"));
  return (
    <>
      {text.split(re).map((part, i) => {
        const v = impact.get(foldQuotes(part.toLowerCase()));
        if (v == null) return part;
        return (
          <mark
            key={i}
            title={`driver ${signed(v)}`}
            className={cx("bg-transparent text-ink underline decoration-2 underline-offset-[3px]", v > 0 ? "decoration-bull/80" : v < 0 ? "decoration-bear/80" : "decoration-neu/70")}
          >
            {part}
          </mark>
        );
      })}
    </>
  );
}
