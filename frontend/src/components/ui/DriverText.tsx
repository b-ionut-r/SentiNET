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
const STOP = new Set(["the", "and", "for", "from", "with", "to", "of", "on", "in", "at", "by", "a", "an", "is", "are", "its", "into", "after", "over"]);

function patterns(text: string, drivers: Driver[]): Map<string, number> {
  const lower = text.toLowerCase();
  const out = new Map<string, number>();
  for (const d of drivers) {
    const term = d.term.trim().toLowerCase();
    if (!term) continue;
    if (lower.includes(term)) {
      out.set(term, d.impact);
      continue;
    }
    for (const tok of term.split(/\s+/)) {
      const t = tok.replace(/^[^\p{L}\p{N}$%]+|[^\p{L}\p{N}%]+$/gu, "");
      if (t.length >= 3 && !STOP.has(t) && lower.includes(t) && !out.has(t)) out.set(t, d.impact);
    }
  }
  return out;
}

export function DriverText({ text, drivers }: { text: string; drivers: Driver[] }) {
  const impact = patterns(text, drivers);
  if (!impact.size) return <>{text}</>;
  const re = new RegExp(`(${[...impact.keys()].sort((a, b) => b.length - a.length).map(escapeRe).join("|")})`, "giu");
  return (
    <>
      {text.split(re).map((part, i) => {
        const v = impact.get(part.toLowerCase());
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
