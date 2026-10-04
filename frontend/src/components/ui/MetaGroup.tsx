import type { ReactNode } from "react";

/**
 * A dot-separated run of metadata ("CNBC · Google News · 3d ago"). Each item
 * carries its own leading separator and never breaks internally, and the run
 * wraps as a unit inside a flex-wrap parent — so a "·" never starts a line.
 */
export function MetaGroup({ items, className }: { items: ReactNode[]; className?: string }) {
  const parts = items.filter((x) => x != null && x !== false && x !== "");
  if (!parts.length) return null;
  return (
    <span className={className ?? "inline-flex flex-wrap items-center gap-x-1.5"}>
      {parts.map((m, i) => (
        <span key={i} className="inline-flex items-center gap-x-1.5 whitespace-nowrap">
          {i > 0 && (
            <span className="text-faint" aria-hidden>
              ·
            </span>
          )}
          {m}
        </span>
      ))}
    </span>
  );
}
