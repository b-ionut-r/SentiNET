/**
 * "What's moving it": story clusters ranked by impact, a coverage-mix bar
 * that shows how much of the conversation leans bull vs. bear, and
 * expandable member articles.
 */
import { ChevronRight, ExternalLink } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { Analysis, Narrative, Signal } from "../../api/types";
import { Pulse } from "../../components/charts/Pulse";
import { Chip, ScoreChip } from "../../components/ui/Badges";
import { Empty } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { useTooltip } from "../../components/ui/Tooltip";
import { cx } from "../../lib/cx";
import { plural, signed, timeAgo } from "../../lib/format";
import { divergingFill, polarityOf } from "../../lib/sentiment";
import { themeLabel } from "./themes";

const INITIAL = 8;

export function Narratives({ a, membersOf, className }: { a: Analysis; membersOf: Map<string, Signal[]>; className?: string }) {
  const [showAll, setShowAll] = useState(false);
  const list = showAll ? a.narratives : a.narratives.slice(0, INITIAL);
  const items = a.narratives.reduce((s, n) => s + n.count, 0);
  return (
    <Panel
      id="narratives"
      title="What's moving it"
      subtitle={a.narratives.length ? `${a.narratives.length} stories from ${plural(items, "item")} · ranked by coverage × tone × recency` : undefined}
      className={className}
      flush
    >
      {a.narratives.length === 0 ? (
        <Empty title="No story clusters">Not enough related coverage to form narratives — see individual signals below.</Empty>
      ) : (
        <>
          <CoverageMix narratives={a.narratives} />
          {a.timeline.length > 1 && (
            <div className="px-4 pb-3.5">
              <div className="mb-1.5 text-2xs text-muted">Signal pulse · all kept items over time</div>
              <Pulse buckets={a.timeline} />
            </div>
          )}
          <ol className="divide-hair hairline-t">
            {list.map((n, i) => (
              <NarrativeRow key={n.id} n={n} rank={i + 1} members={membersOf.get(n.id) ?? []} defaultOpen={false} />
            ))}
          </ol>
          {a.narratives.length > INITIAL && (
            <button className="w-full px-4 py-2.5 text-left text-xs font-medium text-muted hover:text-ink hairline-t" onClick={() => setShowAll((s) => !s)}>
              {showAll ? "Show fewer" : `Show all ${a.narratives.length} stories`}
            </button>
          )}
        </>
      )}
    </Panel>
  );
}

/** Share of story coverage by narrative, colored on the diverging tone ramp. */
function CoverageMix({ narratives }: { narratives: Narrative[] }) {
  const { showAt, hide } = useTooltip();
  const total = narratives.reduce((s, n) => s + n.count, 0) || 1;
  const bull = narratives.filter((n) => polarityOf(n.score) === "bull").reduce((s, n) => s + n.count, 0);
  const bear = narratives.filter((n) => polarityOf(n.score) === "bear").reduce((s, n) => s + n.count, 0);
  return (
    <div className="px-4 pb-3.5">
      <div className="mb-1.5 flex items-baseline justify-between text-2xs text-muted">
        <span>Coverage mix by story</span>
        <span>
          <span className="font-semibold text-bull">▲ {Math.round((bull / total) * 100)}%</span> bullish-toned ·{" "}
          <span className="font-semibold text-bear">▼ {Math.round((bear / total) * 100)}%</span> bearish-toned
        </span>
      </div>
      <div className="flex h-2.5 gap-[2px]" role="img" aria-label={`Coverage: ${Math.round((bull / total) * 100)}% bullish-toned, ${Math.round((bear / total) * 100)}% bearish-toned`}>
        {narratives.map((n, i) => (
          <div
            key={n.id}
            tabIndex={0}
            className="h-full outline-none transition-[filter] hover:brightness-125 focus-visible:brightness-125"
            style={{
              flexGrow: n.count,
              flexBasis: 0,
              minWidth: 3,
              background: divergingFill(Math.max(-1, Math.min(1, n.score / 0.5))),
              borderRadius: `${i === 0 ? 3 : 1}px ${i === narratives.length - 1 ? 3 : 1}px ${i === narratives.length - 1 ? 3 : 1}px ${i === 0 ? 3 : 1}px`,
            }}
            onMouseEnter={(e) =>
              showAt(
                <div className="max-w-[260px]">
                  <div className="font-semibold text-ink">
                    #{i + 1} · {n.count} items · tone {signed(n.score)}
                  </div>
                  <div className="mt-0.5">{n.headline}</div>
                </div>,
                e.currentTarget,
              )
            }
            onMouseLeave={hide}
            onFocus={(e) => showAt(`#${i + 1} ${n.headline} — ${n.count} items, tone ${signed(n.score)}`, e.currentTarget)}
            onBlur={hide}
          />
        ))}
      </div>
    </div>
  );
}

function NarrativeRow({ n, rank, members, defaultOpen }: { n: Narrative; rank: number; members: Signal[]; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [flash, setFlash] = useState(false);
  const ref = useRef<HTMLLIElement>(null);

  // Verdict reasons can point here: open and briefly highlight.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onFocus = () => {
      setOpen(true);
      setFlash(true);
      window.setTimeout(() => setFlash(false), 1400);
    };
    el.addEventListener("sentinet:focus", onFocus);
    return () => el.removeEventListener("sentinet:focus", onFocus);
  }, []);

  const outlets = n.publishers.slice(0, 3).join(", ") + (n.publishers.length > 3 ? ` +${n.publishers.length - 3}` : "");
  const sorted = [...members].sort((x, y) => (y.timestamp ?? "").localeCompare(x.timestamp ?? ""));
  return (
    <li ref={ref} id={`narr-${n.id}`} className={cx("scroll-mt-32 transition-colors duration-700", flash && "bg-accent/8")}>
      <button className="group flex w-full gap-3 px-4 py-3 text-left" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="mt-0.5 w-5 shrink-0 font-mono text-xs text-faint num">{String(rank).padStart(2, "0")}</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-2">
            <p className="min-w-0 flex-1 text-[14px] font-medium leading-5 text-ink">
              {n.headline}
              {n.is_new && <span className="ml-2 inline-block translate-y-[-1px] rounded bg-accent/15 px-1 py-px align-middle text-2xs font-semibold uppercase tracking-wider text-accent">New</span>}
            </p>
            <ScoreChip score={n.score} className="mt-px shrink-0" />
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-muted">
            <span className="font-medium text-ink-2">{plural(n.count, "item")}</span>
            <span>{outlets || "—"}</span>
            {n.velocity_24h > 0 && <span className="text-ink-2">+{n.velocity_24h} in 24h</span>}
            {n.last_seen && <span>latest {timeAgo(n.last_seen)}</span>}
            {n.themes.slice(0, 2).map((t) => (
              <Chip key={t} tone="muted">
                {themeLabel(t)}
              </Chip>
            ))}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <div className="h-1 w-24 overflow-hidden rounded-full bg-[rgb(var(--grid))]" title={`Impact ${Math.round(n.impact * 100)}/100`}>
              <div className="h-full rounded-full bg-[rgb(var(--ink-2))]" style={{ width: `${Math.max(4, n.impact * 100)}%` }} />
            </div>
            <span className="text-2xs text-faint">impact {Math.round(n.impact * 100)}</span>
            <span className="ml-auto inline-flex items-center gap-0.5 text-2xs text-muted group-hover:text-ink-2">
              {open ? "Hide" : "Sources"}
              <ChevronRight className={cx("size-3 transition-transform", open && "rotate-90")} />
            </span>
          </div>
        </div>
      </button>
      {open && (
        <ul className="mb-3 ml-12 mr-4 space-y-1.5 rounded-lg bg-sunken p-2.5 animate-fade-in">
          {sorted.length === 0 && n.url && (
            <li>
              <a className="link text-xs" href={n.url} target="_blank" rel="noreferrer">
                Open the representative article
              </a>
            </li>
          )}
          {sorted.slice(0, 12).map((s) => (
            <li key={s.id} className="flex items-start gap-2 text-xs">
              <ScoreChip score={s.score} className="mt-px shrink-0" />
              <div className="min-w-0 flex-1">
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noreferrer" className="group/a text-ink-2 hover:text-ink">
                    {s.title}
                    <ExternalLink className="ml-1 inline size-3 text-faint group-hover/a:text-muted" aria-hidden />
                  </a>
                ) : (
                  <span className="text-ink-2">{s.title}</span>
                )}
                <div className="text-2xs text-muted">
                  {s.publisher ?? s.source_label} · {timeAgo(s.timestamp)}
                  {s.duplicates > 0 && ` · ×${s.duplicates + 1} syndicated`}
                </div>
              </div>
            </li>
          ))}
          {sorted.length > 12 && <li className="text-2xs text-muted">+{sorted.length - 12} more in the signal explorer</li>}
        </ul>
      )}
    </li>
  );
}
