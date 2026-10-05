/**
 * First-load experience: every source and intel task lights up as the SSE
 * stream reports it — a terminal-style scan with counts and latencies.
 */
import { OctagonAlert, Check, KeyRound, Minus } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useSources, type TrackedProgress } from "../../api/hooks";
import type { ProgressEvent, SourceInfo } from "../../api/types";
import { Skeleton } from "../../components/ui/Misc";
import { cx } from "../../lib/cx";
import { ms } from "../../lib/format";
import { SKIP_BADGE, skipReason, type SkipReason } from "./scanChip";

type ChipStatus = ProgressEvent["status"] | "queued";

interface Chip {
  key: string;
  label: string;
  status: ChipStatus;
  count: number | null;
  ms: number | null;
  detail: string | null;
  /** Set when status is "skipped": only "key" means an API key would unlock it. */
  skip: SkipReason | null;
}

const STAGE_TITLE: Record<string, string> = { source: "Sources", intel: "Market intel", synth: "Synthesis" };

export function ScanPanel({ ticker, progress }: { ticker: string; progress: TrackedProgress[] }) {
  const sources = useSources();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const id = window.setInterval(() => setElapsed(performance.now() - start), 100);
    return () => window.clearInterval(id);
  }, [ticker]);

  const groups = useMemo(() => {
    const byKey = new Map(progress.map((p) => [`${p.stage}:${p.key}`, p]));
    const info = new Map((sources.data ?? []).map((s) => [s.key, s]));
    const src: Chip[] = (sources.data ?? [])
      .filter((s) => s.enabled)
      .map((s) => {
        const ev = byKey.get(`source:${s.key}`);
        if (ev) return toChip(ev, s);
        const keyless = s.requires_key && !s.configured;
        return { key: s.key, label: s.label, status: keyless ? "skipped" : "queued", count: null, ms: null, detail: keyless ? "needs API key" : null, skip: keyless ? "key" : null };
      });
    for (const p of progress) if (p.stage === "source" && !src.some((c) => c.key === p.key)) src.push(toChip(p, info.get(p.key)));
    const intel = progress.filter((p) => p.stage === "intel").map((p) => toChip(p));
    const synth = progress.filter((p) => p.stage === "resolve" || p.stage === "nlp" || p.stage === "analytics" || p.stage === "done").map((p) => toChip(p));
    return { source: src, intel, synth };
  }, [progress, sources.data]);

  const all = [...groups.source, ...groups.intel];
  const done = all.filter((c) => c.status !== "queued" && c.status !== "running").length;
  const total = Math.max(all.length, 1);
  const items = all.reduce((s, c) => s + (c.status === "ok" ? c.count ?? 0 : 0), 0);
  const resolved = progress.find((p) => p.stage === "resolve" && p.status === "ok");
  // Terminal log: the six most recent completions, oldest first.
  const log = progress
    .filter((p) => p.status !== "running")
    .sort((x, y) => x.seq - y.seq)
    .slice(-6);

  return (
    <div className="space-y-4" aria-busy="true">
      {/* Screen readers hear task completions only — never the 100 ms timer. */}
      <p className="sr-only" aria-live="polite">
        {all.length ? `${done} of ${all.length} tasks done, ${items} items collected` : `Scanning ${ticker}`}
      </p>
      <section className="panel relative overflow-hidden">
        <div className="absolute inset-x-0 top-0 h-px overflow-hidden" aria-hidden>
          <div className="h-full w-1/3 animate-scan-sweep bg-accent" />
        </div>
        <div className="flex flex-wrap items-end justify-between gap-3 px-5 pt-5">
          <div>
            <p className="eyebrow">Scanning</p>
            <h1 className="mt-1 text-[22px] font-semibold tracking-[-0.01em] text-ink">
              {ticker}
              {resolved?.detail && <span className="ml-2 text-base font-normal text-ink-2">{resolved.detail}</span>}
            </h1>
            <p className="mt-1 text-sm text-muted">Fusing news, crowd, analysts, insiders, filings and global news tone.</p>
          </div>
          <div className="text-right font-mono text-xs text-muted" aria-hidden>
            <div className="text-lg font-medium text-ink">{(elapsed / 1000).toFixed(1)}s</div>
            {done}/{all.length || "…"} tasks · {items} items
          </div>
        </div>
        <div className="mx-5 mt-4 h-1 overflow-hidden rounded-full bg-[rgb(var(--grid))]">
          <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${(done / total) * 100}%` }} />
        </div>
        {/* Explicit minmax(0,1fr) tracks at every width: an implicit auto track grows to the
            max-content of the nowrap chip labels and pushes the badges out of the card on phones. */}
        <div className="grid grid-cols-1 gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,0.8fr)]">
          {(["source", "intel", "synth"] as const).map((g) => (
            <div key={g} className="min-w-0">
              <h2 className="eyebrow mb-2">{STAGE_TITLE[g]}</h2>
              {groups[g].length === 0 ? (
                <div className="space-y-1.5">
                  {Array.from({ length: g === "synth" ? 2 : 5 }, (_, i) => (
                    <Skeleton key={i} className="h-7 w-full" />
                  ))}
                </div>
              ) : (
                <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                  {groups[g].map((c) => (
                    <ScanChip key={c.key} c={c} />
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
        <div className="hairline-t bg-sunken px-5 py-3 font-mono text-[11px] leading-[18px] text-muted">
          {log.length === 0 ? (
            <span className="animate-pulse-soft">connecting to /api/analyze/{ticker}/stream …</span>
          ) : (
            log.map((p, i) => (
              <div key={`${p.stage}:${p.key}:${i}`} className="flex gap-3 truncate">
                <span className="w-16 shrink-0 text-muted">{p.stage}</span>
                <span className="w-28 shrink-0 truncate text-ink-2">{p.key}</span>
                <span className={cx("w-14 shrink-0", statusText[p.status])}>{p.status}</span>
                <span className="hidden w-16 shrink-0 text-right sm:inline">{p.count != null ? `${p.count} items` : ""}</span>
                <span className="hidden w-14 shrink-0 text-right sm:inline">{p.ms != null ? ms(p.ms) : ""}</span>
                <span className="truncate text-muted">{p.detail ?? ""}</span>
              </div>
            ))
          )}
        </div>
      </section>
      <div className="space-y-4 opacity-60" aria-hidden>
        <Skeleton className="h-64 w-full rounded-xl" />
        <div className="grid gap-4 lg:grid-cols-12">
          <Skeleton className="h-72 rounded-xl lg:col-span-8" />
          <Skeleton className="h-72 rounded-xl lg:col-span-4" />
        </div>
      </div>
    </div>
  );
}

const statusText: Record<ChipStatus, string> = {
  queued: "text-faint",
  running: "text-accent",
  ok: "text-good",
  empty: "text-muted",
  error: "text-critical",
  skipped: "text-muted",
};

function toChip(p: ProgressEvent, source?: SourceInfo): Chip {
  return { key: p.key, label: p.label, status: p.status, count: p.count, ms: p.ms, detail: p.detail, skip: p.status === "skipped" ? skipReason(p.detail, source) : null };
}

function ScanChip({ c }: { c: Chip }) {
  const icon =
    c.status === "ok" ? (
      <Check className="size-3.5 text-good" />
    ) : c.status === "error" ? (
      <OctagonAlert className="size-3.5 text-critical" />
    ) : c.status === "empty" ? (
      <Minus className="size-3.5 text-muted" />
    ) : c.status === "skipped" ? (
      c.skip === "key" ? <KeyRound className="size-3.5 text-faint" /> : <Minus className="size-3.5 text-faint" />
    ) : c.status === "running" ? (
      <span className="size-2 animate-pulse-soft rounded-full bg-accent" />
    ) : (
      <span className="size-1.5 rounded-full bg-[rgb(var(--faint))]" />
    );
  const showDetail = c.detail && c.status !== "running" && c.status !== "queued";
  return (
    <li
      className={cx(
        "rounded-md px-2 py-1.5 text-xs transition-colors duration-300",
        c.status === "queued" ? "text-faint" : c.status === "skipped" ? "text-muted" : "bg-raised text-ink-2",
        c.status === "running" && "text-ink",
      )}
      title={c.detail ?? undefined}
    >
      <div className="flex items-center gap-2">
        <span className="flex size-3.5 shrink-0 items-center justify-center">{icon}</span>
        <span className="min-w-0 flex-1 truncate">{c.label}</span>
        <span className="shrink-0 font-mono text-2xs text-muted">
          {c.status === "ok" && c.count != null ? c.count : c.status === "error" ? "err" : c.status === "skipped" ? SKIP_BADGE[c.skip ?? "na"] : ""}
          {c.ms != null && c.status !== "running" ? <span className="ml-1.5 text-muted">{ms(c.ms)}</span> : null}
        </span>
      </div>
      {showDetail && <div className={cx("mt-0.5 truncate pl-[22px] font-mono text-[10.5px] leading-4", c.status === "error" ? "text-critical" : "text-muted")}>{c.detail}</div>}
    </li>
  );
}
