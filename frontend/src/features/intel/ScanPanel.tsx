/**
 * First-load experience: every source and intel task lights up as the SSE
 * stream reports it — a terminal-style scan with counts and latencies.
 */
import { OctagonAlert, Check, KeyRound, Minus } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useSources } from "../../api/hooks";
import type { ProgressEvent } from "../../api/types";
import { Skeleton } from "../../components/ui/Misc";
import { cx } from "../../lib/cx";
import { ms } from "../../lib/format";

type ChipStatus = ProgressEvent["status"] | "queued";

interface Chip {
  key: string;
  label: string;
  status: ChipStatus;
  count: number | null;
  ms: number | null;
  detail: string | null;
}

const STAGE_TITLE: Record<string, string> = { source: "Sources", intel: "Market intel", synth: "Synthesis" };

export function ScanPanel({ ticker, progress }: { ticker: string; progress: ProgressEvent[] }) {
  const sources = useSources();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const id = window.setInterval(() => setElapsed(performance.now() - start), 100);
    return () => window.clearInterval(id);
  }, [ticker]);

  const groups = useMemo(() => {
    const byKey = new Map(progress.map((p) => [`${p.stage}:${p.key}`, p]));
    const src: Chip[] = (sources.data ?? [])
      .filter((s) => s.enabled)
      .map((s) => {
        const ev = byKey.get(`source:${s.key}`);
        return ev ? toChip(ev) : { key: s.key, label: s.label, status: s.requires_key && !s.configured ? "skipped" : "queued", count: null, ms: null, detail: s.requires_key && !s.configured ? "needs API key" : null };
      });
    for (const p of progress) if (p.stage === "source" && !src.some((c) => c.key === p.key)) src.push(toChip(p));
    const intel = progress.filter((p) => p.stage === "intel").map(toChip);
    const synth = progress.filter((p) => p.stage === "resolve" || p.stage === "nlp" || p.stage === "analytics" || p.stage === "done").map(toChip);
    return { source: src, intel, synth };
  }, [progress, sources.data]);

  const all = [...groups.source, ...groups.intel];
  const done = all.filter((c) => c.status !== "queued" && c.status !== "running").length;
  const total = Math.max(all.length, 1);
  const items = all.reduce((s, c) => s + (c.status === "ok" ? c.count ?? 0 : 0), 0);
  const resolved = progress.find((p) => p.stage === "resolve" && p.status === "ok");
  const log = progress.filter((p) => p.status !== "running").slice(-6);

  return (
    <div className="space-y-4" aria-busy="true" aria-live="polite">
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
          <div className="text-right font-mono text-xs text-muted">
            <div className="text-lg font-medium text-ink">{(elapsed / 1000).toFixed(1)}s</div>
            {done}/{all.length || "…"} tasks · {items} items
          </div>
        </div>
        <div className="mx-5 mt-4 h-1 overflow-hidden rounded-full bg-[rgb(var(--grid))]">
          <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${(done / total) * 100}%` }} />
        </div>
        <div className="grid gap-5 p-5 lg:grid-cols-[1fr_1fr_0.8fr]">
          {(["source", "intel", "synth"] as const).map((g) => (
            <div key={g}>
              <h2 className="eyebrow mb-2">{STAGE_TITLE[g]}</h2>
              {groups[g].length === 0 ? (
                <div className="space-y-1.5">
                  {Array.from({ length: g === "synth" ? 2 : 5 }, (_, i) => (
                    <Skeleton key={i} className="h-7 w-full" />
                  ))}
                </div>
              ) : (
                <ul className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
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
                <span className="w-16 shrink-0 text-faint">{p.stage}</span>
                <span className="w-28 shrink-0 truncate text-ink-2">{p.key}</span>
                <span className={cx("w-14 shrink-0", statusText[p.status])}>{p.status}</span>
                <span className="w-16 shrink-0 text-right">{p.count != null ? `${p.count} items` : ""}</span>
                <span className="w-14 shrink-0 text-right">{p.ms != null ? ms(p.ms) : ""}</span>
                <span className="truncate text-faint">{p.detail ?? ""}</span>
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
  skipped: "text-faint",
};

function toChip(p: ProgressEvent): Chip {
  return { key: p.key, label: p.label, status: p.status, count: p.count, ms: p.ms, detail: p.detail };
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
      <KeyRound className="size-3.5 text-faint" />
    ) : c.status === "running" ? (
      <span className="size-2 animate-pulse-soft rounded-full bg-accent" />
    ) : (
      <span className="size-1.5 rounded-full bg-[rgb(var(--faint))]" />
    );
  return (
    <li
      className={cx(
        "flex h-7 items-center gap-2 rounded-md px-2 text-xs transition-colors duration-300",
        c.status === "queued" || c.status === "skipped" ? "text-faint" : "bg-raised text-ink-2",
        c.status === "running" && "text-ink",
      )}
      title={c.detail ?? undefined}
    >
      <span className="flex size-3.5 shrink-0 items-center justify-center">{icon}</span>
      <span className="min-w-0 flex-1 truncate">{c.label}</span>
      <span className="shrink-0 font-mono text-2xs text-muted">
        {c.status === "ok" && c.count != null ? c.count : c.status === "error" ? "err" : c.status === "skipped" ? "key" : ""}
        {c.ms != null && c.status !== "running" ? <span className="ml-1.5 text-faint">{ms(c.ms)}</span> : null}
      </span>
    </li>
  );
}
