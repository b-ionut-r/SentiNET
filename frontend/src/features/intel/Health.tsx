/** SEC filings (importance + polarity) and per-source health with honest statuses. */
import { ArrowUpRight, FileText, Radio } from "lucide-react";

import { useSources } from "../../api/hooks";
import type { Analysis, Filing, SourceReport } from "../../api/types";
import { Mark, ScoreChip, StatusBadge } from "../../components/ui/Badges";
import { Empty } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { ms, shortDate } from "../../lib/format";

const IMPORTANCE: Record<Filing["importance"], { label: string; dots: number }> = {
  high: { label: "High importance", dots: 3 },
  medium: { label: "Medium importance", dots: 2 },
  low: { label: "Low importance", dots: 1 },
};

export function FilingsPanel({ a, className }: { a: Analysis; className?: string }) {
  const cik = a.profile?.cik;
  return (
    <Panel
      id="filings"
      title="SEC filings"
      icon={<FileText />}
      subtitle="Material 8-K items decoded; insider-form floods summarized"
      className={className}
      flush
      actions={
        cik ? (
          <a className="btn h-7 px-2 text-xs" href={`https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${cik}&type=&dateb=&owner=include&count=40`} target="_blank" rel="noreferrer">
            EDGAR <ArrowUpRight className="size-3" />
          </a>
        ) : undefined
      }
    >
      {a.filings.length === 0 ? (
        <Empty title="No filings" className="hairline-t">
          {a.profile?.quote_type === "CRYPTOCURRENCY" ? "Crypto assets don't file with the SEC." : "No recent SEC filings found for this issuer."}
        </Empty>
      ) : (
        <ul className="divide-hair hairline-t">
          {a.filings.slice(0, 10).map((f, i) => {
            const imp = IMPORTANCE[f.importance] ?? IMPORTANCE.low;
            const row = (
              <>
                <span className="w-12 shrink-0 pt-px text-2xs text-muted num">{shortDate(f.date)}</span>
                <span className="w-16 shrink-0">
                  <span className="inline-flex h-5 max-w-full items-center truncate whitespace-nowrap rounded bg-raised px-1.5 font-mono text-2xs font-medium text-ink-2" title={f.form}>
                    {f.form.replace(/^SCHEDULE /, "SC ")}
                  </span>
                </span>
                <span className="min-w-0 flex-1">
                  <span className="line-clamp-3 block text-xs leading-[18px] text-ink" title={f.title}>
                    {f.title}
                  </span>
                  {f.items.length > 0 && <span className="text-2xs text-muted">Items {f.items.join(", ")}</span>}
                </span>
                <span className="flex shrink-0 items-center gap-2 pt-1" title={imp.label}>
                  {f.polarity !== "neutral" && <Mark p={f.polarity} />}
                  <span className="flex gap-0.5" aria-label={imp.label}>
                    {[1, 2, 3].map((d) => (
                      <span key={d} className={cx("size-1.5 rounded-full", d <= imp.dots ? "bg-[rgb(var(--ink-2))]" : "bg-[rgb(var(--grid))]")} />
                    ))}
                  </span>
                </span>
              </>
            );
            return (
              <li key={i}>
                {f.url ? (
                  <a href={f.url} target="_blank" rel="noreferrer" className="flex gap-2.5 px-4 py-2 hover:bg-raised/60">
                    {row}
                  </a>
                ) : (
                  <div className="flex gap-2.5 px-4 py-2">{row}</div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}

export function SourcesPanel({ a, className }: { a: Analysis; className?: string }) {
  const info = useSources();
  const docs = new Map((info.data ?? []).map((s) => [s.key, s.docs_url]));
  const ok = a.sources.filter((s) => s.status === "ok").length;
  const needKey = a.sources.filter((s) => s.status === "unconfigured").length;
  const failed = a.sources.filter((s) => s.status === "error").length;
  const order: Record<SourceReport["status"], number> = { ok: 0, empty: 1, error: 2, unconfigured: 3, disabled: 4 };
  const list = [...a.sources].filter((s) => s.status !== "unconfigured" && s.status !== "disabled").sort((x, y) => order[x.status] - order[y.status] || y.kept - x.kept);
  const keyless = a.sources.filter((s) => s.status === "unconfigured");
  const disabled = a.sources.filter((s) => s.status === "disabled");
  return (
    <Panel
      id="sources"
      title="Source health"
      icon={<Radio />}
      subtitle={`${ok} of ${a.sources.length} returned data${failed ? ` · ${failed} failed` : ""}${needKey ? ` · ${needKey} need a free key` : ""} · analysis ${ms(a.elapsed_ms)}`}
      className={className}
    >
      <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
        {list.map((s) => (
          <li key={s.key} className="rounded-lg bg-sunken p-2.5" style={{ boxShadow: "inset 0 0 0 1px var(--hairline)" }}>
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-xs font-medium text-ink">{s.label}</span>
              <StatusBadge status={s.status} />
            </div>
            <div className="mt-1 flex items-center justify-between gap-2 text-2xs text-muted">
              <span className="capitalize">
                {s.kind}
                {s.has_metrics && " · metrics"}
              </span>
              <span className="num">{s.latency_ms != null ? ms(s.latency_ms) : ""}</span>
            </div>
            {s.has_metrics && s.fetched === 0 && s.kept === 0 && s.status === "ok" ? (
              <p className="mt-1.5 text-2xs text-ink-2">Structured metrics only (ranks, counts)</p>
            ) : s.status === "ok" || s.status === "empty" ? (
              <div className="mt-1.5 flex items-center justify-between gap-2 text-2xs">
                <span className="text-ink-2 num">
                  {s.fetched} fetched → <span className="font-semibold text-ink">{s.kept}</span> kept
                </span>
                {s.score != null && s.kept > 0 && <ScoreChip score={s.score} />}
              </div>
            ) : (
              <p className="mt-1.5 truncate text-2xs text-critical" title={s.error ?? undefined}>
                {s.error ?? "Failed"}
              </p>
            )}
          </li>
        ))}
      </ul>
      {keyless.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg px-2.5 py-2 text-2xs" style={{ boxShadow: "inset 0 0 0 1px var(--hairline)" }}>
          <StatusBadge status="unconfigured" />
          <span className="text-muted">free key unlocks:</span>
          <span className="min-w-0">
            {keyless.map((s, i) => (
              <span key={s.key}>
                {docs.get(s.key) ? (
                  <a className="whitespace-nowrap font-medium text-accent hover:underline" href={docs.get(s.key) ?? undefined} target="_blank" rel="noreferrer" title={`Get a free ${s.label} key`}>
                    {s.label}
                  </a>
                ) : (
                  <span className="whitespace-nowrap font-medium text-ink-2">{s.label}</span>
                )}
                {i < keyless.length - 1 && <span className="text-faint">{"\u00a0· "}</span>}
              </span>
            ))}
          </span>
          <span className="text-muted">— set it in backend/.env</span>
        </div>
      )}
      {disabled.length > 0 && (
        <p className="mt-2 flex flex-wrap items-center gap-x-2 text-2xs text-muted">
          <StatusBadge status="disabled" />
          {disabled.map((s) => s.label).join(" · ")}
        </p>
      )}
    </Panel>
  );
}
