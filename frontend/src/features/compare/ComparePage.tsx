/**
 * Compare up to four tickers: verdict dials, a component matrix, news tone on
 * one shared axis (same unit), and a key-stats table. URL is the state:
 * /compare?t=AAPL,NVDA,MSFT
 */
import { Plus, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { useAnalysisPlain } from "../../api/hooks";
import type { Analysis, ComponentKey } from "../../api/types";
import { Dial, VERDICT_DIAL } from "../../components/charts/Dial";
import { LineChart, type LineSeries } from "../../components/charts/LineChart";
import { Mark, ScoreChip } from "../../components/ui/Badges";
import { Empty, Skeleton, TickerLogo } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { money, pct, price, ratioPct, signed } from "../../lib/format";
import { divergingFill, polarityOf100, scoreCell100, textTone } from "../../lib/sentiment";

const MAX = 4;
/** Categorical identity colours, by slot (never by rank or position). */
const SLOT = ["rgb(var(--cat-1))", "rgb(var(--cat-2))", "rgb(var(--cat-3))", "rgb(var(--cat-4))"];
const COMPONENTS: Array<{ key: ComponentKey; label: string }> = [
  { key: "news", label: "News" },
  { key: "social", label: "Social" },
  { key: "analysts", label: "Analysts" },
  { key: "insiders", label: "Insiders" },
  { key: "momentum", label: "Momentum" },
  { key: "technicals", label: "Technicals" },
];

/**
 * Stable ticker → colour slot. A ticker keeps its hue for as long as it stays
 * on the page (removing another never repaints it); a newcomer takes the
 * lowest free slot.
 */
function useStableSlots(tickers: string[]): string[] {
  const assigned = useRef(new Map<string, number>());
  const key = tickers.join(",");
  return useMemo(() => {
    const prev = assigned.current;
    const next = new Map<string, number>();
    for (const t of tickers) if (prev.has(t)) next.set(t, prev.get(t)!);
    for (const t of tickers) {
      if (next.has(t)) continue;
      const used = new Set(next.values());
      let s = 0;
      while (used.has(s)) s++;
      next.set(t, s);
    }
    assigned.current = next;
    return tickers.map((t) => SLOT[next.get(t)!]);
  }, [key]); // `key` is the value identity of `tickers` (a fresh array every render)
}

function parseTickers(raw: string | null): string[] {
  return [...new Set((raw ?? "").split(/[,\s]+/).map((t) => t.trim().toUpperCase().replace(/^\$/, "")).filter(Boolean))].slice(0, MAX);
}

export default function ComparePage() {
  const [params, setParams] = useSearchParams();
  const tickers = parseTickers(params.get("t"));
  const [draft, setDraft] = useState("");
  const colors = useStableSlots(tickers);

  useEffect(() => {
    document.title = tickers.length ? `Compare ${tickers.join(" · ")} — SentiNET` : "Compare — SentiNET";
  }, [tickers]);

  const setTickers = (list: string[]) => setParams(list.length ? { t: list.join(",") } : {}, { replace: true });
  const add = (e: React.FormEvent) => {
    e.preventDefault();
    const t = draft.trim().toUpperCase().replace(/^\$/, "");
    if (t && !tickers.includes(t) && tickers.length < MAX) setTickers([...tickers, t]);
    setDraft("");
  };

  // Fixed number of hooks: one per slot.
  const q0 = useAnalysisPlain(tickers[0] ?? "", !!tickers[0]);
  const q1 = useAnalysisPlain(tickers[1] ?? "", !!tickers[1]);
  const q2 = useAnalysisPlain(tickers[2] ?? "", !!tickers[2]);
  const q3 = useAnalysisPlain(tickers[3] ?? "", !!tickers[3]);
  const queries = [q0, q1, q2, q3].slice(0, tickers.length);
  const loaded = queries.map((q) => q.data ?? null);
  // A ticker that failed to analyze has no column data, ever: its cells say so ("—")
  // instead of a loading ellipsis that never resolves.
  const failed = queries.map((q) => !q.data && !!q.error);
  const remove = (t: string) => setTickers(tickers.filter((x) => x !== t));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[22px] font-semibold tracking-[-0.01em] text-ink">Compare</h1>
          <p className="mt-0.5 text-sm text-muted">Up to four tickers side by side — same scales, same evidence.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {tickers.map((t, i) => (
            <span key={t} className="inline-flex h-8 items-center gap-2 rounded-lg bg-raised pl-2.5 pr-1 text-sm font-medium text-ink" style={{ boxShadow: "0 0 0 1px var(--hairline)" }}>
              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colors[i] }} aria-hidden />
              <span className="font-mono">{t}</span>
              <button className="rounded p-1 text-muted hover:bg-panel hover:text-ink" onClick={() => remove(t)} aria-label={`Remove ${t}`}>
                <X className="size-3.5" />
              </button>
            </span>
          ))}
          {tickers.length < MAX && (
            <form onSubmit={add} className="flex items-center gap-1.5">
              <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Add ticker" className="field w-28 uppercase placeholder:normal-case" aria-label="Add ticker to compare" />
              <button className="btn px-2" type="submit" aria-label="Add">
                <Plus className="size-4" />
              </button>
            </form>
          )}
        </div>
      </div>

      {tickers.length === 0 ? (
        <Panel>
          <Empty title="Pick tickers to compare">
            Try{" "}
            <Link className="link" to="/compare?t=AAPL,MSFT,NVDA">
              AAPL · MSFT · NVDA
            </Link>{" "}
            or{" "}
            <Link className="link" to="/compare?t=BTC-USD,COIN,MSTR">
              BTC-USD · COIN · MSTR
            </Link>
            .
          </Empty>
        </Panel>
      ) : (
        <>
          <div className={cx("grid gap-4", colsClass(tickers.length))}>
            {tickers.map((t, i) => (
              <VerdictCard key={t} ticker={t} color={colors[i]} q={queries[i]} onRemove={() => remove(t)} />
            ))}
          </div>
          {loaded.some(Boolean) && (
            <>
              <ComponentMatrix tickers={tickers} colors={colors} data={loaded} failed={failed} />
              <ToneCompare tickers={tickers} colors={colors} data={loaded} />
              <StatsTable tickers={tickers} colors={colors} data={loaded} failed={failed} />
            </>
          )}
        </>
      )}
    </div>
  );
}

const colsClass = (n: number) => (n <= 1 ? "" : n === 2 ? "md:grid-cols-2" : n === 3 ? "md:grid-cols-3" : "md:grid-cols-2 xl:grid-cols-4");

/** Cell for a column with no analysis: still loading, or never coming (the run failed). */
function NoData({ ticker, failed }: { ticker: string; failed: boolean }) {
  return failed ? (
    <span className="text-xs text-muted" title={`Couldn't analyze ${ticker}`}>
      —
    </span>
  ) : (
    <span className="text-xs text-muted">…</span>
  );
}

function VerdictCard({ ticker, color, q, onRemove }: { ticker: string; color: string; q: ReturnType<typeof useAnalysisPlain>; onRemove: () => void }) {
  if (q.isPending) {
    return (
      <div className="panel space-y-3 p-4" aria-busy="true">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="mx-auto h-28 w-40 rounded-full" />
        <Skeleton className="h-12 w-full" />
        <p className="text-center text-2xs text-muted">Analyzing {ticker}… first runs take ~10s</p>
      </div>
    );
  }
  if (q.error || !q.data) {
    return (
      <div className="panel p-4">
        <Empty title={`Couldn't analyze ${ticker}`}>
          {q.error?.message}
          <div className="mt-3">
            <button className="btn h-7 px-2.5 text-xs" onClick={onRemove}>
              <X className="size-3.5" /> Remove {ticker}
            </button>
          </div>
        </Empty>
      </div>
    );
  }
  const a = q.data;
  const v = a.verdict;
  const p = polarityOf100(v.score);
  const top = a.narratives[0];
  return (
    <section className="panel flex flex-col p-4">
      <header className="flex items-center gap-2.5">
        <span className="h-8 w-1 shrink-0 rounded-full" style={{ background: color }} aria-hidden />
        <TickerLogo symbol={a.ticker} url={a.profile?.logo_url} size={28} />
        <div className="min-w-0 flex-1">
          <Link to={`/t/${encodeURIComponent(a.ticker)}`} className="block truncate text-sm font-semibold text-ink hover:underline">
            {a.profile?.short_name ?? a.profile?.name ?? a.ticker}
          </Link>
          <div className="flex items-baseline gap-2 text-2xs text-muted">
            <span className="font-mono">{a.ticker}</span>
            <span className="text-ink-2 num">{price(a.quote?.price, a.quote?.currency)}</span>
            {a.quote?.change_pct != null && <span className={textTone[a.quote.change_pct > 0 ? "bull" : a.quote.change_pct < 0 ? "bear" : "neutral"]}>{pct(a.quote.change_pct, 2)}</span>}
          </div>
        </div>
      </header>
      <div className="mt-3">
        <Dial value={v.score} bands={VERDICT_DIAL} size={160} thickness={9} ticks={false} ariaLabel={`${a.ticker} SentiNET ${v.score}`}>
          <span className="text-[40px] font-semibold leading-none tracking-[-0.04em] text-ink">{v.score}</span>
          <span className={cx("mt-1 flex items-center gap-1 text-xs font-semibold", textTone[p])}>
            <Mark p={p} />
            {v.label}
          </span>
        </Dial>
        <p className="-mt-1 text-center text-2xs text-muted">{v.confidence[0].toUpperCase() + v.confidence.slice(1)} confidence</p>
      </div>
      <p className="mt-3 line-clamp-3 text-[13px] leading-[19px] text-ink-2" title={v.headline}>
        {v.headline}
      </p>
      {top && (
        <div className="mt-auto pt-3">
          <div className="rounded-md bg-sunken p-2.5">
            <div className="mb-1 flex items-center justify-between">
              <span className="eyebrow">Top story</span>
              <ScoreChip score={top.score} />
            </div>
            <p className="line-clamp-2 text-xs text-ink">{top.headline}</p>
            <p className="mt-0.5 text-2xs text-muted">{top.count} items</p>
          </div>
        </div>
      )}
    </section>
  );
}

function ComponentMatrix({ tickers, colors, data, failed }: { tickers: string[]; colors: string[]; data: Array<Analysis | null>; failed: boolean[] }) {
  const rows: Array<{ key: string; label: string; get: (a: Analysis) => number | null }> = [
    { key: "sentinet", label: "SentiNET", get: (a) => a.verdict.score },
    ...COMPONENTS.map((c) => ({ key: c.key, label: c.label, get: (a: Analysis) => a.verdict.components.find((x) => x.key === c.key && x.available)?.score ?? null })),
  ];
  return (
    <Panel title="Component matrix" subtitle="0–100 per input · cell tint = distance from neutral 50" flush>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" style={{ minWidth: 104 + tickers.length * 76 }}>
          <thead>
            <tr className="hairline-t hairline-b">
              <th className="w-24 py-2 pl-4 text-left text-2xs font-normal text-muted sm:w-32">Component</th>
              {tickers.map((t, i) => (
                <th key={t} className="py-2 text-center text-xs font-semibold text-ink">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="size-2 rounded-sm" style={{ background: colors[i] }} />
                    <span className="font-mono">{t}</span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-hair">
            {rows.map((r) => (
              <tr key={r.key} className={cx(r.key === "sentinet" && "font-semibold")}>
                <td className="py-1.5 pl-4 text-xs text-ink-2">{r.label}</td>
                {data.map((a, i) => {
                  const v = a ? r.get(a) : null;
                  const cell = v == null ? null : scoreCell100(v);
                  const comp = a?.verdict.components.find((c) => c.key === r.key);
                  return (
                    <td key={tickers[i]} className="px-1.5 py-1">
                      <div
                        className="flex h-8 items-center justify-center rounded-md text-sm num"
                        style={{ background: cell == null ? "transparent" : `color-mix(in oklab, ${divergingFill(cell.tint)} 30%, transparent)` }}
                        title={comp?.detail ?? undefined}
                      >
                        {cell == null ? (
                          a ? <span className="text-xs text-muted">n/a</span> : <NoData ticker={tickers[i]} failed={failed[i]} />
                        ) : (
                          <span className="inline-flex items-center gap-1 text-ink">
                            {cell.polarity !== "neutral" && <Mark p={cell.polarity} className="text-[8px]" />}
                            {cell.value}
                          </span>
                        )}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

const SMOOTH_DAYS = 7;
const SMOOTH_MIN = 3;

/**
 * One point per calendar day carrying a 7-day trailing mean of GDELT tone (needs ≥ 3
 * reported days in the window, so long outages stay gaps instead of being bridged).
 * The raw daily value rides along as a tooltip note.
 */
function smoothedDaily(series: Array<{ date: string; tone: number | null }>): Array<{ x: number; y: number | null; note?: string }> {
  const byDay = new Map(series.filter((p) => p.tone != null).map((p) => [Date.parse(`${p.date}T12:00:00Z`), p.tone as number]));
  const days = [...byDay.keys()].filter(Number.isFinite).sort((x, y) => x - y);
  if (!days.length) return [];
  const out: Array<{ x: number; y: number | null; note?: string }> = [];
  for (let x = days[0]; x <= days[days.length - 1]; x += 864e5) {
    const win: number[] = [];
    for (let k = 0; k < SMOOTH_DAYS; k++) {
      const v = byDay.get(x - k * 864e5);
      if (v != null) win.push(v);
    }
    const day = byDay.get(x);
    out.push({ x, y: win.length >= SMOOTH_MIN ? win.reduce((s, v) => s + v, 0) / win.length : null, note: day != null ? `day ${signed(day)}` : "no report that day" });
  }
  return out;
}

function ToneCompare({ tickers, colors, data }: { tickers: string[]; colors: string[]; data: Array<Analysis | null> }) {
  const series = useMemo<LineSeries[]>(
    () =>
      data.flatMap((a, i) =>
        a?.tone?.series.length
          ? [{ key: tickers[i], label: tickers[i], color: colors[i], points: smoothedDaily(a.tone.series) }]
          : [],
      ),
    [data, tickers, colors],
  );
  const missing = tickers.filter((_, i) => data[i] && !data[i]?.tone?.series.length);
  return (
    <Panel title="Global news tone" subtitle={`GDELT tone, ${SMOOTH_DAYS}-day trailing mean — one shared axis, same unit for every ticker; hover for the daily value`}>
      {series.length === 0 ? (
        <Empty title="No tone history">GDELT returned no series for these tickers.</Empty>
      ) : (
        <LineChart series={series} height={230} baseline={0} endLabels yFormat={(v) => signed(v, 1)} valueFormat={(v) => signed(v)} ariaLabel="News tone comparison" />
      )}
      {missing.length > 0 && <p className="mt-2 text-2xs text-muted">No GDELT series for {missing.join(", ")} in this run — GDELT allows one request every 5 s, so tone can lag behind; refresh later.</p>}
    </Panel>
  );
}

function StatsTable({ tickers, colors, data, failed }: { tickers: string[]; colors: string[]; data: Array<Analysis | null>; failed: boolean[] }) {
  const tone = (v: number | null | undefined, digits = 1, suffix = "%") =>
    v == null ? <span className="text-muted">—</span> : <span className={textTone[v > 0 ? "bull" : v < 0 ? "bear" : "neutral"]}>{suffix === "%" ? pct(v, digits) : signed(v, digits)}</span>;
  const rows: Array<[string, (a: Analysis) => ReactNode]> = [
    ["Price", (a) => price(a.quote?.price, a.quote?.currency)],
    ["Today", (a) => tone(a.quote?.change_pct, 2)],
    ["1 month", (a) => tone(a.technicals?.return_1m)],
    ["3 months", (a) => tone(a.technicals?.return_3m)],
    ["vs 200-DMA", (a) => tone(a.technicals?.vs_200dma_pct)],
    ["RSI 14", (a) => (a.technicals?.rsi_14 != null ? a.technicals.rsi_14.toFixed(0) : "—")],
    ["Market cap", (a) => money(a.quote?.market_cap, a.quote?.currency)],
    ["News tone", (a) => (a.news.n ? <ScoreChip score={a.news.score} /> : "—")],
    ["Social tone", (a) => (a.social.n ? <ScoreChip score={a.social.score} /> : "—")],
    ["Analyst consensus", (a) => (a.analysts?.consensus ? a.analysts.consensus.replace("_", " ") : "—")],
    ["Upside to target", (a) => tone(a.analysts?.upside_pct)],
    ["StockTwits bullish", (a) => ratioPct(a.crowd?.stocktwits_bull_ratio)],
    ["Reddit rank", (a) => (a.crowd?.reddit_rank != null ? `#${a.crowd.reddit_rank}` : "—")],
    ["Attention", (a) => (a.attention ? `${a.attention.heat} · ${a.attention.label}` : "—")],
    ["Next earnings", (a) => (a.earnings?.days_until != null ? `${a.earnings.days_until}d` : "—")],
    ["Signals", (a) => String(a.sentiment.n)],
  ];
  return (
    <Panel title="Key stats" flush>
      <div className="overflow-x-auto">
        <table className="w-full text-xs num" style={{ minWidth: 104 + tickers.length * 76 }}>
          <thead>
            <tr className="hairline-t hairline-b">
              <th className="py-2 pl-4 text-left font-normal text-muted sm:w-40" />
              {tickers.map((t, i) => (
                <th key={t} className="py-2 pr-4 text-right font-semibold text-ink">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="size-2 rounded-sm" style={{ background: colors[i] }} />
                    <span className="font-mono">{t}</span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-hair">
            {rows.map(([label, get]) => (
              <tr key={label}>
                <td className="py-1.5 pl-4 text-ink-2">{label}</td>
                {data.map((a, i) => (
                  <td key={tickers[i]} className="py-1.5 pr-4 text-right capitalize text-ink">
                    {a ? get(a) : <NoData ticker={tickers[i]} failed={failed[i]} />}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
