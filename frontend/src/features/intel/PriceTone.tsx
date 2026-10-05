/**
 * Price × tone: candlesticks (TradingView lightweight-charts) with catalyst
 * markers, and daily GDELT news tone in a SEPARATE synced pane below — same
 * time axis, its own value scale (never a dual axis). Beside it, the
 * tone→return lead/lag readout from /api/history.
 */
import {
  ColorType,
  CrosshairMode,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LogicalRange,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
  type WhitespaceData,
} from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";

import { useHistory, usePrice } from "../../api/hooks";
import type { Analysis, Candle, HistoryResponse, PriceRange, TonePoint } from "../../api/types";
import { Columns } from "../../components/charts/Columns";
import { Meter } from "../../components/charts/Bars";
import { Empty, Segmented, Skeleton } from "../../components/ui/Misc";
import { Panel, SubHead } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { longDate, MINUS, ordinal, pct, price as fmtPrice, signed } from "../../lib/format";
import { textTone } from "../../lib/sentiment";
import { tokenColor, useTheme } from "../../lib/theme";

const RANGES: PriceRange[] = ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y"];
/** Ranges with one bar per session or week (date-keyed time axis). */
const DAILY: ReadonlySet<PriceRange> = new Set(["1M", "3M", "6M", "1Y", "5Y"]);
/** Ranges where the daily tone pane is meaningful (GDELT history is ~90 days). */
const TONE_RANGES: ReadonlySet<PriceRange> = new Set(["1M", "3M", "6M", "1Y"]);
const TZ_SHIFT = -new Date().getTimezoneOffset() * 60;

interface Row {
  time: Time;
  candle: Candle | null;
  tone: number | null;
  volume: number | null;
}

/**
 * Calendar day of a daily candle. Bars are stamped at local midnight of the
 * exchange (04:00Z for New York, 15:00Z the day before for Tokyo); shifting by
 * 12h lands every exchange from UTC−12 to UTC+12 on its own session date.
 */
function sessionDay(t: string): string {
  const ms = Date.parse(t);
  return Number.isFinite(ms) ? new Date(ms + 12 * 3600_000).toISOString().slice(0, 10) : t.slice(0, 10);
}

/** Build one shared time index for both panes (trading days + trailing news-only days). */
function buildRows(candles: Candle[], tone: TonePoint[], daily: boolean): Row[] {
  if (!daily) {
    return candles.map((c) => ({ time: (Math.floor(new Date(c.t).getTime() / 1000) + TZ_SHIFT) as UTCTimestamp, candle: c, tone: null, volume: null }));
  }
  const rows: Row[] = candles.map((c) => ({ time: sessionDay(c.t), candle: c, tone: null, volume: null }));
  const days = rows.map((r) => r.time as string);
  const bucket = new Map<string, number[]>();
  const vol = new Map<string, number>();
  const trailing: TonePoint[] = [];
  for (const p of tone) {
    if (p.tone == null) continue;
    // Weekend/holiday tone rolls forward to the next trading day it could move.
    const target = days.find((d) => d >= p.date);
    if (!target) {
      trailing.push(p);
      continue;
    }
    bucket.set(target, [...(bucket.get(target) ?? []), p.tone]);
    vol.set(target, (vol.get(target) ?? 0) + (p.volume ?? 0));
  }
  for (const r of rows) {
    const vals = bucket.get(r.time as string);
    if (vals?.length) {
      r.tone = vals.reduce((s, v) => s + v, 0) / vals.length;
      r.volume = vol.get(r.time as string) ?? null;
    }
  }
  if (candles.length && tone.length && rows.length) {
    for (const p of trailing) rows.push({ time: p.date, candle: null, tone: p.tone, volume: p.volume });
  }
  return rows;
}

export default function PriceTone({ a }: { a: Analysis }) {
  const [range, setRange] = useState<PriceRange>("3M");
  const priceQ = usePrice(a.ticker, range);
  const history = useHistory(a.ticker, 90);
  const daily = DAILY.has(range);
  const toneSeries = useMemo(() => a.tone?.series ?? [], [a.tone]);
  const withTone = TONE_RANGES.has(range);
  const rows = useMemo(() => buildRows(priceQ.data?.candles ?? [], withTone ? toneSeries : [], daily), [priceQ.data, toneSeries, daily, withTone]);
  const showTone = withTone && toneSeries.length > 0;
  const currency = priceQ.data?.currency ?? a.quote?.currency;

  return (
    <div id="price" className="grid scroll-mt-36 md:scroll-mt-28 gap-4 lg:grid-cols-12">
      <Panel
        title="Price × news tone"
        subtitle={
          showTone
            ? "Daily global news tone (GDELT) in its own pane below — same dates, separate scale"
            : toneSeries.length === 0
              ? "No GDELT tone series for this ticker yet — price only"
              : "Tone is daily — switch to 1M–1Y to see it under the price"
        }
        actions={<Segmented options={RANGES.map((r) => ({ value: r, label: r }))} value={range} onChange={setRange} size="xs" label="Price range" />}
        className="lg:col-span-8"
        footer={
          <div className="flex flex-wrap items-center justify-between gap-2">
            <MarkerLegend />
            <a className="text-faint hover:text-muted" href="https://www.tradingview.com/" target="_blank" rel="noreferrer">
              Charts by TradingView
            </a>
          </div>
        }
      >
        {priceQ.isPending ? (
          <Skeleton className="h-[392px] w-full" />
        ) : priceQ.error || !priceQ.data?.available || !priceQ.data.candles.length ? (
          <Empty title="Price history unavailable" className="h-[392px]">
            {priceQ.data?.error ?? (priceQ.error instanceof Error ? priceQ.error.message : "No candles returned for this range.")}
          </Empty>
        ) : (
          <div className={cx(priceQ.isPlaceholderData && "is-refetching")}>
            <Charts a={a} rows={rows} showTone={showTone} daily={daily} currency={currency} />
          </div>
        )}
      </Panel>
      <ToneLeadPanel a={a} history={history.data} loading={history.isPending} error={history.error} className="lg:col-span-4" />
    </div>
  );
}

function MarkerLegend() {
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <span className="inline-flex items-center gap-1">
        <span className="text-[9px] text-bull">▲</span> bullish event
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="text-[9px] text-bear">▼</span> bearish event
      </span>
      <span>A analyst · E earnings · F filing · I insider · N news</span>
    </span>
  );
}

/** Comparable key for any lightweight-charts time (string, timestamp or BusinessDay). */
function timeKey(t: Time): string {
  if (typeof t === "object") return `${t.year}-${String(t.month).padStart(2, "0")}-${String(t.day).padStart(2, "0")}`;
  return String(t);
}

const MARK_CODE: Record<string, string> = { analyst: "A", earnings: "E", filing: "F", insider: "I", news: "N", dividend: "D" };

function Charts({ a, rows, showTone, daily, currency }: { a: Analysis; rows: Row[]; showTone: boolean; daily: boolean; currency?: string | null }) {
  const priceEl = useRef<HTMLDivElement>(null);
  const toneEl = useRef<HTMLDivElement>(null);
  const charts = useRef<{ price?: IChartApi; tone?: IChartApi; candles?: ISeriesApi<"Candlestick">; vol?: ISeriesApi<"Histogram">; toneBars?: ISeriesApi<"Histogram"> }>({});
  const { theme } = useTheme();
  const [hover, setHover] = useState<Row | null>(null);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;

  // Create both charts once; wire range + crosshair sync.
  useEffect(() => {
    if (!priceEl.current || !toneEl.current) return;
    const common = {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, fontFamily: "Inter Variable, Inter, system-ui, sans-serif", fontSize: 11, attributionLogo: false },
      rightPriceScale: { borderVisible: false, minimumWidth: 64 },
      timeScale: { borderVisible: false, rightOffset: 2, fixLeftEdge: true, fixRightEdge: true },
      crosshair: { mode: CrosshairMode.Normal },
      handleScale: { axisPressedMouseMove: false },
      // Pin the locale: some environments report tags Intl rejects (e.g. "en-US@posix").
      localization: { locale: "en-US" },
    };
    const price = createChart(priceEl.current, { ...common, rightPriceScale: { ...common.rightPriceScale, scaleMargins: { top: 0.08, bottom: 0.2 } } });
    const tone = createChart(toneEl.current, { ...common, rightPriceScale: { ...common.rightPriceScale, scaleMargins: { top: 0.12, bottom: 0.08 } } });
    const candles = price.addCandlestickSeries({ borderVisible: false, priceLineVisible: false, lastValueVisible: true });
    const vol = price.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
    price.priceScale("vol").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 }, visible: false });
    const toneBars = tone.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false, base: 0, priceFormat: { type: "price", precision: 2, minMove: 0.01 } });
    charts.current = { price, tone, candles, vol, toneBars };

    let syncing = false;
    const sync = (target: IChartApi) => (r: LogicalRange | null) => {
      if (syncing || !r) return;
      syncing = true;
      target.timeScale().setVisibleLogicalRange(r);
      syncing = false;
    };
    const toTone = sync(tone);
    const toPrice = sync(price);
    price.timeScale().subscribeVisibleLogicalRangeChange(toTone);
    tone.timeScale().subscribeVisibleLogicalRangeChange(toPrice);

    const rowAt = (t: Time | undefined) => {
      if (t == null) return null;
      const k = timeKey(t);
      return rowsRef.current.find((r) => timeKey(r.time) === k) ?? null;
    };
    const onMove = (src: "price" | "tone") => (p: MouseEventParams) => {
      const row = rowAt(p.time);
      setHover(row);
      const other = src === "price" ? tone : price;
      const series = src === "price" ? toneBars : candles;
      if (!row || p.time == null) {
        other.clearCrosshairPosition();
        return;
      }
      const v = src === "price" ? row.tone : row.candle?.c;
      if (v == null) other.clearCrosshairPosition();
      else other.setCrosshairPosition(v, p.time, series);
    };
    const mp = onMove("price");
    const mt = onMove("tone");
    price.subscribeCrosshairMove(mp);
    tone.subscribeCrosshairMove(mt);

    return () => {
      price.unsubscribeCrosshairMove(mp);
      tone.unsubscribeCrosshairMove(mt);
      price.remove();
      tone.remove();
      charts.current = {};
    };
  }, []);

  // Theme colors (re-applied when the theme flips).
  useEffect(() => {
    const { price, tone, candles, toneBars } = charts.current;
    if (!price || !tone || !candles || !toneBars) return;
    const text = tokenColor("muted");
    const grid = tokenColor("grid");
    const cross = tokenColor("ink-2", 0.45);
    for (const c of [price, tone]) {
      c.applyOptions({
        layout: { textColor: text },
        grid: { vertLines: { visible: false }, horzLines: { color: grid } },
        crosshair: {
          vertLine: { color: cross, width: 1, style: 0, labelBackgroundColor: tokenColor("raised") },
          horzLine: { color: cross, width: 1, style: 0, labelBackgroundColor: tokenColor("raised") },
        },
      });
    }
    candles.applyOptions({ upColor: tokenColor("bull"), downColor: tokenColor("bear"), wickUpColor: tokenColor("bull"), wickDownColor: tokenColor("bear") });
  }, [theme]);

  // Data + markers.
  useEffect(() => {
    const { price, tone, candles, vol, toneBars } = charts.current;
    if (!price || !tone || !candles || !vol || !toneBars) return;
    const bull = tokenColor("bull");
    const bear = tokenColor("bear");
    const volColor = tokenColor("muted", 0.28);
    candles.setData(rows.map((r) => (r.candle ? { time: r.time, open: r.candle.o, high: r.candle.h, low: r.candle.l, close: r.candle.c } : ({ time: r.time } as WhitespaceData))));
    vol.setData(rows.map((r) => (r.candle?.v != null ? { time: r.time, value: r.candle.v, color: volColor } : ({ time: r.time } as WhitespaceData))));
    toneBars.setData(
      rows.map((r) => (r.tone != null ? { time: r.time, value: r.tone, color: r.tone >= 0 ? tokenColor("bull", 0.85) : tokenColor("bear", 0.85) } : ({ time: r.time } as WhitespaceData))),
    );

    // Catalyst + past-earnings markers, snapped to the nearest bar at/after the event.
    const markers: SeriesMarker<Time>[] = [];
    if (daily || rows.length) {
      const times = rows.filter((r) => r.candle).map((r) => r.time);
      const snap = (iso: string): Time | null => {
        if (!times.length) return null;
        if (daily) {
          const d = iso.slice(0, 10);
          if (d < (times[0] as string) || d > (times[times.length - 1] as string)) return null;
          return (times.find((t) => (t as string) >= d) as Time) ?? null;
        }
        const s = (Math.floor(new Date(iso).getTime() / 1000) + TZ_SHIFT) as number;
        if (s < (times[0] as number) || s > (times[times.length - 1] as number) + 86400) return null;
        return (times.find((t) => (t as number) >= s) as Time) ?? (times[times.length - 1] as Time);
      };
      const add = (iso: string, kind: string, pol: "bull" | "bear" | "neutral") => {
        const t = snap(iso);
        if (t == null) return;
        markers.push({
          time: t,
          position: pol === "bull" ? "belowBar" : "aboveBar",
          shape: pol === "bull" ? "arrowUp" : pol === "bear" ? "arrowDown" : "circle",
          color: pol === "bull" ? bull : pol === "bear" ? bear : tokenColor("neu"),
          text: MARK_CODE[kind] ?? "",
          size: 0.8,
        });
      };
      a.catalysts.filter((c) => !c.upcoming).forEach((c) => add(c.date, c.kind, c.polarity));
      a.earnings?.history.forEach((e) => add(e.date, "earnings", e.surprise_pct == null ? "neutral" : e.surprise_pct > 0 ? "bull" : "bear"));
    }
    markers.sort((x, y) => {
      const kx = timeKey(x.time);
      const ky = timeKey(y.time);
      return typeof x.time === "number" && typeof y.time === "number" ? x.time - y.time : kx < ky ? -1 : kx > ky ? 1 : 0;
    });
    candles.setMarkers(markers);

    price.applyOptions({ timeScale: { visible: !showTone, timeVisible: !daily, secondsVisible: false } });
    tone.applyOptions({ timeScale: { visible: showTone, timeVisible: false } });
    price.timeScale().fitContent();
    tone.timeScale().fitContent();
  }, [rows, showTone, daily, a.catalysts, a.earnings, theme]);

  const last = [...rows].reverse().find((r) => r.candle) ?? null;
  const shown = hover ?? last;
  const prevClose = useMemo(() => {
    if (!shown?.candle) return null;
    const i = rows.indexOf(shown);
    for (let k = i - 1; k >= 0; k--) if (rows[k].candle) return rows[k].candle!.c;
    return null;
  }, [shown, rows]);
  const chg = shown?.candle && prevClose ? (shown.candle.c / prevClose - 1) * 100 : null;

  return (
    <div>
      <div className="mb-2 flex min-h-[20px] flex-wrap items-baseline gap-x-3 gap-y-0.5 text-xs num" aria-live="off">
        {shown ? (
          <>
            <span className="font-medium text-ink-2">{typeof shown.time === "string" ? longDate(shown.time) : new Date(((shown.time as number) - TZ_SHIFT) * 1000).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</span>
            {shown.candle && (
              <>
                <span className="text-muted">
                  O <span className="text-ink-2">{fmtPrice(shown.candle.o, currency)}</span>
                </span>
                <span className="text-muted">
                  H <span className="text-ink-2">{fmtPrice(shown.candle.h, currency)}</span>
                </span>
                <span className="text-muted">
                  L <span className="text-ink-2">{fmtPrice(shown.candle.l, currency)}</span>
                </span>
                <span className="text-muted">
                  C <span className="font-semibold text-ink">{fmtPrice(shown.candle.c, currency)}</span>
                </span>
                {chg != null && <span className={cx("font-medium", textTone[chg > 0 ? "bull" : chg < 0 ? "bear" : "neutral"])}>{pct(chg, 2)}</span>}
              </>
            )}
            {!shown.candle && <span className="text-muted">market closed</span>}
            {showTone && (
              <span className="text-muted">
                Tone <span className={cx("font-semibold", shown.tone == null ? "text-faint" : textTone[shown.tone > 0.25 ? "bull" : shown.tone < -0.25 ? "bear" : "neutral"])}>{shown.tone == null ? "—" : signed(shown.tone)}</span>
              </span>
            )}
          </>
        ) : null}
      </div>
      <div ref={priceEl} className="h-[300px] w-full max-sm:h-[240px]" role="img" aria-label="Candlestick price chart" />
      <div className={cx("mt-1", !showTone && "hidden")}>
        <div className="mb-0.5 flex items-center justify-between text-2xs text-muted">
          <span>News tone · GDELT avg (≈ {MINUS}3 … +3)</span>
          <span className="hidden sm:inline">weekend news rolls into the next session</span>
        </div>
        <div ref={toneEl} className="h-[92px] w-full" role="img" aria-label="Daily news tone histogram" />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------- */

function ToneLeadPanel({ a, history, loading, error, className }: { a: Analysis; history?: HistoryResponse; loading: boolean; error: Error | null; className?: string }) {
  const t = a.tone;
  return (
    <Panel title={`Does news tone lead ${a.ticker}?`} subtitle="Pearson r of daily tone vs. returns at −3…+3 day lags (90d)" className={className}>
      {loading ? (
        <div>
          <Skeleton className="h-40 w-full" />
          <p className="mt-2 text-2xs text-muted">Loading 90 days of GDELT tone and prices — GDELT is rate-limited, this can take a few seconds.</p>
        </div>
      ) : error || !history || history.lags.length === 0 ? (
        <Empty title="Lead/lag unavailable">{error?.message ?? "Not enough overlapping tone and price history yet."}</Empty>
      ) : (
        <LagView h={history} />
      )}
      {t && (
        <div className="mt-4 pt-3.5 hairline-t">
          <SubHead right={t.query ? <span className="truncate" title={t.query}>GDELT</span> : undefined}>Global news tone</SubHead>
          <div className="grid grid-cols-3 gap-2 text-center">
            {(
              [
                ["7 days", t.tone_7d],
                ["30 days", t.tone_30d],
                ["90 days", t.tone_90d],
              ] as const
            ).map(([l, v]) => (
              <div key={l} className="rounded-md bg-sunken px-2 py-2">
                <div className={cx("text-base font-semibold", v == null ? "text-faint" : textTone[v > 0.25 ? "bull" : v < -0.25 ? "bear" : "neutral"])}>{signed(v)}</div>
                <div className="text-2xs text-muted">{l}</div>
              </div>
            ))}
          </div>
          {t.percentile_7d != null && (
            <div className="mt-3">
              <div className="mb-1 flex justify-between text-2xs text-muted">
                <span>7-day tone within its 90-day range</span>
                <span className="font-medium text-ink-2">{ordinal(t.percentile_7d * 100)} pct</span>
              </div>
              <Meter value={t.percentile_7d} color="rgb(var(--ink-2))" height={6} />
              <div className="mt-1 flex justify-between text-2xs text-faint">
                <span>90d low</span>
                <span>90d high</span>
              </div>
            </div>
          )}
          {t.change_7d_vs_30d != null && (
            <p className="mt-2 text-xs text-ink-2">
              7d vs 30d: <span className={cx("font-semibold", textTone[t.change_7d_vs_30d > 0.1 ? "bull" : t.change_7d_vs_30d < -0.1 ? "bear" : "neutral"])}>{signed(t.change_7d_vs_30d)}</span>{" "}
              {t.change_7d_vs_30d > 0.1 ? "— improving" : t.change_7d_vs_30d < -0.1 ? "— deteriorating" : "— steady"}
            </p>
          )}
        </div>
      )}
    </Panel>
  );
}

/**
 * Smallest |r| that is significant (two-sided p < 0.05) for n paired points:
 * r = t / sqrt(df + t²) with the Student-t 97.5% quantile approximated by a
 * Cornish–Fisher series (within ~1% of exact for df ≥ 4).
 */
function criticalR(n: number): number {
  const df = n - 2;
  const t = 1.96 + 2.37 / df + 2.8 / (df * df);
  return t / Math.sqrt(df + t * t);
}

function LagView({ h }: { h: HistoryResponse }) {
  const best = h.best_lag;
  const sig = (p: number) => p < 0.05;
  const items = [...h.lags]
    .sort((x, y) => x.lag_days - y.lag_days)
    .map((l) => ({
      key: String(l.lag_days),
      label: l.lag_days === 0 ? "0" : `${l.lag_days > 0 ? "+" : MINUS}${Math.abs(l.lag_days)}d`,
      value: l.r,
      emphasis: best?.lag_days === l.lag_days,
      color: sig(l.p_value) ? "rgb(var(--ink))" : "rgb(var(--ink-2) / 0.55)",
      tip: (
        <div className="space-y-0.5">
          <div className="font-semibold text-ink">r = {signed(l.r)}</div>
          <div>
            lag {l.lag_days > 0 ? `tone leads by ${l.lag_days}d` : l.lag_days < 0 ? `returns lead by ${-l.lag_days}d` : "same day"} · p = {l.p_value.toFixed(2)} · n = {l.n}
          </div>
          <div className="text-muted">{sig(l.p_value) ? "statistically significant (p < 0.05)" : "not significant"}</div>
        </div>
      ),
    }));
  const ns = h.lags.map((l) => l.n).filter((n) => n > 3).sort((x, y) => x - y);
  const nMid = ns.length ? ns[Math.floor(ns.length / 2)] : 0;
  const rCrit = nMid > 3 ? criticalR(nMid) : null;
  const maxAbs = Math.max(0.5, (rCrit ?? 0) * 1.6, ...h.lags.map((l) => Math.abs(l.r)));
  return (
    <div>
      <p className="mb-3 text-sm leading-5 text-ink">{h.interpretation || "No interpretation available."}</p>
      <Columns
        items={items}
        height={84}
        max={maxAbs}
        band={rCrit != null ? { value: rCrit, label: `|r| < ${rCrit.toFixed(2)} is indistinguishable from noise at n = ${nMid}` } : null}
        format={(v) => signed(v)}
        labels="emphasis"
        ariaLabel="Correlation of news tone with returns by lag"
      />
      <div className="mt-1.5 flex justify-between text-2xs text-faint">
        <span>← returns lead tone</span>
        <span>tone leads returns →</span>
      </div>
      {rCrit != null && (
        <p className="mt-1.5 flex items-center gap-1.5 text-2xs text-muted">
          <span className="inline-block h-2.5 w-3 rounded-sm bg-[rgb(var(--ink-2)/0.09)]" aria-hidden />
          Shaded band = noise: |r| below {rCrit.toFixed(2)} isn't significant at n = {nMid} (p ≥ 0.05)
        </p>
      )}
      {best && (
        <div className="mt-3 grid grid-cols-4 gap-2 text-center text-xs">
          {(
            [
              ["best lag", best.lag_days === 0 ? "0d" : `${best.lag_days > 0 ? "+" : MINUS}${Math.abs(best.lag_days)}d`],
              ["r", signed(best.r)],
              ["p", best.p_value.toFixed(2)],
              ["n", String(best.n)],
            ] as const
          ).map(([l, v]) => (
            <div key={l}>
              <div className="font-semibold text-ink num">{v}</div>
              <div className="text-2xs text-muted">{l}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
