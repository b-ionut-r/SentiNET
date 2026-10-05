/**
 * Price × tone: candlesticks (TradingView lightweight-charts) with catalyst
 * markers, then volume and daily GDELT news tone in SEPARATE synced panes
 * below — same time axis, each with its own value scale (never a dual axis).
 * Beside it, the tone→return lead/lag readout from /api/history.
 */
import {
  ColorType,
  CrosshairMode,
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type LogicalRange,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type WhitespaceData,
} from "lightweight-charts";
import { Loader2, RotateCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useHistory, usePrice } from "../../api/hooks";
import type { Analysis, HistoryResponse, PriceRange } from "../../api/types";
import { Columns } from "../../components/charts/Columns";
import { Meter } from "../../components/charts/Bars";
import { Empty, Segmented, Skeleton } from "../../components/ui/Misc";
import { Panel, SubHead } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { compact, longDate, MINUS, ordinal, pct, price as fmtPrice, signed } from "../../lib/format";
import { textTone } from "../../lib/sentiment";
import { tokenColor, useTheme } from "../../lib/theme";
import { criticalR, fmtP, LAGS_TESTED, lagMissingReason, lagVerdict, MIN_RELIABLE_N, reliableLag } from "./lagRule";
import { buildRows, type Row, toneOf, TZ_SHIFT } from "./priceRows";

const RANGES: PriceRange[] = ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y"];
/** Ranges with one bar per session or week (date-keyed time axis). */
const DAILY: ReadonlySet<PriceRange> = new Set(["1M", "3M", "6M", "1Y", "5Y"]);
/** Ranges where the daily tone pane is meaningful (GDELT history is ~90 days). */
const TONE_RANGES: ReadonlySet<PriceRange> = new Set(["1M", "3M", "6M", "1Y"]);
export default function PriceTone({ a }: { a: Analysis }) {
  const [range, setRange] = useState<PriceRange>("3M");
  const priceQ = usePrice(a.ticker, range);
  const history = useHistory(a.ticker, 90);
  // The axis follows the candles on screen, not the tab just clicked: while a new range
  // loads, the previous range's bars stay up (5-minute bars must never be keyed by date).
  const shownRange = priceQ.data?.range ?? range;
  const daily = DAILY.has(shownRange);
  const toneSeries = useMemo(() => toneOf(a, history.data), [a, history.data]);
  const withTone = TONE_RANGES.has(shownRange);
  const rows = useMemo(() => buildRows(priceQ.data?.candles ?? [], withTone ? toneSeries : [], daily), [priceQ.data, toneSeries, daily, withTone]);
  const showTone = withTone && toneSeries.length > 0;
  const currency = priceQ.data?.currency ?? a.quote?.currency;

  // The lead/lag column only earns its space when it has something to show; otherwise the
  // price chart takes the full width and one honest line explains what's missing. Its slot
  // is held while the history loads, so the chart doesn't narrow under a fitted range.
  const t = a.tone;
  const hasLags = !!history.data && history.data.lags.length > 0;
  const hasToneSummary = !!t && [t.tone_7d, t.tone_30d, t.tone_90d].some((v) => v != null);
  const side = hasLags || hasToneSummary || history.isPending;

  return (
    <div className="grid gap-4 lg:grid-cols-12">
      <Panel
        title="Price × news tone"
        subtitle={
          showTone
            ? "Daily global news tone (GDELT) in its own pane below — same dates, separate scale"
            : toneSeries.length === 0
              ? history.isPending
                ? "Loading the daily GDELT tone series…"
                : "No GDELT tone series for this ticker yet — price only"
              : "Tone is daily — switch to 1M–1Y to see it under the price"
        }
        actions={<Segmented options={RANGES.map((r) => ({ value: r, label: r }))} value={range} onChange={setRange} size="xs" label="Price range" />}
        className={side ? "lg:col-span-8" : "lg:col-span-12"}
        footer={
          <div className="flex flex-wrap items-center justify-between gap-2">
            <MarkerLegend />
            <a className="text-muted hover:text-ink-2" href="https://www.tradingview.com/" target="_blank" rel="noreferrer">
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
      {side ? (
        <ToneLeadPanel a={a} history={history.data} loading={history.isPending} error={history.error} onRetry={() => history.refetch()} className="lg:col-span-4" />
      ) : (
        <LeadLagNotice a={a} history={history.data} loading={history.isPending} error={history.error} onRetry={() => history.refetch()} className="lg:col-span-12" />
      )}
    </div>
  );
}

function MarkerLegend() {
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <span className="inline-flex items-center gap-1">
        <span className="text-[9px] text-bull-ink">▲</span> bullish event
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="text-[9px] text-bear-ink">▼</span> bearish event
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

/** Axis decimals for a price level: none for ≥ 1,000 (BTC reads 88,000), cents above 1, more for sub-dollar. */
function axisDigits(ref: number): number {
  const v = Math.abs(ref);
  return v >= 1000 ? 0 : v >= 1 ? 2 : v >= 0.01 ? 4 : 6;
}

/** Narrower or shorter than this, a pane is hidden or mid-relayout: it keeps its last size. */
const MIN_PANE_PX = 8;

interface Pane {
  chart: IChartApi;
  series: ISeriesApi<"Candlestick"> | ISeriesApi<"Histogram">;
  value: (r: Row) => number | null;
}

/**
 * Three synced panes on one time index: candles (+ catalyst markers), volume, and
 * daily news tone. Separate panes keep every value scale honest — the price axis
 * never labels the volume band, and tone has its own zero-centred scale.
 */
function Charts({ a, rows, showTone, daily, currency }: { a: Analysis; rows: Row[]; showTone: boolean; daily: boolean; currency?: string | null }) {
  const priceEl = useRef<HTMLDivElement>(null);
  const volEl = useRef<HTMLDivElement>(null);
  const toneEl = useRef<HTMLDivElement>(null);
  const charts = useRef<{ price?: IChartApi; vol?: IChartApi; tone?: IChartApi; candles?: ISeriesApi<"Candlestick">; volBars?: ISeriesApi<"Histogram">; toneBars?: ISeriesApi<"Candlestick">; zero?: IPriceLine }>({});
  const { theme } = useTheme();
  const [hover, setHover] = useState<Row | null>(null);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  const showVol = rows.some((r) => r.candle?.v != null && r.candle.v > 0);

  // Create the panes once; wire range + crosshair sync across all of them.
  useEffect(() => {
    if (!priceEl.current || !volEl.current || !toneEl.current) return;
    const common = {
      // Sized by the observer below, not `autoSize`: autoSize follows every size the host
      // reports, including a transient 0 px (Playwright's full-page capture, print layout),
      // and lockVisibleTimeRangeOnResize then carries a degenerate range back — a 3M chart
      // came back showing its last 3 bars.
      autoSize: false,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, fontFamily: "Inter Variable, Inter, system-ui, sans-serif", fontSize: 11, attributionLogo: false },
      rightPriceScale: { borderVisible: false, minimumWidth: 64 },
      // No edge clamps: each pane would clamp to its OWN last non-empty bar (tone can run past the
      // last session), which shifts the panes against each other. One shared range, applied as-is.
      // The lead/lag column can appear after the first fit and narrow the chart: keep the
      // fitted dates on screen (squeeze the bars) instead of dropping the oldest third.
      timeScale: { borderVisible: false, rightOffset: 1, lockVisibleTimeRangeOnResize: true },
      crosshair: { mode: CrosshairMode.Normal },
      handleScale: { axisPressedMouseMove: false },
      // Pin the locale: some environments report tags Intl rejects (e.g. "en-US@posix").
      localization: { locale: "en-US" },
    };
    const price = createChart(priceEl.current, { ...common, rightPriceScale: { ...common.rightPriceScale, scaleMargins: { top: 0.08, bottom: 0.06 } } });
    const vol = createChart(volEl.current, { ...common, rightPriceScale: { ...common.rightPriceScale, scaleMargins: { top: 0.18, bottom: 0 } } });
    const tone = createChart(toneEl.current, { ...common, rightPriceScale: { ...common.rightPriceScale, scaleMargins: { top: 0.12, bottom: 0.08 } } });
    const candles = price.addCandlestickSeries({ borderVisible: false, priceLineVisible: false, lastValueVisible: true });
    const volBars = vol.addHistogramSeries({ priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
    // Tone as zero-based candle bodies: diverging bars whose width tracks the price candles at every range.
    const toneBars = tone.addCandlestickSeries({ borderVisible: false, wickVisible: false, priceLineVisible: false, lastValueVisible: false, priceFormat: { type: "price", precision: 2, minMove: 0.01 } });
    const zero = toneBars.createPriceLine({ price: 0, color: tokenColor("axis"), lineWidth: 1, lineStyle: 0, axisLabelVisible: false });
    charts.current = { price, vol, tone, candles, volBars, toneBars, zero };

    const all = [price, vol, tone];
    const hosts = new Map<IChartApi, HTMLElement>([
      [price, priceEl.current],
      [vol, volEl.current],
      [tone, toneEl.current],
    ]);
    // Follow the host's size, but never down to nothing: a hidden pane (display: none) or a
    // momentarily collapsed viewport keeps its last real size, and with it its date range.
    const byHost = new Map<Element, IChartApi>([...hosts].map(([c, el]) => [el, c]));
    const fit = (el: Element, width: number, height: number) => {
      const w = Math.floor(width);
      const h = Math.floor(height);
      if (w < MIN_PANE_PX || h < MIN_PANE_PX) return;
      const c = byHost.get(el);
      const o = c?.options();
      if (c && o && (o.width !== w || o.height !== h)) c.resize(w, h);
    };
    for (const [el] of byHost) fit(el, el.clientWidth, el.clientHeight);
    const sizer = new ResizeObserver((entries) => entries.forEach((e) => fit(e.target, e.contentRect.width, e.contentRect.height)));
    for (const [el] of byHost) sizer.observe(el);
    // A hidden pane (0px wide) computes a degenerate range on fitContent; it must never drive the others.
    const shown = (c: IChartApi) => (hosts.get(c)?.clientWidth ?? 0) > 0;
    let syncing = false;
    for (const src of all) {
      src.timeScale().subscribeVisibleLogicalRangeChange((r: LogicalRange | null) => {
        if (syncing || !r || !shown(src)) return;
        syncing = true;
        for (const c of all) if (c !== src) c.timeScale().setVisibleLogicalRange(r);
        syncing = false;
      });
    }

    const rowAt = (t: Time | undefined) => {
      if (t == null) return null;
      const k = timeKey(t);
      return rowsRef.current.find((r) => timeKey(r.time) === k) ?? null;
    };
    const panes: Pane[] = [
      { chart: price, series: candles, value: (r) => r.candle?.c ?? null },
      { chart: vol, series: volBars, value: (r) => r.candle?.v ?? null },
      { chart: tone, series: toneBars, value: (r) => r.tone },
    ];
    for (const src of panes) {
      src.chart.subscribeCrosshairMove((p: MouseEventParams) => {
        const row = rowAt(p.time);
        setHover(row);
        for (const o of panes) {
          if (o === src) continue;
          const v = row && p.time != null ? o.value(row) : null;
          // Only mirror the crosshair where that pane can place it: lightweight-charts throws
          // ("Value is null") for a pane with no bar on screen yet — e.g. the tone pane while
          // the 90-day history is still arriving under a hovering cursor — or a hidden pane.
          const placeable = v != null && p.time != null && shown(o.chart) && o.series.priceToCoordinate(v) != null && o.chart.timeScale().timeToCoordinate(p.time) != null;
          if (placeable) o.chart.setCrosshairPosition(v, p.time!, o.series);
          else o.chart.clearCrosshairPosition();
        }
      });
    }

    return () => {
      sizer.disconnect();
      for (const c of all) c.remove(); // also drops every subscription
      charts.current = {};
    };
  }, []);

  // Theme colors (re-applied when the theme flips).
  useEffect(() => {
    const { price, vol, tone, candles, toneBars, zero } = charts.current;
    if (!price || !vol || !tone || !candles || !toneBars) return;
    const text = tokenColor("muted");
    const grid = tokenColor("grid");
    const cross = tokenColor("ink-2", 0.45);
    for (const c of [price, vol, tone]) {
      c.applyOptions({
        layout: { textColor: text },
        grid: { vertLines: { visible: false }, horzLines: { color: grid } },
        crosshair: {
          vertLine: { color: cross, width: 1, style: 0, labelBackgroundColor: tokenColor("raised") },
          horzLine: { color: cross, width: 1, style: 0, labelBackgroundColor: tokenColor("raised") },
        },
      });
    }
    vol.applyOptions({ grid: { horzLines: { visible: false } } });
    candles.applyOptions({ upColor: tokenColor("bull"), downColor: tokenColor("bear"), wickUpColor: tokenColor("bull"), wickDownColor: tokenColor("bear") });
    toneBars.applyOptions({ upColor: tokenColor("bull", 0.85), downColor: tokenColor("bear", 0.85) });
    zero?.applyOptions({ color: tokenColor("axis") });
  }, [theme]);

  // Data + markers.
  useEffect(() => {
    const { price, vol, tone, candles, volBars, toneBars } = charts.current;
    if (!price || !vol || !tone || !candles || !volBars || !toneBars) return;
    const bull = tokenColor("bull");
    const bear = tokenColor("bear");
    const volColor = tokenColor("muted", 0.4);
    const lastClose = [...rows].reverse().find((r) => r.candle)?.candle?.c ?? 1;
    const digits = axisDigits(lastClose);
    candles.applyOptions({
      priceFormat: { type: "custom", minMove: 10 ** -digits, formatter: (v: number) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }) },
    });
    candles.setData(rows.map((r) => (r.candle ? { time: r.time, open: r.candle.o, high: r.candle.h, low: r.candle.l, close: r.candle.c } : ({ time: r.time } as WhitespaceData))));
    volBars.setData(rows.map((r) => (r.candle?.v != null ? { time: r.time, value: r.candle.v, color: volColor } : ({ time: r.time } as WhitespaceData))));
    toneBars.setData(
      rows.map((r) => (r.tone != null ? { time: r.time, open: 0, close: r.tone, high: Math.max(0, r.tone), low: Math.min(0, r.tone) } : ({ time: r.time } as WhitespaceData))),
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

    // Only the lowest visible pane carries the date axis.
    price.applyOptions({ timeScale: { visible: !showVol && !showTone, timeVisible: !daily, secondsVisible: false } });
    vol.applyOptions({ timeScale: { visible: showVol && !showTone, timeVisible: !daily, secondsVisible: false } });
    tone.applyOptions({ timeScale: { visible: showTone, timeVisible: false } });
    // Fit the price pane; the sync carries its range to the visible panes below.
    price.timeScale().fitContent();
  }, [rows, showTone, showVol, daily, a.catalysts, a.earnings, theme]);

  const last = [...rows].reverse().find((r) => r.candle) ?? null;
  const shown = hover ?? last;
  const prevClose = useMemo(() => {
    if (!shown?.candle) return null;
    const i = rows.indexOf(shown);
    for (let k = i - 1; k >= 0; k--) if (rows[k].candle) return rows[k].candle!.c;
    return null;
  }, [shown, rows]);
  const chg = shown?.candle && prevClose ? (shown.candle.c / prevClose - 1) * 100 : null;
  const crypto = a.profile?.quote_type === "CRYPTOCURRENCY";

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
                {shown.candle.v != null && shown.candle.v > 0 && (
                  <span className="text-muted">
                    Vol <span className="text-ink-2">{compact(shown.candle.v)}</span>
                  </span>
                )}
              </>
            )}
            {!shown.candle && <span className="text-muted">market closed</span>}
            {showTone && (
              <span className="text-muted">
                Tone <span className={cx("font-semibold", shown.tone == null ? "text-muted" : textTone[shown.tone > 0.25 ? "bull" : shown.tone < -0.25 ? "bear" : "neutral"])}>{shown.tone == null ? "—" : signed(shown.tone)}</span>
              </span>
            )}
          </>
        ) : null}
      </div>
      <div ref={priceEl} className={cx("w-full", showVol ? "h-[248px] max-sm:h-[196px]" : "h-[300px] max-sm:h-[240px]")} role="img" aria-label="Candlestick price chart" />
      <div className={cx("relative mt-1", !showVol && "hidden")}>
        <span className="pointer-events-none absolute left-0 top-0 z-[1] text-2xs text-muted">Volume</span>
        <div ref={volEl} className="h-[56px] w-full max-sm:h-[48px]" role="img" aria-label="Daily volume" />
      </div>
      <div className={cx("mt-1", !showTone && "hidden")}>
        <div className="mb-0.5 flex items-center justify-between text-2xs text-muted">
          <span>News tone · GDELT avg (≈ {MINUS}3 … +3)</span>
          {!crypto && <span className="hidden sm:inline">weekend news rolls into the next session</span>}
        </div>
        <div ref={toneEl} className="h-[92px] w-full" role="img" aria-label="Daily news tone histogram" />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------- */

interface LeadLagProps {
  a: Analysis;
  history?: HistoryResponse;
  loading: boolean;
  error: Error | null;
  onRetry: () => void;
  className?: string;
}

/** Compact stand-in for the lead/lag panel when there is nothing to chart. */
function LeadLagNotice({ a, history, loading, error, onRetry, className }: LeadLagProps) {
  return (
    <div className={cx("panel flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3 text-xs", className)}>
      <span className="font-semibold text-ink">Does news tone lead {a.ticker}?</span>
      {loading ? (
        <span className="inline-flex items-center gap-1.5 text-muted">
          <Loader2 className="size-3.5 animate-spin" aria-hidden />
          Checking 90 days of GDELT tone against returns — GDELT is rate-limited, this can take a few seconds.
        </span>
      ) : (
        <>
          {/* Its own line on phones (a flex-basis of 0 squeezed it into a ~90px column beside Retry). */}
          <span className="min-w-0 grow basis-full text-muted sm:basis-0">{lagMissingReason(a.ticker, history, error)}</span>
          <button className="btn h-7 text-xs" onClick={onRetry}>
            <RotateCw className="size-3.5" /> Retry
          </button>
        </>
      )}
    </div>
  );
}

function ToneLeadPanel({ a, history, loading, error, onRetry, className }: LeadLagProps) {
  const t = a.tone;
  return (
    <Panel title={`Does news tone lead ${a.ticker}?`} subtitle="Pearson r of daily tone vs. returns at −3…+3 day lags (90d)" className={className}>
      {loading ? (
        <div>
          <Skeleton className="h-[84px] w-full" />
          <p className="mt-2 inline-flex items-center gap-1.5 text-2xs text-muted">
            <Loader2 className="size-3 animate-spin" aria-hidden />
            Loading 90 days of GDELT tone and prices — GDELT is rate-limited, this can take a few seconds.
          </p>
        </div>
      ) : error || !history || history.lags.length === 0 ? (
        <div className="flex items-start justify-between gap-3 rounded-md bg-sunken px-3 py-2.5 text-xs text-muted">
          <span>{lagMissingReason(a.ticker, history, error)}</span>
          <button className="btn h-7 shrink-0 text-xs" onClick={onRetry}>
            <RotateCw className="size-3.5" /> Retry
          </button>
        </div>
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
                <div className={cx("text-base font-semibold", v == null ? "text-muted" : textTone[v > 0.25 ? "bull" : v < -0.25 ? "bear" : "neutral"])}>{signed(v)}</div>
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
              <div className="mt-1 flex justify-between text-2xs text-muted">
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

function LagView({ h }: { h: HistoryResponse }) {
  const best = h.best_lag;
  const items = [...h.lags]
    .sort((x, y) => x.lag_days - y.lag_days)
    .map((l) => ({
      key: String(l.lag_days),
      label: l.lag_days === 0 ? "0" : `${l.lag_days > 0 ? "+" : MINUS}${Math.abs(l.lag_days)}d`,
      value: l.r,
      emphasis: best?.lag_days === l.lag_days,
      color: reliableLag(l) ? "rgb(var(--ink))" : "rgb(var(--ink-2) / 0.55)",
      tip: (
        <div className="space-y-0.5">
          <div className="font-semibold text-ink">r = {signed(l.r)}</div>
          <div>
            lag {l.lag_days > 0 ? `tone leads by ${l.lag_days}d` : l.lag_days < 0 ? `returns lead by ${-l.lag_days}d` : "same day"} · p = {fmtP(l.p_value)} · n = {l.n}
          </div>
          <div className="text-muted">{lagVerdict(l)}</div>
        </div>
      ),
    }));
  const ns = h.lags.map((l) => l.n).filter((n) => n > 3).sort((x, y) => x - y);
  const nMid = ns.length ? ns[Math.floor(ns.length / 2)] : 0;
  const rCrit = nMid > 3 ? criticalR(nMid) : null;
  const maxAbs = Math.min(1, Math.max(0.5, (rCrit ?? 0) * 1.6, ...h.lags.map((l) => Math.abs(l.r))));
  return (
    <div>
      <p className="mb-3 text-sm leading-5 text-ink">{h.interpretation || "No interpretation available."}</p>
      <Columns
        items={items}
        height={84}
        max={maxAbs}
        band={rCrit != null ? { value: rCrit, label: `|r| < ${rCrit.toFixed(2)} doesn't survive testing ${LAGS_TESTED} lags at n = ${nMid}` } : null}
        format={(v) => signed(v)}
        labels="emphasis"
        ariaLabel="Correlation of news tone with returns by lag"
      />
      <div className="mt-1.5 flex justify-between text-2xs text-muted">
        <span>← returns lead tone</span>
        <span>tone leads returns →</span>
      </div>
      {rCrit != null && (
        <p className="mt-1.5 flex items-center gap-1.5 text-2xs text-muted">
          <span className="inline-block h-2.5 w-3 rounded-sm bg-[rgb(var(--ink-2)/0.09)]" aria-hidden />
          {nMid < MIN_RELIABLE_N
            ? `Shaded band = noise. Only ${nMid} paired days — under ${MIN_RELIABLE_N}, no lag counts as reliable yet`
            : `Shaded band = noise: at n = ${nMid}, |r| below ${rCrit.toFixed(2)} doesn't survive testing ${LAGS_TESTED} lags`}
        </p>
      )}
      {best && (
        <div className="mt-3 grid grid-cols-4 gap-2 text-center text-xs">
          {(
            [
              ["best lag", best.lag_days === 0 ? "0d" : `${best.lag_days > 0 ? "+" : MINUS}${Math.abs(best.lag_days)}d`],
              ["r", signed(best.r)],
              ["p", fmtP(best.p_value)],
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
