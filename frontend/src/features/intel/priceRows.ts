/**
 * Shared time index for the price, volume and tone panes (pure, unit-tested in
 * e2e/unit.mjs): one row per trading session plus trailing news-only days.
 */
import type { Time, UTCTimestamp } from "lightweight-charts";

import type { Analysis, Candle, HistoryResponse, TonePoint } from "../../api/types";

/** Seconds to shift UTC timestamps so intraday bars read in the viewer's local time. */
export const TZ_SHIFT = -new Date().getTimezoneOffset() * 60;

export interface Row {
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
export function sessionDay(t: string): string {
  const ms = Date.parse(t);
  return Number.isFinite(ms) ? new Date(ms + 12 * 3600_000).toISOString().slice(0, 10) : t.slice(0, 10);
}

/**
 * Build one shared time index for both panes (trading days + trailing news-only days).
 * Tone from a weekend/holiday rolls into the next session, volume-weighted like the
 * backend's alignment; tone dated before the first candle is dropped so the first
 * bar never silently averages weeks of history.
 */
export function buildRows(candles: Candle[], tone: TonePoint[], daily: boolean): Row[] {
  if (!daily) {
    const bars = uniqueBy(candles, (c) => Math.floor(new Date(c.t).getTime() / 1000) + TZ_SHIFT, (x, y) => x - y);
    return bars.map(([t, c]) => ({ time: t as UTCTimestamp, candle: c, tone: null, volume: null }));
  }
  const rows: Row[] = uniqueBy(candles, (c) => sessionDay(c.t), (x, y) => (x < y ? -1 : x > y ? 1 : 0)).map(([t, c]) => ({ time: t, candle: c, tone: null, volume: null }));
  const days = rows.map((r) => r.time as string);
  const acc = new Map<string, { sum: number; w: number; vol: number }>();
  const trailing: TonePoint[] = [];
  for (const p of tone) {
    if (p.tone == null || !days.length || p.date < days[0]) continue;
    const target = days.find((d) => d >= p.date);
    if (!target) {
      trailing.push(p);
      continue;
    }
    const w = p.volume != null && p.volume > 0 ? p.volume : 1;
    const a = acc.get(target) ?? { sum: 0, w: 0, vol: 0 };
    acc.set(target, { sum: a.sum + p.tone * w, w: a.w + w, vol: a.vol + (p.volume ?? 0) });
  }
  for (const r of rows) {
    const a = acc.get(r.time as string);
    if (a && a.w > 0) {
      r.tone = a.sum / a.w;
      r.volume = a.vol || null;
    }
  }
  // News-only days after the last session; the chart needs strictly ascending, unique times.
  const tail = new Map(trailing.map((p) => [p.date, p]));
  for (const d of [...tail.keys()].sort()) {
    const p = tail.get(d)!;
    rows.push({ time: d, candle: null, tone: p.tone, volume: p.volume });
  }
  return rows;
}

/**
 * Candles keyed and sorted by `key`, bars sharing a key merged into one (first open,
 * extreme high/low, last close, summed volume). The chart rejects repeated or
 * descending times outright, so this holds whatever the feed sends.
 */
function uniqueBy<K extends string | number>(candles: Candle[], key: (c: Candle) => K, cmp: (x: K, y: K) => number): Array<[K, Candle]> {
  const out = new Map<K, Candle>();
  for (const c of [...candles].sort((x, y) => Date.parse(x.t) - Date.parse(y.t))) {
    const k = key(c);
    const prev = out.get(k);
    out.set(
      k,
      prev
        ? { t: prev.t, o: prev.o, h: Math.max(prev.h, c.h), l: Math.min(prev.l, c.l), c: c.c, v: prev.v == null && c.v == null ? null : (prev.v ?? 0) + (c.v ?? 0) }
        : c,
    );
  }
  return [...out.entries()].sort((x, y) => cmp(x[0], y[0]));
}

/**
 * The daily GDELT tone to draw: the analysis's own series, or — when GDELT outlived that
 * run's time budget — the same series from the 90-day history fetched for the lead/lag
 * panel, so the chart never says "no tone" beside a panel that is correlating it.
 */
export function toneOf(a: Pick<Analysis, "tone">, history: Pick<HistoryResponse, "points"> | undefined): TonePoint[] {
  if (a.tone?.series.length) return a.tone.series;
  return (history?.points ?? []).filter((p) => p.tone != null).map((p) => ({ date: p.date, tone: p.tone, volume: p.volume }));
}
