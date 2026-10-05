/**
 * Signal pulse: per time bucket, bullish items rise above the axis and
 * bearish items fall below it (neutral items are left out of the bars but
 * kept in the tooltip). Shows *when* the tone shifted and how loud it was.
 */
import type { TimelineBucket } from "../../api/types";
import { bucketTime, signed } from "../../lib/format";
import { useRoving } from "../../lib/useRoving";

/** Smallest gap between consecutive buckets (ms): the series' bucket size. */
function stepOf(buckets: TimelineBucket[]): number {
  const ts = buckets.map((b) => Date.parse(b.t));
  let step = Infinity;
  for (let i = 1; i < ts.length; i++) step = Math.min(step, ts[i] - ts[i - 1]);
  return step;
}

/** Place buckets on an even time grid so quiet stretches show as gaps. */
function onTimeGrid(buckets: TimelineBucket[], step: number): Array<TimelineBucket | { t: string; empty: true }> {
  const ts = buckets.map((b) => Date.parse(b.t));
  if (!Number.isFinite(step) || step < 15 * 60_000) return buckets;
  const slots = Math.round((ts[ts.length - 1] - ts[0]) / step) + 1;
  if (slots > 240) return buckets;
  const byT = new Map(buckets.map((b, i) => [Math.round((ts[i] - ts[0]) / step), b]));
  return Array.from({ length: slots }, (_, i) => byT.get(i) ?? { t: new Date(ts[0] + i * step).toISOString(), empty: true as const });
}

export function Pulse({ buckets: raw, height = 56 }: { buckets: TimelineBucket[]; height?: number }) {
  const sorted = [...raw].sort((x, y) => x.t.localeCompare(y.t));
  const step = stepOf(sorted);
  const buckets = raw.length < 2 ? [] : onTimeGrid(sorted, step);
  const filled = buckets.filter((b): b is TimelineBucket => !("empty" in b));
  const span = sorted.length > 1 ? Date.parse(sorted[sorted.length - 1].t) - Date.parse(sorted[0].t) : 0;
  const fmt = (t: string, detail = false) => bucketTime(t, span, step, detail);
  const tips = filled.map((b) => (
    <div className="space-y-0.5">
      <div className="text-2xs text-muted">{fmt(b.t, true)}</div>
      <div>
        <span className="font-semibold text-ink">{b.count}</span> items · {b.news} news · {b.social} social
      </div>
      <div>
        <span className="text-bull-ink">▲ {b.bullish}</span> · <span className="text-bear-ink">▼ {b.bearish}</span> · mean {signed(b.score)}
      </div>
    </div>
  ));
  const { container, mark } = useRoving(tips, `Signal pulse, ${filled.length} time buckets`);
  if (raw.length < 2) return null;
  const max = Math.max(1, ...sorted.map((b) => Math.max(b.bullish, b.bearish)));
  const half = height / 2;
  const first = buckets[0].t;
  const last = buckets[buckets.length - 1].t;
  let k = -1;
  return (
    <div>
      <div {...container} className={`relative flex gap-px ${container.className}`} style={{ height }}>
        <div className="absolute inset-x-0 h-px bg-[rgb(var(--axis))]" style={{ top: half }} />
        {buckets.map((b) => {
          if ("empty" in b) return <div key={b.t} className="flex-1" aria-hidden />;
          k += 1;
          const up = (b.bullish / max) * (half - 2);
          const down = (b.bearish / max) * (half - 2);
          return (
            <div key={b.t} {...mark(k)} className="group relative flex-1">
              {up > 0 && <div className="absolute inset-x-[15%] rounded-t-[2px] bg-bull group-hover:brightness-125" style={{ bottom: half + 1, height: Math.max(2, up) }} />}
              {down > 0 && <div className="absolute inset-x-[15%] rounded-b-[2px] bg-bear group-hover:brightness-125" style={{ top: half + 1, height: Math.max(2, down) }} />}
            </div>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-2xs text-muted">
        <span>{fmt(first)}</span>
        <span>
          <span className="text-bull-ink">▲</span> bullish items above · <span className="text-bear-ink">▼</span> bearish below
        </span>
        <span>{fmt(last)}</span>
      </div>
    </div>
  );
}
