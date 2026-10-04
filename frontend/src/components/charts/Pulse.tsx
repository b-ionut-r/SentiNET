/**
 * Signal pulse: per time bucket, bullish items rise above the axis and
 * bearish items fall below it (neutral items are left out of the bars but
 * kept in the tooltip). Shows *when* the tone shifted and how loud it was.
 */
import type { TimelineBucket } from "../../api/types";
import { signed } from "../../lib/format";
import { useTooltip } from "../ui/Tooltip";

/** Place buckets on an even time grid so quiet stretches show as gaps. */
function onTimeGrid(buckets: TimelineBucket[]): Array<TimelineBucket | { t: string; empty: true }> {
  const ts = buckets.map((b) => Date.parse(b.t));
  let step = Infinity;
  for (let i = 1; i < ts.length; i++) step = Math.min(step, ts[i] - ts[i - 1]);
  if (!Number.isFinite(step) || step < 15 * 60_000) return buckets;
  const slots = Math.round((ts[ts.length - 1] - ts[0]) / step) + 1;
  if (slots > 240) return buckets;
  const byT = new Map(buckets.map((b, i) => [Math.round((ts[i] - ts[0]) / step), b]));
  return Array.from({ length: slots }, (_, i) => byT.get(i) ?? { t: new Date(ts[0] + i * step).toISOString(), empty: true as const });
}

export function Pulse({ buckets: raw, height = 56 }: { buckets: TimelineBucket[]; height?: number }) {
  const { showAt, hide } = useTooltip();
  if (raw.length < 2) return null;
  const sorted = [...raw].sort((x, y) => x.t.localeCompare(y.t));
  const buckets = onTimeGrid(sorted);
  const max = Math.max(1, ...sorted.map((b) => Math.max(b.bullish, b.bearish)));
  const half = height / 2;
  const fmt = (t: string) => new Date(t).toLocaleString("en-US", { weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false });
  const first = buckets[0].t;
  const last = buckets[buckets.length - 1].t;
  return (
    <div>
      <div className="relative flex gap-px" style={{ height }} role="img" aria-label={`Signal pulse, ${buckets.length} buckets`}>
        <div className="absolute inset-x-0 h-px bg-[rgb(var(--axis))]" style={{ top: half }} />
        {buckets.map((b) => {
          if ("empty" in b) return <div key={b.t} className="flex-1" aria-hidden />;
          const up = (b.bullish / max) * (half - 2);
          const down = (b.bearish / max) * (half - 2);
          const tip = (
            <div className="space-y-0.5">
              <div className="text-2xs text-muted">{fmt(b.t)}</div>
              <div>
                <span className="font-semibold text-ink">{b.count}</span> items · {b.news} news · {b.social} social
              </div>
              <div>
                <span className="text-bull">▲ {b.bullish}</span> · <span className="text-bear">▼ {b.bearish}</span> · mean {signed(b.score)}
              </div>
            </div>
          );
          return (
            <div
              key={b.t}
              tabIndex={0}
              className="group relative flex-1 outline-none"
              onMouseEnter={(e) => showAt(tip, e.currentTarget)}
              onMouseLeave={hide}
              onFocus={(e) => showAt(tip, e.currentTarget)}
              onBlur={hide}
            >
              {up > 0 && <div className="absolute inset-x-[15%] rounded-t-[2px] bg-bull group-hover:brightness-125" style={{ bottom: half + 1, height: Math.max(2, up) }} />}
              {down > 0 && <div className="absolute inset-x-[15%] rounded-b-[2px] bg-bear group-hover:brightness-125" style={{ top: half + 1, height: Math.max(2, down) }} />}
            </div>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-2xs text-faint">
        <span>{fmt(first)}</span>
        <span>
          <span className="text-bull">▲</span> bullish items above · <span className="text-bear">▼</span> bearish below
        </span>
        <span>{fmt(last)}</span>
      </div>
    </div>
  );
}
