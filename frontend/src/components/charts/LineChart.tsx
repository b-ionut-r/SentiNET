/**
 * Multi-series time line chart on ONE y-axis. A crosshair snaps to the
 * nearest date and a single tooltip lists every series there; arrow keys move
 * it for keyboard users. Legend for ≥ 2 series, optional end labels.
 */
import { useMemo, useState, type ReactNode } from "react";

import { useSize } from "../../lib/useSize";
import { useTooltip } from "../ui/Tooltip";
import { extent, linear, linePath, niceTicks } from "./scale";

export interface LineSeries {
  key: string;
  label: string;
  color: string;
  /** `note` is extra tooltip context for that point (e.g. the raw daily value behind a smoothed line). */
  points: Array<{ x: number; y: number | null; note?: string }>;
  area?: boolean;
}

export interface Zone {
  from: number;
  to: number;
  label: string;
  color: string;
}

interface LineChartProps {
  series: LineSeries[];
  height?: number;
  yDomain?: [number, number];
  yTicks?: number[];
  yFormat?: (v: number) => string;
  valueFormat?: (v: number) => string;
  xFormat?: (ms: number) => string;
  xTipFormat?: (ms: number) => string;
  baseline?: number | null;
  zones?: Zone[];
  endLabels?: boolean;
  legend?: boolean;
  ariaLabel: string;
  footer?: ReactNode;
}

const M = { top: 10, bottom: 22, left: 34 };

const dayFmt = (ms: number) => new Date(ms).toLocaleDateString("en-US", { month: "short", day: "numeric" });

export function LineChart({
  series,
  height = 200,
  yDomain,
  yTicks,
  yFormat = (v) => String(v),
  valueFormat,
  xFormat = dayFmt,
  xTipFormat = (ms) => new Date(ms).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" }),
  baseline = null,
  zones,
  endLabels = false,
  legend,
  ariaLabel,
  footer,
}: LineChartProps) {
  const [ref, { width }] = useSize();
  const { show, hide } = useTooltip();
  const [hover, setHover] = useState<number | null>(null);
  const fmtV = valueFormat ?? yFormat;
  const right = endLabels ? 64 : 10;

  const xs = useMemo(() => {
    const set = new Set<number>();
    series.forEach((s) => s.points.forEach((p) => set.add(p.x)));
    return [...set].sort((a, b) => a - b);
  }, [series]);

  const lookup = useMemo(() => series.map((s) => new Map(s.points.map((p) => [p.x, p.y]))), [series]);
  const notes = useMemo(() => series.map((s) => new Map(s.points.filter((p) => p.note).map((p) => [p.x, p.note as string]))), [series]);

  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  const ext = yDomain ?? extent(baseline != null ? [...ys, baseline] : ys) ?? [0, 1];
  const ticks = yTicks ?? niceTicks(ext[0], ext[1], height < 160 ? 3 : 4);
  const y0 = yDomain ? yDomain[0] : Math.min(ext[0], ticks[0] ?? ext[0]);
  const y1 = yDomain ? yDomain[1] : Math.max(ext[1], ticks[ticks.length - 1] ?? ext[1]);
  const innerW = Math.max(0, width - M.left - right);
  const innerH = height - M.top - M.bottom;
  const x = linear(xs[0] ?? 0, xs[xs.length - 1] ?? 1, M.left, M.left + innerW);
  const y = linear(y0, y1, M.top + innerH, M.top);

  // Evenly spaced date ticks, never more than there are dates (no duplicate labels on short series).
  const xTickCount = Math.min(xs.length, Math.max(2, Math.min(6, Math.floor(innerW / 110))));
  const xTicks = xs.length > 1 ? [...new Set(Array.from({ length: xTickCount }, (_, i) => Math.round((i * (xs.length - 1)) / (xTickCount - 1))))].map((i) => xs[i]) : xs;

  const showLegend = legend ?? series.length >= 2;

  const tipFor = (idx: number) => {
    const t = xs[idx];
    return (
      <div className="space-y-1">
        <div className="text-2xs text-muted">{xTipFormat(t)}</div>
        {series.map((s, i) => {
          const v = lookup[i].get(t);
          return (
            <div key={s.key} className="flex items-center gap-2 whitespace-nowrap">
              <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: s.color }} />
              <span className="font-semibold text-ink num">{v == null ? "—" : fmtV(v)}</span>
              <span className="text-muted">{s.label}</span>
              {notes[i].get(t) && <span className="text-muted">· {notes[i].get(t)}</span>}
            </div>
          );
        })}
      </div>
    );
  };

  const place = (idx: number, svg: SVGSVGElement) => {
    setHover(idx);
    const r = svg.getBoundingClientRect();
    show(tipFor(idx), r.left + x(xs[idx]), r.top + M.top);
  };

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (xs.length === 0) return;
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(x(xs[i]) - px);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    place(best, e.currentTarget);
  };

  const onKey = (e: React.KeyboardEvent<SVGSVGElement>) => {
    if (!xs.length) return;
    const cur = hover ?? xs.length - 1;
    const next: Record<string, number> = { ArrowLeft: cur - 1, ArrowRight: cur + 1, Home: 0, End: xs.length - 1 };
    if (!(e.key in next)) return;
    e.preventDefault();
    place(Math.max(0, Math.min(xs.length - 1, next[e.key])), e.currentTarget);
  };

  const leave = () => {
    setHover(null);
    hide();
  };

  // End labels: last finite value per series, nudged apart minimally.
  const ends = endLabels
    ? series
        .map((s) => {
          const last = [...s.points].reverse().find((p) => p.y != null);
          return last ? { s, yv: last.y as number, py: y(last.y as number) } : null;
        })
        .filter((v): v is { s: LineSeries; yv: number; py: number } => v != null)
        .sort((a, b) => a.py - b.py)
    : [];
  for (let i = 1; i < ends.length; i++) if (ends[i].py - ends[i - 1].py < 13) ends[i].py = ends[i - 1].py + 13;

  return (
    <figure className="min-w-0">
      {showLegend && (
        <figcaption className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
          {series.map((s) => (
            <span key={s.key} className="inline-flex items-center gap-1.5">
              <span className="h-0.5 w-3.5 rounded-full" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </figcaption>
      )}
      <div ref={ref} style={{ height }} className="relative">
        {width > 0 && xs.length > 0 && (
          <svg
            width={width}
            height={height}
            className="block touch-none"
            role="img"
            aria-label={`${ariaLabel} — use arrow keys to read each date`}
            tabIndex={0}
            onPointerMove={onMove}
            onPointerLeave={leave}
            onBlur={leave}
            onKeyDown={onKey}
          >
            {zones?.map((z) => (
              <g key={z.label}>
                <rect x={M.left} width={innerW} y={y(Math.min(z.to, y1))} height={Math.max(0, y(Math.max(z.from, y0)) - y(Math.min(z.to, y1)))} fill={z.color} />
              </g>
            ))}
            {ticks.map((t) => (
              <g key={t}>
                <line x1={M.left} x2={M.left + innerW} y1={y(t)} y2={y(t)} stroke="rgb(var(--grid))" strokeWidth={1} shapeRendering="crispEdges" />
                <text x={M.left - 6} y={y(t) + 3.5} textAnchor="end" className="fill-[rgb(var(--muted))] text-[10px] num">
                  {yFormat(t)}
                </text>
              </g>
            ))}
            {baseline != null && baseline >= y0 && baseline <= y1 && (
              <line x1={M.left} x2={M.left + innerW} y1={y(baseline)} y2={y(baseline)} stroke="rgb(var(--axis))" strokeWidth={1} shapeRendering="crispEdges" />
            )}
            {xTicks.map((t, i) => (
              <text
                key={`${t}-${i}`}
                x={x(t)}
                y={height - 6}
                textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}
                className="fill-[rgb(var(--muted))] text-[10px] num"
              >
                {xFormat(t)}
              </text>
            ))}
            {series.map((s) => {
              const pts: Array<[number, number | null]> = s.points.map((p) => [x(p.x), p.y == null ? null : y(p.y)]);
              const d = linePath(pts);
              const finite = pts.filter((p): p is [number, number] => p[1] != null);
              return (
                <g key={s.key}>
                  {s.area && finite.length > 1 && (
                    <path
                      d={`${linePath(finite)}L${finite[finite.length - 1][0]},${M.top + innerH}L${finite[0][0]},${M.top + innerH}Z`}
                      fill={s.color}
                      opacity={0.1}
                    />
                  )}
                  <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                </g>
              );
            })}
            {ends.map(({ s, yv, py }) => {
              const lastX = x(xs[xs.length - 1]);
              return (
                <g key={`end-${s.key}`}>
                  <line x1={lastX + 4} x2={lastX + 10} y1={y(yv)} y2={py} stroke={s.color} strokeWidth={1} />
                  <text x={lastX + 13} y={py + 3.5} className="fill-[rgb(var(--ink-2))] text-[10.5px] font-medium">
                    {s.label}
                  </text>
                </g>
              );
            })}
            {hover != null && (
              <g pointerEvents="none">
                <line x1={x(xs[hover])} x2={x(xs[hover])} y1={M.top} y2={M.top + innerH} stroke="rgb(var(--ink-2))" strokeWidth={1} opacity={0.5} />
                {series.map((s, i) => {
                  const v = lookup[i].get(xs[hover]);
                  return v == null ? null : (
                    <circle key={s.key} cx={x(xs[hover])} cy={y(v)} r={4} fill={s.color} stroke="rgb(var(--panel))" strokeWidth={2} />
                  );
                })}
              </g>
            )}
          </svg>
        )}
      </div>
      {footer}
    </figure>
  );
}
