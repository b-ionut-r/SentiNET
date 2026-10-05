/**
 * Small bar forms: diverging bar (centered on a neutral value), meter with a
 * baseline marker, stacked part-to-whole bar, and a low–high range bar.
 * Marks follow the house spec: thin, 2px surface gaps, recessive tracks.
 */
import type { ReactNode } from "react";

import { cx } from "../../lib/cx";
import { toneVar } from "../../lib/sentiment";
import { useRoving } from "../../lib/useRoving";

/* ------------------------------------------------------------------------- */

interface DivergingBarProps {
  value: number | null;
  center?: number;
  min?: number;
  max?: number;
  height?: number;
  className?: string;
  /** Neutral dead-zone half-width around center (drawn gray). */
  deadZone?: number;
}

/** Bar growing from a center line toward bull (right) or bear (left). */
export function DivergingBar({ value, center = 50, min = 0, max = 100, height = 6, className, deadZone = 4 }: DivergingBarProps) {
  const span = max - min;
  const c = ((center - min) / span) * 100;
  const v = value == null ? null : ((Math.max(min, Math.min(max, value)) - min) / span) * 100;
  const left = v == null ? c : Math.min(c, v);
  const width = v == null ? 0 : Math.abs(v - c);
  const p = value == null || Math.abs(value - center) <= deadZone ? "neutral" : value > center ? "bull" : "bear";
  const right = v != null && v >= c;
  return (
    <div className={cx("relative w-full", className)} style={{ height }}>
      <div className="absolute inset-0 rounded-full bg-[rgb(var(--grid))]" />
      {v != null && (
        <div
          className="absolute top-0 h-full transition-[left,width] duration-500 ease-out"
          style={{
            left: `${left}%`,
            width: `${Math.max(width, 0.8)}%`,
            background: toneVar(p),
            borderRadius: right ? "0 3px 3px 0" : "3px 0 0 3px",
          }}
        />
      )}
      <div className="absolute -top-1 -bottom-1 w-px bg-[rgb(var(--axis))]" style={{ left: `${c}%` }} aria-hidden />
    </div>
  );
}

/* ------------------------------------------------------------------------- */

interface MeterProps {
  /** 0..1 fill. */
  value: number | null;
  color: string;
  /** Track color — a lighter step of the same ramp. */
  track?: string;
  height?: number;
  marker?: { at: number; label: string } | null;
  className?: string;
}

/** Same-ramp meter with an optional baseline marker (e.g. typical level). */
export function Meter({ value, color, track, height = 8, marker, className }: MeterProps) {
  const v = value == null ? null : Math.max(0, Math.min(1, value));
  return (
    <div className={cx("relative w-full", className)} style={{ height }}>
      <div className="absolute inset-0 rounded-full" style={{ background: track ?? "rgb(var(--grid))" }} />
      {v != null && (
        <div
          className="absolute inset-y-0 left-0 transition-[width] duration-500 ease-out"
          style={{ width: `${v * 100}%`, background: color, borderRadius: v >= 0.99 ? 9999 : "9999px 3px 3px 9999px" }}
        />
      )}
      {marker && (
        <div className="absolute -top-1 -bottom-1 w-0.5 rounded-full bg-[rgb(var(--ink))]" style={{ left: `calc(${marker.at * 100}% - 1px)`, boxShadow: "0 0 0 2px rgb(var(--panel))" }} title={marker.label} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------------- */

export interface Segment {
  key: string;
  label: string;
  value: number;
  color: string;
}

/** Horizontal part-to-whole bar with 2px surface gaps and per-segment tooltips (one tab stop, ←/→ to read). */
export function StackedBar({ segments, height = 10, className, format = (n: number) => String(n) }: { segments: Segment[]; height?: number; className?: string; format?: (n: number) => string }) {
  const total = segments.reduce((s, x) => s + Math.max(0, x.value), 0);
  const visible = segments.filter((s) => s.value > 0);
  const tips = visible.map((s) => (
    <div className="flex items-center gap-2 whitespace-nowrap">
      <span className="size-2 rounded-sm" style={{ background: s.color }} />
      <span className="font-semibold text-ink num">{format(s.value)}</span>
      <span className="text-muted">
        {s.label} · {Math.round((s.value / total) * 100)}%
      </span>
    </div>
  ));
  const { container, mark } = useRoving(tips, visible.map((s) => `${s.label} ${format(s.value)}`).join(", "));
  if (total <= 0) return <div className={cx("rounded-full bg-[rgb(var(--grid))]", className)} style={{ height }} />;
  return (
    <div {...container} className={cx("flex w-full gap-[2px]", container.className, className)} style={{ height }}>
      {visible.map((s, i) => (
        <div
          key={s.key}
          {...mark(i)}
          className="h-full transition-[filter] hover:brightness-125 data-[kb-active=true]:brightness-125"
          style={{
            flexGrow: s.value,
            flexBasis: 0,
            minWidth: 3,
            background: s.color,
            borderRadius: `${i === 0 ? 3 : 1}px ${i === visible.length - 1 ? 3 : 1}px ${i === visible.length - 1 ? 3 : 1}px ${i === 0 ? 3 : 1}px`,
          }}
        />
      ))}
    </div>
  );
}

/** Legend row for a StackedBar (swatch mirrors the mark: a rect). */
export function SegmentLegend({ segments, format = (n: number) => String(n), className }: { segments: Segment[]; format?: (n: number) => string; className?: string }) {
  return (
    <div className={cx("flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted", className)}>
      {segments.map((s) => (
        <span key={s.key} className="inline-flex items-center gap-1.5 whitespace-nowrap">
          <span className="size-2 rounded-[2px]" style={{ background: s.color }} />
          {s.label}
          <span className="font-medium text-ink-2 num">{format(s.value)}</span>
        </span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------------- */

export interface RangeMarker {
  value: number;
  label: ReactNode;
  kind: "current" | "mean" | "median";
}

/** Low–high band with labelled markers (e.g. analyst targets vs. current price). */
export function RangeBar({
  low,
  high,
  markers,
  lowLabel,
  highLabel,
  className,
}: {
  low: number;
  high: number;
  markers: RangeMarker[];
  lowLabel: ReactNode;
  highLabel: ReactNode;
  className?: string;
}) {
  const values = [low, high, ...markers.map((m) => m.value)];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo) * 0.04 || 1;
  const d0 = lo - pad;
  const d1 = hi + pad;
  const x = (v: number) => ((v - d0) / (d1 - d0)) * 100;
  const current = markers.find((m) => m.kind === "current");
  const others = markers.filter((m) => m.kind !== "current");
  return (
    <div className={cx("relative pt-6 pb-6", className)}>
      {current && (
        <div className="absolute top-0 -translate-x-1/2 whitespace-nowrap text-2xs font-semibold text-ink" style={{ left: `${x(current.value)}%` }}>
          {current.label}
        </div>
      )}
      <div className="relative h-2">
        <div className="absolute inset-y-0 rounded-full bg-[rgb(var(--grid))]" style={{ left: 0, right: 0 }} />
        <div className="absolute inset-y-0 rounded-full bg-[rgb(var(--ink-2)/0.26)]" style={{ left: `${x(low)}%`, width: `${x(high) - x(low)}%` }} />
        {others.map((m) => (
          <div key={m.kind} className="absolute -top-1 -bottom-1 w-0.5 -translate-x-1/2 rounded-full bg-[rgb(var(--ink-2))]" style={{ left: `${x(m.value)}%` }} />
        ))}
        {current && (
          <div
            className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[rgb(var(--ink))]"
            style={{ left: `${x(current.value)}%`, boxShadow: "0 0 0 2px rgb(var(--panel))" }}
          />
        )}
      </div>
      <div className="absolute bottom-0 left-0 text-2xs text-muted num" style={{ left: `${Math.max(0, x(low) - 2)}%` }}>
        {lowLabel}
      </div>
      {others
        .filter((m) => x(m.value) - x(low) > 16 && x(high) - x(m.value) > 16)
        .map((m) => (
          <div key={m.kind} className="absolute bottom-0 -translate-x-1/2 whitespace-nowrap text-2xs font-medium text-ink-2" style={{ left: `${x(m.value)}%` }}>
            {m.label}
          </div>
        ))}
      <div className="absolute bottom-0 right-0 text-2xs text-muted num" style={{ right: `${Math.max(0, 100 - x(high) - 2)}%` }}>
        {highLabel}
      </div>
    </div>
  );
}
