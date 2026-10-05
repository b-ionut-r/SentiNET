/**
 * 240° gauge on a diverging bear ↔ neutral ↔ bull scale. Bands are separated
 * by surface gaps; the band holding the value is drawn at full strength and a
 * ringed marker pins the exact value. Center content is HTML (same sans).
 */
import type { ReactNode } from "react";

import { divergingFill, VERDICT_BANDS } from "../../lib/sentiment";

export interface DialBand {
  min: number;
  max: number;
  /** -1..1 position on the diverging ramp. */
  t: number;
  label: string;
}

interface DialProps {
  value: number | null;
  bands: ReadonlyArray<DialBand>;
  size?: number;
  thickness?: number;
  children?: ReactNode;
  ariaLabel: string;
  /** Show 0 / 50 / 100 tick labels. */
  ticks?: boolean;
  /** Label of the band to highlight, when the caller's band rule decides edges (e.g. 55 is Neutral). */
  activeLabel?: string;
}

const SWEEP = 240;

function polar(cx: number, cy: number, r: number, v: number) {
  const phi = ((-SWEEP / 2 + (SWEEP * v) / 100) * Math.PI) / 180;
  return { x: cx + r * Math.sin(phi), y: cy - r * Math.cos(phi) };
}

function arcPath(cx: number, cy: number, r: number, v0: number, v1: number) {
  const a = polar(cx, cy, r, v0);
  const b = polar(cx, cy, r, v1);
  const large = ((v1 - v0) / 100) * SWEEP > 180 ? 1 : 0;
  return `M ${a.x} ${a.y} A ${r} ${r} 0 ${large} 1 ${b.x} ${b.y}`;
}

export function Dial({ value, bands, size = 200, thickness = 9, children, ariaLabel, ticks = true, activeLabel }: DialProps) {
  const r = size / 2 - thickness / 2 - (ticks ? 14 : 4);
  const cx = size / 2;
  const cy = size / 2;
  const height = Math.ceil(cy + r * 0.5 + thickness / 2 + (ticks ? 16 : 6));
  // Surface gap between bands, expressed in value units (2px of arc length).
  const gapV = (2 / (2 * Math.PI * r * (SWEEP / 360))) * 100;
  const v = value == null ? null : Math.max(0, Math.min(100, value));
  const marker = v == null ? null : polar(cx, cy, r, v);

  return (
    <div className="relative mx-auto select-none" style={{ width: size, height }} role="img" aria-label={ariaLabel}>
      <svg width={size} height={height} className="absolute inset-0 overflow-visible" aria-hidden>
        {bands.map((b, i) => {
          const lo = i === 0 ? b.min : b.min + gapV / 2;
          const hi = i === bands.length - 1 ? b.max : b.max - gapV / 2;
          const active = activeLabel != null ? b.label === activeLabel : v != null && v >= b.min && (v < b.max || (i === bands.length - 1 && v <= b.max));
          return (
            <path
              key={b.label}
              d={arcPath(cx, cy, r, lo, hi)}
              fill="none"
              stroke={divergingFill(b.t)}
              strokeWidth={thickness}
              strokeLinecap="butt"
              opacity={v == null ? 0.3 : active ? 1 : 0.42}
            >
              <title>{`${b.label} (${b.min}–${b.max})`}</title>
            </path>
          );
        })}
        {marker && (
          <circle cx={marker.x} cy={marker.y} r={thickness / 2 + 3} fill="rgb(var(--ink))" stroke="rgb(var(--panel))" strokeWidth={2.5} />
        )}
        {ticks &&
          [0, 50, 100].map((t) => {
            const p = polar(cx, cy, r + thickness / 2 + 9, t);
            return (
              <text
                key={t}
                x={p.x}
                y={t === 50 ? p.y + 2 : p.y + 8}
                textAnchor={t === 0 ? "end" : t === 100 ? "start" : "middle"}
                className="fill-[rgb(var(--muted))] text-[10px] num"
              >
                {t}
              </text>
            );
          })}
      </svg>
      <div className="absolute inset-x-0 flex flex-col items-center" style={{ top: cy - r * 0.62 }}>
        {children}
      </div>
    </div>
  );
}

/** SentiNET verdict bands as a continuous dial scale. */
export const VERDICT_DIAL: DialBand[] = VERDICT_BANDS.map((b) => ({
  min: b.min === 0 ? 0 : b.min - 0.5,
  max: b.max === 100 ? 100 : b.max + 0.5,
  t: b.level / 3,
  label: b.label,
}));

/** CNN-style Fear & Greed bands. */
export const FEAR_GREED_DIAL: DialBand[] = [
  { min: 0, max: 25, t: -1, label: "Extreme Fear" },
  { min: 25, max: 45, t: -0.5, label: "Fear" },
  { min: 45, max: 55, t: 0, label: "Neutral" },
  { min: 55, max: 75, t: 0.5, label: "Greed" },
  { min: 75, max: 100, t: 1, label: "Extreme Greed" },
];
