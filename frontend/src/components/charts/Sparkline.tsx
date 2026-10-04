import { useSize } from "../../lib/useSize";
import { extent, linear, linePath } from "./scale";

interface SparklineProps {
  values: number[];
  height?: number;
  color?: string;
  /** Soft 10% wash under the line. */
  area?: boolean;
  /** Reference value drawn as a hairline (e.g. first value, 50). */
  reference?: number | null;
  domain?: [number, number];
  className?: string;
  ariaLabel?: string;
}

/** Compact trend line with an end dot; fills its container's width. */
export function Sparkline({ values, height = 28, color = "rgb(var(--ink-2))", area, reference, domain, className, ariaLabel }: SparklineProps) {
  const [ref, { width }] = useSize();
  const clean = values.filter((v) => Number.isFinite(v));
  const ext = domain ?? extent(reference != null ? [...clean, reference] : clean);
  const pad = 3;
  let body = null;
  if (width > 0 && ext && clean.length >= 2) {
    const [lo, hi] = ext[0] === ext[1] ? [ext[0] - 1, ext[1] + 1] : ext;
    const x = linear(0, clean.length - 1, pad, width - pad);
    const y = linear(lo, hi, height - pad, pad);
    const pts: Array<[number, number]> = clean.map((v, i) => [x(i), y(v)]);
    const d = linePath(pts);
    const last = pts[pts.length - 1];
    body = (
      <svg width={width} height={height} className="block overflow-visible" aria-hidden>
        {reference != null && <line x1={0} x2={width} y1={y(reference)} y2={y(reference)} stroke="rgb(var(--axis))" strokeWidth={1} />}
        {area && <path d={`${d}L${last[0]},${height}L${pts[0][0]},${height}Z`} fill={color} opacity={0.1} />}
        <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={last[0]} cy={last[1]} r={2.5} fill={color} stroke="rgb(var(--panel))" strokeWidth={1.5} />
      </svg>
    );
  }
  return (
    <div ref={ref} className={className} style={{ height }} role={ariaLabel ? "img" : undefined} aria-label={ariaLabel}>
      {body}
    </div>
  );
}
