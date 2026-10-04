/**
 * Vertical columns around a zero baseline (diverging when values cross 0).
 * Thin (≤ 24px), 4px rounded data-end, per-column hover/focus tooltip, labels
 * only where asked (selective labelling).
 */
import type { ReactNode } from "react";

import { cx } from "../../lib/cx";
import { toneVar } from "../../lib/sentiment";
import { useTooltip } from "../ui/Tooltip";

export interface ColumnItem {
  key: string;
  label: string;
  value: number | null;
  tip?: ReactNode;
  /** Draw label + full strength; others are muted when any item is emphasized. */
  emphasis?: boolean;
  color?: string;
}

interface ColumnsProps {
  items: ColumnItem[];
  height?: number;
  format?: (v: number) => string;
  /** Which value labels to draw. */
  labels?: "all" | "emphasis" | "none";
  /** Symmetric domain magnitude; defaults to max |value|. */
  max?: number;
  className?: string;
  ariaLabel: string;
}

export function Columns({ items, height = 96, format = (v) => v.toFixed(1), labels = "emphasis", max, className, ariaLabel }: ColumnsProps) {
  const { showAt, hide } = useTooltip();
  const vals = items.map((i) => i.value).filter((v): v is number => v != null && Number.isFinite(v));
  const hasNeg = vals.some((v) => v < 0);
  const hasPos = vals.some((v) => v > 0);
  const m = max ?? Math.max(1e-9, ...vals.map(Math.abs));
  const labelBand = 14;
  const plotH = height - labelBand * (hasNeg && hasPos ? 2 : 1);
  const zeroTop = hasNeg && hasPos ? labelBand + plotH / 2 : hasNeg ? labelBand * 0 : labelBand + plotH;
  const scale = hasNeg && hasPos ? plotH / 2 / m : plotH / m;
  const anyEmph = items.some((i) => i.emphasis);

  return (
    <div className={cx("min-w-0", className)}>
      <div className="relative" style={{ height }} role="img" aria-label={ariaLabel}>
        <div className="absolute inset-x-0 h-px bg-[rgb(var(--axis))]" style={{ top: zeroTop }} />
        <div className="absolute inset-0 flex">
          {items.map((it) => {
            const v = it.value;
            const h = v == null ? 0 : Math.max(2, Math.abs(v) * scale);
            const up = v != null && v >= 0;
            const color = it.color ?? toneVar(v == null ? "neutral" : v > 0 ? "bull" : v < 0 ? "bear" : "neutral");
            const showLabel = v != null && (labels === "all" || (labels === "emphasis" && it.emphasis));
            return (
              <div
                key={it.key}
                tabIndex={0}
                className="group relative flex-1 outline-none"
                onMouseEnter={(e) => it.tip && showAt(it.tip, e.currentTarget)}
                onMouseLeave={hide}
                onFocus={(e) => it.tip && showAt(it.tip, e.currentTarget)}
                onBlur={hide}
              >
                {v != null && (
                  <div
                    className="absolute left-1/2 w-[min(24px,60%)] -translate-x-1/2 transition-[filter] group-hover:brightness-125 group-focus-visible:brightness-125"
                    style={{
                      top: up ? zeroTop - h : zeroTop,
                      height: h,
                      background: color,
                      opacity: anyEmph && !it.emphasis ? 0.45 : 1,
                      borderRadius: up ? "4px 4px 0 0" : "0 0 4px 4px",
                    }}
                  />
                )}
                {v == null && <div className="absolute left-1/2 -translate-x-1/2 text-2xs text-faint" style={{ top: zeroTop - 14 }}>—</div>}
                {showLabel && (
                  <div
                    className="absolute inset-x-0 text-center text-2xs font-semibold text-ink num"
                    style={{ top: up ? zeroTop - h - labelBand : zeroTop + h + 1 }}
                  >
                    {format(v)}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
      <div className="mt-1 flex">
        {items.map((it) => (
          <div key={it.key} className={cx("flex-1 truncate text-center text-2xs num", it.emphasis ? "font-semibold text-ink-2" : "text-muted")}>
            {it.label}
          </div>
        ))}
      </div>
    </div>
  );
}
