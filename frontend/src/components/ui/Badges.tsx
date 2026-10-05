/** Chips and badges that carry meaning with text + glyph, never color alone. */
import { OctagonAlert, TriangleAlert, CircleCheck, CircleSlash, Eye, Info, KeyRound, CircleMinus } from "lucide-react";
import type { ReactNode } from "react";

import type { Polarity, SourceStatus } from "../../api/types";
import { cx } from "../../lib/cx";
import { signed } from "../../lib/format";
import { glyph, labelOf, polarityOf, textTone, tintFor, toneVar } from "../../lib/sentiment";

type ChipTone = "neutral" | "accent" | "muted" | "bull" | "bear";

const chipTone: Record<ChipTone, string> = {
  neutral: "bg-raised text-ink-2",
  muted: "bg-transparent text-muted",
  accent: "bg-accent/12 text-accent",
  bull: "bg-bull/12 text-bull-ink",
  bear: "bg-bear/12 text-bear-ink",
};

export function Chip({ children, tone = "neutral", className, title }: { children: ReactNode; tone?: ChipTone; className?: string; title?: string }) {
  return (
    <span
      title={title}
      className={cx("inline-flex h-5 items-center gap-1 whitespace-nowrap rounded px-1.5 text-2xs font-medium", chipTone[tone], className)}
      style={tone === "muted" ? { boxShadow: "0 0 0 1px var(--hairline)" } : undefined}
    >
      {children}
    </span>
  );
}

/** Direction glyph in the polarity color. */
export function Mark({ p, className }: { p: Polarity; className?: string }) {
  return (
    <span aria-hidden className={cx("inline-block shrink-0 text-[9px] leading-none", textTone[p], className)}>
      {glyph(p)}
    </span>
  );
}

/** Signed sentiment score with glyph, tinted by strength: "▲ +0.31". */
export function ScoreChip({ score, label, className, digits = 2 }: { score: number; label?: boolean; className?: string; digits?: number }) {
  const p = polarityOf(score);
  return (
    <span
      className={cx("inline-flex h-5 items-center gap-1 whitespace-nowrap rounded px-1.5 text-2xs font-semibold num", textTone[p], className)}
      style={{ background: tintFor(score) }}
      title={`${labelOf(p)} · score ${signed(score)} (−1 to +1)`}
    >
      <span className="text-[8px] leading-none">{glyph(p)}</span>
      {signed(score, digits)}
      {label && <span className="font-medium opacity-90">{labelOf(p)}</span>}
    </span>
  );
}

/** Polarity label chip ("▲ Bullish"). */
export function PolarityChip({ p, children, className }: { p: Polarity; children?: ReactNode; className?: string }) {
  return (
    <span
      className={cx("inline-flex h-5 items-center gap-1 whitespace-nowrap rounded px-1.5 text-2xs font-semibold", textTone[p], className)}
      style={{ background: toneVar(p, p === "neutral" ? 0.12 : 0.12) }}
    >
      <span className="text-[8px] leading-none">{glyph(p)}</span>
      {children ?? labelOf(p)}
    </span>
  );
}

/** Change badge: "▲ 8", colored by direction (up = bull unless inverted). */
export function Delta({
  value,
  digits = 0,
  suffix = "",
  invert = false,
  plain = false,
  className,
}: {
  value: number | null | undefined;
  digits?: number;
  suffix?: string;
  invert?: boolean;
  /** Direction only, in neutral ink — for changes that aren't sentiment (ranks, attention). */
  plain?: boolean;
  className?: string;
}) {
  if (value == null || !Number.isFinite(value)) return <span className={cx("text-muted", className)}>—</span>;
  const r = Number(value.toFixed(digits));
  const p: Polarity = r === 0 ? "neutral" : (r > 0) !== invert ? "bull" : "bear";
  const g = r === 0 ? "●" : r > 0 ? "▲" : "▼";
  return (
    <span className={cx("inline-flex items-center gap-1 whitespace-nowrap font-medium", plain ? "text-ink-2" : textTone[p], className)}>
      <span className="text-[0.65em] leading-none">{g}</span>
      {Math.abs(r).toFixed(digits)}
      {suffix}
    </span>
  );
}

/* ------------------------------------------------------------------------- */
/* Status & severity (reserved status colors, always icon + label)            */
/* ------------------------------------------------------------------------- */

const statusMeta: Record<SourceStatus, { label: string; color: string; Icon: typeof CircleCheck }> = {
  ok: { label: "OK", color: "text-good", Icon: CircleCheck },
  empty: { label: "No data", color: "text-muted", Icon: CircleMinus },
  error: { label: "Error", color: "text-critical", Icon: OctagonAlert },
  disabled: { label: "Disabled", color: "text-muted", Icon: CircleSlash },
  unconfigured: { label: "Needs key", color: "text-warn", Icon: KeyRound },
};

export function StatusBadge({ status, className }: { status: SourceStatus; className?: string }) {
  const m = statusMeta[status] ?? statusMeta.empty;
  return (
    <span className={cx("inline-flex items-center gap-1 text-2xs font-medium text-ink-2", className)}>
      <m.Icon className={cx("size-3.5", m.color)} aria-hidden />
      {m.label}
    </span>
  );
}

export type Severity = "info" | "watch" | "alert";

export const severityMeta: Record<Severity, { label: string; color: string; ring: string; Icon: typeof Info }> = {
  alert: { label: "Alert", color: "text-critical", ring: "bg-critical", Icon: TriangleAlert },
  watch: { label: "Watch", color: "text-warn", ring: "bg-warn", Icon: Eye },
  info: { label: "Info", color: "text-muted", ring: "bg-neu", Icon: Info },
};

export function SeverityBadge({ severity, className }: { severity: Severity; className?: string }) {
  const m = severityMeta[severity] ?? severityMeta.info;
  return (
    <span className={cx("inline-flex items-center gap-1 text-2xs font-semibold uppercase tracking-wider text-ink", className)}>
      <m.Icon className={cx("size-3.5", m.color)} aria-hidden />
      {m.label}
    </span>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="kbd">{children}</kbd>;
}

/** "New" marker for stories/narratives: a neutral outlined tag (accent stays reserved for interaction). */
export function NewBadge({ className }: { className?: string }) {
  return (
    <span className={cx("inline-block rounded px-1 py-px align-middle text-2xs font-semibold uppercase tracking-wider text-ink", className)} style={{ boxShadow: "inset 0 0 0 1px var(--hairline-strong)" }}>
      New
    </span>
  );
}
