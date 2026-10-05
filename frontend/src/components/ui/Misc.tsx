/** Small shared building blocks: states, segmented control, logo, count-up. */
import { CircleAlert, RotateCw, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { cx } from "../../lib/cx";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("skeleton", className)} aria-hidden />;
}

/** Honest "nothing here" message: says what is missing and why if known. */
export function Empty({ icon, title, children, className }: { icon?: ReactNode; title: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <div className={cx("flex flex-col items-center justify-center gap-1.5 px-4 py-6 text-center", className)}>
      {icon && <div className="mb-1 text-faint [&>svg]:size-5">{icon}</div>}
      <p className="text-sm font-medium text-ink-2">{title}</p>
      {children && <p className="max-w-sm text-xs text-muted">{children}</p>}
    </div>
  );
}

export function ErrorState({ title, message, onRetry, className }: { title: string; message?: string; onRetry?: () => void; className?: string }) {
  return (
    <div className={cx("flex flex-col items-center justify-center gap-2 px-4 py-10 text-center", className)}>
      <CircleAlert className="size-6 text-critical" aria-hidden />
      <p className="text-base font-semibold text-ink">{title}</p>
      {message && <p className="max-w-md text-sm text-ink-2">{message}</p>}
      {onRetry && (
        <button className="btn mt-2" onClick={onRetry}>
          <RotateCw className="size-3.5" /> Try again
        </button>
      )}
    </div>
  );
}

/**
 * Inline, dismissible-by-fix warning for a failed action while older data
 * stays on screen (status colour + icon + words, never colour alone).
 */
export function InlineAlert({ children, action, className }: { children: ReactNode; action?: ReactNode; className?: string }) {
  return (
    <div role="alert" className={cx("flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg bg-serious/10 px-3 py-2 text-sm text-ink", className)} style={{ boxShadow: "inset 0 0 0 1px rgb(var(--serious) / 0.35)" }}>
      <TriangleAlert className="size-4 shrink-0 text-serious" aria-hidden />
      <div className="min-w-0 flex-1">{children}</div>
      {action}
    </div>
  );
}

export interface SegOption<T extends string> {
  value: T;
  label: ReactNode;
  title?: string;
}

/** Compact segmented control (radio group semantics). */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  className,
  size = "sm",
  label,
}: {
  options: ReadonlyArray<SegOption<T>>;
  value: T;
  onChange: (v: T) => void;
  className?: string;
  size?: "xs" | "sm";
  label?: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className={cx("inline-flex items-center gap-0.5 rounded-lg bg-sunken p-0.5", className)} style={{ boxShadow: "0 0 0 1px var(--hairline)" }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            role="radio"
            aria-checked={active}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={cx(
              "rounded-md font-medium transition-colors whitespace-nowrap",
              size === "xs" ? "h-6 px-2 text-2xs" : "h-7 px-2.5 text-xs",
              active ? "bg-raised text-ink shadow-[0_0_0_1px_var(--hairline-strong)]" : "text-muted hover:text-ink-2",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Company logo with a monogram fallback (logos are best-effort upstream). */
export function TickerLogo({ symbol, url, size = 32, className }: { symbol: string; url?: string | null; size?: number; className?: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [url]);
  const mono = symbol.replace(/[^A-Z0-9]/gi, "").slice(0, symbol.length > 4 ? 3 : 2).toUpperCase() || "?";
  const style = { width: size, height: size, boxShadow: "0 0 0 1px var(--hairline)" };
  if (url && !failed) {
    return (
      <img
        src={url}
        alt=""
        width={size}
        height={size}
        loading="lazy"
        onError={() => setFailed(true)}
        className={cx("shrink-0 rounded-lg bg-white object-contain", className)}
        style={style}
      />
    );
  }
  return (
    <span
      aria-hidden
      className={cx("inline-flex shrink-0 items-center justify-center rounded-lg bg-raised font-mono font-semibold text-ink-2", className)}
      style={{ ...style, fontSize: Math.max(9, size * 0.32) }}
    >
      {mono}
    </span>
  );
}

const reduceMotion = () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** Counts up to `value` once on mount (first load only), then tracks it. */
export function CountUp({ value, format = (n) => String(Math.round(n)), duration = 700 }: { value: number; format?: (n: number) => string; duration?: number }) {
  const [shown, setShown] = useState(() => (reduceMotion() ? value : 0));
  const animated = useRef(reduceMotion());

  useEffect(() => {
    if (animated.current) {
      setShown(value);
      return;
    }
    animated.current = true;
    let raf = 0;
    const start = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - k, 3);
      setShown(value * eased);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);

  return <>{format(shown)}</>;
}

/** Thin indeterminate bar for background refreshes. */
export function TopProgress({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-50 h-0.5 overflow-hidden" aria-hidden>
      <div className="h-full w-1/4 animate-scan-sweep bg-accent" />
    </div>
  );
}
