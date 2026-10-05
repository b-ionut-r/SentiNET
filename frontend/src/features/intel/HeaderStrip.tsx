/** Identity + quote strip with the page's actions and data freshness. */
import { Check, ChevronDown, Columns3, Download, Link2, RefreshCw, Star } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../../api/client";
import type { Analysis } from "../../api/types";
import { TickerLogo } from "../../components/ui/Misc";
import { Tip } from "../../components/ui/Tooltip";
import { cx } from "../../lib/cx";
import { compact, dayTime, money, ms, pct, price, signed, timeAgo } from "../../lib/format";
import { glyph, polarityOf, textTone } from "../../lib/sentiment";

interface Props {
  a: Analysis;
  watched: boolean;
  onWatch: () => void;
  onRefresh: () => void;
  refreshing: boolean;
  /** Progress events received during a background re-scan. */
  progressCount?: number;
}

export function HeaderStrip({ a, watched, onWatch, onRefresh, refreshing, progressCount = 0 }: Props) {
  const p = a.profile;
  const qt = a.quote;
  const chg = polarityOf(qt?.change_pct ?? 0, 0.005);
  const ok = a.sources.filter((s) => s.status === "ok").length;
  const live = a.sources.filter((s) => s.status !== "unconfigured" && s.status !== "disabled").length;
  const meta = [p?.exchange, p?.quote_type === "EQUITY" ? null : p?.quote_type?.toLowerCase(), p?.sector, p?.industry].filter(Boolean);

  return (
    <header className="flex flex-col gap-4 lg:flex-row lg:items-center">
      <div className="flex min-w-0 flex-1 items-center gap-3.5">
        <TickerLogo symbol={a.ticker} url={p?.logo_url} size={44} className="rounded-xl" />
        <div className="min-w-0">
          <div className="flex items-baseline gap-2.5">
            <h1 className="truncate text-[22px] font-semibold leading-7 tracking-[-0.01em] text-ink">{p?.name ?? a.ticker}</h1>
            <span className="shrink-0 font-mono text-sm font-medium text-ink-2">{a.ticker}</span>
          </div>
          <p className="mt-0.5 truncate text-xs text-muted" title={meta.join(" · ") || undefined}>
            {meta.length ? meta.join(" · ") : "Profile unavailable"}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-[26px] font-semibold leading-none tracking-[-0.02em] text-ink">{price(qt?.price, qt?.currency)}</span>
            {qt?.change_pct != null && (
              <span className={cx("text-sm font-semibold", textTone[chg])}>
                <span className="mr-0.5 text-[10px]">{glyph(chg)}</span>
                {signed(qt.change, 2)} ({pct(qt.change_pct, 2)})
              </span>
            )}
          </div>
          <p className="mt-1.5 text-2xs text-muted" title={qt?.as_of ?? undefined}>
            {qt?.as_of ? `as of ${quoteAge(qt.as_of)}` : qt?.price != null ? "latest quote" : "quote unavailable"}
          </p>
        </div>
        <DayRange low={qt?.day_low ?? null} high={qt?.day_high ?? null} last={qt?.price ?? null} currency={qt?.currency} />
        <Stat label="Mkt cap" value={money(qt?.market_cap, qt?.currency)} />
        {a.technicals && (a.technicals.return_1m != null || a.technicals.return_ytd != null) && (
          <Stat
            label={a.technicals.pct_from_52w_high != null ? `${pct(a.technicals.pct_from_52w_high)} from 52w high` : "returns"}
            value={
              <span className="flex gap-2.5">
                {(
                  [
                    ["1M", a.technicals.return_1m],
                    ["YTD", a.technicals.return_ytd],
                  ] as const
                )
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <span key={k}>
                      <span className="mr-1 text-2xs font-normal text-muted">{k}</span>
                      <span className={textTone[polarityOf(v ?? 0, 0.05)]}>{pct(v)}</span>
                    </span>
                  ))}
              </span>
            }
          />
        )}
        <Stat
          label="Volume"
          value={
            <>
              {compact(qt?.volume)}
              {a.technicals?.volume_ratio != null && <span className="ml-1 text-xs font-normal text-muted">{a.technicals.volume_ratio.toFixed(2)}× avg</span>}
            </>
          }
        />
      </div>

      <div className="flex flex-col items-start gap-1.5 lg:items-end">
        <div className="flex items-center gap-1.5">
          <button className={cx("btn", watched && "text-ink")} onClick={onWatch} aria-pressed={watched} title="Watch (w)">
            <Star className={cx("size-3.5", watched && "fill-current text-warn")} />
            {watched ? "Watching" : "Watch"}
          </button>
          <button className="btn" onClick={onRefresh} disabled={refreshing} title="Refresh (r)" aria-label="Refresh analysis">
            <RefreshCw className={cx("size-3.5", refreshing && "animate-spin")} />
            <span className="hidden sm:inline">Refresh</span>
          </button>
          <ExportMenu ticker={a.ticker} />
          <CopyLink />
          <Link className="btn hidden px-2 sm:inline-flex" to={`/compare?t=${encodeURIComponent(a.ticker)}`} title="Compare">
            <Columns3 className="size-3.5" />
          </Link>
        </div>
        <p className="text-2xs text-muted" aria-live="polite">
          {refreshing ? (
            <span className="text-accent">Re-scanning{progressCount > 0 ? ` · ${progressCount} updates` : "…"}</span>
          ) : (
            <>Updated {timeAgo(a.generated_at)}</>
          )}{" "}
          · {a.cached ? "cached" : `fresh run ${ms(a.elapsed_ms)}`} · {ok}/{live} sources ok · engine {a.engine}
        </p>
      </div>
    </header>
  );
}

/** Recent quotes read as "5m ago"; stale ones (weekend, halted) show the exact session time. */
function quoteAge(asOf: string): string {
  const t = Date.parse(asOf);
  return Number.isFinite(t) && Date.now() - t > 6 * 3600_000 ? dayTime(asOf) : timeAgo(asOf);
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-sm font-semibold leading-none text-ink">{value}</div>
      <p className="mt-1.5 text-2xs text-muted">{label}</p>
    </div>
  );
}

function DayRange({ low, high, last, currency }: { low: number | null; high: number | null; last: number | null; currency?: string | null }) {
  if (low == null || high == null || last == null || high <= low) return null;
  const at = Math.max(0, Math.min(1, (last - low) / (high - low)));
  return (
    <Tip content={`Day range ${price(low, currency)} – ${price(high, currency)} · last ${price(last, currency)}`}>
      <div className="w-36">
        <div className="flex justify-between text-2xs text-ink-2 num">
          <span>{price(low, currency)}</span>
          <span>{price(high, currency)}</span>
        </div>
        <div className="relative mt-1.5 h-1 rounded-full bg-[rgb(var(--grid))]">
          <div className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[rgb(var(--ink))]" style={{ left: `${at * 100}%`, boxShadow: "0 0 0 2px rgb(var(--page))" }} />
        </div>
        <p className="mt-1 text-2xs text-muted">Day range</p>
      </div>
    </Tip>
  );
}

function ExportMenu({ ticker }: { ticker: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div className="relative" ref={ref}>
      <button className="btn" onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open}>
        <Download className="size-3.5" />
        <span className="hidden sm:inline">Export</span>
        <ChevronDown className="size-3 text-muted" />
      </button>
      {open && (
        <div role="menu" className="absolute right-0 top-9 z-30 w-44 rounded-lg bg-raised p-1 shadow-pop animate-fade-in">
          {(["csv", "json"] as const).map((f) => (
            <a
              key={f}
              role="menuitem"
              href={api.exportUrl(ticker, f)}
              download={`${ticker}-signals.${f}`}
              onClick={() => setOpen(false)}
              className="block rounded-md px-2.5 py-1.5 text-sm text-ink-2 hover:bg-panel hover:text-ink"
            >
              Signals as {f.toUpperCase()}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function CopyLink() {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked — nothing to do */
    }
  };
  return (
    <button className="btn px-2" onClick={copy} title="Copy link" aria-label="Copy link">
      {copied ? <Check className="size-3.5 text-good" /> : <Link2 className="size-3.5" />}
    </button>
  );
}
