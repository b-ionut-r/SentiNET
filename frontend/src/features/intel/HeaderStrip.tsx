/** Identity + quote strip with the page's actions and data freshness. */
import { Check, ChevronDown, Columns3, Download, Link2, RefreshCw, Star, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../../api/client";
import type { Analysis } from "../../api/types";
import { TickerLogo } from "../../components/ui/Misc";
import { Tip } from "../../components/ui/Tooltip";
import { cx } from "../../lib/cx";
import { compact, dayTime, money, ms, pct, price, priceMove, timeAgo } from "../../lib/format";
import { glyph, polarityOf, textTone } from "../../lib/sentiment";
import { useNow } from "../../lib/useNow";

interface Props {
  a: Analysis;
  watched: boolean;
  onWatch: () => void;
  onRefresh: () => void;
  refreshing: boolean;
  /** Progress events received during a background re-scan. */
  progressCount?: number;
  /** The last refresh failed, so this run is older than requested. */
  stale?: boolean;
}

export function HeaderStrip({ a, watched, onWatch, onRefresh, refreshing, progressCount = 0, stale = false }: Props) {
  const now = useNow(30_000);
  const p = a.profile;
  const qt = a.quote;
  const chg = polarityOf(qt?.change_pct ?? 0, 0.005);
  const ok = a.sources.filter((s) => s.status === "ok").length;
  const live = a.sources.filter((s) => s.status !== "unconfigured" && s.status !== "disabled").length;
  // Title uses the brand name; the legal name (when different) leads the subline.
  const title = p?.short_name || p?.name || a.ticker;
  const legal = p?.name && p.name !== title ? p.name : null;
  const meta = [legal, p?.exchange, p?.quote_type === "EQUITY" ? null : p?.quote_type?.toLowerCase(), p?.sector, p?.industry].filter(Boolean);

  return (
    <header className="intel-head">
      <div className="flex min-w-0 items-center gap-3 [grid-area:id] sm:gap-3.5">
        <TickerLogo symbol={a.ticker} url={p?.logo_url} size={44} className="rounded-xl max-sm:!size-9" />
        <div className="min-w-0">
          <div className="flex items-baseline gap-2">
            <h1 className="truncate text-lg font-semibold leading-6 tracking-[-0.01em] text-ink sm:text-[22px] sm:leading-7" title={p?.name ?? undefined}>{title}</h1>
            <span className="shrink-0 font-mono text-xs font-medium text-ink-2 sm:text-sm">{a.ticker}</span>
          </div>
          <p className="mt-0.5 truncate text-xs text-muted" title={meta.join(" · ") || undefined}>
            {meta.length ? meta.join(" · ") : "Profile unavailable"}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 [grid-area:quote] lg:block">
        <div className="flex items-baseline gap-2">
          <span className="text-[22px] font-semibold leading-none tracking-[-0.02em] text-ink sm:text-[26px]">{price(qt?.price, qt?.currency)}</span>
          {qt?.change_pct != null && (
            <span className={cx("text-sm font-semibold", textTone[chg])}>
              <span className="mr-0.5 text-[10px]">{glyph(chg)}</span>
              {qt.change != null && `${priceMove(qt.change, qt.currency, qt.price)} `}({pct(qt.change_pct, 2)})
            </span>
          )}
        </div>
        <p className="text-2xs text-muted lg:mt-1.5" title={qt?.as_of ?? undefined}>
          {qt?.as_of ? `as of ${quoteAge(qt.as_of)}` : qt?.price != null ? "latest quote" : "quote unavailable"}
        </p>
      </div>

      {/* Phones: one swipeable line of secondary stats so the verdict lands on the first screen. */}
      <div className="no-scrollbar scroll-fade-x -mx-4 flex items-end gap-x-6 gap-y-3 overflow-x-auto px-4 [grid-area:stats] sm:mx-0 sm:flex-wrap sm:overflow-visible sm:px-0 lg:gap-x-8">
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

      <div className="flex items-center gap-1.5 justify-self-end [grid-area:actions]">
        <button className={cx("btn max-sm:px-2", watched && "text-ink")} onClick={onWatch} aria-pressed={watched} aria-label={watched ? "Watching — remove from watchlist" : "Add to watchlist"} title="Watch (w)">
          <Star className={cx("size-3.5", watched && "fill-current text-ink")} />
          <span className="hidden sm:inline">{watched ? "Watching" : "Watch"}</span>
        </button>
        <button className="btn max-sm:px-2" onClick={onRefresh} disabled={refreshing} title="Refresh (r)" aria-label="Refresh analysis">
          <RefreshCw className={cx("size-3.5", refreshing && "animate-spin")} />
          <span className="hidden sm:inline">Refresh</span>
        </button>
        <ExportMenu ticker={a.ticker} />
        <CopyLink />
        <Link className="btn hidden px-2 sm:inline-flex" to={`/compare?t=${encodeURIComponent(a.ticker)}`} title="Compare" aria-label={`Compare ${a.ticker}`}>
          <Columns3 className="size-3.5" />
        </Link>
      </div>
      <p className="text-2xs text-muted [grid-area:fresh] lg:justify-self-end lg:text-right" aria-live="polite">
        {refreshing ? (
          <span className="text-accent">Re-scanning{progressCount > 0 ? ` · ${progressCount} updates` : "…"}</span>
        ) : stale ? (
          <span className="inline-flex items-center gap-1 font-medium text-serious">
            <TriangleAlert className="size-3" aria-hidden />
            Stale · updated {timeAgo(a.generated_at, now)}
          </span>
        ) : (
          <>Updated {timeAgo(a.generated_at, now)}</>
        )}{" "}
        · {a.cached ? "cached" : `fresh run ${ms(a.elapsed_ms)}`} · {ok}/{live} sources ok<span className="max-sm:hidden"> · engine {a.engine}</span>
      </p>
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
      <button className="btn max-sm:px-2" onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open} aria-label="Export signals">
        <Download className="size-3.5" />
        <span className="hidden sm:inline">Export</span>
        <ChevronDown className="size-3 text-muted max-sm:hidden" />
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
