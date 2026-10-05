/**
 * Watchlist (stored SentiNET scores with change and history) plus alert rules
 * CRUD and the recent alert feed from the background monitor.
 */
import { Bell, BellRing, CircleCheck, CircleMinus, Plus, Star, Trash } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAlertCreate, useAlertDelete, useAlertEvents, useAlerts, useHealth, useWatchlist, useWatchToggle } from "../../api/hooks";
import type { AlertKind, AlertRule, WatchItem } from "../../api/types";
import { Sparkline } from "../../components/charts/Sparkline";
import { Delta, Mark } from "../../components/ui/Badges";
import { Empty, ErrorState, InlineAlert, Skeleton, TickerLogo } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { dayTime, pct, price, timeAgo } from "../../lib/format";
import { polarityOf100, textTone, toneVar, verdictBand } from "../../lib/sentiment";

const KINDS: Record<AlertKind, { label: string; unit?: string; needsThreshold: boolean; placeholder?: string }> = {
  score_above: { label: "SentiNET rises above", needsThreshold: true, placeholder: "65" },
  score_below: { label: "SentiNET falls below", needsThreshold: true, placeholder: "40" },
  score_change: { label: "SentiNET moves by at least", unit: "pts", needsThreshold: true, placeholder: "8" },
  attention_spike: { label: "Attention spikes", needsThreshold: false },
  new_narrative: { label: "A new narrative appears", needsThreshold: false },
  analyst_action: { label: "An analyst rating/target changes", needsThreshold: false },
};

export default function WatchlistPage() {
  useEffect(() => {
    document.title = "Watchlist — SentiNET";
  }, []);
  return (
    <div className="space-y-4">
      <div>
        <h1 className="flex items-center gap-2 text-[22px] font-semibold tracking-[-0.01em] text-ink">
          <Star className="size-5 text-muted" /> Watchlist & alerts
        </h1>
        <p className="mt-0.5 text-sm text-muted">Every analysis stores a SentiNET snapshot; the background monitor refreshes watched tickers and evaluates alert rules.</p>
        <MonitorStatus />
      </div>
      <WatchTable />
      <div className="grid gap-4 lg:grid-cols-12">
        <Rules className="lg:col-span-7" />
        <Events className="lg:col-span-5" />
      </div>
    </div>
  );
}

/** What will actually happen in the background, read from /api/health — never assumed. */
function MonitorStatus() {
  const h = useHealth();
  const f = h.data?.features;
  if (!f) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        {f.monitor ? <CircleCheck className="size-3.5 text-good" aria-hidden /> : <CircleMinus className="size-3.5 text-muted" aria-hidden />}
        {f.monitor ? "Monitor on — refreshes watched tickers on a schedule" : "Monitor off — scores update only when you open a ticker, and rules don't fire"}
      </span>
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        {f.alert_webhook ? <CircleCheck className="size-3.5 text-good" aria-hidden /> : <CircleMinus className="size-3.5 text-muted" aria-hidden />}
        {f.alert_webhook ? "Webhook delivery on" : "No webhook — alerts appear here only (set ALERT_WEBHOOK_URL)"}
      </span>
    </div>
  );
}

function WatchTable() {
  const wl = useWatchlist();
  const toggle = useWatchToggle();
  const navigate = useNavigate();
  const [t, setT] = useState("");
  const add = (e: React.FormEvent) => {
    e.preventDefault();
    const sym = t.trim().toUpperCase().replace(/^\$/, "");
    // Keep what was typed until the server accepts it, so a rejected symbol can be corrected.
    if (sym) toggle.mutate({ ticker: sym, watched: false }, { onSuccess: () => setT("") });
  };
  const remove = (ticker: string) => toggle.mutate({ ticker, watched: true });
  const failed = toggle.error && toggle.variables;
  return (
    <Panel
      title="Watching"
      subtitle={wl.data ? `${wl.data.length} tickers` : undefined}
      flush
      actions={
        <form onSubmit={add} className="flex items-center gap-1.5">
          <input
            value={t}
            onChange={(e) => {
              setT(e.target.value);
              if (toggle.error) toggle.reset();
            }}
            placeholder="Add ticker"
            className="field h-7 w-32 text-xs uppercase placeholder:normal-case"
            aria-label="Add ticker"
            aria-invalid={!!failed && !failed.watched}
          />
          <button className="btn h-7 px-2" type="submit" aria-label="Add ticker" disabled={toggle.isPending}>
            <Plus className="size-3.5" />
          </button>
        </form>
      }
    >
      {failed && (
        <div className="px-4 pb-3">
          <InlineAlert action={<button className="btn h-7" onClick={() => toggle.reset()}>Dismiss</button>}>
            <span className="font-medium">
              Couldn't {failed.watched ? "remove" : "add"} {failed.ticker}
            </span>
            <span className="text-ink-2"> — {toggle.error.message}</span>
          </InlineAlert>
        </div>
      )}
      {wl.isPending ? (
        <div className="space-y-2 p-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : wl.error ? (
        <ErrorState title="Watchlist unavailable" message={wl.error.message} onRetry={() => void wl.refetch()} />
      ) : !wl.data?.length ? (
        <Empty title="Your watchlist is empty" className="hairline-t">
          Add a ticker above, or press <kbd className="kbd">w</kbd> on any ticker page.
        </Empty>
      ) : (
        <>
        <ul className="divide-hair hairline-t sm:hidden">
          {wl.data.map((w) => (
            <WatchCard key={w.ticker} w={w} onOpen={() => navigate(`/t/${encodeURIComponent(w.ticker)}`)} onRemove={() => remove(w.ticker)} />
          ))}
        </ul>
        <div className="hidden overflow-x-auto sm:block">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="text-2xs text-muted hairline-t hairline-b">
                <th className="py-2 pl-4 text-left font-normal">Ticker</th>
                <th className="py-2 text-left font-normal">SentiNET</th>
                <th className="py-2 text-left font-normal">Label</th>
                <th className="w-40 py-2 text-left font-normal">Score history</th>
                <th className="py-2 text-right font-normal">Price</th>
                <th className="py-2 text-right font-normal">Updated</th>
                <th className="w-12 py-2 pr-4" />
              </tr>
            </thead>
            <tbody className="divide-hair">
              {wl.data.map((w) => (
                <WatchRow key={w.ticker} w={w} onOpen={() => navigate(`/t/${encodeURIComponent(w.ticker)}`)} onRemove={() => remove(w.ticker)} />
              ))}
            </tbody>
          </table>
        </div>
        </>
      )}
    </Panel>
  );
}

/** Compact phone layout of a watchlist row. */
function WatchCard({ w, onOpen, onRemove }: { w: WatchItem; onOpen: () => void; onRemove: () => void }) {
  const s = w.last?.sentinel_score ?? null;
  const prev = w.previous?.sentinel_score ?? null;
  const p = polarityOf100(s);
  return (
    <li className="flex items-center pr-2">
      <button onClick={onOpen} className="flex min-w-0 flex-1 items-center gap-3 py-3 pl-4 pr-2 text-left">
        <TickerLogo symbol={w.ticker} url={`https://logos.stocktwits-cdn.com/${w.ticker}.png`} size={32} />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-sm font-semibold text-ink">{w.ticker}</span>
            {s != null && <span className={cx("text-sm font-semibold", textTone[p])}>{s}</span>}
            {s != null && prev != null && <Delta value={s - prev} className="text-2xs" />}
          </div>
          <div className="truncate text-2xs text-muted">{s != null ? `${verdictBand(s).label} · ${price(w.last?.price)} · ${timeAgo(w.last?.at)}` : "not analyzed yet"}</div>
        </div>
        <div className="w-20">{w.spark.length > 1 && <Sparkline values={w.spark} height={26} domain={[0, 100]} reference={50} color={toneVar(p)} />}</div>
      </button>
      <button className="grid size-9 shrink-0 place-items-center rounded-lg text-muted hover:bg-raised hover:text-critical" onClick={onRemove} aria-label={`Remove ${w.ticker}`}>
        <Trash className="size-4" />
      </button>
    </li>
  );
}

function WatchRow({ w, onOpen, onRemove }: { w: WatchItem; onOpen: () => void; onRemove: () => void }) {
  const s = w.last?.sentinel_score ?? null;
  const prev = w.previous?.sentinel_score ?? null;
  const p = polarityOf100(s);
  const priceChg = w.last?.price != null && w.previous?.price ? (w.last.price / w.previous.price - 1) * 100 : null;
  return (
    <tr className="group cursor-pointer hover:bg-raised/60" onClick={onOpen}>
      <td className="py-2.5 pl-4">
        <div className="flex items-center gap-2.5">
          <TickerLogo symbol={w.ticker} url={`https://logos.stocktwits-cdn.com/${w.ticker}.png`} size={28} />
          <div className="min-w-0">
            <Link to={`/t/${encodeURIComponent(w.ticker)}`} className="font-mono text-sm font-semibold text-ink group-hover:underline" onClick={(e) => e.stopPropagation()}>
              {w.ticker}
            </Link>
            <div className="max-w-[220px] truncate text-2xs text-muted">{w.name ?? ""}</div>
          </div>
        </div>
      </td>
      <td className="py-2.5">
        {s == null ? (
          <span className="text-xs text-muted">not analyzed yet</span>
        ) : (
          <span className="flex items-baseline gap-2">
            <span className={cx("text-lg font-semibold", textTone[p])}>{s}</span>
            {prev != null && <Delta value={s - prev} className="text-xs" />}
          </span>
        )}
      </td>
      <td className="py-2.5 text-xs">
        {s != null && (
          <span className={cx("inline-flex items-center gap-1.5 font-medium", textTone[p])}>
            <Mark p={p} />
            {verdictBand(s).label}
          </span>
        )}
      </td>
      <td className="py-2.5 pr-4">{w.spark.length > 1 ? <Sparkline values={w.spark} height={28} domain={[0, 100]} reference={50} color={toneVar(p)} ariaLabel={`${w.ticker} stored scores`} /> : <span className="text-2xs text-muted">—</span>}</td>
      <td className="py-2.5 text-right text-xs num">
        <div className="text-ink">{price(w.last?.price)}</div>
        {priceChg != null && <div className={cx("text-2xs", textTone[priceChg > 0 ? "bull" : priceChg < 0 ? "bear" : "neutral"])}>{pct(priceChg)}</div>}
      </td>
      <td className="py-2.5 text-right text-2xs text-muted">{w.last ? timeAgo(w.last.at) : `added ${timeAgo(w.added_at)}`}</td>
      <td className="py-2.5 pr-4 text-right">
        <button
          className="rounded p-1.5 text-muted opacity-0 transition-opacity hover:bg-panel hover:text-critical focus-visible:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          aria-label={`Remove ${w.ticker}`}
        >
          <Trash className="size-3.5" />
        </button>
      </td>
    </tr>
  );
}

function Rules({ className }: { className?: string }) {
  const rules = useAlerts();
  const create = useAlertCreate();
  const del = useAlertDelete();
  const [ticker, setTicker] = useState("");
  const [kind, setKind] = useState<AlertKind>("score_change");
  const [threshold, setThreshold] = useState("");
  const meta = KINDS[kind];
  const valid = ticker.trim().length > 0 && (!meta.needsThreshold || (threshold.trim() !== "" && Number.isFinite(Number(threshold))));

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    create.mutate(
      { ticker: ticker.trim().toUpperCase().replace(/^\$/, ""), kind, threshold: meta.needsThreshold ? Number(threshold) : null },
      { onSuccess: () => (setTicker(""), setThreshold("")) },
    );
  };

  return (
    <Panel title="Alert rules" icon={<Bell />} subtitle="Evaluated by the monitor after each refresh; optional webhook delivery" className={className} flush>
      <form onSubmit={submit} className="flex flex-wrap items-center gap-2 px-4 pb-3">
        <input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Ticker" className="field w-24 uppercase placeholder:normal-case" aria-label="Alert ticker" />
        <select value={kind} onChange={(e) => setKind(e.target.value as AlertKind)} className="field order-first w-full min-w-0 pr-7 sm:order-none sm:w-auto sm:flex-1" aria-label="Alert condition">
          {(Object.keys(KINDS) as AlertKind[]).map((k) => (
            <option key={k} value={k}>
              {KINDS[k].label}
            </option>
          ))}
        </select>
        {meta.needsThreshold && (
          <input value={threshold} onChange={(e) => setThreshold(e.target.value)} placeholder={meta.placeholder} inputMode="decimal" className="field w-20 num" aria-label="Threshold" />
        )}
        <button className="btn btn-primary" type="submit" disabled={!valid || create.isPending}>
          Add rule
        </button>
      </form>
      {create.error && <p className="px-4 pb-2 text-xs text-critical">{create.error.message}</p>}
      {rules.isPending ? (
        <div className="p-4">
          <Skeleton className="h-24 w-full" />
        </div>
      ) : rules.error ? (
        <Empty title="Alert rules unavailable" className="hairline-t">{rules.error.message}</Empty>
      ) : !rules.data?.length ? (
        <Empty title="No alert rules" className="hairline-t">Create one above — e.g. “NVDA · SentiNET moves by at least 8”.</Empty>
      ) : (
        <ul className="divide-hair hairline-t">
          {rules.data.map((r) => (
            <RuleRow key={r.id} r={r} onDelete={() => del.mutate(r.id)} />
          ))}
        </ul>
      )}
    </Panel>
  );
}

function RuleRow({ r, onDelete }: { r: AlertRule; onDelete: () => void }) {
  const meta = KINDS[r.kind];
  return (
    <li className="group flex items-center gap-3 px-4 py-2.5">
      <Link to={`/t/${encodeURIComponent(r.ticker)}`} className="w-16 shrink-0 font-mono text-sm font-semibold text-ink hover:underline">
        {r.ticker}
      </Link>
      <span className="min-w-0 flex-1 text-sm text-ink-2">
        {meta?.label ?? r.kind}
        {r.threshold != null && (
          <span className="ml-1 font-semibold text-ink num">
            {r.kind === "score_change" ? "±" : ""}
            {r.threshold}
            {meta?.unit ? ` ${meta.unit}` : ""}
          </span>
        )}
        {!r.enabled && <span className="ml-2 text-2xs text-muted">paused</span>}
      </span>
      <span className="hidden text-2xs text-muted sm:block">{r.last_triggered_at ? `fired ${timeAgo(r.last_triggered_at)}` : "never fired"}</span>
      <button className="rounded p-1.5 text-muted hover:bg-raised hover:text-critical" onClick={onDelete} aria-label={`Delete alert for ${r.ticker}`}>
        <Trash className="size-3.5" />
      </button>
    </li>
  );
}

function Events({ className }: { className?: string }) {
  const ev = useAlertEvents();
  return (
    <Panel title="Recent alerts" icon={<BellRing />} subtitle="Newest first" className={className} flush>
      {ev.isPending ? (
        <div className="p-4">
          <Skeleton className="h-24 w-full" />
        </div>
      ) : ev.error ? (
        <Empty title="Alert feed unavailable" className="hairline-t">{ev.error.message}</Empty>
      ) : !ev.data?.length ? (
        <Empty title="No alerts yet" className="hairline-t">They'll appear here when a rule fires.</Empty>
      ) : (
        <ul className="divide-hair hairline-t">
          {ev.data.map((e) => (
            <li key={e.id} className="px-4 py-2.5">
              <div className="flex items-center justify-between gap-2">
                <Link to={`/t/${encodeURIComponent(e.ticker)}`} className="font-mono text-xs font-semibold text-ink hover:underline">
                  {e.ticker}
                </Link>
                <span className="text-2xs text-muted" title={e.at}>
                  {dayTime(e.at)} · {e.delivered ? "delivered" : "not delivered"}
                </span>
              </div>
              <p className="mt-0.5 text-sm font-medium text-ink">{e.title}</p>
              <p className="text-xs text-ink-2">{e.detail}</p>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
