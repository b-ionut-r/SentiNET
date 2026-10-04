/** Smart money: analyst consensus & revisions, insider flow, earnings track record. */
import { Briefcase, CalendarClock, UserRound } from "lucide-react";

import type { AnalystAction, Analysis, InsiderTxn, RatingCounts } from "../../api/types";
import { RangeBar, SegmentLegend, StackedBar, type Segment } from "../../components/charts/Bars";
import { Columns } from "../../components/charts/Columns";
import { Chip, Mark } from "../../components/ui/Badges";
import { Empty } from "../../components/ui/Misc";
import { Panel, SubHead } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { compact, countdown, int, money, pct, price, shortDate, signed } from "../../lib/format";
import { divergingFill, textTone } from "../../lib/sentiment";

/* ------------------------------------------------------------------------- */
/* Analysts                                                                    */
/* ------------------------------------------------------------------------- */

const BUCKETS: Array<{ key: keyof Omit<RatingCounts, "period">; label: string; t: number }> = [
  { key: "strong_buy", label: "Strong buy", t: 1 },
  { key: "buy", label: "Buy", t: 0.55 },
  { key: "hold", label: "Hold", t: 0 },
  { key: "sell", label: "Sell", t: -0.55 },
  { key: "strong_sell", label: "Strong sell", t: -1 },
];

const segmentsOf = (c: RatingCounts): Segment[] => BUCKETS.map((b) => ({ key: b.key, label: b.label, value: c[b.key], color: divergingFill(b.t) }));

const CONSENSUS: Record<string, string> = { strong_buy: "Strong Buy", buy: "Buy", hold: "Hold", sell: "Sell", strong_sell: "Strong Sell", underperform: "Underperform", outperform: "Outperform" };

export function AnalystsPanel({ a }: { a: Analysis }) {
  const v = a.analysts;
  const cur = a.quote?.price ?? null;
  const ccy = a.quote?.currency;
  if (!v || (v.total === 0 && v.actions.length === 0 && v.target_mean == null)) {
    return (
      <Panel title="Analysts" icon={<Briefcase />}>
        <Empty title="No analyst coverage">{a.profile?.quote_type === "CRYPTOCURRENCY" ? "Crypto assets aren't covered by sell-side analysts." : "No ratings, targets or rating changes were found."}</Empty>
      </Panel>
    );
  }
  const consensus = v.consensus ? CONSENSUS[v.consensus] ?? v.consensus : null;
  const up = v.upside_pct;
  return (
    <Panel title="Analysts" icon={<Briefcase />} subtitle={`${v.total} analysts${v.counts ? ` · ${v.counts.period === "0m" ? "this month" : v.counts.period}` : ""}`}>
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="text-[22px] font-semibold leading-none tracking-[-0.01em] text-ink">{consensus ?? "—"}</div>
          <div className="mt-1.5 text-2xs text-muted">
            consensus{v.mean_rating != null && <> · mean {v.mean_rating.toFixed(2)} (1 = strong buy, 5 = strong sell)</>}
          </div>
        </div>
        {up != null && (
          <div className="text-right">
            <div className={cx("text-lg font-semibold leading-none", textTone[up > 2 ? "bull" : up < -2 ? "bear" : "neutral"])}>{pct(up)}</div>
            <div className="mt-1.5 text-2xs text-muted">to mean target</div>
          </div>
        )}
      </div>

      {v.counts && (
        <div className="mt-4">
          <StackedBar segments={segmentsOf(v.counts)} height={10} />
          <SegmentLegend segments={segmentsOf(v.counts)} className="mt-2" />
        </div>
      )}

      {v.trend.length > 1 && (
        <div className="mt-4">
          <SubHead right="share rating Buy or better">Rating drift</SubHead>
          <ul className="space-y-1.5">
            {v.trend.slice(0, 4).map((t) => {
              const tot = BUCKETS.reduce((s, b) => s + t[b.key], 0);
              const buyShare = tot ? (t.strong_buy + t.buy) / tot : 0;
              return (
                <li key={t.period} className="grid grid-cols-[38px_minmax(0,1fr)_36px] items-center gap-2">
                  <span className="text-2xs text-muted num">{t.period === "0m" ? "now" : t.period.replace("-", "−")}</span>
                  <StackedBar segments={segmentsOf(t)} height={6} />
                  <span className="text-right text-2xs font-medium text-ink-2 num">{Math.round(buyShare * 100)}%</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {v.target_low != null && v.target_high != null && cur != null && (
        <div className="mt-4">
          <SubHead right={v.target_median != null ? `median ${price(v.target_median, ccy)}` : undefined}>Price targets</SubHead>
          <RangeBar
            low={v.target_low}
            high={v.target_high}
            lowLabel={`low ${price(v.target_low, ccy)}`}
            highLabel={`high ${price(v.target_high, ccy)}`}
            markers={[
              { value: cur, label: `now ${price(cur, ccy)}`, kind: "current" },
              ...(v.target_mean != null ? [{ value: v.target_mean, label: `mean ${price(v.target_mean, ccy)}`, kind: "mean" as const }] : []),
            ]}
          />
        </div>
      )}

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <Revision label="Rating changes · 90d" up={v.upgrades_90d} down={v.downgrades_90d} upLabel="up" downLabel="down" />
        <Revision label="Target changes · 30d" up={v.pt_raises_30d} down={v.pt_cuts_30d} upLabel="raised" downLabel="cut" />
      </div>

      {v.actions.length > 0 && (
        <div className="mt-4">
          <SubHead>Recent actions</SubHead>
          <ul className="divide-hair">
            {v.actions.slice(0, 6).map((x, i) => (
              <ActionRow key={i} x={x} ccy={ccy} />
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}

function Revision({ label, up, down, upLabel, downLabel }: { label: string; up: number; down: number; upLabel: string; downLabel: string }) {
  return (
    <div className="rounded-md bg-sunken px-2.5 py-2">
      <div className="text-2xs text-muted">{label}</div>
      <div className="mt-1 flex items-center gap-3 font-medium">
        <span className={up > 0 ? "text-bull" : "text-muted"}>
          <span className="text-[8px]">▲</span> {up} {upLabel}
        </span>
        <span className={down > 0 ? "text-bear" : "text-muted"}>
          <span className="text-[8px]">▼</span> {down} {downLabel}
        </span>
      </div>
    </div>
  );
}

function actionPolarity(x: AnalystAction): "bull" | "bear" | "neutral" {
  if (x.action === "up") return "bull";
  if (x.action === "down") return "bear";
  if (x.price_target != null && x.prior_target != null && x.price_target !== x.prior_target) return x.price_target > x.prior_target ? "bull" : "bear";
  return "neutral";
}

const ACTION_LABEL: Record<AnalystAction["action"], string> = { up: "Upgrade", down: "Downgrade", init: "Initiates", main: "Maintains", reit: "Reiterates", other: "Update" };

function ActionRow({ x, ccy }: { x: AnalystAction; ccy?: string | null }) {
  const p = actionPolarity(x);
  const grade = x.from_grade && x.to_grade && x.from_grade !== x.to_grade ? `${x.from_grade} → ${x.to_grade}` : x.to_grade ?? "";
  const pt =
    x.price_target != null
      ? x.prior_target != null && x.prior_target !== x.price_target
        ? `${price(x.prior_target, ccy)} → ${price(x.price_target, ccy)}`
        : price(x.price_target, ccy)
      : null;
  return (
    <li className="grid grid-cols-[44px_minmax(0,1fr)_auto] items-start gap-2 py-1.5 text-xs">
      <span className="pt-px text-2xs text-muted num">{shortDate(x.date)}</span>
      <span className="min-w-0">
        <span className="flex items-center gap-1.5">
          <Mark p={p} />
          <span className="truncate font-medium text-ink">{x.firm}</span>
        </span>
        <span className="block truncate text-2xs text-muted">
          {ACTION_LABEL[x.action]}
          {grade && ` · ${grade}`}
        </span>
      </span>
      {pt && <span className={cx("pt-px text-right text-2xs font-medium num", textTone[p])}>{pt}</span>}
    </li>
  );
}

/* ------------------------------------------------------------------------- */
/* Insiders                                                                    */
/* ------------------------------------------------------------------------- */

const KIND_TONE: Record<InsiderTxn["kind"], "bull" | "bear" | "neutral" | "accent" | "muted"> = {
  buy: "bull",
  sell: "bear",
  award: "muted",
  exercise: "muted",
  gift: "muted",
  other: "muted",
};

export function InsidersPanel({ a }: { a: Analysis }) {
  const v = a.insiders;
  if (!v) {
    return (
      <Panel title="Insiders" icon={<UserRound />}>
        <Empty title="No insider data">{a.profile?.quote_type === "CRYPTOCURRENCY" ? "Crypto assets have no insider filings." : "No Form 4 transactions were found."}</Empty>
      </Panel>
    );
  }
  const maxV = Math.max(v.buy_value, v.sell_value, 1);
  const net = v.net_value;
  const p = net > 0 ? "bull" : net < 0 ? "bear" : "neutral";
  const ccy = a.quote?.currency;
  const mcap = a.quote?.market_cap;
  return (
    <Panel title="Insiders" icon={<UserRound />} subtitle={`Open-market trades · last ${v.window_days} days`}>
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className={cx("text-[22px] font-semibold leading-none tracking-[-0.01em]", textTone[p])}>
            {net === 0 ? money(0, ccy) : `${net > 0 ? "+" : "−"}${money(Math.abs(net), ccy)}`}
          </div>
          <div className="mt-1.5 text-2xs text-muted">net insider flow</div>
        </div>
        {v.ratio != null && (
          <div className="text-right">
            <div className="text-lg font-semibold leading-none text-ink">{signed(v.ratio)}</div>
            <div className="mt-1.5 text-2xs text-muted">buy/sell balance</div>
          </div>
        )}
      </div>
      <div className="mt-4 space-y-2.5">
        {(
          [
            ["Buys", v.buys, v.buy_value, "bull"],
            ["Sells", v.sells, v.sell_value, "bear"],
          ] as const
        ).map(([l, n, val, tone]) => (
          <div key={l} className="grid grid-cols-[44px_minmax(0,1fr)_88px] items-center gap-2.5 text-xs">
            <span className="text-ink-2">{l}</span>
            <div className="h-2 rounded-full bg-[rgb(var(--grid))]">
              <div className={cx("h-full rounded-full", tone === "bull" ? "bg-bull" : "bg-bear")} style={{ width: `${val > 0 ? Math.max(2, (val / maxV) * 100) : 0}%` }} />
            </div>
            <span className="text-right num">
              <span className="font-semibold text-ink">{money(val, ccy)}</span> <span className="text-muted">· {n}</span>
            </span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-2xs leading-4 text-muted">
        {v.buys === 0 && v.sells > 0
          ? `Selling only.${mcap ? ` Total sales equal ${((v.sell_value / mcap) * 100).toPrecision(1)}% of market cap — ` : " "}scheduled sales by large-cap executives are usually routine; clustered buying is the signal that matters.`
          : v.buys > 0
            ? "Open-market buying is the strongest insider signal — executives buy for one reason."
            : "Awards, option exercises and gifts are shown but don't count toward flow."}
      </p>
      {v.transactions.length > 0 && (
        <div className="mt-3">
          <SubHead>Recent filings</SubHead>
          <ul className="divide-hair">
            {v.transactions.slice(0, 6).map((t, i) => (
              <li key={i} className="grid grid-cols-[44px_minmax(0,1fr)_auto] items-start gap-2 py-1.5 text-xs">
                <span className="pt-px text-2xs text-muted num">{shortDate(t.date)}</span>
                <span className="min-w-0">
                  <span className="block truncate font-medium text-ink">{t.insider}</span>
                  <span className="block truncate text-2xs text-muted">{t.position ?? "—"}</span>
                </span>
                <span className="flex flex-col items-end gap-0.5">
                  <Chip tone={KIND_TONE[t.kind]} className="capitalize">
                    {t.kind}
                  </Chip>
                  <span className="text-2xs text-muted num">
                    {t.shares != null ? `${compact(t.shares)} sh` : ""}
                    {t.value != null ? ` · ${money(t.value, ccy)}` : ""}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------------- */
/* Earnings                                                                    */
/* ------------------------------------------------------------------------- */

export function EarningsPanel({ a }: { a: Analysis }) {
  const e = a.earnings;
  if (!e || (!e.next_date && e.history.length === 0)) {
    return (
      <Panel title="Earnings" icon={<CalendarClock />}>
        <Empty title="No earnings calendar">{a.profile?.quote_type === "CRYPTOCURRENCY" || a.profile?.quote_type === "ETF" ? "This asset doesn't report earnings." : "No upcoming date or reported history found."}</Empty>
      </Panel>
    );
  }
  const hist = [...e.history].slice(0, 8).reverse();
  const beats = e.history.filter((h) => (h.surprise_pct ?? 0) > 0).length;
  const avgSurprise = e.history.length ? e.history.reduce((s, h) => s + (h.surprise_pct ?? 0), 0) / e.history.length : null;
  const ccy = a.quote?.currency;
  return (
    <Panel title="Earnings" icon={<CalendarClock />} subtitle="Next report and track record">
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="text-[22px] font-semibold leading-none tracking-[-0.01em] text-ink">{e.next_date ? shortDate(e.next_date) : "Not scheduled"}</div>
          <div className="mt-1.5 text-2xs text-muted">{e.next_date ? `next report · ${countdown(e.days_until)}` : "no confirmed date"}</div>
        </div>
        {e.beat_rate != null && (
          <div className="text-right">
            <div className="text-lg font-semibold leading-none text-ink">
              {beats}/{e.history.length}
            </div>
            <div className="mt-1.5 text-2xs text-muted">beats</div>
          </div>
        )}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-md bg-sunken px-2.5 py-2">
          <div className="text-2xs text-muted">EPS estimate</div>
          <div className="mt-1 font-semibold text-ink num">{e.eps_estimate != null ? price(e.eps_estimate, ccy) : "—"}</div>
          {e.eps_low != null && e.eps_high != null && (
            <div className="text-2xs text-muted num">
              {price(e.eps_low, ccy)} – {price(e.eps_high, ccy)}
            </div>
          )}
        </div>
        <div className="rounded-md bg-sunken px-2.5 py-2">
          <div className="text-2xs text-muted">Revenue estimate</div>
          <div className="mt-1 font-semibold text-ink num">{money(e.revenue_estimate, ccy)}</div>
          {avgSurprise != null && <div className="text-2xs text-muted">avg EPS surprise {pct(avgSurprise)}</div>}
        </div>
      </div>
      {hist.length > 0 && (
        <div className="mt-4">
          <SubHead right="EPS surprise vs. estimate">Last {hist.length} quarters</SubHead>
          <Columns
            items={hist.map((h, i) => ({
              key: h.date,
              label: quarterLabel(h.date),
              value: h.surprise_pct,
              emphasis: i === hist.length - 1,
              tip: (
                <div className="space-y-0.5">
                  <div className="font-semibold text-ink">{h.surprise_pct != null ? `${pct(h.surprise_pct)} surprise` : "no surprise data"}</div>
                  <div>
                    {shortDate(h.date)} · actual {price(h.eps_actual, ccy)} vs est. {price(h.eps_estimate, ccy)}
                  </div>
                </div>
              ),
            }))}
            height={84}
            format={(v) => pct(v, 1)}
            labels="emphasis"
            ariaLabel="Earnings surprise by quarter"
          />
        </div>
      )}
      {e.history.length > 0 && (
        <table className="mt-3 w-full text-xs num">
          <thead>
            <tr className="text-2xs text-muted">
              <th className="py-1 text-left font-normal">Reported</th>
              <th className="py-1 text-right font-normal">Est.</th>
              <th className="py-1 text-right font-normal">Actual</th>
              <th className="py-1 text-right font-normal">Surprise</th>
            </tr>
          </thead>
          <tbody className="divide-hair">
            {e.history.slice(0, 4).map((h) => (
              <tr key={h.date}>
                <td className="py-1 text-ink-2">{shortDate(h.date)}</td>
                <td className="py-1 text-right text-muted">{price(h.eps_estimate, ccy)}</td>
                <td className="py-1 text-right text-ink">{price(h.eps_actual, ccy)}</td>
                <td className={cx("py-1 text-right font-medium", textTone[(h.surprise_pct ?? 0) > 0 ? "bull" : (h.surprise_pct ?? 0) < 0 ? "bear" : "neutral"])}>{pct(h.surprise_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="mt-2 text-2xs text-faint">{int(e.history.length)} reported quarters on record.</p>
    </Panel>
  );
}

/** Report month, e.g. "Jul ’26" (fiscal quarters differ by company, so label by date). */
function quarterLabel(d: string): string {
  const [y, m] = d.split("-").map(Number);
  return `${new Date(y, m - 1, 1).toLocaleDateString("en-US", { month: "short" })} ’${String(y).slice(2)}`;
}
