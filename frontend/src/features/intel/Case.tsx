/** Bull case vs. bear case (from the brief) and the "watch next" catalyst timeline. */
import { Briefcase, CalendarClock, Coins, ExternalLink, FileText, Newspaper, UserRound } from "lucide-react";

import type { Analysis, Catalyst } from "../../api/types";
import { Mark } from "../../components/ui/Badges";
import { Empty } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";
import { countdown, daysUntil, shortDate } from "../../lib/format";

const KIND_ICON: Record<Catalyst["kind"], typeof Coins> = {
  earnings: CalendarClock,
  dividend: Coins,
  analyst: Briefcase,
  filing: FileText,
  insider: UserRound,
  news: Newspaper,
};

/** The brief: one numbers-backed paragraph, then the bull and bear evidence side by side. */
export function CasePanel({ a, className }: { a: Analysis; className?: string }) {
  const b = a.brief;
  return (
    <Panel id="case" title="Bull case vs. bear case" subtitle="Deterministic brief — every point is backed by a number" className={className} bodyClassName="flex flex-col">
      {b.summary && <p className="mb-4 max-w-[78ch] text-sm leading-[22px] text-ink-2">{b.summary}</p>}
      <div className="grid flex-1 gap-4 sm:grid-cols-2">
        <CaseColumn title="Bull case" p="bull" points={b.bull_points} empty="No bullish evidence cleared the bar." />
        <CaseColumn title="Bear case" p="bear" points={b.bear_points} empty="No bearish evidence cleared the bar." />
      </div>
    </Panel>
  );
}

function CaseColumn({ title, p, points, empty }: { title: string; p: "bull" | "bear"; points: string[]; empty: string }) {
  return (
    <div className="rounded-lg bg-sunken p-3.5" style={{ boxShadow: "inset 0 0 0 1px var(--hairline)" }}>
      <div className="mb-2.5 flex items-center gap-2">
        <span className={cx("h-3.5 w-0.5 rounded-full", p === "bull" ? "bg-bull" : "bg-bear")} aria-hidden />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-2">{title}</h3>
        <span className="text-2xs text-muted">{points.length}</span>
      </div>
      {points.length === 0 ? (
        <p className="text-xs text-muted">{empty}</p>
      ) : (
        <ul className="space-y-2">
          {points.map((t, i) => (
            <li key={i} className="flex gap-2 text-[13px] leading-[19px] text-ink-2">
              <Mark p={p} className="mt-[6px]" />
              <span>{t}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function WatchNext({ a, className }: { a: Analysis; className?: string }) {
  const upcoming = a.catalysts.filter((c) => c.upcoming).sort((x, y) => x.date.localeCompare(y.date));
  const recent = a.catalysts.filter((c) => !c.upcoming).sort((x, y) => y.date.localeCompare(x.date)).slice(0, 5);
  const e = a.earnings;
  const earningsDays = e?.days_until ?? daysUntil(e?.next_date);
  const hasEarningsCatalyst = upcoming.some((c) => c.kind === "earnings");
  const showEarnings = !!e?.next_date && earningsDays != null && earningsDays >= 0;

  return (
    <Panel title="Watch next" subtitle="Upcoming catalysts, then what just happened" className={className}>
      {showEarnings && e?.next_date && earningsDays != null && (
        <div className="mb-3 flex items-center gap-3.5 rounded-lg bg-sunken p-3" style={{ boxShadow: "inset 0 0 0 1px var(--hairline)" }}>
          <div className="text-center">
            <div className="text-[28px] font-semibold leading-none tracking-[-0.02em] text-ink">{earningsDays}</div>
            <div className="mt-1 text-2xs text-muted">{earningsDays === 1 ? "day" : "days"}</div>
          </div>
          <div className="min-w-0 text-xs">
            <div className="font-semibold text-ink">Earnings · {shortDate(e.next_date)}</div>
            <div className="mt-0.5 text-ink-2">
              {e.eps_estimate != null && <>EPS est. ${e.eps_estimate.toFixed(2)}</>}
              {e.eps_low != null && e.eps_high != null && <span className="text-muted"> (${e.eps_low.toFixed(2)}–${e.eps_high.toFixed(2)})</span>}
            </div>
            {e.beat_rate != null && e.history.length > 0 && (
              <div className="mt-0.5 text-muted">
                Beat {Math.round(e.beat_rate * e.history.length)} of last {e.history.length}
              </div>
            )}
          </div>
        </div>
      )}
      {upcoming.length + recent.length === 0 ? (
        showEarnings ? (
          <p className="text-xs text-muted">No other dated catalysts — no dividends, rating changes or material filings in range.</p>
        ) : (
          <Empty title="No dated catalysts">No earnings date, dividends, rating changes or material filings in range.</Empty>
        )
      ) : (
        <ol className="relative space-y-0.5">
          {upcoming
            .filter((c) => !(hasEarningsCatalyst && c.kind === "earnings" && e?.next_date))
            .map((c, i) => (
              <CatalystRow key={`u${i}`} c={c} />
            ))}
          {recent.length > 0 && upcoming.length > 0 && <li className="eyebrow px-1 pb-1 pt-2">Recent</li>}
          {recent.map((c, i) => (
            <CatalystRow key={`r${i}`} c={c} />
          ))}
        </ol>
      )}
      {a.brief.watch.length > 0 && (
        <div className="mt-3 pt-3 hairline-t">
          <p className="eyebrow mb-1.5">On watch</p>
          <ul className="space-y-1">
            {a.brief.watch.map((w, i) => (
              <li key={i} className="flex gap-2 text-xs text-ink-2">
                <span className="mt-[7px] size-1 shrink-0 rounded-full bg-[rgb(var(--muted))]" />
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}

function CatalystRow({ c }: { c: Catalyst }) {
  const Icon = KIND_ICON[c.kind] ?? Newspaper;
  const d = daysUntil(c.date);
  const body = (
    <>
      <span className="w-12 shrink-0 pt-px text-2xs text-muted num">{shortDate(c.date)}</span>
      <Icon className="mt-0.5 size-3.5 shrink-0 text-muted" aria-hidden />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5 text-xs font-medium text-ink">
          <span className="truncate">{c.title}</span>
          {c.url && <ExternalLink className="size-3 shrink-0 text-faint" aria-hidden />}
        </span>
        {(c.detail || c.upcoming) && (
          <span className="block truncate text-2xs text-muted">
            {c.upcoming && d != null ? `${countdown(d)}${c.detail ? " · " : ""}` : ""}
            {c.detail}
          </span>
        )}
      </span>
      {c.polarity !== "neutral" && <Mark p={c.polarity} className="mt-1" />}
    </>
  );
  return (
    <li>
      {c.url ? (
        <a href={c.url} target="_blank" rel="noreferrer" className="flex gap-2.5 rounded-md px-1 py-1.5 hover:bg-raised">
          {body}
        </a>
      ) : (
        <div className="flex gap-2.5 px-1 py-1.5">{body}</div>
      )}
    </li>
  );
}
