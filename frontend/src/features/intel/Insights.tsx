/** Severity-ranked insights: the things worth acting on or watching. */
import { CalendarClock, Database, Flame, Landmark, RotateCcw, ShieldAlert, Split, TrendingUp, Users } from "lucide-react";

import type { Insight, InsightKind } from "../../api/types";
import { Mark, SeverityBadge, severityMeta } from "../../components/ui/Badges";
import { Empty } from "../../components/ui/Misc";
import { Panel } from "../../components/ui/Panel";
import { cx } from "../../lib/cx";

const KIND: Record<InsightKind, { label: string; Icon: typeof Flame }> = {
  divergence: { label: "Divergence", Icon: Split },
  attention: { label: "Attention", Icon: Flame },
  reversal: { label: "Reversal", Icon: RotateCcw },
  crowding: { label: "Crowding", Icon: Users },
  catalyst: { label: "Catalyst", Icon: CalendarClock },
  smart_money: { label: "Smart money", Icon: Landmark },
  risk: { label: "Risk", Icon: ShieldAlert },
  momentum: { label: "Momentum", Icon: TrendingUp },
  quality: { label: "Data quality", Icon: Database },
};

const RANK: Record<string, number> = { alert: 0, watch: 1, info: 2 };

export function InsightsRail({ insights, className }: { insights: Insight[]; className?: string }) {
  const sorted = [...insights].sort((a, b) => (RANK[a.severity] ?? 3) - (RANK[b.severity] ?? 3));
  const counts: Record<string, number> = { alert: 0, watch: 0, info: 0 };
  insights.forEach((i) => (counts[i.severity] = (counts[i.severity] ?? 0) + 1));
  return (
    <Panel
      id="insights"
      title="Insights"
      subtitle={insights.length ? `${counts.alert} alert · ${counts.watch} watch · ${counts.info} info` : "Only shown when evidence clears the bar"}
      className={className}
      flush
    >
      {sorted.length === 0 ? (
        <Empty title="Nothing unusual">No divergences, spikes, crowding or risk flags cleared their thresholds.</Empty>
      ) : (
        <ul className="divide-hair hairline-t">
          {sorted.map((ins, i) => {
            const k = KIND[ins.kind] ?? KIND.quality;
            return (
              <li key={i} className="relative flex gap-3 px-4 py-3">
                <span className={cx("absolute inset-y-3 left-0 w-0.5 rounded-r", (severityMeta[ins.severity] ?? severityMeta.info).ring, ins.severity === "info" && "opacity-40")} aria-hidden />
                <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-raised text-ink-2">
                  <k.Icon className="size-3.5" aria-hidden />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                    <SeverityBadge severity={ins.severity} />
                    <span className="text-2xs text-muted">{k.label}</span>
                    {ins.polarity !== "neutral" && <Mark p={ins.polarity} />}
                  </div>
                  <p className="mt-1 text-sm font-semibold leading-5 text-ink">{ins.title}</p>
                  <p className="mt-0.5 text-xs leading-[18px] text-ink-2">{ins.detail}</p>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
