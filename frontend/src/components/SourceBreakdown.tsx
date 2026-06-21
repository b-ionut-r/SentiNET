import type { SourceBreakdown as SB } from "../api/types";
import { SOURCE_LABELS } from "../api/types";
import { colorForScore } from "../lib/sentiment";

interface Props {
  sources: SB[];
}

const KIND_HINT: Record<string, string> = {
  yahoo: "news",
  google_news: "news",
  hackernews: "news",
  reddit: "social",
  tradestie: "social",
  apewisdom: "social",
  stocktwits: "social",
};

export function SourceBreakdown({ sources }: Props) {
  // Show responding sources first, ordered by volume.
  const ordered = [...sources].sort((a, b) => b.count - a.count);

  return (
    <div className="card">
      <div className="card-title">Source Breakdown — news vs. social</div>
      <div className="space-y-3">
        {ordered.map((s) => {
          const ok = s.status === "ok" && s.count > 0;
          const label = SOURCE_LABELS[s.source] ?? s.source;
          // Position of the avg marker on a -1..1 track.
          const pos = ((Math.max(-1, Math.min(1, s.avg_score)) + 1) / 2) * 100;
          return (
            <div key={s.source}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-slate-200">{label}</span>
                  <span className="rounded bg-panel-2 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-500">
                    {KIND_HINT[s.source] ?? "src"}
                  </span>
                </div>
                {ok ? (
                  <span className="font-mono text-xs text-slate-400">
                    {s.count} • {s.avg_score >= 0 ? "+" : ""}
                    {s.avg_score.toFixed(2)}
                  </span>
                ) : (
                  <span
                    className="text-[11px] text-slate-500"
                    title={s.error ?? undefined}
                  >
                    {s.status === "empty" ? "no mentions" : s.status === "disabled" ? "disabled" : "unavailable"}
                  </span>
                )}
              </div>
              {ok ? (
                <div className="relative h-2.5 w-full rounded-full bg-gradient-to-r from-bear/30 via-neutral/20 to-bull/30">
                  <div
                    className="absolute top-1/2 h-4 w-1.5 -translate-y-1/2 rounded-full"
                    style={{
                      left: `calc(${pos}% - 3px)`,
                      background: colorForScore(s.avg_score),
                    }}
                  />
                </div>
              ) : (
                <div className="h-2.5 w-full rounded-full bg-panel-2" />
              )}
              {ok && (
                <div className="mt-1 flex gap-3 text-[11px] text-slate-500">
                  <span className="text-bull">{s.bullish_pct}% bull</span>
                  <span className="text-bear">{s.bearish_pct}% bear</span>
                  <span className="text-neutral">{s.neutral_pct}% neutral</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
