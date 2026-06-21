import type { TrendingKeyword } from "../api/types";

interface Props {
  trending: TrendingKeyword[];
}

export function TrendingMentions({ trending }: Props) {
  const max = trending.reduce((m, t) => Math.max(m, t.count), 1);
  return (
    <div className="card">
      <div className="card-title">Trending in the conversation</div>
      {trending.length === 0 ? (
        <div className="py-4 text-sm text-slate-500">No recurring themes yet.</div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {trending.map((t) => {
            const scale = 0.8 + (t.count / max) * 0.6;
            return (
              <span
                key={t.keyword}
                className="chip border border-edge bg-panel-2"
                style={{ fontSize: `${scale * 0.8}rem` }}
              >
                <span className="text-slate-200">{t.keyword}</span>
                <span className="text-slate-500">{t.count}</span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
