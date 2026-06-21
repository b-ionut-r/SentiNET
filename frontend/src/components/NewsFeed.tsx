import type { Signal } from "../api/types";
import { SOURCE_LABELS } from "../api/types";
import { labelColor, labelText, timeAgo } from "../lib/sentiment";

interface Props {
  signals: Signal[];
}

export function NewsFeed({ signals }: Props) {
  return (
    <div className="card flex h-full flex-col">
      <div className="card-title">Live Signal Feed — {signals.length} items</div>
      {signals.length === 0 ? (
        <div className="flex flex-1 items-center justify-center py-10 text-sm text-slate-500">
          No signals available from connected sources.
        </div>
      ) : (
        <div className="scroll-thin -mr-2 max-h-[520px] space-y-2 overflow-y-auto pr-2">
          {signals.map((s, i) => {
            const color = labelColor(s.label);
            const body = (
              <div className="rounded-xl border border-edge bg-panel-2/50 p-3 transition hover:border-slate-600">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="rounded bg-panel px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                    {SOURCE_LABELS[s.source] ?? s.source}
                  </span>
                  <span className="flex items-center gap-2">
                    <span
                      className="chip"
                      style={{ background: `${color}22`, color }}
                    >
                      {labelText(s.label)} {s.score >= 0 ? "+" : ""}
                      {s.score.toFixed(2)}
                    </span>
                    <span className="text-[11px] text-slate-500">
                      {timeAgo(s.timestamp)}
                    </span>
                  </span>
                </div>
                <p className="text-sm leading-snug text-slate-200">{s.text}</p>
                {s.author && (
                  <div className="mt-1 text-[11px] text-slate-500">{s.author}</div>
                )}
              </div>
            );
            return s.url ? (
              <a key={i} href={s.url} target="_blank" rel="noopener noreferrer" className="block">
                {body}
              </a>
            ) : (
              <div key={i}>{body}</div>
            );
          })}
        </div>
      )}
    </div>
  );
}
