import type { SourceBreakdown } from "../api/types";
import { SOURCE_LABELS } from "../api/types";
import { timeAgo } from "../lib/sentiment";

interface Props {
  sources: SourceBreakdown[];
  generatedAt: string;
  cached: boolean;
}

const DOT: Record<string, string> = {
  ok: "bg-bull",
  empty: "bg-slate-500",
  error: "bg-bear",
  disabled: "bg-slate-700",
};

export function SourceStatus({ sources, generatedAt, cached }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-500">
      <span className="font-medium text-slate-400">Sources:</span>
      {sources.map((s) => (
        <span key={s.source} className="flex items-center gap-1" title={s.error ?? s.status}>
          <span className={`h-2 w-2 rounded-full ${DOT[s.status] ?? "bg-slate-600"}`} />
          {SOURCE_LABELS[s.source] ?? s.source}
        </span>
      ))}
      <span className="ml-auto">
        {cached ? "cached • " : ""}updated {timeAgo(generatedAt)}
      </span>
    </div>
  );
}
