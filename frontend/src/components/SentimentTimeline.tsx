import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TimelinePoint } from "../api/types";
import { colorForScore } from "../lib/sentiment";

interface Props {
  timeline: TimelinePoint[];
}

export function SentimentTimeline({ timeline }: Props) {
  const data = timeline.map((p) => ({
    t: new Date(p.bucket).getTime(),
    score: p.avg_score,
    count: p.count,
  }));

  return (
    <div className="card">
      <div className="card-title">Sentiment Over Time</div>
      <div className="h-40">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-500">
            Not enough timestamped signals yet.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
              <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} hide />
              <YAxis domain={[-1, 1]} width={36} tick={{ fontSize: 11, fill: "#64748b" }} />
              <ReferenceLine y={0} stroke="#334155" />
              <Tooltip
                cursor={{ fill: "rgba(255,255,255,0.04)" }}
                contentStyle={{
                  background: "#111726",
                  border: "1px solid #1f2937",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelFormatter={(t) => new Date(t as number).toLocaleString()}
                formatter={(v: number, _n, p) => [
                  `${(v as number).toFixed(2)} (${(p?.payload as { count: number }).count} signals)`,
                  "Avg sentiment",
                ]}
              />
              <Bar dataKey="score" radius={[2, 2, 0, 0]}>
                {data.map((d, i) => (
                  <Cell key={i} fill={colorForScore(d.score)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
