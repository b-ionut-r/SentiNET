import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PriceResponse } from "../api/types";

interface Props {
  data: PriceResponse | undefined;
  loading: boolean;
  range: string;
  onRange: (r: string) => void;
}

const RANGES = ["1D", "5D", "1M", "6M", "1Y"];

export function PriceChart({ data, loading, range, onRange }: Props) {
  const up = (data?.change ?? 0) >= 0;
  const color = up ? "#22c55e" : "#ef4444";
  const points = (data?.points ?? []).map((p) => ({
    t: new Date(p.t).getTime(),
    close: p.close,
  }));

  return (
    <div className="card">
      <div className="mb-3 flex items-center justify-between">
        <div className="card-title mb-0">Price</div>
        <div className="flex gap-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => onRange(r)}
              className={`rounded-md px-2 py-1 text-xs font-medium transition ${
                r === range
                  ? "bg-accent/20 text-accent"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {data?.current_price != null && (
        <div className="mb-2 flex items-baseline gap-3">
          <span className="font-mono text-2xl font-bold text-slate-100">
            {data.currency === "USD" || !data.currency ? "$" : ""}
            {data.current_price.toFixed(2)}
          </span>
          <span className="font-mono text-sm font-semibold" style={{ color }}>
            {up ? "▲" : "▼"} {data.change?.toFixed(2)} ({data.change_pct?.toFixed(2)}%)
          </span>
        </div>
      )}

      <div className="h-48">
        {loading ? (
          <div className="skeleton h-full w-full" />
        ) : points.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-500">
            {data?.error ? "Price data unavailable" : "No price data"}
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={points} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="t"
                type="number"
                domain={["dataMin", "dataMax"]}
                hide
              />
              <YAxis domain={["auto", "auto"]} width={48} tick={{ fontSize: 11, fill: "#64748b" }} />
              <Tooltip
                contentStyle={{
                  background: "#111726",
                  border: "1px solid #1f2937",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelFormatter={(t) => new Date(t as number).toLocaleString()}
                formatter={(v: number) => [v.toFixed(2), "Close"]}
              />
              <Area
                type="monotone"
                dataKey="close"
                stroke={color}
                strokeWidth={2}
                fill="url(#priceFill)"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
