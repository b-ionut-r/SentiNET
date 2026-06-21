interface Props {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
  trend?: "up" | "down" | null;
}

export function StatCard({ label, value, sub, accent, trend }: Props) {
  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div className="flex items-baseline gap-2">
        <span
          className="font-mono text-2xl font-bold"
          style={{ color: accent ?? "#e2e8f0" }}
        >
          {value}
        </span>
        {trend && (
          <span className={trend === "up" ? "text-bull" : "text-bear"}>
            {trend === "up" ? "▲" : "▼"}
          </span>
        )}
      </div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}
