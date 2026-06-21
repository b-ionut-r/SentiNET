import { colorForScore, scoreHeadline } from "../lib/sentiment";

interface Props {
  score: number; // -1 .. 1
  totalSignals: number;
}

// Semicircular gauge: maps score [-1,1] to a 180deg sweep (left=bearish).
export function SentimentGauge({ score, totalSignals }: Props) {
  const clamped = Math.max(-1, Math.min(1, score));
  const angle = (clamped + 1) * 90; // 0 .. 180 degrees
  const color = colorForScore(clamped);

  const cx = 130;
  const cy = 130;
  const r = 100;
  const needleLen = 92;
  const rad = (Math.PI * (180 - angle)) / 180;
  const nx = cx + needleLen * Math.cos(rad);
  const ny = cy - needleLen * Math.sin(rad);

  const arc = (start: number, end: number) => {
    const s = (Math.PI * (180 - start)) / 180;
    const e = (Math.PI * (180 - end)) / 180;
    const x1 = cx + r * Math.cos(s);
    const y1 = cy - r * Math.sin(s);
    const x2 = cx + r * Math.cos(e);
    const y2 = cy - r * Math.sin(e);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 260 165" className="w-full max-w-[320px]">
        {/* colored zones */}
        <path d={arc(0, 60)} stroke="#ef4444" strokeWidth="16" fill="none" strokeLinecap="round" opacity="0.85" />
        <path d={arc(62, 118)} stroke="#eab308" strokeWidth="16" fill="none" opacity="0.85" />
        <path d={arc(120, 180)} stroke="#22c55e" strokeWidth="16" fill="none" strokeLinecap="round" opacity="0.85" />
        {/* needle */}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth="4" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="8" fill={color} />
      </svg>

      <div className="-mt-4 text-center">
        <div className="font-mono text-4xl font-bold" style={{ color }}>
          {clamped >= 0 ? "+" : ""}
          {clamped.toFixed(2)}
        </div>
        <div className="mt-1 text-lg font-semibold" style={{ color }}>
          {scoreHeadline(clamped)}
        </div>
        <div className="mt-1 text-xs text-slate-500">
          weighted across {totalSignals} signal{totalSignals === 1 ? "" : "s"}
        </div>
      </div>
    </div>
  );
}
