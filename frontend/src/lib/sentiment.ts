import type { SentimentLabel } from "../api/types";

export function labelColor(label: SentimentLabel): string {
  switch (label) {
    case "bullish":
      return "#22c55e";
    case "bearish":
      return "#ef4444";
    default:
      return "#eab308";
  }
}

export function labelText(label: SentimentLabel): string {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

// A descriptive headline for a score in [-1, 1].
export function scoreHeadline(score: number): string {
  if (score >= 0.5) return "Strongly Bullish";
  if (score >= 0.15) return "Bullish";
  if (score > 0.05) return "Leaning Bullish";
  if (score <= -0.5) return "Strongly Bearish";
  if (score <= -0.15) return "Bearish";
  if (score < -0.05) return "Leaning Bearish";
  return "Neutral";
}

export function colorForScore(score: number): string {
  if (score > 0.05) return "#22c55e";
  if (score < -0.05) return "#ef4444";
  return "#eab308";
}

export function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  const secs = Math.max(0, (Date.now() - then) / 1000);
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}
