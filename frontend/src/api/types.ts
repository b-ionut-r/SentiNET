// Mirrors the backend Pydantic schemas (app/schemas.py).

export type SentimentLabel = "bullish" | "bearish" | "neutral";
export type SourceStatus = "ok" | "empty" | "error" | "disabled";

export interface Signal {
  source: string;
  text: string;
  url: string | null;
  author: string | null;
  timestamp: string | null;
  engagement: number;
  score: number;
  label: SentimentLabel;
}

export interface SourceBreakdown {
  source: string;
  status: SourceStatus;
  count: number;
  avg_score: number;
  bullish_pct: number;
  bearish_pct: number;
  neutral_pct: number;
  weight: number;
  error: string | null;
}

export interface TimelinePoint {
  bucket: string;
  avg_score: number;
  count: number;
}

export interface TrendingKeyword {
  keyword: string;
  count: number;
}

export interface AnalyzeResponse {
  ticker: string;
  company: string | null;
  generated_at: string;
  cached: boolean;
  overall_score: number;
  overall_label: SentimentLabel;
  total_signals: number;
  active_sources: number;
  bullish_pct: number;
  bearish_pct: number;
  neutral_pct: number;
  sources: SourceBreakdown[];
  timeline: TimelinePoint[];
  trending: TrendingKeyword[];
  signals: Signal[];
}

export interface PricePoint {
  t: string;
  close: number;
}

export interface PriceResponse {
  ticker: string;
  company: string | null;
  currency: string | null;
  current_price: number | null;
  previous_close: number | null;
  change: number | null;
  change_pct: number | null;
  range: string;
  points: PricePoint[];
  available: boolean;
  error: string | null;
}

export const SOURCE_LABELS: Record<string, string> = {
  yahoo: "Yahoo Finance",
  google_news: "Google News",
  reddit: "Reddit",
  tradestie: "WSB (Tradestie)",
  apewisdom: "ApeWisdom",
  stocktwits: "StockTwits",
  hackernews: "Hacker News",
};
