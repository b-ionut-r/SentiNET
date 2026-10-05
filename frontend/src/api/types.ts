// Mirrors backend/app/schemas.py — keep in sync.
// Scores: sentiment in [-1, 1]; SentiNET / component scores in [0, 100] (50 = neutral).
// Timestamps are ISO-8601 UTC strings. `null` = not available (never faked).

export type SentimentLabel = "bullish" | "bearish" | "neutral";
export type Polarity = "bull" | "bear" | "neutral";
export type SignalKind = "news" | "social" | "analysis" | "filing";
export type SourceStatus = "ok" | "empty" | "error" | "disabled" | "unconfigured";
export type ConfidenceLevel = "low" | "medium" | "high";

export interface Driver {
  term: string;
  impact: number;
}

export interface Signal {
  id: string;
  source: string;
  source_label: string;
  kind: SignalKind;
  title: string;
  body: string | null;
  url: string | null;
  author: string | null;
  publisher: string | null;
  timestamp: string | null;
  engagement: number;
  score: number;
  label: SentimentLabel;
  confidence: number;
  relevance: number;
  weight: number;
  themes: string[];
  events: string[];
  drivers: Driver[];
  user_label: SentimentLabel | null;
  narrative_id: string | null;
  duplicates: number;
}

export interface SentimentStat {
  score: number;
  label: SentimentLabel;
  n: number;
  bullish: number;
  bearish: number;
  neutral: number;
  confidence: number;
}

export interface SourceReport {
  key: string;
  label: string;
  kind: SignalKind;
  status: SourceStatus;
  fetched: number;
  kept: number;
  score: number | null;
  bullish_pct: number;
  bearish_pct: number;
  weight: number;
  latency_ms: number | null;
  error: string | null;
  requires_key: boolean;
  has_metrics: boolean;
}

export interface Narrative {
  id: string;
  headline: string;
  count: number;
  publishers: string[];
  score: number;
  label: SentimentLabel;
  first_seen: string | null;
  last_seen: string | null;
  velocity_24h: number;
  themes: string[];
  events: string[];
  impact: number;
  url: string | null;
  signal_ids: string[];
  is_new: boolean;
}

export interface ThemeStat {
  theme: string;
  label: string;
  count: number;
  score: number;
  share: number;
}

export interface Keyword {
  term: string;
  count: number;
  score: number;
}

export interface TimelineBucket {
  t: string;
  score: number;
  count: number;
  news: number;
  social: number;
  bullish: number;
  bearish: number;
}

export interface Reason {
  text: string;
  polarity: Polarity;
  weight: number;
  ref: string | null;
}

export type ComponentKey = "news" | "social" | "analysts" | "insiders" | "momentum" | "technicals";

export interface Component {
  key: ComponentKey;
  label: string;
  score: number | null;
  weight: number;
  available: boolean;
  detail: string;
  confidence: number;
}

export interface Verdict {
  score: number;
  label: string;
  stance: SentimentLabel;
  confidence: ConfidenceLevel;
  confidence_value: number;
  headline: string;
  reasons: Reason[];
  components: Component[];
}

export type InsightKind =
  | "divergence"
  | "attention"
  | "reversal"
  | "crowding"
  | "catalyst"
  | "smart_money"
  | "risk"
  | "momentum"
  | "quality"
  | "deal";

export interface Insight {
  kind: InsightKind;
  severity: "info" | "watch" | "alert";
  polarity: Polarity;
  title: string;
  detail: string;
}

export interface Brief {
  summary: string;
  bull_points: string[];
  bear_points: string[];
  watch: string[];
}

export interface DeltaView {
  previous_at: string | null;
  score_change: number | null;
  sentinel_change: number | null;
  price_change_pct: number | null;
  new_narratives: string[];
  note: string | null;
}

export interface Profile {
  symbol: string;
  name: string;
  short_name: string | null;
  quote_type: string | null;
  exchange: string | null;
  sector: string | null;
  industry: string | null;
  country: string | null;
  website: string | null;
  summary: string | null;
  employees: number | null;
  logo_url: string | null;
  cik: string | null;
  /** Reporting currency of EPS/revenue (Yahoo financialCurrency) — often not the quote's: "USD" for SHOP.TO. */
  financial_currency?: string | null;
}

export interface Quote {
  price: number | null;
  change: number | null;
  change_pct: number | null;
  currency: string | null;
  previous_close: number | null;
  open: number | null;
  day_high: number | null;
  day_low: number | null;
  year_high: number | null;
  year_low: number | null;
  volume: number | null;
  avg_volume: number | null;
  market_cap: number | null;
  fifty_day_avg: number | null;
  two_hundred_day_avg: number | null;
  as_of: string | null;
}

export interface Technicals {
  return_1d: number | null;
  return_5d: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_ytd: number | null;
  vs_50dma_pct: number | null;
  vs_200dma_pct: number | null;
  rsi_14: number | null;
  volatility_30d: number | null;
  pct_from_52w_high: number | null;
  volume_ratio: number | null;
  trend: "uptrend" | "downtrend" | "sideways" | null;
}

export interface AnalystAction {
  date: string;
  firm: string;
  action: "up" | "down" | "init" | "main" | "reit" | "other";
  from_grade: string | null;
  to_grade: string | null;
  price_target: number | null;
  prior_target: number | null;
}

export interface RatingCounts {
  period: string;
  strong_buy: number;
  buy: number;
  hold: number;
  sell: number;
  strong_sell: number;
}

export interface AnalystView {
  consensus: string | null;
  mean_rating: number | null;
  counts: RatingCounts | null;
  total: number;
  target_mean: number | null;
  target_median: number | null;
  target_high: number | null;
  target_low: number | null;
  upside_pct: number | null;
  actions: AnalystAction[];
  upgrades_90d: number;
  downgrades_90d: number;
  pt_raises_30d: number;
  pt_cuts_30d: number;
  trend: RatingCounts[];
}

export interface InsiderTxn {
  date: string;
  insider: string;
  position: string | null;
  kind: "buy" | "sell" | "award" | "exercise" | "gift" | "other";
  shares: number | null;
  value: number | null;
  text: string | null;
}

export interface InsiderView {
  window_days: number;
  buys: number;
  sells: number;
  buy_value: number;
  sell_value: number;
  net_value: number;
  ratio: number | null;
  transactions: InsiderTxn[];
}

export interface EarningsEvent {
  date: string;
  eps_estimate: number | null;
  eps_actual: number | null;
  surprise_pct: number | null;
}

export interface EarningsView {
  next_date: string | null;
  days_until: number | null;
  eps_estimate: number | null;
  eps_low: number | null;
  eps_high: number | null;
  revenue_estimate: number | null;
  history: EarningsEvent[];
  beat_rate: number | null;
}

export interface Filing {
  form: string;
  date: string;
  title: string;
  items: string[];
  url: string | null;
  importance: "high" | "medium" | "low";
  polarity: Polarity;
}

export interface CrowdView {
  stocktwits_bullish: number | null;
  stocktwits_bearish: number | null;
  stocktwits_bull_ratio: number | null;
  stocktwits_bull_authors: number | null;
  stocktwits_bear_authors: number | null;
  stocktwits_messages: number | null;
  stocktwits_watchers: number | null;
  reddit_mentions: number | null;
  reddit_mentions_prev: number | null;
  reddit_rank: number | null;
  reddit_rank_prev: number | null;
  reddit_upvotes: number | null;
  wsb_sentiment: number | null;
  wsb_label: SentimentLabel | null;
  wsb_comments: number | null;
  bluesky_posts: number | null;
}

export interface AttentionView {
  heat: number;
  label: "Quiet" | "Normal" | "Elevated" | "Spiking";
  news_volume_z: number | null;
  wiki_views_7d: number | null;
  wiki_views_z: number | null;
  reddit_change_pct: number | null;
  signals_24h: number;
}

export interface Catalyst {
  date: string;
  kind: "earnings" | "dividend" | "analyst" | "filing" | "insider" | "news";
  title: string;
  detail: string | null;
  polarity: Polarity;
  upcoming: boolean;
  url: string | null;
}

export interface TonePoint {
  date: string;
  tone: number | null;
  volume: number | null;
}

export interface ToneTrend {
  query: string;
  tone_7d: number | null;
  tone_30d: number | null;
  tone_90d: number | null;
  change_7d_vs_30d: number | null;
  percentile_7d: number | null;
  series: TonePoint[];
}

export interface Analysis {
  ticker: string;
  generated_at: string;
  cached: boolean;
  elapsed_ms: number;
  engine: string;
  profile: Profile | null;
  quote: Quote | null;
  technicals: Technicals | null;
  verdict: Verdict;
  brief: Brief;
  delta: DeltaView;
  insights: Insight[];
  sentiment: SentimentStat;
  news: SentimentStat;
  social: SentimentStat;
  narratives: Narrative[];
  themes: ThemeStat[];
  keywords: Keyword[];
  timeline: TimelineBucket[];
  tone: ToneTrend | null;
  analysts: AnalystView | null;
  insiders: InsiderView | null;
  earnings: EarningsView | null;
  filings: Filing[];
  crowd: CrowdView | null;
  attention: AttentionView | null;
  catalysts: Catalyst[];
  sources: SourceReport[];
  signals: Signal[];
}

export interface ProgressEvent {
  stage: "resolve" | "source" | "intel" | "nlp" | "analytics" | "done";
  key: string;
  label: string;
  status: "running" | "ok" | "empty" | "error" | "skipped";
  count: number | null;
  ms: number | null;
  detail: string | null;
}

export interface Candle {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number | null;
}

export type PriceRange = "1D" | "5D" | "1M" | "3M" | "6M" | "1Y" | "5Y";

export interface PriceResponse {
  ticker: string;
  range: PriceRange;
  interval: string;
  currency: string | null;
  candles: Candle[];
  available: boolean;
  error: string | null;
}

export interface LagStat {
  lag_days: number;
  r: number;
  n: number;
  p_value: number;
}

export interface HistoryPoint {
  date: string;
  tone: number | null;
  volume: number | null;
  close: number | null;
  ret_pct: number | null;
  snapshot_score: number | null;
  wiki_views: number | null;
}

export interface HistoryResponse {
  ticker: string;
  days: number;
  points: HistoryPoint[];
  lags: LagStat[];
  best_lag: LagStat | null;
  interpretation: string;
  status: Record<string, string>;
}

export interface Snapshot {
  ticker: string;
  at: string;
  sentinel_score: number;
  score: number;
  label: SentimentLabel;
  n_signals: number;
  price: number | null;
  currency?: string | null;
  news_score: number | null;
  social_score: number | null;
  narratives: string[];
}

export interface GaugeComponent {
  key: string;
  label: string;
  score: number | null;
  rating: string | null;
}

export interface HistoryValue {
  t: string;
  v: number;
}

export interface FearGreed {
  score: number;
  rating: string;
  previous_close: number | null;
  week_ago: number | null;
  month_ago: number | null;
  year_ago: number | null;
  components: GaugeComponent[];
  history: HistoryValue[];
}

export interface IndexQuote {
  symbol: string;
  name: string;
  price: number | null;
  change_pct: number | null;
  spark: number[];
}

export interface TrendingTicker {
  symbol: string;
  name: string | null;
  source: "reddit" | "stocktwits" | "yahoo";
  rank: number | null;
  rank_prev: number | null;
  mentions: number | null;
  mentions_prev: number | null;
  change_pct: number | null;
  sentiment: number | null;
}

export interface MarketOverview {
  generated_at: string;
  regime: string;
  regime_detail: string;
  fear_greed: FearGreed | null;
  crypto_fear_greed: FearGreed | null;
  indices: IndexQuote[];
  trending: TrendingTicker[];
  headlines: SentimentStat;
  narratives: Narrative[];
  status: Record<string, string>;
}

export interface SymbolMatch {
  symbol: string;
  name: string;
  exchange: string | null;
  type: string | null;
  logo_url: string | null;
}

export interface ScoreRequest {
  texts: string[];
  ticker?: string | null;
}

export interface ScoredText {
  text: string;
  score: number;
  label: SentimentLabel;
  confidence: number;
  drivers: Driver[];
  themes: string[];
  events: string[];
  relevance: number | null;
}

export interface ScoreResponse {
  engine: string;
  results: ScoredText[];
  summary: SentimentStat;
  themes: ThemeStat[];
}

export interface WatchItem {
  ticker: string;
  added_at: string;
  name: string | null;
  last: Snapshot | null;
  previous: Snapshot | null;
  spark: number[];
}

export type AlertKind =
  | "score_above"
  | "score_below"
  | "score_change"
  | "attention_spike"
  | "new_narrative"
  | "analyst_action";

export interface AlertRuleIn {
  ticker: string;
  kind: AlertKind;
  threshold?: number | null;
}

export interface AlertRule extends AlertRuleIn {
  id: number;
  enabled: boolean;
  created_at: string;
  last_triggered_at: string | null;
}

export interface AlertEvent {
  id: number;
  rule_id: number | null;
  ticker: string;
  at: string;
  title: string;
  detail: string;
  delivered: boolean;
}

export interface SourceInfo {
  key: string;
  label: string;
  kind: SignalKind;
  enabled: boolean;
  requires_key: boolean;
  configured: boolean;
  description: string;
  docs_url: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  engine: string;
  sources: SourceInfo[];
  features: Record<string, boolean>;
}
