/**
 * Display labels for theme and event keys that arrive bare on signals and
 * narratives (ThemeStat rows carry their own labels from the backend).
 * Mirrors backend app/nlp/themes.py THEMES and app/nlp/events.py EVENT_LABELS;
 * unknown keys are humanized, so a new backend key still renders sensibly.
 */
const THEMES: Record<string, string> = {
  earnings: "Earnings",
  guidance: "Guidance & outlook",
  analyst: "Analyst ratings",
  product: "Products & launches",
  ai: "AI",
  legal: "Legal",
  regulatory: "Regulation & policy",
  deals: "M&A & partnerships",
  management: "Management",
  capital_return: "Buybacks & dividends",
  macro: "Macro & markets",
  supply_chain: "Supply chain & production",
  competition: "Competition",
  labor: "Labor & layoffs",
  trading: "Trading & flows",
  valuation: "Valuation",
  insider: "Insider activity",
};

const EVENTS: Record<string, string> = {
  analyst_upgrade: "Analyst upgrade",
  analyst_downgrade: "Analyst downgrade",
  analyst_initiate: "Coverage initiated",
  analyst_top_pick: "Named top pick",
  pt_raise: "Price target raised",
  pt_cut: "Price target cut",
  earnings_beat: "Earnings beat",
  earnings_miss: "Earnings miss",
  guidance_raise: "Guidance raised",
  guidance_cut: "Guidance cut",
  record_results: "Record results",
  buyback: "Buyback",
  dividend_raise: "Dividend raised",
  dividend_cut: "Dividend cut",
  layoffs: "Layoffs",
  lawsuit: "Lawsuit",
  investigation: "Investigation",
  settlement: "Settlement",
  m_and_a: "M&A",
  partnership: "Partnership",
  contract_win: "Contract win",
  product_launch: "Product launch",
  recall: "Recall",
  exec_departure: "Executive departure",
  exec_hire: "Executive hire",
  offering: "Share offering",
  bankruptcy: "Bankruptcy risk",
  delisting: "Delisting risk",
  short_report: "Short-seller report",
  insider_buy: "Insider buying",
  insider_sell: "Insider selling",
  all_time_high: "All-time high",
  low_52w: "52-week low",
  stock_split: "Stock split",
  price_up: "Price jump",
  price_down: "Price drop",
  regulatory_approval: "Regulatory approval",
  regulatory_setback: "Regulatory/trial setback",
  index_inclusion: "Index inclusion",
  data_breach: "Breach/outage",
};

const humanize = (k: string) => k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const themeLabel = (k: string): string => THEMES[k] ?? humanize(k);
export const eventLabel = (k: string): string => EVENTS[k] ?? humanize(k);
