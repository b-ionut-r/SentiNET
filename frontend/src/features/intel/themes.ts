/**
 * Display labels for theme and event keys that arrive bare on signals and
 * narratives (ThemeStat rows carry their own labels from the backend).
 */
const THEMES: Record<string, string> = {
  earnings: "Earnings",
  guidance: "Guidance",
  analyst: "Analyst actions",
  product: "Products",
  ai: "AI",
  legal: "Legal",
  regulatory: "Regulatory",
  deals: "Deals",
  management: "Management",
  capital_return: "Buybacks & dividends",
  macro: "Macro",
  supply_chain: "Supply chain",
  competition: "Competition",
  labor: "Labor",
  trading: "Trading & flows",
  valuation: "Valuation",
};

const EVENTS: Record<string, string> = {
  analyst_upgrade: "Upgrade",
  analyst_downgrade: "Downgrade",
  analyst_initiate: "Initiation",
  pt_raise: "PT raise",
  pt_cut: "PT cut",
  earnings_beat: "Earnings beat",
  earnings_miss: "Earnings miss",
  guidance_raise: "Guidance raise",
  guidance_cut: "Guidance cut",
  record_results: "Record results",
  buyback: "Buyback",
  dividend_raise: "Dividend hike",
  dividend_cut: "Dividend cut",
  layoffs: "Layoffs",
  lawsuit: "Lawsuit",
  investigation: "Investigation",
  settlement: "Settlement",
  m_and_a: "M&A",
  partnership: "Partnership",
  contract_win: "Contract win",
  product_launch: "Launch",
  recall: "Recall",
  exec_departure: "Exec departure",
  exec_hire: "Exec hire",
  offering: "Offering",
  bankruptcy: "Bankruptcy",
  delisting: "Delisting",
  short_report: "Short report",
  insider_buy: "Insider buy",
  insider_sell: "Insider sell",
  all_time_high: "All-time high",
  low_52w: "52-week low",
  stock_split: "Stock split",
  price_up: "Price up",
  price_down: "Price down",
};

const humanize = (k: string) => k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const themeLabel = (k: string): string => THEMES[k] ?? humanize(k);
export const eventLabel = (k: string): string => EVENTS[k] ?? humanize(k);
