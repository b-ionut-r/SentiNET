/**
 * Display formatting. Every signed number uses a real minus sign (U+2212) and
 * an explicit "+" so sentiment never relies on color alone.
 */
export const MINUS = "−";
export const DASH = "—";

const isNum = (n: number | null | undefined): n is number => typeof n === "number" && Number.isFinite(n);

/** "+0.31" / "−0.42" / "0.00". */
export function signed(n: number | null | undefined, digits = 2): string {
  if (!isNum(n)) return DASH;
  const r = Number(n.toFixed(digits));
  if (r === 0) return (0).toFixed(digits);
  return `${r > 0 ? "+" : MINUS}${Math.abs(r).toFixed(digits)}`;
}

/** Signed integer, e.g. "+8" / "−3" / "0". */
export function signedInt(n: number | null | undefined): string {
  if (!isNum(n)) return DASH;
  const r = Math.round(n);
  return r === 0 ? "0" : `${r > 0 ? "+" : MINUS}${Math.abs(r)}`;
}

/** Percent value already in percent units: 1.234 -> "+1.2%". */
export function pct(n: number | null | undefined, digits = 1, withSign = true): string {
  if (!isNum(n)) return DASH;
  return withSign ? `${signed(n, digits)}%` : `${n.toFixed(digits).replace("-", MINUS)}%`;
}

/** Ratio 0..1 -> "62%". */
export function ratioPct(n: number | null | undefined, digits = 0): string {
  if (!isNum(n)) return DASH;
  return `${(n * 100).toFixed(digits)}%`;
}

const currencySymbol: Record<string, string> = {
  USD: "$",
  EUR: "€",
  GBP: "£",
  JPY: "¥",
  CAD: "C$",
  AUD: "A$",
  NZD: "NZ$",
  HKD: "HK$",
  SGD: "S$",
  CHF: "CHF ",
  CNY: "CN¥",
  INR: "₹",
  KRW: "₩",
  ILS: "₪",
  ZAR: "R",
};

/**
 * Yahoo's minor-unit codes are case-sensitive: "GBp"/"GBX" is pence, "ZAc" South
 * African cents, "ILA" Israeli agorot. Upper-casing them reads pence as pounds (100×).
 * Per-share figures (price, day range, analyst targets) arrive in the minor unit;
 * aggregates (market cap, insider values) arrive in the major one — VOD.L quotes
 * 126.8 GBp with a 29.4e9 GBP market cap.
 */
const MINOR_UNITS: Record<string, { major: string; suffix: string }> = {
  GBp: { major: "GBP", suffix: "p" },
  GBX: { major: "GBP", suffix: "p" },
  GBx: { major: "GBP", suffix: "p" },
  ZAc: { major: "ZAR", suffix: "c" },
  ZAC: { major: "ZAR", suffix: "c" },
  ILA: { major: "ILS", suffix: " ag." },
};

/** ISO code of the major currency ("GBp" → "GBP", "usd" → "USD"); null when unknown. */
export function majorCurrency(currency: string | null | undefined): string | null {
  if (!currency) return null;
  return MINOR_UNITS[currency]?.major ?? currency.toUpperCase();
}

/** Prefix for an amount in the major unit ("$", "£", "SEK "); "" when the currency is unknown. */
export function symbolFor(currency: string | null | undefined): string {
  const major = majorCurrency(currency);
  if (!major) return "";
  return currencySymbol[major] ?? `${major} `;
}

function withUnit(n: number, body: string, currency: string | null | undefined): string {
  const minor = currency ? MINOR_UNITS[currency] : undefined;
  const sign = n < 0 ? MINUS : "";
  return minor ? `${sign}${body}${minor.suffix}` : `${sign}${symbolFor(currency)}${body}`;
}

/**
 * Per-share price in its quote currency with sensible precision for the magnitude
 * ($0.000123, $12.34, $86,152, 126.80p). No currency → the bare number: a symbol is
 * never guessed.
 */
export function price(n: number | null | undefined, currency?: string | null): string {
  if (!isNum(n)) return DASH;
  const abs = Math.abs(n);
  const digits = abs >= 10000 ? 0 : abs >= 1 ? 2 : abs >= 0.01 ? 4 : 6;
  return withUnit(n, abs.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }), currency);
}

/**
 * A per-share estimate or target (EPS, price target): cents precision ("−$0.14",
 * "$0.50", "$0.04", "121.80p"); sub-dime values keep up to 4 decimals only when they
 * carry them ("$0.0123", never "$0.0400").
 */
export function perShare(n: number | null | undefined, currency?: string | null): string {
  if (!isNum(n)) return DASH;
  const abs = Math.abs(n);
  if (abs > 0 && abs < 0.0001) return price(n, currency);
  const [min, max] = abs >= 10000 ? [0, 0] : abs >= 0.1 ? [2, 2] : [2, 4];
  return withUnit(n, abs.toLocaleString("en-US", { minimumFractionDigits: min, maximumFractionDigits: max }), currency);
}

/**
 * Currency that EPS and revenue estimates are reported in — often not the quote's:
 * SHOP.TO quotes CAD but reports USD, ASML quotes USD but reports EUR, VOD.L quotes
 * pence but reports EUR. Uses the API's profile.financial_currency when present;
 * otherwise labels only the case that is safe to infer (a US company quoted in USD)
 * and returns null, so the number shows without a symbol rather than a wrong one.
 */
export function reportingCurrency(a: {
  quote: { currency: string | null } | null;
  profile: { country: string | null; financial_currency?: string | null } | null;
}): string | null {
  const stated = a.profile?.financial_currency;
  if (stated) return stated;
  const quoted = a.quote?.currency;
  return quoted === "USD" && a.profile?.country === "United States" ? "USD" : null;
}

/** Compact magnitude with ~3 significant digits: 4.87T, 33.3M, 996K, 812. */
export function compact(n: number | null | undefined): string {
  if (!isNum(n)) return DASH;
  const abs = Math.abs(n);
  const sign = n < 0 ? MINUS : "";
  const units: Array<[number, string]> = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  for (const [v, u] of units) {
    if (abs >= v * 0.9995) {
      const x = abs / v;
      const d = x >= 100 ? 0 : x >= 10 ? 1 : 2;
      return `${sign}${Number(x.toFixed(d))}${u}`;
    }
  }
  return `${sign}${Math.round(abs).toLocaleString("en-US")}`;
}

/**
 * Compact aggregate amount (market cap, deal value, revenue) in the MAJOR unit: a
 * pence-quoted listing's market cap is in pounds, so "GBp" reads as "£29.4B".
 */
export function money(n: number | null | undefined, currency?: string | null): string {
  if (!isNum(n)) return DASH;
  return `${n < 0 ? MINUS : ""}${symbolFor(currency)}${compact(Math.abs(n))}`;
}

export function int(n: number | null | undefined): string {
  if (!isNum(n)) return DASH;
  return Math.round(n).toLocaleString("en-US");
}

/* ------------------------------------------------------------------------- */
/* Dates                                                                       */
/* ------------------------------------------------------------------------- */

/**
 * Parse "YYYY-MM-DD" as a local calendar date (no timezone drift); ISO
 * datetimes as-is. Exactly-midnight-UTC datetimes are how date-only events
 * (earnings day, ex-dividend, filing date) arrive when the API types them as
 * datetimes — read those as calendar dates too, or the Americas see them a
 * day early.
 */
export function parseDate(s: string | null | undefined): Date | null {
  if (!s) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:T00:00(?::00(?:\.0+)?)?(?:Z|[+-]00:?00))?$/.exec(s);
  const d = m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function timeAgo(s: string | null | undefined, now = Date.now()): string {
  const d = parseDate(s);
  if (!d) return DASH;
  const sec = Math.round((now - d.getTime()) / 1000);
  if (sec < 0) return "just now";
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const h = Math.floor(min / 60);
  if (h < 24) return `${h}h ago`;
  const days = Math.floor(h / 24);
  if (days < 14) return `${days}d ago`;
  return shortDate(s);
}

/** "Oct 2" (adds the year when it differs from now). */
export function shortDate(s: string | null | undefined): string {
  const d = parseDate(s);
  if (!d) return DASH;
  const sameYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", ...(sameYear ? {} : { year: "numeric" }) });
}

/** "Oct 2, 2026". */
export function longDate(s: string | null | undefined): string {
  const d = parseDate(s);
  if (!d) return DASH;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

/** "Tue 14:02" within a week, else "Sep 12 14:02". */
export function dayTime(s: string | null | undefined, now = Date.now()): string {
  const d = parseDate(s);
  if (!d) return DASH;
  const time = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  const within = Math.abs(now - d.getTime()) < 6 * 864e5;
  const day = within
    ? d.toLocaleDateString("en-US", { weekday: "short" })
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  return `${day} ${time}`;
}

/**
 * Label for one bucket of a bucketed series spanning `spanMs`, cut every `stepMs`.
 * Weekday + time only pins a moment inside a window shorter than a week — a 7-day
 * window starts and ends on the same weekday and hour — so wider windows carry the
 * date. Daily buckets carry only their (UTC) calendar date; `detail` (tooltips)
 * always gives weekday and date.
 */
export function bucketTime(t: string, spanMs: number, stepMs: number, detail = false): string {
  const DAY = 864e5;
  if (stepMs >= DAY) {
    const d = parseDate(t.slice(0, 10));
    if (!d) return DASH;
    return d.toLocaleDateString("en-US", detail ? { weekday: "short", month: "short", day: "numeric" } : { month: "short", day: "numeric" });
  }
  const d = new Date(t);
  if (Number.isNaN(d.getTime())) return DASH;
  const time = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
  const day = detail
    ? d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" })
    : spanMs < 6 * DAY
      ? d.toLocaleDateString("en-US", { weekday: "short" })
      : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  return `${day} ${time}`;
}

/** Whole calendar days from today until the date (negative = past). */
export function daysUntil(s: string | null | undefined, now = new Date()): number | null {
  const d = parseDate(s);
  if (!d) return null;
  const a = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  const b = Date.UTC(d.getFullYear(), d.getMonth(), d.getDate());
  return Math.round((b - a) / 864e5);
}

export function countdown(days: number | null | undefined): string {
  if (!isNum(days)) return DASH;
  if (days === 0) return "today";
  if (days === 1) return "tomorrow";
  if (days === -1) return "yesterday";
  return days > 0 ? `in ${days} days` : `${-days} days ago`;
}

export function ms(n: number | null | undefined): string {
  if (!isNum(n)) return DASH;
  return n >= 1000 ? `${(n / 1000).toFixed(1)}s` : `${Math.round(n)}ms`;
}

/** 1 → "1st", 12 → "12th", 23 → "23rd". */
export function ordinal(n: number): string {
  const r = Math.round(n);
  const tens = Math.abs(r) % 100;
  const suffix = tens >= 11 && tens <= 13 ? "th" : (["th", "st", "nd", "rd"][Math.abs(r) % 10] ?? "th");
  return `${r}${suffix}`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${int(n)} ${n === 1 ? one : many}`;
}

/**
 * A provider-supplied link that is safe to put in an href: absolute http(s) only,
 * normalised by the URL parser. Feed URLs come from RSS <link>s, redirect params and
 * third-party APIs, and React 18 renders `javascript:` hrefs as-is in production — so
 * anything else (javascript:, data:, vbscript:, relative paths) is not linked at all.
 */
export function safeHref(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : undefined;
  } catch {
    return undefined;
  }
}

export function hostOf(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

export { isNum };
