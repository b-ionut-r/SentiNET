/**
 * Defensive localStorage helpers. Storage can be unavailable (private mode,
 * blocked site data), so every access is wrapped and falls back silently.
 */
export function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

export function writeJSON(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable — non-critical */
  }
}

const RECENT_KEY = "sentinet.recent";
const RECENT_MAX = 8;

/** Recently opened tickers, most recent first. */
export function getRecent(): string[] {
  const list = readJSON<unknown>(RECENT_KEY, []);
  return Array.isArray(list) ? list.filter((x): x is string => typeof x === "string").slice(0, RECENT_MAX) : [];
}

export function pushRecent(ticker: string): void {
  const t = ticker.toUpperCase();
  writeJSON(RECENT_KEY, [t, ...getRecent().filter((x) => x !== t)].slice(0, RECENT_MAX));
}
