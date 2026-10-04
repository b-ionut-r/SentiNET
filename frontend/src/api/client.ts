/**
 * Typed fetchers for the SentiNET API (`/api/*`) plus the SSE helper used by
 * the Intel page's live scan. All functions throw `ApiError` on failure.
 */
import type {
  AlertEvent,
  AlertRule,
  AlertRuleIn,
  Analysis,
  HealthResponse,
  HistoryResponse,
  MarketOverview,
  PriceRange,
  PriceResponse,
  ProgressEvent,
  ScoreRequest,
  ScoreResponse,
  Snapshot,
  SourceInfo,
  SymbolMatch,
  WatchItem,
} from "./types";

const BASE: string = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function detailOf(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d) && d[0] && typeof d[0] === "object" && "msg" in d[0]) return String((d[0] as { msg: unknown }).msg);
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit & { signal?: AbortSignal }): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("Can't reach the SentiNET API. Is the backend running?", 0);
  }
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detailOf(body, `${res.status} ${res.statusText || "error"}`), res.status);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  analyze: (ticker: string, refresh = false, signal?: AbortSignal) =>
    request<Analysis>(`/api/analyze/${enc(ticker)}${refresh ? "?refresh=true" : ""}`, { signal }),
  price: (ticker: string, range: PriceRange, signal?: AbortSignal) =>
    request<PriceResponse>(`/api/price/${enc(ticker)}?range=${enc(range)}`, { signal }),
  history: (ticker: string, days = 90, signal?: AbortSignal) =>
    request<HistoryResponse>(`/api/history/${enc(ticker)}?days=${days}`, { signal }),
  market: (signal?: AbortSignal) => request<MarketOverview>("/api/market", { signal }),
  search: (q: string, signal?: AbortSignal) => request<SymbolMatch[]>(`/api/search?q=${enc(q)}`, { signal }),
  labScore: (body: ScoreRequest) => request<ScoreResponse>("/api/lab/score", { method: "POST", body: JSON.stringify(body) }),
  watchlist: (signal?: AbortSignal) => request<WatchItem[]>("/api/watchlist", { signal }),
  watchAdd: (ticker: string) => request<WatchItem[]>("/api/watchlist", { method: "POST", body: JSON.stringify({ ticker }) }),
  watchRemove: (ticker: string) => request<WatchItem[]>(`/api/watchlist/${enc(ticker)}`, { method: "DELETE" }),
  snapshots: (ticker: string, limit = 60, signal?: AbortSignal) =>
    request<Snapshot[]>(`/api/snapshots/${enc(ticker)}?limit=${limit}`, { signal }),
  alerts: (signal?: AbortSignal) => request<AlertRule[]>("/api/alerts", { signal }),
  alertCreate: (rule: AlertRuleIn) => request<AlertRule>("/api/alerts", { method: "POST", body: JSON.stringify(rule) }),
  alertDelete: (id: number) => request<unknown>(`/api/alerts/${id}`, { method: "DELETE" }),
  alertEvents: (limit = 50, signal?: AbortSignal) => request<AlertEvent[]>(`/api/alerts/events?limit=${limit}`, { signal }),
  health: (signal?: AbortSignal) => request<HealthResponse>("/api/health", { signal }),
  sources: (signal?: AbortSignal) => request<SourceInfo[]>("/api/sources", { signal }),
  exportUrl: (ticker: string, format: "csv" | "json") => `${BASE}/api/export/${enc(ticker)}.${format}`,
};

/* ------------------------------------------------------------------------- */
/* Streaming analysis (Server-Sent Events)                                    */
/* ------------------------------------------------------------------------- */

export interface StreamOptions {
  refresh?: boolean;
  signal?: AbortSignal;
  onProgress?: (ev: ProgressEvent) => void;
}

function parseJSON<T>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

/**
 * Run an analysis over `/api/analyze/{t}/stream`, reporting each progress
 * event. A server-sent `error` rejects with its message; a transport failure
 * (no SSE support, proxy buffering, dropped connection) falls back to the
 * plain GET so the user still gets a result.
 */
export function streamAnalysis(ticker: string, opts: StreamOptions = {}): Promise<Analysis> {
  const { refresh = false, signal, onProgress } = opts;
  if (typeof EventSource === "undefined") return api.analyze(ticker, refresh, signal);

  return new Promise<Analysis>((resolve, reject) => {
    const url = `${BASE}/api/analyze/${enc(ticker)}/stream${refresh ? "?refresh=true" : ""}`;
    const es = new EventSource(url);
    let settled = false;

    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      es.close();
      signal?.removeEventListener("abort", onAbort);
      fn();
    };
    const onAbort = () => finish(() => reject(new DOMException("Aborted", "AbortError")));
    signal?.addEventListener("abort", onAbort);

    es.addEventListener("progress", (e) => {
      const ev = parseJSON<ProgressEvent>((e as MessageEvent<string>).data);
      if (ev && !settled) onProgress?.(ev);
    });
    es.addEventListener("result", (e) => {
      const data = parseJSON<Analysis>((e as MessageEvent<string>).data);
      if (data) finish(() => resolve(data));
      else finish(() => reject(new ApiError("The analysis stream returned malformed data.", 502)));
    });
    es.addEventListener("error", (e) => {
      const raw = e instanceof MessageEvent && typeof e.data === "string" ? e.data : null;
      if (raw) {
        // Server-sent `event: error` — a real failure (e.g. unknown ticker).
        const body = parseJSON<unknown>(raw);
        const msg = body ? detailOf(body, raw) : raw;
        const status = body && typeof body === "object" && "status" in body ? Number((body as { status: unknown }).status) || 400 : 400;
        finish(() => reject(new ApiError(msg, status)));
        return;
      }
      // Transport error: fall back to the plain request (shares the server's single-flight run).
      finish(() => {
        api.analyze(ticker, refresh, signal).then(resolve, reject);
      });
    });
  });
}
