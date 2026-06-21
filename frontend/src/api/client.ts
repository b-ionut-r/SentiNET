import type { AnalyzeResponse, PriceResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function fetchAnalysis(ticker: string, refresh = false): Promise<AnalyzeResponse> {
  const q = refresh ? "?refresh=true" : "";
  return getJSON<AnalyzeResponse>(`/api/analyze/${encodeURIComponent(ticker)}${q}`);
}

export function fetchPrice(ticker: string, range: string): Promise<PriceResponse> {
  return getJSON<PriceResponse>(
    `/api/price/${encodeURIComponent(ticker)}?range=${encodeURIComponent(range)}`,
  );
}
