/** TanStack Query hooks, one per resource. */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

import { type BusyWait, retryWhenBusy } from "./busy";
import { api, streamAnalysis } from "./client";
import { HISTORY_TONE_RETRIES, HISTORY_TONE_RETRY_MS, historyTonePending, TONE_FOLLOW_UP_MS, toneArrivedViaHistory, tonePending } from "./pending";
import type { AlertRuleIn, Analysis, PriceRange, ProgressEvent, ScoreRequest, WatchItem } from "./types";

const MIN = 60_000;

export const keys = {
  analysis: (t: string) => ["analysis", t] as const,
  price: (t: string, r: PriceRange) => ["price", t, r] as const,
  history: (t: string, d: number) => ["history", t, d] as const,
  market: ["market"] as const,
  search: (q: string) => ["search", q] as const,
  watchlist: ["watchlist"] as const,
  snapshots: (t: string) => ["snapshots", t] as const,
  alerts: ["alerts"] as const,
  alertEvents: ["alert-events"] as const,
  sources: ["sources"] as const,
  health: ["health"] as const,
};

/** A progress event plus its arrival order (the latest update per task wins). */
export type TrackedProgress = ProgressEvent & { seq: number };

/**
 * The Intel page's analysis: streamed over SSE with live progress, cached per
 * ticker. `refresh()` forces a fresh server run while the previous result
 * stays on screen.
 */
export function useAnalysis(ticker: string) {
  const [progress, setProgress] = useState<Record<string, TrackedProgress>>({});
  const seq = useRef(0);
  const refreshNext = useRef(false);
  const qc = useQueryClient();

  useEffect(() => setProgress({}), [ticker]);

  const query = useQuery({
    queryKey: keys.analysis(ticker),
    queryFn: ({ signal }) => {
      const refresh = refreshNext.current;
      refreshNext.current = false;
      setProgress({});
      return streamAnalysis(ticker, {
        refresh,
        signal,
        onProgress: (ev) => setProgress((p) => ({ ...p, [`${ev.stage}:${ev.key}`]: { ...ev, seq: ++seq.current } })),
      });
    },
    staleTime: 5 * MIN,
    gcTime: 30 * MIN,
    retry: false,
    refetchOnWindowFocus: false,
    enabled: ticker.length > 0,
  });

  const refresh = useCallback(() => {
    refreshNext.current = true;
    void query.refetch();
    void qc.invalidateQueries({ queryKey: keys.history(ticker, 90) });
    void qc.invalidateQueries({ queryKey: ["price", ticker] });
  }, [query, qc, ticker]);

  return { ...query, progress: Object.values(progress), refresh };
}

/**
 * A result that went out while GDELT was still loading is superseded on the server
 * once the tone lands. Re-read it quietly (plain GET, no forced refresh, no dimming)
 * on a short schedule and swap it in as soon as a read carries the tone; stop on
 * ticker change, a manual refresh or the last attempt.
 */
export function useLateToneFollowUp(ticker: string, data: Analysis | undefined, progress: ProgressEvent[], busy: boolean) {
  const qc = useQueryClient();
  // Observes the tone ↔ price history the page loads anyway (never fetches it itself): once
  // that call has tone and this result does not, the server has superseded it — re-read now.
  const history = useHistory(ticker, 90, false).data;
  const arrived = toneArrivedViaHistory(data, history);
  const pending = !!data && data.ticker === ticker && (arrived || tonePending(data, progress));
  const stamp = data?.generated_at;
  useEffect(() => {
    if (!pending || busy) return;
    const ctrl = new AbortController();
    let timer = 0;
    const attempt = (i: number) => {
      api
        .analyze(ticker, false, ctrl.signal)
        .then((fresh) => {
          if (ctrl.signal.aborted) return;
          if (fresh.tone?.series.length) qc.setQueryData(keys.analysis(ticker), fresh);
          else if (i + 1 < TONE_FOLLOW_UP_MS.length) timer = window.setTimeout(() => attempt(i + 1), TONE_FOLLOW_UP_MS[i + 1] - TONE_FOLLOW_UP_MS[i]);
        })
        .catch(() => undefined); // best effort: the page already shows a complete result
    };
    timer = window.setTimeout(() => attempt(0), arrived ? 0 : TONE_FOLLOW_UP_MS[0]);
    return () => {
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [pending, arrived, busy, ticker, stamp, qc]);
}

/** Plain (non-streamed) analysis — shares the Intel page's cache entry. */
export function useAnalysisPlain(ticker: string, enabled = true) {
  return useQuery({
    queryKey: keys.analysis(ticker),
    queryFn: ({ signal }) => api.analyze(ticker, false, signal),
    staleTime: 5 * MIN,
    gcTime: 30 * MIN,
    retry: false,
    refetchOnWindowFocus: false,
    enabled: enabled && ticker.length > 0,
  });
}

export function usePrice(ticker: string, range: PriceRange) {
  return useQuery({
    queryKey: keys.price(ticker, range),
    queryFn: ({ signal }) => api.price(ticker, range, signal),
    staleTime: range === "1D" ? MIN : 10 * MIN,
    placeholderData: keepPreviousData,
    retry: 1,
  });
}

/**
 * Tone ↔ price history. A response that went out while its GDELT call was still
 * running (the server keeps it alive) is re-read a few times on its own, so the tone
 * pane and the lead/lag readout fill in without a manual Retry.
 */
export function useHistory(ticker: string, days = 90, enabled = true) {
  return useQuery({
    queryKey: keys.history(ticker, days),
    queryFn: ({ signal }) => api.history(ticker, days, signal),
    staleTime: 30 * MIN,
    retry: 1,
    enabled,
    refetchInterval: (q) => (historyTonePending(q.state.data) && q.state.dataUpdateCount <= HISTORY_TONE_RETRIES ? HISTORY_TONE_RETRY_MS : false),
  });
}

export function useMarket() {
  return useQuery({
    queryKey: keys.market,
    queryFn: ({ signal }) => api.market(signal),
    staleTime: 2 * MIN,
    refetchInterval: 5 * MIN,
    retry: 1,
  });
}

export function useSearch(q: string) {
  const term = q.trim();
  return useQuery({
    queryKey: keys.search(term.toLowerCase()),
    queryFn: ({ signal }) => api.search(term, signal),
    enabled: term.length > 0,
    staleTime: 5 * MIN,
    placeholderData: keepPreviousData,
    retry: false,
  });
}

export function useHealth() {
  return useQuery({ queryKey: keys.health, queryFn: ({ signal }) => api.health(signal), staleTime: 10 * MIN, retry: 1 });
}

export function useSources() {
  return useQuery({ queryKey: keys.sources, queryFn: ({ signal }) => api.sources(signal), staleTime: 10 * MIN, retry: 1 });
}

export function useSnapshots(ticker: string, enabled = true) {
  return useQuery({
    queryKey: keys.snapshots(ticker),
    queryFn: ({ signal }) => api.snapshots(ticker, 60, signal),
    staleTime: 5 * MIN,
    enabled,
  });
}

/* ------------------------------------------------------------------------- */
/* Watchlist & alerts                                                          */
/* ------------------------------------------------------------------------- */

export function useWatchlist() {
  return useQuery({ queryKey: keys.watchlist, queryFn: ({ signal }) => api.watchlist(signal), staleTime: MIN });
}

export function useWatchToggle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ ticker, watched }: { ticker: string; watched: boolean }) =>
      watched ? api.watchRemove(ticker) : api.watchAdd(ticker),
    onSuccess: (list: WatchItem[] | undefined) => {
      if (Array.isArray(list)) qc.setQueryData(keys.watchlist, list);
      void qc.invalidateQueries({ queryKey: keys.watchlist });
    },
  });
}

export function useAlerts() {
  return useQuery({ queryKey: keys.alerts, queryFn: ({ signal }) => api.alerts(signal), staleTime: MIN });
}

export function useAlertEvents() {
  return useQuery({
    queryKey: keys.alertEvents,
    queryFn: ({ signal }) => api.alertEvents(50, signal),
    staleTime: MIN,
    refetchInterval: 2 * MIN,
  });
}

export function useAlertCreate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (rule: AlertRuleIn) => api.alertCreate(rule),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.alerts }),
  });
}

export function useAlertDelete() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.alertDelete(id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.alerts }),
  });
}

/**
 * Sentiment-lab run. A busy lab (503 + Retry-After) is waited out and retried up to
 * `BUSY_RETRIES` times; `busy` describes the pending retry for a countdown, and
 * `cancel()` stops the wait (or the request) and returns the hook to idle. A newer run
 * or leaving the page aborts the older one.
 */
export function useLabScore() {
  const [busy, setBusy] = useState<BusyWait | null>(null);
  const ctrl = useRef<AbortController | null>(null);
  const mutation = useMutation({
    mutationFn: (req: ScoreRequest) => {
      ctrl.current?.abort();
      const own = new AbortController();
      ctrl.current = own;
      const mine = (w: BusyWait | null) => ctrl.current === own && setBusy(w);
      return retryWhenBusy((signal) => api.labScore(req, signal), { signal: own.signal, onWait: mine }).finally(() => mine(null));
    },
  });
  const { reset } = mutation;
  const cancel = useCallback(() => {
    ctrl.current?.abort();
    ctrl.current = null;
    setBusy(null);
    reset();
  }, [reset]);
  useEffect(() => () => ctrl.current?.abort(), []);
  return { ...mutation, busy, cancel };
}

export type { Analysis };
