import { useQuery } from "@tanstack/react-query";

import { fetchAnalysis, fetchPrice } from "../api/client";

export function useAnalysis(ticker: string | null) {
  return useQuery({
    queryKey: ["analysis", ticker],
    queryFn: () => fetchAnalysis(ticker as string),
    enabled: !!ticker,
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}

export function usePrice(ticker: string | null, range: string) {
  return useQuery({
    queryKey: ["price", ticker, range],
    queryFn: () => fetchPrice(ticker as string, range),
    enabled: !!ticker,
    staleTime: 2 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}
