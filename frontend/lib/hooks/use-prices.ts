/**
 * React Query hooks for price data
 */

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { priceApi } from "@/lib/api";
import type { LivePricesParams, LivePricesResponse } from "@/types";

/**
 * Query keys for prices
 */
export const priceKeys = {
  all: ["prices"] as const,
  live: (params: LivePricesParams) => [...priceKeys.all, "live", params] as const,
};

/**
 * Hook to fetch live prices
 */
export function useLivePrices(
  params: LivePricesParams = {},
  enabled = true
): UseQueryResult<LivePricesResponse> {
  return useQuery({
    queryKey: priceKeys.live(params),
    queryFn: () => priceApi.getLivePrices(params),
    enabled,
  });
}

/**
 * Hook to fetch live prices with auto-refresh
 */
export function useLivePricesWithRefresh(
  params: LivePricesParams = {},
  refetchInterval = 5 * 60 * 1000 // 5 minutes
): UseQueryResult<LivePricesResponse> {
  return useQuery({
    queryKey: priceKeys.live(params),
    queryFn: () => priceApi.getLivePrices(params),
    refetchInterval,
  });
}



