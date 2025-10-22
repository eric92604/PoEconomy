/**
 * React Query hooks for currency data
 */

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { currencyApi } from "@/lib/api";
import type { CurrenciesResponse, LeaguesResponse } from "@/types";

/**
 * Query keys for currencies
 */
export const currencyKeys = {
  all: ["currencies"] as const,
  list: () => [...currencyKeys.all, "list"] as const,
  leagues: () => ["leagues"] as const,
};

/**
 * Hook to fetch all currencies
 * Metadata changes rarely, so we cache for longer
 */
export function useCurrencies(): UseQueryResult<CurrenciesResponse> {
  return useQuery({
    queryKey: currencyKeys.list(),
    queryFn: () => currencyApi.getCurrencies(),
    staleTime: 30 * 60 * 1000, // 30 minutes - metadata rarely changes
    gcTime: 60 * 60 * 1000, // 1 hour
  });
}

/**
 * Hook to fetch all leagues
 * Metadata changes rarely, so we cache for longer
 */
export function useLeagues(): UseQueryResult<LeaguesResponse> {
  return useQuery({
    queryKey: currencyKeys.leagues(),
    queryFn: () => currencyApi.getLeagues(),
    staleTime: 30 * 60 * 1000, // 30 minutes - metadata rarely changes
    gcTime: 60 * 60 * 1000, // 1 hour
  });
}



