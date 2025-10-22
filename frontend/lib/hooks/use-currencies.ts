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
 */
export function useCurrencies(): UseQueryResult<CurrenciesResponse> {
  return useQuery({
    queryKey: currencyKeys.list(),
    queryFn: () => currencyApi.getCurrencies(),
  });
}

/**
 * Hook to fetch all leagues
 */
export function useLeagues(): UseQueryResult<LeaguesResponse> {
  return useQuery({
    queryKey: currencyKeys.leagues(),
    queryFn: () => currencyApi.getLeagues(),
  });
}



