/**
 * React Query hooks for league history data
 */

import { useQueries, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { leagueHistoryApi } from "@/lib/api";
import type { LeagueHistoricalSeries, HistoricalLeaguesResponse } from "@/types/api";

export const leagueHistoryKeys = {
  all: ["league-history"] as const,
  leagues: () => [...leagueHistoryKeys.all, "leagues"] as const,
  series: (currency: string, league: string) =>
    [...leagueHistoryKeys.all, "series", currency, league] as const,
};

/**
 * Fetch available historical leagues from the archive table.
 * Cached for 30 minutes — the league list rarely changes.
 */
export function useHistoricalLeagues(): UseQueryResult<HistoricalLeaguesResponse> {
  return useQuery({
    queryKey: leagueHistoryKeys.leagues(),
    queryFn: () => leagueHistoryApi.getHistoricalLeagues(),
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}

/**
 * Fetch historical price series for a set of previous leagues.
 *
 * Fires one independent React Query per league so:
 * - Each league's data is cached under its own stable key
 * - Adding a new league only fetches that league's data; the rest are served from cache
 * - All requests are issued in parallel (React Query batches concurrent useQueries calls)
 */
export function useLeagueHistoricalPrices(
  currency: string,
  leagues: string[],
  enabled = true
): { seriesMap: Record<string, LeagueHistoricalSeries | undefined>; isFetching: boolean } {
  const results = useQueries({
    queries: leagues.map((league) => ({
      queryKey: leagueHistoryKeys.series(currency, league),
      queryFn: () => leagueHistoryApi.getLeagueHistoricalPrices({ currency, league }),
      enabled: enabled && !!currency,
      staleTime: 10 * 60 * 1000,
      gcTime: 20 * 60 * 1000,
      refetchOnWindowFocus: false,
    })),
  });

  return {
    seriesMap: Object.fromEntries(leagues.map((league, i) => [league, results[i]?.data])),
    isFetching: results.some((r) => r.isFetching),
  };
}
