/**
 * React Query hook for latest predictions (dashboard data)
 */

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { latestPredictionsApi } from "@/lib/api";
import type { LatestPredictionsResponse } from "@/types/api";

/**
 * Query keys for latest predictions
 */
export const latestPredictionsKeys = {
  all: ["latest-predictions"] as const,
  byParams: (league?: string, horizons?: string[], limit?: number) =>
    [...latestPredictionsKeys.all, "by-params", league, horizons, limit] as const,
};

/**
 * Hook to fetch latest predictions for dashboard
 */
export function useLatestPredictions(params: {
  league?: string;
  horizons?: string[];
  limit?: number;
} = {}): UseQueryResult<LatestPredictionsResponse> {
  return useQuery({
    queryKey: latestPredictionsKeys.byParams(params.league, params.horizons, params.limit),
    queryFn: () => latestPredictionsApi.getLatestPredictions(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
  });
}

/**
 * Hook to fetch latest predictions for a specific league
 */
export function useLatestPredictionsByLeague(
  league: string,
  options: {
    horizons?: string[];
    limit?: number;
  } = {}
): UseQueryResult<LatestPredictionsResponse> {
  return useLatestPredictions({
    league,
    horizons: options.horizons || ["1d", "3d", "7d"],
    limit: options.limit || 100,
  });
}
