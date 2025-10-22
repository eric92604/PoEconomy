/**
 * React Query hook for latest predictions (dashboard data)
 */

import { useState, useEffect } from "react";
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
  enabled?: boolean;
} = {}): UseQueryResult<LatestPredictionsResponse> {
  return useQuery({
    queryKey: latestPredictionsKeys.byParams(params.league, params.horizons, params.limit),
    queryFn: () => latestPredictionsApi.getLatestPredictions(params),
    enabled: params.enabled !== false, // Only fetch when enabled
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    refetchInterval: false, // Disable automatic refetching to prevent multiple calls
    refetchOnWindowFocus: false, // Disable refetch on window focus
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

/**
 * Hook for paginated latest predictions with streaming-like behavior
 * Loads data in chunks to improve perceived performance
 */
export function usePaginatedLatestPredictions(params: {
  league?: string;
  horizons?: string[];
  pageSize?: number;
  maxPages?: number;
} = {}) {
  const [currentPage, setCurrentPage] = useState(0);
  const [allData, setAllData] = useState<Record<string, Record<string, any>>>({});
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  
  const pageSize = params.pageSize || 20;
  const maxPages = params.maxPages || 5;
  
  // Fetch current page
  const { data: currentPageData, isLoading, error } = useLatestPredictions({
    league: params.league,
    horizons: params.horizons || ["1d"],
    limit: pageSize,
  });
  
  // Load more data when current page loads
  useEffect(() => {
    if (currentPageData?.predictions && currentPage < maxPages) {
      setAllData(prev => ({
        ...prev,
        ...currentPageData.predictions
      }));
      
      // Auto-load next page after a short delay for streaming effect
      if (currentPage < maxPages - 1) {
        setIsLoadingMore(true);
        const timer = setTimeout(() => {
          setCurrentPage(prev => prev + 1);
          setIsLoadingMore(false);
        }, 500); // 500ms delay for streaming effect
        
        return () => clearTimeout(timer);
      }
    }
  }, [currentPageData, currentPage, maxPages]);
  
  const hasMore = currentPage < maxPages - 1;
  const totalLoaded = Object.keys(allData).length;
  
  return {
    data: {
      predictions: allData,
      metadata: currentPageData?.metadata
    },
    isLoading: isLoading && currentPage === 0,
    isLoadingMore,
    error,
    hasMore,
    totalLoaded,
    loadMore: () => setCurrentPage(prev => prev + 1),
    reset: () => {
      setCurrentPage(0);
      setAllData({});
      setIsLoadingMore(false);
    }
  };
}