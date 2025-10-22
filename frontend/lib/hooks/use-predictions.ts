/**
 * React Query hooks for prediction data
 */

import { useQuery, useMutation, type UseQueryResult } from "@tanstack/react-query";
import { predictionApi } from "@/lib/api";
import type {
  PredictionRequest,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
} from "@/types";

/**
 * Query keys for predictions
 */
export const predictionKeys = {
  all: ["predictions"] as const,
  single: (currency: string, league: string, horizon: string) =>
    [...predictionKeys.all, "single", currency, league, horizon] as const,
  batch: (requests: PredictionRequest[]) =>
    [...predictionKeys.all, "batch", JSON.stringify(requests)] as const,
};

/**
 * Hook to fetch a single prediction
 */
export function useSinglePrediction(
  request: PredictionRequest,
  enabled = true
): UseQueryResult<PredictionResponse> {
  return useQuery({
    queryKey: predictionKeys.single(
      request.currency,
      request.league || "default",
      request.horizon || "1d"
    ),
    queryFn: () => predictionApi.getPredictions(request),
    enabled: enabled && !!request.currency,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Hook to fetch batch predictions
 */
export function useBatchPredictions(
  request: BatchPredictionRequest,
  enabled = true
): UseQueryResult<BatchPredictionResponse> {
  return useQuery({
    queryKey: predictionKeys.batch(request.requests),
    queryFn: () => predictionApi.getBatchPredictions(request),
    enabled: enabled && request.requests.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Mutation hook for fetching a single prediction on demand
 */
export function usePredictionMutation() {
  return useMutation({
    mutationFn: (request: PredictionRequest) => predictionApi.getPredictions(request),
  });
}

/**
 * Mutation hook for fetching batch predictions on demand
 */
export function useBatchPredictionMutation() {
  return useMutation({
    mutationFn: (request: BatchPredictionRequest) => predictionApi.getBatchPredictions(request),
  });
}



