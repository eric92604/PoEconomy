import { apiClient } from "./client";
import type { LatestPredictionsResponse } from "@/types/api";

/**
 * Get latest predictions for dashboard
 */
export async function getLatestPredictions(params: {
  league?: string;
  horizons?: string[];
  limit?: number;
} = {}): Promise<LatestPredictionsResponse> {
  const searchParams = new URLSearchParams();
  
  if (params.league) searchParams.append("league", params.league);
  if (params.horizons) searchParams.append("horizons", params.horizons.join(","));
  if (params.limit) searchParams.append("limit", params.limit.toString());
  
  const response = await apiClient.get<LatestPredictionsResponse>(`/predict/latest?${searchParams}`);
  return response;
}

/**
 * Latest predictions API object
 */
export const latestPredictionsApi = {
  getLatestPredictions,
};
