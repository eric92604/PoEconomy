import { apiClient } from "./client";
import type { HealthResponse } from "@/types/api";

/**
 * Check API health
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/health");
  return response;
}

/**
 * Health API object
 */
export const healthApi = {
  checkHealth
};