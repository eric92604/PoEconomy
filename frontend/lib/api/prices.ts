import { apiClient } from "./client";
import type { LivePricesResponse } from "@/types/api";


/**
 * Get live prices with parameters
 */
export async function getLivePrices(params: {
  currency?: string;
  league?: string;
  limit?: number;
  hours?: number;
} = {}): Promise<LivePricesResponse> {
  const searchParams = new URLSearchParams();
  
  if (params.currency) searchParams.append("currency", params.currency);
  if (params.league) searchParams.append("league", params.league);
  if (params.limit) searchParams.append("limit", params.limit.toString());
  if (params.hours) searchParams.append("hours", params.hours.toString());
  
  const response = await apiClient.get<LivePricesResponse>(`/prices/live?${searchParams}`);
  return response;
}

/**
 * Price API object
 */
export const priceApi = {
  getLivePrices
};