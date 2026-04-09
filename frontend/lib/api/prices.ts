import { apiClient } from "./client";
import type { LivePricesResponse } from "@/types/api";

/**
 * Historical price data point
 */
export interface HistoricalPrice {
  date: string;
  avg_price: number;
}

/**
 * Historical prices response
 */
export interface HistoricalPricesResponse {
  currency: string;
  league: string;
  start_date: string;
  end_date?: string;
  count: number;
  prices: HistoricalPrice[];
  source: string;
  last_updated: string;
  error?: string;
}

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
 * Get historical prices with parameters
 */
export async function getHistoricalPrices(params: {
  currency: string;
  league: string;
  start_date?: string;
  end_date?: string;
  limit?: number;
}): Promise<HistoricalPricesResponse> {
  const searchParams = new URLSearchParams();
  
  searchParams.append("currency", params.currency);
  searchParams.append("league", params.league);
  if (params.start_date) searchParams.append("start_date", params.start_date);
  if (params.end_date) searchParams.append("end_date", params.end_date);
  if (params.limit) searchParams.append("limit", params.limit.toString());
  
  const response = await apiClient.get<HistoricalPricesResponse>(`/prices/history?${searchParams}`);
  return response;
}

/**
 * Price API object
 */
export const priceApi = {
  getLivePrices,
  getHistoricalPrices
};