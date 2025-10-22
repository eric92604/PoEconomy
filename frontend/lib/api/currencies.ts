import { apiClient } from "./client";
import type { CurrencyMetadata } from "@/types/api";

/**
 * Currency metadata type for API responses
 */
type CurrencyMetadataResponse = Record<string, Record<string, CurrencyMetadata>>;

/**
 * Fetch all available currencies
 */
export async function getCurrencies(): Promise<{ currencies: CurrencyMetadataResponse }> {
  const response = await apiClient.get<{ currencies: CurrencyMetadataResponse }>("/predict/currencies");
  return response;
}


/**
 * Currency API object
 */
export const currencyApi = {
  getCurrencies,
  getLeagues: async () => {
    const response = await apiClient.get<{ leagues: Record<string, { currency_count: number }> }>("/predict/leagues");
    return response;
  }
};