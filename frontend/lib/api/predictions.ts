import { apiClient } from "./client";
import type { PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, CurrencyMetadata, LeagueInfo } from "@/types/api";

/**
 * Currency metadata response type
 */
type CurrencyMetadataResponse = Record<string, Record<string, CurrencyMetadata>>;

/**
 * League info response type
 */
type LeagueInfoResponse = Record<string, LeagueInfo>;

/**
 * Get predictions for currencies
 */
export async function getPredictions(
  request: PredictionRequest
): Promise<PredictionResponse> {
  const response = await apiClient.post<PredictionResponse>("/predict/single", request);
  return response;
}

/**
 * Get batch predictions for multiple currencies
 */
export async function getBatchPredictions(
  request: BatchPredictionRequest
): Promise<BatchPredictionResponse> {
  const response = await apiClient.post<BatchPredictionResponse>("/predict/batch", request);
  return response;
}

/**
 * Get available currencies for predictions
 */
export async function getAvailableCurrencies(): Promise<CurrencyMetadataResponse> {
  const response = await apiClient.get<{ currencies: CurrencyMetadataResponse }>("/predict/currencies");
  return response.currencies;
}

/**
 * Get available leagues for predictions
 */
export async function getAvailableLeagues(): Promise<LeagueInfoResponse> {
  const response = await apiClient.get<{ leagues: LeagueInfoResponse }>("/predict/leagues");
  return response.leagues;
}


/**
 * Prediction API object
 */
export const predictionApi = {
  getPredictions,
  getBatchPredictions,
  getAvailableCurrencies,
  getAvailableLeagues
};