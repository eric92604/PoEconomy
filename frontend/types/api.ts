/**
 * Core API response types for PoEconomy backend
 */

/**
 * Prediction horizons supported by the API
 */
export type PredictionHorizon = "1d" | "3d" | "7d";

/**
 * Data source for predictions
 */
export type PredictionSource = "cache" | "live";

/**
 * Currency metadata from the API
 */
export interface CurrencyMetadata {
  currency_id: string;
  category: string;
  icon_url?: string;
  [key: string]: unknown;
}

/**
 * Response from /predict/currencies endpoint
 */
export interface CurrenciesResponse {
  currencies: Record<string, Record<string, CurrencyMetadata>>;
}

/**
 * League information
 */
export interface LeagueInfo {
  currency_count: number;
}

/**
 * Response from /predict/leagues endpoint
 */
export interface LeaguesResponse {
  leagues: Record<string, LeagueInfo>;
}

/**
 * Prediction data for a specific horizon
 */
export interface PredictionData {
  currency: string;
  league: string;
  horizon: string;
  current_price: number;
  predicted_price: number;
  price_change_percent: number;
  confidence: number;
  timestamp: number;
  source: PredictionSource;
  prediction_lower?: number;
  prediction_upper?: number;
}

/**
 * Latest predictions response for dashboard
 */
export interface LatestPredictionsResponse {
  predictions: Record<string, Record<string, PredictionData>>;
  metadata: {
    league: string;
    total_currencies: number;
    horizons_requested: string[];
    latest_prediction_time: string;
    query_efficiency: string;
  };
}

/**
 * Request body for single prediction
 */
export interface PredictionRequest {
  currency: string;
  league?: string;
  horizon?: PredictionHorizon;
}

/**
 * Prediction metadata
 */
export interface PredictionMetadata {
  model_version?: string;
  features_used?: number;
  last_training?: string;
  prediction_lower?: number;
  prediction_upper?: number;
}

/**
 * Single prediction response
 */
export interface PredictionResponse {
  currency: string;
  league: string;
  horizon: PredictionHorizon;
  predicted_price: number;
  current_price: number;
  price_change: number;
  price_change_percent: number;
  confidence: number;
  prediction_lower?: number;
  prediction_upper?: number;
  source: PredictionSource;
  timestamp: string;
  metadata?: PredictionMetadata;
}

/**
 * Batch prediction request
 */
export interface BatchPredictionRequest {
  requests: PredictionRequest[];
}

/**
 * Batch prediction response
 */
export interface BatchPredictionResponse {
  results: PredictionResponse[];
}

/**
 * Live price data point
 */
export interface LivePrice {
  currency: string;
  league: string;
  price: number;
  confidence: number;
  timestamp: number;
}

/**
 * Live prices metadata
 */
export interface LivePricesMetadata {
  total_count: number;
  returned_count: number;
  time_range_hours: number;
  filters: {
    currency?: string;
    league?: string;
  };
  last_updated: string;
  cache_info: {
    cached: boolean;
    cache_ttl_minutes: number;
  };
}

/**
 * Response from /prices/live endpoint
 */
export interface LivePricesResponse {
  prices: LivePrice[];
  metadata: LivePricesMetadata;
}

/**
 * Query parameters for live prices
 */
export interface LivePricesParams {
  currency?: string;
  league?: string;
  limit?: number;
  hours?: number;
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
}

/**
 * API error response
 */
export interface ApiError {
  message: string;
  detail?: string;
  status?: number;
}

/**
 * Generic API response wrapper
 */
export type ApiResponse<T> = {
  data: T;
  error: null;
} | {
  data: null;
  error: ApiError;
};

/**
 * A single price point in a historical league series
 */
export interface LeagueHistoricalPrice {
  date: string;
  avg_price: number;
}

/**
 * Data for one league within a league-history response
 */
export interface LeagueHistoricalSeries {
  league_start_date: string | null;
  count: number;
  prices: LeagueHistoricalPrice[];
  error?: string;
}

/**
 * Response from GET /prices/league-history
 */
export interface LeagueHistoricalResponse {
  currency: string;
  leagues: Record<string, LeagueHistoricalSeries>;
}

/**
 * Response from GET /prices/leagues
 */
export interface HistoricalLeaguesResponse {
  leagues: Record<string, { league_start_date: string | null }>;
}

