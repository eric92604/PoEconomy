/**
 * API constants and configuration
 */

/**
 * Get the API base URL from environment variables
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://api.poeconomy.com";

/**
 * Get the API key from environment variables
 */
export const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  HEALTH: "/health",
  CURRENCIES: "/predict/currencies",
  LEAGUES: "/predict/leagues",
  PREDICT_SINGLE: "/predict/single",
  PREDICT_BATCH: "/predict/batch",
  PRICES_LIVE: "/prices/live",
} as const;

/**
 * API request timeout in milliseconds
 */
export const API_TIMEOUT = 15000;

/**
 * Maximum retries for failed requests
 */
export const MAX_RETRIES = 1;

/**
 * Retry delay in milliseconds (exponential backoff)
 */
export const RETRY_DELAY = 1000;

/**
 * Default cache time for React Query (30 minutes)
 * Longer cache for historical data to reduce API calls
 */
export const DEFAULT_CACHE_TIME = 30 * 60 * 1000;

/**
 * Default stale time for React Query (15 minutes)
 * Data is considered fresh for 15 minutes to reduce unnecessary requests
 */
export const DEFAULT_STALE_TIME = 15 * 60 * 1000;

/**
 * Maximum items per batch request
 */
export const MAX_BATCH_SIZE = 50;



