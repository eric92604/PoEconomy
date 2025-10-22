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
export const API_TIMEOUT = 30000;

/**
 * Maximum retries for failed requests
 */
export const MAX_RETRIES = 3;

/**
 * Retry delay in milliseconds (exponential backoff)
 */
export const RETRY_DELAY = 1000;

/**
 * Default cache time for React Query (10 minutes)
 * This is longer than browser cache to avoid unnecessary re-fetches
 */
export const DEFAULT_CACHE_TIME = 10 * 60 * 1000;

/**
 * Default stale time for React Query (5 minutes)
 * Data is considered fresh for 5 minutes, matching browser cache
 */
export const DEFAULT_STALE_TIME = 5 * 60 * 1000;

/**
 * Maximum items per batch request
 */
export const MAX_BATCH_SIZE = 50;



