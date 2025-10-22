/**
 * Data validation utilities
 */

import { z } from "zod";

/**
 * Validate prediction horizon
 */
export function isValidHorizon(horizon: string): horizon is "1d" | "3d" | "7d" {
  return ["1d", "3d", "7d"].includes(horizon);
}

/**
 * Validate confidence value (0-1)
 */
export function isValidConfidence(confidence: number): boolean {
  return confidence >= 0 && confidence <= 1;
}

/**
 * Validate price value (must be positive)
 */
export function isValidPrice(price: number): boolean {
  return price > 0 && isFinite(price);
}

/**
 * Validate percentage value
 */
export function isValidPercentage(percentage: number): boolean {
  return isFinite(percentage);
}

/**
 * Validate timestamp (Unix timestamp in seconds)
 */
export function isValidTimestamp(timestamp: number): boolean {
  return timestamp > 0 && timestamp < Date.now() / 1000 + 365 * 24 * 60 * 60; // Within next year
}

/**
 * Zod schema for prediction request
 */
export const predictionRequestSchema = z.object({
  currency: z.string().min(1, "Currency name is required"),
  league: z.string().optional(),
  horizon: z.enum(["1d", "3d", "7d"]).optional(),
});

/**
 * Zod schema for batch prediction request
 */
export const batchPredictionRequestSchema = z.object({
  requests: z
    .array(predictionRequestSchema)
    .min(1, "At least one request is required")
    .max(50, "Maximum 50 requests per batch"),
});

/**
 * Zod schema for live prices params
 */
export const livePricesParamsSchema = z.object({
  currency: z.string().optional(),
  league: z.string().optional(),
  limit: z.number().int().min(1).max(1000).optional(),
  hours: z.number().int().min(1).max(168).optional(),
});

/**
 * Zod schema for currency filters
 */
export const currencyFiltersSchema = z.object({
  search: z.string().optional(),
  leagues: z.array(z.string()).optional(),
  categories: z.array(z.string()).optional(),
  minConfidence: z.number().min(0).max(1).optional(),
  minProfit: z.number().optional(),
  minPrice: z.number().positive().optional(),
  maxPrice: z.number().positive().optional(),
  onlyWithPredictions: z.boolean().optional(),
});

/**
 * Validate and parse prediction request
 */
export function validatePredictionRequest(data: unknown) {
  return predictionRequestSchema.safeParse(data);
}

/**
 * Validate and parse batch prediction request
 */
export function validateBatchPredictionRequest(data: unknown) {
  return batchPredictionRequestSchema.safeParse(data);
}

/**
 * Validate and parse live prices params
 */
export function validateLivePricesParams(data: unknown) {
  return livePricesParamsSchema.safeParse(data);
}

/**
 * Validate and parse currency filters
 */
export function validateCurrencyFilters(data: unknown) {
  return currencyFiltersSchema.safeParse(data);
}

/**
 * Sanitize string input (remove dangerous characters)
 */
export function sanitizeString(input: string): string {
  return input.replace(/[<>\"']/g, "");
}

/**
 * Validate currency name (alphanumeric, spaces, and common punctuation)
 */
export function isValidCurrencyName(name: string): boolean {
  return /^[a-zA-Z0-9\s\-'\.]+$/.test(name);
}

/**
 * Validate league name
 */
export function isValidLeagueName(name: string): boolean {
  return /^[a-zA-Z0-9\s\-_]+$/.test(name);
}



