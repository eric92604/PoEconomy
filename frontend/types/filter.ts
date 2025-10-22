/**
 * Filter and sorting types for the application
 */

import { CurrencyCategory, CurrencySortField, SortDirection } from "./currency";

/**
 * Filter configuration for currencies
 */
export interface CurrencyFilters {
  // Text search
  search?: string;

  // League filter
  leagues?: string[];

  // Category filter
  categories?: CurrencyCategory[];

  // Confidence threshold (0-1)
  minConfidence?: number;

  // Profit threshold (percentage)
  minProfit?: number;

  // Price range
  minPrice?: number;
  maxPrice?: number;

  // Only show currencies with predictions
  onlyWithPredictions?: boolean;
}

/**
 * Sort configuration
 */
export interface SortConfig {
  field: CurrencySortField;
  direction: SortDirection;
}




