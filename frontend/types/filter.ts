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

/**
 * Filter preset for quick access
 */
export interface FilterPreset {
  id: string;
  name: string;
  description: string;
  filters: CurrencyFilters;
  sort?: SortConfig;
}

/**
 * Predefined filter presets
 */
export const DEFAULT_FILTER_PRESETS: FilterPreset[] = [
  {
    id: "high-profit",
    name: "High Profit",
    description: "Currencies with >5% predicted profit",
    filters: {
      minProfit: 5,
      minConfidence: 0.7,
    },
    sort: {
      field: "profit_1d",
      direction: "desc",
    },
  },
  {
    id: "safe-bets",
    name: "Safe Bets",
    description: "High confidence predictions",
    filters: {
      minConfidence: 0.85,
      minProfit: 1,
    },
    sort: {
      field: "average_confidence",
      direction: "desc",
    },
  },
  {
    id: "long-term",
    name: "Long Term",
    description: "Best 7-day predictions",
    filters: {
      minConfidence: 0.7,
      minProfit: 3,
    },
    sort: {
      field: "profit_7d",
      direction: "desc",
    },
  },
  {
    id: "quick-flip",
    name: "Quick Flip",
    description: "Best 1-day opportunities",
    filters: {
      minConfidence: 0.75,
      minProfit: 2,
    },
    sort: {
      field: "profit_1d",
      direction: "desc",
    },
  },
];



