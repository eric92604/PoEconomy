/**
 * Currency-related types for the application
 */

import { PredictionHorizon } from "./api";

/**
 * Currency with enriched data for display
 */
export interface Currency {
  name: string;
  category: string;
  icon_url?: string;
  leagues: string[];
}

/**
 * Currency with current price and prediction data
 */
export interface CurrencyWithPredictions {
  currency: string;
  league: string;
  current_price: number;
  icon_url?: string;
  category?: string;
  predictions: {
    "1d"?: CurrencyPrediction;
    "3d"?: CurrencyPrediction;
    "7d"?: CurrencyPrediction;
  };
  average_confidence: number;
}

/**
 * Individual currency prediction
 */
export interface CurrencyPrediction {
  predicted_price: number;
  price_change_percent: number;
  confidence: number;
  prediction_lower?: number;
  prediction_upper?: number;
  horizon: PredictionHorizon;
}

/**
 * Currency selection state
 */
export interface CurrencySelection {
  currency: string;
  league: string;
}

/**
 * Currency price history point
 */
export interface PriceHistoryPoint {
  timestamp: number;
  date: Date;
  price: number;
  confidence: number;
}

/**
 * Currency category for filtering
 */
export type CurrencyCategory = "currency" | "fragment" | "essence" | "fossil" | "other";

/**
 * Sorting options for currency table
 */
export type CurrencySortField =
  | "currency"
  | "current_price"
  | "profit_1d"
  | "profit_3d"
  | "profit_7d"
  | "average_confidence"
  | "volatility";

/**
 * Sort direction
 */
export type SortDirection = "asc" | "desc";

/**
 * Currency sort configuration
 */
export interface CurrencySort {
  field: CurrencySortField;
  direction: SortDirection;
}



