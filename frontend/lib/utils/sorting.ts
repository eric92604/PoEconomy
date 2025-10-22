/**
 * Data sorting utilities
 */

import type { CurrencyWithPredictions, CurrencySortField, SortDirection } from "@/types";

/**
 * Sort currencies based on field and direction
 */
export function sortCurrencies(
  currencies: CurrencyWithPredictions[],
  field: CurrencySortField,
  direction: SortDirection
): CurrencyWithPredictions[] {
  const sorted = [...currencies].sort((a, b) => {
    let aValue: number | string;
    let bValue: number | string;

    switch (field) {
      case "currency":
        aValue = a.currency;
        bValue = b.currency;
        break;

      case "current_price":
        aValue = a.current_price;
        bValue = b.current_price;
        break;

      case "profit_1d":
        aValue = a.predictions["1d"]?.price_change_percent ?? -Infinity;
        bValue = b.predictions["1d"]?.price_change_percent ?? -Infinity;
        break;

      case "profit_3d":
        aValue = a.predictions["3d"]?.price_change_percent ?? -Infinity;
        bValue = b.predictions["3d"]?.price_change_percent ?? -Infinity;
        break;

      case "profit_7d":
        aValue = a.predictions["7d"]?.price_change_percent ?? -Infinity;
        bValue = b.predictions["7d"]?.price_change_percent ?? -Infinity;
        break;

      case "average_confidence":
        aValue = a.average_confidence;
        bValue = b.average_confidence;
        break;

      case "volatility":
        // Calculate volatility from price ranges
        aValue = calculateVolatilityScore(a);
        bValue = calculateVolatilityScore(b);
        break;

      default:
        aValue = 0;
        bValue = 0;
    }

    // Compare values
    if (typeof aValue === "string" && typeof bValue === "string") {
      return direction === "asc"
        ? aValue.localeCompare(bValue)
        : bValue.localeCompare(aValue);
    }

    if (typeof aValue === "number" && typeof bValue === "number") {
      return direction === "asc" ? aValue - bValue : bValue - aValue;
    }

    return 0;
  });

  return sorted;
}

/**
 * Calculate volatility score from predictions
 */
function calculateVolatilityScore(currency: CurrencyWithPredictions): number {
  const predictions = Object.values(currency.predictions).filter((p) => p !== undefined);
  
  if (predictions.length === 0) return 0;

  // Calculate range as percentage of current price
  const ranges = predictions.map((pred) => {
    if (pred.prediction_lower && pred.prediction_upper) {
      return ((pred.prediction_upper - pred.prediction_lower) / currency.current_price) * 100;
    }
    return 0;
  });

  // Return average range
  return ranges.reduce((sum, r) => sum + r, 0) / ranges.length;
}

/**
 * Get sort icon based on field and direction
 */
export function getSortIcon(
  currentField: CurrencySortField,
  targetField: CurrencySortField,
  direction: SortDirection
): "asc" | "desc" | "none" {
  if (currentField !== targetField) return "none";
  return direction;
}

/**
 * Toggle sort direction
 */
export function toggleSortDirection(direction: SortDirection): SortDirection {
  return direction === "asc" ? "desc" : "asc";
}



