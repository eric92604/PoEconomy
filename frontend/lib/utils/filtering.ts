/**
 * Data filtering utilities
 */

import type { CurrencyWithPredictions, CurrencyFilters, CurrencyCategory } from "@/types";

/**
 * Filter currencies based on filter configuration
 */
export function filterCurrencies(
  currencies: CurrencyWithPredictions[],
  filters: CurrencyFilters
): CurrencyWithPredictions[] {
  return currencies.filter((currency) => {
    // Text search
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      if (!currency.currency.toLowerCase().includes(searchLower)) {
        return false;
      }
    }

    // League filter
    if (filters.leagues && filters.leagues.length > 0) {
      if (!filters.leagues.includes(currency.league)) {
        return false;
      }
    }

    // Category filter
    if (filters.categories && filters.categories.length > 0) {
      if (!currency.category || !filters.categories.includes(currency.category as CurrencyCategory)) {
        return false;
      }
    }

    // Confidence threshold
    if (filters.minConfidence !== undefined) {
      if (currency.average_confidence < filters.minConfidence) {
        return false;
      }
    }

    // Profit threshold (check any prediction)
    if (filters.minProfit !== undefined) {
      const hasMinProfit = Object.values(currency.predictions).some(
        (pred) => pred && pred.price_change_percent >= filters.minProfit!
      );
      if (!hasMinProfit) {
        return false;
      }
    }

    // Price range
    if (filters.minPrice !== undefined && currency.current_price < filters.minPrice) {
      return false;
    }
    if (filters.maxPrice !== undefined && currency.current_price > filters.maxPrice) {
      return false;
    }

    // Only show currencies with predictions
    if (filters.onlyWithPredictions) {
      const hasPredictions = Object.keys(currency.predictions).length > 0;
      if (!hasPredictions) {
        return false;
      }
    }

    return true;
  });
}

/**
 * Debounce function for search inputs
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

/**
 * Create filter summary text
 */
export function getFilterSummary(filters: CurrencyFilters): string {
  const parts: string[] = [];

  if (filters.search) {
    parts.push(`"${filters.search}"`);
  }

  if (filters.leagues && filters.leagues.length > 0) {
    parts.push(`${filters.leagues.length} league(s)`);
  }

  if (filters.categories && filters.categories.length > 0) {
    parts.push(`${filters.categories.length} categor${filters.categories.length === 1 ? "y" : "ies"}`);
  }

  if (filters.minConfidence !== undefined) {
    parts.push(`confidence ≥${(filters.minConfidence * 100).toFixed(0)}%`);
  }

  if (filters.minProfit !== undefined) {
    parts.push(`profit ≥${filters.minProfit}%`);
  }

  if (filters.minPrice !== undefined || filters.maxPrice !== undefined) {
    if (filters.minPrice !== undefined && filters.maxPrice !== undefined) {
      parts.push(`price ${filters.minPrice}-${filters.maxPrice}`);
    } else if (filters.minPrice !== undefined) {
      parts.push(`price ≥${filters.minPrice}`);
    } else {
      parts.push(`price ≤${filters.maxPrice}`);
    }
  }

  if (parts.length === 0) {
    return "No filters applied";
  }

  return `Filtered by: ${parts.join(", ")}`;
}

/**
 * Count active filters
 */
export function countActiveFilters(filters: CurrencyFilters): number {
  let count = 0;

  if (filters.search) count++;
  if (filters.leagues && filters.leagues.length > 0) count++;
  if (filters.categories && filters.categories.length > 0) count++;
  if (filters.minConfidence !== undefined) count++;
  if (filters.minProfit !== undefined) count++;
  if (filters.minPrice !== undefined) count++;
  if (filters.maxPrice !== undefined) count++;

  return count;
}

