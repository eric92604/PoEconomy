/**
 * Formatting utilities for numbers, dates, and currencies
 */

/**
 * Format a number as currency (Chaos Orbs)
 */
export function formatPrice(price: number, decimals = 2): string {
  return price.toFixed(decimals);
}

/**
 * Format a chaos price with "k" for thousands (same as ChaosPrice component)
 */
export function formatChaosPrice(price: number): string {
  if (price >= 1000) {
    return (price / 1000).toFixed(1) + 'k';
  }
  return price.toFixed(2);
}

/**
 * Format a percentage with sign
 */
export function formatPercentage(value: number, decimals = 2, includeSign = true): string {
  const formatted = value.toFixed(decimals);
  if (includeSign && value > 0) {
    return `+${formatted}%`;
  }
  return `${formatted}%`;
}

/**
 * Format a confidence score as percentage
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(0)}%`;
}

/**
 * Format a large number with abbreviation (K, M, B)
 */
export function formatLargeNumber(value: number, decimals = 1): string {
  if (value >= 1000000000) {
    return `${(value / 1000000000).toFixed(decimals)}B`;
  }
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(decimals)}M`;
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(decimals)}K`;
  }
  return value.toFixed(decimals);
}

/**
 * Format a date as relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: Date | number): string {
  const now = new Date();
  const target = typeof date === "number" ? new Date(date * 1000) : date;
  const diffMs = now.getTime() - target.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) return "just now";
  if (diffMin < 60) return `${diffMin} minute${diffMin !== 1 ? "s" : ""} ago`;
  if (diffHour < 24) return `${diffHour} hour${diffHour !== 1 ? "s" : ""} ago`;
  if (diffDay < 7) return `${diffDay} day${diffDay !== 1 ? "s" : ""} ago`;
  
  return target.toLocaleDateString();
}

/**
 * Format a timestamp as a date string
 */
export function formatDate(timestamp: number | string | Date, includeTime = false): string {
  const date = typeof timestamp === "number" 
    ? new Date(timestamp * 1000) 
    : new Date(timestamp);

  if (includeTime) {
    return date.toLocaleString();
  }
  return date.toLocaleDateString();
}

/**
 * Format a timestamp for chart display
 */
export function formatChartDate(timestamp: number, range: string): string {
  const date = new Date(timestamp * 1000);

  switch (range) {
    case "24h":
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    case "7d":
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    case "30d":
    case "90d":
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    default:
      return date.toLocaleDateString();
  }
}

/**
 * Format currency name for display
 */
export function formatCurrencyName(name: string): string {
  return name.trim();
}

/**
 * Format league name for display
 */
export function formatLeagueName(name: string): string {
  return name.trim();
}

/**
 * Truncate text to a maximum length
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength - 3)}...`;
}

/**
 * Format a number with thousands separators
 */
export function formatNumber(value: number, decimals = 0): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}



