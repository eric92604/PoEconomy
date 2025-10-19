/**
 * Watchlist types
 */

/**
 * Watchlist item
 */
export interface WatchlistItem {
  id: string;
  currency: string;
  league: string;
  addedAt: Date;
  notes?: string;
  targetPrice?: number;
  alertEnabled: boolean;
}

/**
 * Watchlist with items
 */
export interface Watchlist {
  items: WatchlistItem[];
  lastUpdated: Date;
}

/**
 * Price alert configuration
 */
export interface PriceAlert {
  id: string;
  currency: string;
  league: string;
  condition: "above" | "below" | "change";
  threshold: number;
  enabled: boolean;
  triggered: boolean;
  lastTriggered?: Date;
}

/**
 * Alert notification
 */
export interface AlertNotification {
  id: string;
  alertId: string;
  currency: string;
  league: string;
  message: string;
  timestamp: Date;
  read: boolean;
}



