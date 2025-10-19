/**
 * Chart data types
 */

import { PredictionHorizon } from "./api";

/**
 * Price chart data point
 */
export interface ChartDataPoint {
  timestamp: number;
  date: Date;
  price: number;
  confidence?: number;
  predicted?: boolean;
  prediction_lower?: number;
  prediction_upper?: number;
}

/**
 * Multi-currency chart data
 */
export interface MultiCurrencyChartData {
  currency: string;
  data: ChartDataPoint[];
  color: string;
}

/**
 * Chart configuration
 */
export interface ChartConfig {
  showGrid: boolean;
  showLegend: boolean;
  showTooltip: boolean;
  showPredictionBands: boolean;
  height: number;
  animationDuration: number;
}

/**
 * Time range for charts
 */
export type TimeRange = "24h" | "7d" | "30d" | "90d" | "all";

/**
 * Heat map cell data
 */
export interface HeatMapCell {
  currency: string;
  league: string;
  value: number;
  label: string;
  color: string;
  profitPercent: number;
  confidence: number;
}

/**
 * Heat map configuration
 */
export interface HeatMapConfig {
  colorScale: "profit" | "confidence" | "risk";
  cellSize: "small" | "medium" | "large";
  showLabels: boolean;
  showValues: boolean;
}

/**
 * Trend indicator
 */
export interface TrendIndicator {
  direction: "up" | "down" | "stable";
  strength: number; // 0-1
  color: string;
}

/**
 * Chart tooltip data
 */
export interface ChartTooltipData {
  currency: string;
  date: Date;
  price: number;
  predicted?: boolean;
  confidence?: number;
  priceChange?: number;
  priceChangePercent?: number;
  prediction_lower?: number;
  prediction_upper?: number;
}



