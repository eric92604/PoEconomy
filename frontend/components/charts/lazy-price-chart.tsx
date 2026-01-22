"use client";

/**
 * Lazy-loaded Price Chart - Reduces initial bundle size by loading recharts on demand
 */

import dynamic from "next/dynamic";
import { PriceChartSkeleton } from "./price-chart-skeleton";
import type { ChartDataPoint } from "@/types";

interface PriceChartProps {
  data: ChartDataPoint[];
  currencyName: string;
  timeRange?: string;
  showPredictionBands?: boolean;
}

// Dynamically import the PriceChart component with recharts
const DynamicPriceChart = dynamic(
  () => import("./price-chart").then((mod) => ({ default: mod.PriceChart })),
  {
    loading: () => <PriceChartSkeleton />,
    ssr: false, // Disable SSR for chart (canvas-based rendering)
  }
);

export function LazyPriceChart(props: PriceChartProps) {
  return <DynamicPriceChart {...props} />;
}
