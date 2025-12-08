"use client";

/**
 * Dynamic Price Chart - Shows historical prices and predictions
 */

import { useMemo } from "react";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { ChartDataPoint } from "@/types";
import { formatPrice, formatChaosPrice, formatChartDate } from "@/lib/utils";
import { useTheme } from "@/lib/providers";
import Image from "next/image";

interface PriceChartProps {
  data: ChartDataPoint[];
  currencyName: string;
  timeRange?: string;
  showPredictionBands?: boolean;
}

// Custom Y-axis tick with chaos orb icon
const CustomYAxisTick = ({ x, y, payload, isDark }: any) => {
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={-18}
        y={0}
        dy={4}
        textAnchor="end"
        fill={isDark ? "#9ca3af" : "#6b7280"}
        fontSize={12}
      >
        {formatChaosPrice(payload.value)}
      </text>
      <image
        x={-14}
        y={-6}
        width={12}
        height={12}
        href="/images/chaos-orb.png"
      />
    </g>
  );
};

export function PriceChart({
  data,
  currencyName,
  timeRange = "7d",
  showPredictionBands = true,
}: PriceChartProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  // Separate historical and predicted data
  const chartData = useMemo(() => {
    return data.map((point, index) => {
      const isLastHistorical = !point.predicted && index < data.length - 1 && data[index + 1]?.predicted;
      const isFirstPrediction = point.predicted && (index === 0 || !data[index - 1]?.predicted);
      
      return {
        timestamp: point.timestamp,
        date: formatChartDate(point.timestamp, timeRange),
        price: point.price,
        predicted: point.predicted ? point.price : null,
        historical: !point.predicted ? point.price : null,
        // Connector line: show price at both last historical and first prediction
        connector: (isLastHistorical || isFirstPrediction) ? point.price : null,
        // Only include prediction bounds if they're valid and reasonable
        // Filter out extreme values that would skew the Y-axis
        lower: (point.prediction_lower != null && 
                point.prediction_lower >= 0 && 
                isFinite(point.prediction_lower)) 
          ? point.prediction_lower 
          : null,
        upper: (point.prediction_upper != null && 
                point.prediction_upper >= 0 && 
                isFinite(point.prediction_upper)) 
          ? point.prediction_upper 
          : null,
      };
    });
  }, [data, timeRange]);

  // Calculate price range for Y-axis
  // IMPORTANT: Only use actual price values, ignore prediction bounds to avoid incorrect scaling
  const priceRange = useMemo(() => {
    if (data.length === 0) {
      return { min: 0, max: 100 };
    }

    // Only extract actual price values, filter out null/NaN/invalid values
    const prices = data
      .map((d) => d.price)
      .filter((p) => p != null && !isNaN(p) && isFinite(p) && p >= 0);
    
    if (prices.length === 0) {
      return { min: 0, max: 100 };
    }

    const min = Math.min(...prices);
    const max = Math.max(...prices);
    
    // Handle case where all prices are the same
    if (min === max) {
      const value = min || 1; // Use 1 if min is 0
      // Create a range around the single value (10% above and below)
      const padding = Math.max(value * 0.1, 0.1); // At least 10% or 0.1 unit
      return {
        min: Math.max(0, value - padding),
        max: value + padding,
      };
    }

    // Calculate padding as percentage of range
    const range = max - min;
    // Use 10% of range for padding, with a minimum of 5% of the average value
    // This ensures small variations are visible
    const avgValue = (min + max) / 2;
    const minPaddingFromAvg = avgValue * 0.05; // At least 5% of average
    const padding = Math.max(range * 0.1, minPaddingFromAvg, 0.1); // At least 10% of range, 5% of avg, or 0.1 unit
    
    return {
      min: Math.max(0, min - padding),
      max: max + padding,
    };
  }, [data]);

  // Custom tooltip
  const CustomTooltip = ({ 
    active, 
    payload 
  }: { 
    active?: boolean; 
    payload?: Array<{
      payload: {
        date: string;
        price: number;
        predicted: number | null;
        confidence?: number;
        lower?: number;
        upper?: number;
      };
    }>;
  }) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;
    const isPredicted = data.predicted !== null;

    return (
      <div className="rounded-lg border bg-background p-3 shadow-lg">
        <p className="text-sm font-medium mb-2">{data.date}</p>
        <div className="space-y-1">
          <p className="text-sm">
            <span className="text-muted-foreground">Price: </span>
            <span className="font-mono font-semibold">
              {formatChaosPrice(data.price)}c
            </span>
          </p>
          {isPredicted && (
            <>
              <p className="text-xs text-muted-foreground">Predicted</p>
              {data.lower && data.upper && (
                <p className="text-xs text-muted-foreground">
                  Range: {formatChaosPrice(data.lower)} - {formatChaosPrice(data.upper)}
                </p>
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Price Chart</CardTitle>
          <CardDescription>No data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <p className="text-muted-foreground">No price data to display</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{currencyName} Price History</CardTitle>
        <CardDescription>
          Historical prices and predictions over time
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={chartData}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={isDark ? "#374151" : "#e5e7eb"}
            />
            <XAxis
              dataKey="date"
              stroke={isDark ? "#9ca3af" : "#6b7280"}
              fontSize={12}
            />
            <YAxis
              domain={[priceRange.min, priceRange.max]}
              allowDataOverflow={false}
              allowDecimals={true}
              type="number"
              stroke={isDark ? "#9ca3af" : "#6b7280"}
              fontSize={12}
              tick={<CustomYAxisTick isDark={isDark} />}
              // Explicitly prevent domain recalculation from data
              scale="linear"
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />

            {/* Historical prices */}
            <Line
              type="monotone"
              dataKey="historical"
              stroke={isDark ? "#3b82f6" : "#2563eb"}
              strokeWidth={2}
              dot={false}
              name="Historical"
              connectNulls={false}
            />

            {/* Connector line - bridges historical to predicted */}
            <Line
              type="monotone"
              dataKey="connector"
              stroke={isDark ? "#6b7280" : "#9ca3af"}
              strokeWidth={2}
              strokeDasharray="3 3"
              dot={false}
              legendType="none"
              connectNulls={true}
            />

            {/* Predicted prices */}
            <Line
              type="monotone"
              dataKey="predicted"
              stroke={isDark ? "#f59e0b" : "#d97706"}
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Predicted"
              connectNulls={false}
            />

            {/* Prediction bounds as lines - simpler and doesn't affect Y-axis domain */}
            {showPredictionBands && (
              <Line
                type="monotone"
                dataKey="upper"
                stroke={isDark ? "#f59e0b" : "#fbbf24"}
                strokeWidth={1}
                strokeDasharray="2 2"
                dot={false}
                name="Upper Bound"
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
            {showPredictionBands && (
              <Line
                type="monotone"
                dataKey="lower"
                stroke={isDark ? "#f59e0b" : "#fbbf24"}
                strokeWidth={1}
                strokeDasharray="2 2"
                dot={false}
                name="Lower Bound"
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

