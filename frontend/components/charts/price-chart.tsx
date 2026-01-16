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
import { formatChaosPrice, formatChartDate } from "@/lib/utils";
import { useTheme } from "@/lib/providers";

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
        lower: point.prediction_lower || null,
        upper: point.prediction_upper || null,
        // For the range band, we need [lower, upper] format
        range: point.prediction_lower && point.prediction_upper 
          ? [point.prediction_lower, point.prediction_upper] 
          : null,
      };
    });
  }, [data, timeRange]);

  // Calculate price range for Y-axis
  const priceRange = useMemo(() => {
    const prices = data.map((d) => d.price);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const padding = (max - min) * 0.1;
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
              stroke={isDark ? "#9ca3af" : "#6b7280"}
              fontSize={12}
              tick={<CustomYAxisTick isDark={isDark} />}
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

            {/* Prediction band - using range format [lower, upper] */}
            {showPredictionBands && (
              <Area
                type="monotone"
                dataKey="range"
                stroke="none"
                fill={isDark ? "#f59e0b" : "#fbbf24"}
                fillOpacity={0.2}
                name="Prediction Range"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

