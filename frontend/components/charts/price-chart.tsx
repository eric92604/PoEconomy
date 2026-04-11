"use client";

/**
 * Dynamic Price Chart - Shows historical prices and predictions
 */

import { useMemo, useState, useEffect } from "react";
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
  ReferenceArea,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { ChartDataPoint } from "@/types";
import { formatChaosPrice, formatChartDate } from "@/lib/utils";
import { useTheme } from "@/lib/providers";
import { CustomYAxisTick } from "./chart-primitives";
import { ZoomOut } from "lucide-react";

interface PriceChartProps {
  data: ChartDataPoint[];
  currencyName: string;
  timeRange?: string;
  showPredictionBands?: boolean;
}

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

  // Zoom state
  const [refAreaLeft,  setRefAreaLeft]  = useState<string | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<string | null>(null);
  const [isSelecting,  setIsSelecting]  = useState(false);
  const [zoomDomain,   setZoomDomain]   = useState<{ left: string; right: string } | null>(null);

  // Reset zoom when data changes (e.g. currency switch)
  useEffect(() => { setZoomDomain(null); }, [data]);

  const handleMouseDown = (e: any) => {
    if (!e?.activeLabel) return;
    setRefAreaLeft(e.activeLabel);
    setRefAreaRight(e.activeLabel);
    setIsSelecting(true);
  };

  const handleMouseMove = (e: any) => {
    if (!isSelecting || !e?.activeLabel) return;
    setRefAreaRight(e.activeLabel);
  };

  const handleMouseUp = () => {
    setIsSelecting(false);
    if (!refAreaLeft || !refAreaRight) { setRefAreaLeft(null); setRefAreaRight(null); return; }
    const li = chartData.findIndex((d) => d.date === refAreaLeft);
    const ri = chartData.findIndex((d) => d.date === refAreaRight);
    if (li === -1 || ri === -1 || li === ri) { setRefAreaLeft(null); setRefAreaRight(null); return; }
    const [l, r] = li <= ri ? [refAreaLeft, refAreaRight] : [refAreaRight, refAreaLeft];
    setZoomDomain({ left: l, right: r });
    setRefAreaLeft(null);
    setRefAreaRight(null);
  };

  const zoomedChartData = useMemo(() => {
    if (!zoomDomain) return chartData;
    const li = chartData.findIndex((d) => d.date === zoomDomain.left);
    const ri = chartData.findIndex((d) => d.date === zoomDomain.right);
    if (li === -1 || ri === -1) return chartData;
    return chartData.slice(Math.min(li, ri), Math.max(li, ri) + 1);
  }, [chartData, zoomDomain]);

  // Calculate price range for Y-axis from visible data
  const priceRange = useMemo(() => {
    const prices: number[] = [];
    for (const d of zoomedChartData) {
      if (d.historical != null) prices.push(d.historical);
      if (d.predicted != null) prices.push(d.predicted);
      if (Array.isArray(d.range)) prices.push(d.range[0], d.range[1]);
    }
    if (prices.length === 0) return { min: 0, max: 100 };
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const padding = (max - min) * 0.1;
    return { min: Math.max(0, min - padding), max: max + padding };
  }, [zoomedChartData]);

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
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{currencyName} Price History</CardTitle>
            <CardDescription>Historical prices and predictions over time</CardDescription>
          </div>
          {zoomDomain && (
            <Button variant="outline" size="sm" onClick={() => setZoomDomain(null)}>
              <ZoomOut className="h-4 w-4 mr-1" /> Reset zoom
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart
            data={zoomedChartData}
            style={{ cursor: isSelecting ? "crosshair" : "default" }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { if (isSelecting) { setIsSelecting(false); setRefAreaLeft(null); setRefAreaRight(null); } }}
          >
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

            {/* Drag-to-zoom selection indicator */}
            {isSelecting && refAreaLeft != null && refAreaRight != null && (
              <ReferenceArea
                x1={refAreaLeft}
                x2={refAreaRight}
                stroke={isDark ? "#94a3b8" : "#64748b"}
                strokeOpacity={0.3}
                fill={isDark ? "#94a3b8" : "#64748b"}
                fillOpacity={0.15}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

