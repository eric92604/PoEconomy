"use client";

/**
 * League Comparison Chart - Overlay multiple previous leagues' price histories
 * normalized to "day in league" on the X-axis for pattern comparison.
 */

import { useMemo, useState, useEffect } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TrendingUp, TrendingDown, Minus, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatChaosPrice, formatPercentage, cn } from "@/lib/utils";
import { useTheme } from "@/lib/providers";
import { CustomYAxisTick } from "./chart-primitives";

// Fixed color palette for previous leagues — indexed by selection order
export const LEAGUE_COLORS = [
  "#3b82f6", // blue
  "#10b981", // emerald
  "#ef4444", // red
  "#8b5cf6", // violet
  "#06b6d4", // cyan
  "#f97316", // orange
  "#84cc16", // lime
  "#ec4899", // pink
];

// Current league always uses the PoE gold accent color
export const CURRENT_LEAGUE_COLOR = "#C9A961";

// Average line uses neutral grey
export const AVERAGE_COLOR = "#9ca3af";

// Volatility band fill color
export const VOLATILITY_FILL = "#818cf8";

/**
 * One row in the flattened chart data array.
 * Key "day" is the X-axis value. Each league occupies a numeric key.
 */
export interface ComparisonChartRow {
  day: number;
  [leagueName: string]: number | null | undefined;
}

export interface LeagueSeriesConfig {
  league: string;
  color: string;
  /** Render as a dashed line (e.g. for the Average series) */
  dashed?: boolean;
  /** If true, render as a thicker prominent line (current league) */
  prominent?: boolean;
  /** If true, this is the predicted extension of the current league */
  predicted?: boolean;
}

export interface VolatilityPoint {
  day: number;
  min: number;
  max: number;
}

interface AheadBehindStat {
  currentPrice: number;
  historicalAvg: number;
  delta: number;
  deltaPercent: number;
}

interface LeagueComparisonChartProps {
  data: ComparisonChartRow[];
  series: LeagueSeriesConfig[];
  currencyName: string;
  maxDays: number;
  showVolatility?: boolean;
  volatilityBand?: VolatilityPoint[];
  /** When true, renders prediction lower/upper band from `prediction_range` on rows (current league forecasts). */
  showPredictionBands?: boolean;
  /** When true, fades non-prominent, non-average series lines */
  showAverage?: boolean;
  /** Stats for the vs. Historical Average overlay panel */
  aheadBehindStat?: AheadBehindStat | null;
  currentLeagueDays?: number;
}

export function LeagueComparisonChart({
  data,
  series,
  currencyName,
  maxDays,
  showVolatility = false,
  volatilityBand = [],
  showPredictionBands = false,
  showAverage = false,
  aheadBehindStat,
  currentLeagueDays,
}: LeagueComparisonChartProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  // Zoom state
  const [refAreaLeft,  setRefAreaLeft]  = useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<number | null>(null);
  const [isSelecting,  setIsSelecting]  = useState(false);
  const [zoomDomain,   setZoomDomain]   = useState<{ left: number; right: number } | null>(null);

  // Reset zoom when data changes (e.g. currency switch)
  useEffect(() => { setZoomDomain(null); }, [data]);

  const handleMouseDown = (e: any) => {
    if (!e?.activeLabel) return;
    const val = Number(e.activeLabel);
    setRefAreaLeft(val);
    setRefAreaRight(val);
    setIsSelecting(true);
  };

  const handleMouseMove = (e: any) => {
    if (!isSelecting || !e?.activeLabel) return;
    setRefAreaRight(Number(e.activeLabel));
  };

  const handleMouseUp = () => {
    setIsSelecting(false);
    if (refAreaLeft == null || refAreaRight == null || refAreaLeft === refAreaRight) {
      setRefAreaLeft(null);
      setRefAreaRight(null);
      return;
    }
    const [l, r] = refAreaLeft <= refAreaRight ? [refAreaLeft, refAreaRight] : [refAreaRight, refAreaLeft];
    setZoomDomain({ left: l, right: r });
    setRefAreaLeft(null);
    setRefAreaRight(null);
  };

  const displayData = useMemo(
    () => data.filter((row) => row.day <= maxDays),
    [data, maxDays]
  );

  // Merge volatility and prediction band data into rows for Area rendering
  const chartData = useMemo(() => {
    const bandMap = showVolatility && volatilityBand.length > 0
      ? new Map(volatilityBand.map((v) => [v.day, v]))
      : null;

    return displayData.map((row) => {
      const r = row as ComparisonChartRow & { prediction_range?: [number, number] | null };
      const band = bandMap?.get(row.day);
      return {
        ...row,
        // Expose prediction_range as "range" for the Area dataKey (same as price-chart)
        range: r.prediction_range ?? null,
        volatility_range: band ? [band.min, band.max] : null,
      };
    });
  }, [displayData, showVolatility, volatilityBand]);

  const zoomedChartData = useMemo(() => {
    if (!zoomDomain) return chartData;
    return chartData.filter((row) => row.day >= zoomDomain.left && row.day <= zoomDomain.right);
  }, [chartData, zoomDomain]);

  // Y-axis domain with padding
  const priceRange = useMemo(() => {
    const allPrices: number[] = [];
    for (const row of zoomedChartData) {
      for (const s of series) {
        const v = row[s.league];
        if (typeof v === "number") allPrices.push(v);
      }
      if (showPredictionBands) {
        const r = (row as any).range;
        if (r && typeof r[0] === "number" && typeof r[1] === "number") {
          allPrices.push(r[0], r[1]);
        }
      }
    }
    if (allPrices.length === 0) return { min: 0, max: 100 };
    const min = Math.min(...allPrices);
    const max = Math.max(...allPrices);
    const pad = (max - min) * 0.1;
    return { min: Math.max(0, min - pad), max: max + pad };
  }, [zoomedChartData, series, showPredictionBands]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const relevantPayload = payload.filter(
      (entry: any) =>
        entry.value != null &&
        entry.type !== "none" &&
        entry.dataKey !== "range" &&
        entry.dataKey !== "prediction_range" &&
        entry.dataKey !== "volatility_range"
    );
    if (relevantPayload.length === 0) return null;
    return (
      <div className="rounded-lg border bg-background p-3 shadow-lg">
        <p className="text-sm font-medium mb-2">Day {label}</p>
        <div className="space-y-1">
          {relevantPayload.map((entry: any) => (
            <p key={entry.dataKey} className="text-sm" style={{ color: entry.color }}>
              <span className="font-medium">{entry.name}: </span>
              <span className="font-mono">
                {typeof entry.value === "number"
                  ? `${formatChaosPrice(entry.value)}c`
                  : "—"}
              </span>
            </p>
          ))}
        </div>
      </div>
    );
  };

  if (series.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>League History</CardTitle>
          <CardDescription>
            Select previous leagues from the sidebar to overlay their price histories.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <p className="text-muted-foreground text-sm">No leagues selected</p>
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
            <CardTitle>{currencyName}</CardTitle>
            <CardDescription>Price by day in league (day 1 = league start)</CardDescription>
          </div>
          {zoomDomain && (
            <Button variant="outline" size="sm" onClick={() => setZoomDomain(null)}>
              <ZoomOut className="h-4 w-4 mr-1" /> Reset zoom
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex gap-4">
          {/* Chart */}
          <div className="flex-1 min-w-0">
          <ResponsiveContainer width="100%" height={420}>
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
                dataKey="day"
                stroke={isDark ? "#9ca3af" : "#6b7280"}
                fontSize={12}
                label={{
                  value: "Day in League",
                  position: "insideBottom",
                  offset: -4,
                  fill: isDark ? "#9ca3af" : "#6b7280",
                  fontSize: 12,
                }}
                height={40}
              />
              <YAxis
                domain={[priceRange.min, priceRange.max]}
                stroke={isDark ? "#9ca3af" : "#6b7280"}
                fontSize={12}
                tick={<CustomYAxisTick isDark={isDark} />}
                width={60}
              />
              <Tooltip content={<CustomTooltip />} />

              {/* Volatility band (min/max envelope across previous leagues) */}
              {showVolatility && (
                <Area
                  type="monotone"
                  dataKey="volatility_range"
                  stroke="none"
                  fill={VOLATILITY_FILL}
                  fillOpacity={0.25}
                  legendType="none"
                  connectNulls={false}
                />
              )}

              {/* Current-league prediction range band */}
              <Area
                type="monotone"
                dataKey="range"
                stroke="none"
                fill={isDark ? "#f59e0b" : "#fbbf24"}
                fillOpacity={0.2}
                legendType="none"
                connectNulls={false}
              />

              {/* One Line per selected league — hide previous league lines when both average and range are active */}
              {series.map(({ league, color, dashed, prominent, predicted }) => {
                const isPrevLeague = !prominent && !dashed && !predicted;
                const hiddenByOverlays = isPrevLeague && showAverage && showVolatility;
                if (hiddenByOverlays) return null;
                const opacity = showAverage && isPrevLeague ? 0.25 : 1;
                const stroke = predicted ? (isDark ? "#f59e0b" : "#d97706") : color;
                return (
                  <Line
                    key={league}
                    type="monotone"
                    dataKey={league}
                    stroke={stroke}
                    strokeWidth={prominent ? 3 : 2}
                    strokeDasharray={dashed || predicted ? "5 5" : undefined}
                    strokeOpacity={opacity}
                    dot={false}
                    connectNulls={true}
                    name={predicted ? "Predicted" : league}
                  />
                );
              })}

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
          </div>

          {/* Right column: stat panel + custom legend */}
          <div className="flex flex-col gap-3 justify-center shrink-0 w-36 text-xs">
            {/* vs. Historical Average panel */}
            {aheadBehindStat && (
              <div className="rounded-md border bg-muted/40 px-3 py-2 space-y-0.5">
                <p className="text-muted-foreground">vs. Avg @ day {currentLeagueDays}</p>
                <p className="font-mono font-semibold text-sm">{formatChaosPrice(aheadBehindStat.currentPrice)}c</p>
                <p className="text-muted-foreground">Avg: {formatChaosPrice(aheadBehindStat.historicalAvg)}c</p>
                <div className={cn(
                  "flex items-center gap-1 font-medium pt-0.5",
                  aheadBehindStat.delta > 0 ? "text-green-500" : aheadBehindStat.delta < 0 ? "text-red-500" : "text-muted-foreground"
                )}>
                  {aheadBehindStat.delta > 0 ? <TrendingUp className="h-3 w-3 shrink-0" /> : aheadBehindStat.delta < 0 ? <TrendingDown className="h-3 w-3 shrink-0" /> : <Minus className="h-3 w-3 shrink-0" />}
                  <span className="leading-tight">
                    {aheadBehindStat.delta > 0 ? "+" : ""}{formatChaosPrice(aheadBehindStat.delta)}c ({aheadBehindStat.delta >= 0 ? "+" : ""}{formatPercentage(aheadBehindStat.deltaPercent, 1, false)})
                  </span>
                </div>
              </div>
            )}

            {/* Custom legend */}
            <div className="space-y-1.5">
              {series.map(({ league, color, dashed, prominent, predicted }) => {
                const isPrevLeague = !prominent && !dashed && !predicted;
                if (isPrevLeague && showAverage && showVolatility) return null;
                const label = predicted ? "Predicted" : league;
                const stroke = predicted ? (isDark ? "#f59e0b" : "#d97706") : color;
                const opacity = showAverage && isPrevLeague ? 0.35 : 1;
                return (
                  <div key={league} className="flex items-center gap-1.5" style={{ opacity }}>
                    <svg width="20" height="10" className="shrink-0">
                      <line
                        x1="0" y1="5" x2="20" y2="5"
                        stroke={stroke}
                        strokeWidth={prominent ? 2.5 : 1.5}
                        strokeDasharray={dashed || predicted ? "4 3" : undefined}
                      />
                    </svg>
                    <span className="truncate text-foreground/80">{label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
