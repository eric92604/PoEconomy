"use client";

/**
 * League History Page - Overlay previous leagues' price histories for pattern comparison
 */

import { useState, useMemo, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Check, ChevronsUpDown, RefreshCw } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  useLeagues,
  useLatestPredictions,
  useHistoricalPrices,
  useHistoricalLeagues,
  useLeagueHistoricalPrices,
  useBatchPredictions,
} from "@/lib/hooks";
import {
  LeagueComparisonChart,
  LEAGUE_COLORS,
  CURRENT_LEAGUE_COLOR,
  AVERAGE_COLOR,
} from "@/components/charts/league-comparison-chart";
import { LeagueSelector } from "@/components/currency/league-selector";
import { cn } from "@/lib/utils";
import { LATEST_PREDICTIONS_HORIZONS } from "@/lib/constants/predictions";
import type { ComparisonChartRow, LeagueSeriesConfig, VolatilityPoint } from "@/components/charts/league-comparison-chart";

function dateToDayInLeague(date: string, leagueStartDate: string): number {
  return Math.floor((Date.parse(date) - Date.parse(leagueStartDate)) / 86_400_000) + 1;
}

export default function LeagueHistoryPage() {
  const queryClient = useQueryClient();

  // --- UI state ---
  const [selectedCurrency, setSelectedCurrency] = useState<string>("Divine Orb");
  const [currencyOpen, setCurrencyOpen] = useState(false);
  const [currencySearch, setCurrencySearch] = useState("");
  const [selectedLeagues, setSelectedLeagues] = useState<string[]>([]);
  const [showAverage, setShowAverage] = useState(false);
  const [showVolatility, setShowVolatility] = useState(false);

  // --- Data fetching (mirrors prices page: leagues + latest + batch + history) ---
  const { data: leaguesData } = useLeagues();

  // Same as dashboard / investments: first key is the current league (API orders leagues seasonally).
  const leagues = useMemo(() => {
    if (!leaguesData) return [];
    return Object.keys(leaguesData.leagues);
  }, [leaguesData]);

  const [selectedLeague, setSelectedLeague] = useState<string>("");

  useEffect(() => {
    if (leagues.length > 0 && !selectedLeague) {
      setSelectedLeague(leagues[0]);
    }
  }, [leagues, selectedLeague]);

  const { data: latestPredictionsData } = useLatestPredictions({
    league: selectedLeague || undefined,
    horizons: [...LATEST_PREDICTIONS_HORIZONS],
    limit: 500,
    enabled: !!selectedLeague,
  });

  const { data: batchPredictionsData, isFetching: batchPredictionsFetching } = useBatchPredictions(
    {
      requests: [
        { currency: selectedCurrency, league: selectedLeague, horizon: "1d" },
        { currency: selectedCurrency, league: selectedLeague, horizon: "3d" },
        { currency: selectedCurrency, league: selectedLeague, horizon: "7d" },
      ],
    },
    !!selectedLeague && !!selectedCurrency
  );

  const availableCurrencies = useMemo(() => {
    if (!latestPredictionsData?.predictions || !selectedLeague) return [];
    const names: string[] = [];
    Object.entries(latestPredictionsData.predictions).forEach(([currency, horizonData]) => {
      if (horizonData["1d"]) names.push(currency);
    });
    return names.sort();
  }, [latestPredictionsData, selectedLeague]);

  const { data: currentLeagueData, isFetching: currentLeagueFetching } = useHistoricalPrices(
    { currency: selectedCurrency, league: selectedLeague, limit: 1000 },
    !!selectedCurrency && !!selectedLeague
  );

  // Historical league list
  const { data: historicalLeaguesData, isFetching: leagueListFetching } = useHistoricalLeagues();

  const availableHistoricalLeagues = useMemo(() => {
    const leagues = historicalLeaguesData?.leagues ?? {};
    return Object.entries(leagues)
      .sort(([, a], [, b]) => {
        const aDate = a.league_start_date ?? "";
        const bDate = b.league_start_date ?? "";
        return bDate.localeCompare(aDate); // most recent first
      })
      .map(([name]) => name);
  }, [historicalLeaguesData]);

  // Previous league prices — one query per league, each independently cached
  const { seriesMap: leagueSeriesMap, isFetching: historicalFetching } = useLeagueHistoricalPrices(
    selectedCurrency,
    selectedLeagues,
    selectedLeagues.length > 0
  );

  // --- X-axis cap (maxDays is derived after chart data is built, so it lives in the same useMemo) ---
  const currentLeagueDays = useMemo(
    () => currentLeagueData?.prices.length ?? 0,
    [currentLeagueData]
  );

  // --- Chart data construction ---
  const { chartData, series, volatilityBand, hasPredictionBands, maxDays } = useMemo((): {
    chartData: ComparisonChartRow[];
    series: LeagueSeriesConfig[];
    volatilityBand: VolatilityPoint[];
    hasPredictionBands: boolean;
    maxDays: number;
  } => {
    // Map: day -> { leagueName: price }
    const dayMap = new Map<number, Record<string, number>>();
    const predictionRangeByDay = new Map<number, [number, number]>();

    const addToDayMap = (day: number, leagueName: string, price: number) => {
      const existing = dayMap.get(day) ?? {};
      existing[leagueName] = price;
      dayMap.set(day, existing);
    };

    const normalizeToMidnightUTC = (dateString: string): Date => {
      const date = new Date(dateString);
      return new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    };

    // 1. Current league (always shown)
    if (currentLeagueData?.prices.length && currentLeagueData.prices.length > 0) {
      const startDate = currentLeagueData.prices[0].date;
      for (const { date, avg_price } of currentLeagueData.prices) {
        const day = dateToDayInLeague(date, startDate);
        addToDayMap(day, selectedLeague, avg_price);
      }
    }

    // 2. Previous leagues
    for (const league of selectedLeagues) {
      const leagueSeries = leagueSeriesMap[league];
      if (!leagueSeries?.league_start_date || !leagueSeries.prices.length) continue;
      for (const { date, avg_price } of leagueSeries.prices) {
        const day = dateToDayInLeague(date, leagueSeries.league_start_date);
        addToDayMap(day, league, avg_price);
      }
    }

    // 3. Forward prediction points for current league — stored in a separate key
    //    so they render as a distinct dashed amber line rather than extending the main series.
    const predictedKey = selectedLeague ? `${selectedLeague}__predicted` : null;
    if (
      batchPredictionsData?.results?.length &&
      currentLeagueData?.prices?.length &&
      selectedLeague &&
      predictedKey
    ) {
      const leagueStartDate = currentLeagueData.prices[0].date;
      const lastHistoricalDay = currentLeagueData.prices.length;
      // Bridge: copy the last known historical price into the predicted key so the line connects
      const lastHistoricalPrice = currentLeagueData.prices[currentLeagueData.prices.length - 1]?.avg_price;
      if (lastHistoricalPrice != null) {
        addToDayMap(lastHistoricalDay, predictedKey, lastHistoricalPrice);
      }

      const predictionHorizons = [
        { horizon: "1d" as const, days: 1 },
        { horizon: "3d" as const, days: 3 },
        { horizon: "7d" as const, days: 7 },
      ];
      for (const { horizon, days } of predictionHorizons) {
        const prediction = batchPredictionsData.results.find(
          (p) => p.horizon === horizon && p.currency === selectedCurrency
        );
        if (!prediction?.timestamp) continue;
        const predictionGeneratedDate = normalizeToMidnightUTC(prediction.timestamp);
        const predictedDate = new Date(predictionGeneratedDate);
        predictedDate.setUTCDate(predictedDate.getUTCDate() + days);
        const day = dateToDayInLeague(predictedDate.toISOString(), leagueStartDate);
        addToDayMap(day, predictedKey, prediction.predicted_price);
        if (prediction.prediction_lower != null && prediction.prediction_upper != null) {
          predictionRangeByDay.set(day, [prediction.prediction_lower, prediction.prediction_upper]);
        }
      }
    }

    // Sort rows by day
    const rows: ComparisonChartRow[] = Array.from(dayMap.entries())
      .sort(([a], [b]) => a - b)
      .map(([day, prices]) => {
        const pr = predictionRangeByDay.get(day);
        const row: ComparisonChartRow & { prediction_range?: [number, number] } = { day, ...prices };
        if (pr) row.prediction_range = pr;
        return row;
      });

    // 3. Average line
    if (showAverage && selectedLeagues.length > 0) {
      for (const row of rows) {
        const vals = selectedLeagues
          .map((l) => row[l] as number | undefined)
          .filter((v): v is number => v != null);
        (row as any)["Average"] = vals.length
          ? vals.reduce((s, v) => s + v, 0) / vals.length
          : null;
      }
    }

    // 4. Volatility band (min/max across previous leagues per day)
    const band: VolatilityPoint[] = [];
    if (showVolatility && selectedLeagues.length > 0) {
      for (const row of rows) {
        const vals = selectedLeagues
          .map((l) => row[l] as number | undefined)
          .filter((v): v is number => v != null);
        if (vals.length >= 2) {
          band.push({ day: row.day, min: Math.min(...vals), max: Math.max(...vals) });
        }
      }
    }

    // 5. Series configs
    const seriesConfigs: LeagueSeriesConfig[] = [
      // Current league — always first, gold, prominent
      ...(selectedLeague
        ? [{ league: selectedLeague, color: CURRENT_LEAGUE_COLOR, prominent: true }]
        : []),
      // Predicted extension of current league
      ...(predictedKey && predictionRangeByDay.size > 0
        ? [{ league: predictedKey, color: CURRENT_LEAGUE_COLOR, predicted: true }]
        : []),
      // Previous leagues
      ...selectedLeagues.map((league, i) => ({
        league,
        color: LEAGUE_COLORS[i % LEAGUE_COLORS.length],
      })),
      // Average
      ...(showAverage && selectedLeagues.length > 0
        ? [{ league: "Average", color: AVERAGE_COLOR, dashed: true }]
        : []),
    ];

    const allPredDays = [...predictionRangeByDay.keys()];
    // Also include non-range prediction days (e.g. 7d which may have no bounds)
    // by scanning the predictedKey rows
    const predRows = predictedKey ? rows.filter(r => r[predictedKey] != null) : [];
    const maxPredDay = predRows.length > 0
      ? Math.max(...predRows.map(r => r.day))
      : allPredDays.length > 0 ? Math.max(...allPredDays) : 0;

    return {
      chartData: rows,
      series: seriesConfigs,
      volatilityBand: band,
      hasPredictionBands: predictionRangeByDay.size > 0,
      maxDays: Math.max(currentLeagueData?.prices.length ?? 0, maxPredDay),
    };
  }, [
    currentLeagueData,
    selectedLeague,
    selectedCurrency,
    batchPredictionsData,
    leagueSeriesMap,
    selectedLeagues,
    showAverage,
    showVolatility,
  ]);

  // --- Ahead/behind stat ---
  const aheadBehindStat = useMemo(() => {
    if (!selectedLeague || currentLeagueDays === 0 || selectedLeagues.length === 0) return null;

    // Find the last row that has the current league's actual (non-predicted) price.
    // We can't assume the row day number equals prices.length due to gaps in data.
    const predictedKey = `${selectedLeague}__predicted`;
    const todayRow = [...chartData]
      .reverse()
      .find((r) => typeof r[selectedLeague] === "number" && r[predictedKey] == null);
    if (!todayRow) return null;

    const currentPrice = todayRow[selectedLeague];
    if (typeof currentPrice !== "number") return null;

    const prevPrices = selectedLeagues
      .map((l) => todayRow[l] as number | undefined)
      .filter((v): v is number => v != null);
    if (prevPrices.length === 0) return null;

    const historicalAvg = prevPrices.reduce((s, v) => s + v, 0) / prevPrices.length;
    const delta = currentPrice - historicalAvg;
    const deltaPercent = historicalAvg > 0 ? (delta / historicalAvg) * 100 : 0;

    return { currentPrice, historicalAvg, delta, deltaPercent };
  }, [chartData, selectedLeague, currentLeagueDays, selectedLeagues]);

  const handleLeagueToggle = useCallback((league: string) => {
    setSelectedLeagues((prev) =>
      prev.includes(league) ? prev.filter((l) => l !== league) : [...prev, league]
    );
  }, []);

  const isFetching =
    currentLeagueFetching || leagueListFetching || historicalFetching || batchPredictionsFetching;

  return (
    <div className="py-8 space-y-8">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">League History</h1>
          <p className="text-muted-foreground mt-1">
            Overlay previous leagues&apos; price histories to identify recurring patterns
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="flex items-center gap-2"
          disabled={isFetching}
          onClick={() => {
            queryClient.invalidateQueries({ queryKey: ["league-history"] });
            queryClient.invalidateQueries({ queryKey: ["latest-predictions"] });
            queryClient.invalidateQueries({ queryKey: ["predictions"] });
            queryClient.invalidateQueries({ queryKey: ["prices"] });
          }}
        >
          <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="flex gap-6">
        {/* Sidebar */}
        <div className="w-64 flex-shrink-0 space-y-4">

          {/* Filters card */}
          <Card>
            <CardHeader>
              <CardTitle>Filters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Currency selector */}
              <div className="space-y-2">
                <Label>Currency</Label>
                <Popover open={currencyOpen} onOpenChange={setCurrencyOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={currencyOpen}
                      className="w-full justify-between text-sm"
                    >
                      <span className="truncate">{selectedCurrency || "Select currency…"}</span>
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[240px] p-0">
                    <Command>
                      <CommandInput
                        placeholder="Search currency…"
                        value={currencySearch}
                        onValueChange={setCurrencySearch}
                      />
                      <CommandList>
                        <CommandEmpty>No currency found.</CommandEmpty>
                        <CommandGroup>
                          {(availableCurrencies.length > 0
                            ? availableCurrencies
                            : ["Divine Orb", "Chaos Orb", "Exalted Orb"]
                          )
                            .filter((c) =>
                              c.toLowerCase().includes(currencySearch.toLowerCase())
                            )
                            .map((currency) => (
                              <CommandItem
                                key={currency}
                                value={currency}
                                onSelect={(v) => {
                                  setSelectedCurrency(v);
                                  setCurrencyOpen(false);
                                  setCurrencySearch("");
                                }}
                              >
                                <Check
                                  className={cn(
                                    "mr-2 h-4 w-4",
                                    selectedCurrency === currency ? "opacity-100" : "opacity-0"
                                  )}
                                />
                                {currency}
                              </CommandItem>
                            ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
              </div>

              {/* Average toggle */}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setShowAverage((v) => !v)}
                      disabled={selectedLeagues.length === 0}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors disabled:pointer-events-none disabled:opacity-50",
                        showAverage
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border text-muted-foreground hover:border-muted-foreground hover:text-foreground"
                      )}
                    >
                      <div className={cn("h-2 w-2 rounded-full flex-shrink-0", showAverage ? "bg-primary" : "bg-muted-foreground/40")} />
                      Historical average (selected leagues)
                      {showAverage && <Check className="ml-auto h-3.5 w-3.5" />}
                    </button>
                  </TooltipTrigger>
                  {selectedLeagues.length === 0 && (
                    <TooltipContent>Select at least 1 league to enable</TooltipContent>
                  )}
                </Tooltip>
              </TooltipProvider>

              {/* Volatility toggle */}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setShowVolatility((v) => !v)}
                      disabled={selectedLeagues.length < 2}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors disabled:pointer-events-none disabled:opacity-50",
                        showVolatility
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border text-muted-foreground hover:border-muted-foreground hover:text-foreground"
                      )}
                    >
                      <div className={cn("h-2 w-2 rounded-full flex-shrink-0", showVolatility ? "bg-primary" : "bg-muted-foreground/40")} />
                      Historical price range (selected leagues)
                      {showVolatility && <Check className="ml-auto h-3.5 w-3.5" />}
                    </button>
                  </TooltipTrigger>
                  {selectedLeagues.length < 2 && (
                    <TooltipContent>Select at least 2 leagues to enable</TooltipContent>
                  )}
                </Tooltip>
              </TooltipProvider>
            </CardContent>
          </Card>

          {/* Previous league selector */}
          <Card className="gap-2">
            <CardHeader className="pb-0">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Previous Leagues</CardTitle>
                <Badge variant={selectedLeagues.length > 0 ? "default" : "secondary"} className="text-xs">
                  {selectedLeagues.length} / {availableHistoricalLeagues.length}
                </Badge>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="w-full text-xs h-7 mt-1"
                disabled={availableHistoricalLeagues.length === 0}
                onClick={() =>
                  selectedLeagues.length === availableHistoricalLeagues.length
                    ? setSelectedLeagues([])
                    : setSelectedLeagues([...availableHistoricalLeagues])
                }
              >
                {selectedLeagues.length === availableHistoricalLeagues.length ? "Unselect All" : "Select All"}
              </Button>
            </CardHeader>
            <CardContent className="pt-0">
              <LeagueSelector
                availableLeagues={availableHistoricalLeagues}
                selectedLeagues={selectedLeagues}
                onToggle={handleLeagueToggle}
              />
            </CardContent>
          </Card>
        </div>

        {/* Chart */}
        <div className="flex-1 min-w-0">
          <LeagueComparisonChart
            data={chartData}
            series={series}
            currencyName={selectedCurrency}
            maxDays={maxDays}
            showVolatility={showVolatility}
            volatilityBand={volatilityBand}
            showPredictionBands={hasPredictionBands}
            showAverage={showAverage}
            aheadBehindStat={aheadBehindStat}
            currentLeagueDays={currentLeagueDays}
          />
        </div>
      </div>
    </div>
  );
}
