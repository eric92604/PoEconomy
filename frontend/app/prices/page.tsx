"use client";

/**
 * Prices Page - Live currency prices with interactive historical charts
 */

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { Skeleton } from "@/components/ui/skeleton";
import { Search, RefreshCw, TrendingUp, X, Check, ChevronsUpDown } from "lucide-react";
import { useLeagues, useCurrencies, useHistoricalPrices, useBatchPredictions, useLatestPredictions } from "@/lib/hooks";
import { useQueryClient } from "@tanstack/react-query";
import { PriceChart } from "@/components/charts";
import { formatPrice, formatRelativeTime } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { ChartDataPoint, CurrencyFilters } from "@/types";

export default function PricesPage() {
  const queryClient = useQueryClient();
  const [selectedLeague, setSelectedLeague] = useState<string>("");
  const [selectedCurrency, setSelectedCurrency] = useState<string>("Divine Orb");
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const [filters, setFilters] = useState<CurrencyFilters>({
    search: "",
    minConfidence: undefined,
    minPrice: undefined,
  });

  // Fetch leagues and currencies
  const { data: leaguesData, isLoading: leaguesLoading, isFetching: leaguesFetching } = useLeagues();
  const { data: currenciesData, isLoading: currenciesLoading, isFetching: currenciesFetching } = useCurrencies();

  // Get available leagues
  const leagues = useMemo(() => {
    if (!leaguesData) return [];
    return Object.keys(leaguesData.leagues);
  }, [leaguesData]);

  // Set default league - prefer "Keepers" if available
  useEffect(() => {
    if (leagues.length > 0 && !selectedLeague) {
      // Prefer "Keepers" if available, otherwise use first league
      const preferredLeague = leagues.includes("Keepers") ? "Keepers" : leagues[0];
      setSelectedLeague(preferredLeague);
    }
  }, [leagues, selectedLeague]);

  // Fetch all latest predictions for the league to get list of currencies with predictions
  const { data: latestPredictionsData, isLoading: latestPredictionsLoading, isFetching: latestPredictionsFetching } = useLatestPredictions(
    {
      league: selectedLeague,
      horizons: ["1d"],
      limit: 500,
      enabled: !!selectedLeague,
    }
  );

  // Fetch predictions for the selected currency only (1d, 3d, 7d)
  const { data: batchPredictionsData, isLoading: predictionsLoading, isFetching: predictionsFetching } = useBatchPredictions(
    {
      requests: [
        { currency: selectedCurrency, league: selectedLeague, horizon: "1d" },
        { currency: selectedCurrency, league: selectedLeague, horizon: "3d" },
        { currency: selectedCurrency, league: selectedLeague, horizon: "7d" },
      ],
    },
    !!selectedLeague && !!selectedCurrency
  );

  // Fetch historical prices for the selected currency (no date filtering - get all available data)
  const { data: historicalData, isLoading: historicalLoading, isFetching: historicalFetching } = useHistoricalPrices(
    {
      currency: selectedCurrency,
      league: selectedLeague,
      limit: 1000, // Get more data points
    },
    !!selectedCurrency && !!selectedLeague
  );

  // Get available currencies - only those with predictions
  const availableCurrencies = useMemo(() => {
    if (!latestPredictionsData?.predictions || !selectedLeague) return [];

    const currencies: Array<{ currency: string; price: number; confidence: number }> = [];
    
    // Extract currencies from the predictions data
    Object.entries(latestPredictionsData.predictions).forEach(([currency, horizonData]) => {
      // Check if this currency has at least one prediction
      const has1dPrediction = horizonData["1d"];
      if (has1dPrediction) {
        currencies.push({
          currency,
          price: has1dPrediction.predicted_price || 0,
          confidence: has1dPrediction.confidence || 0,
        });
      }
    });

    return currencies.sort((a, b) => a.currency.localeCompare(b.currency));
  }, [latestPredictionsData, selectedLeague]);


  // Filter currencies (price and confidence filters only, search is handled by Command component)
  const filteredCurrencies = useMemo(() => {
    let filtered = availableCurrencies;

    // Price filter
    if (filters.minPrice !== undefined) {
      filtered = filtered.filter((currency) => currency.price >= filters.minPrice!);
    }

    // Confidence filter
    if (filters.minConfidence !== undefined) {
      filtered = filtered.filter((currency) => currency.confidence >= filters.minConfidence!);
    }

    return filtered;
  }, [availableCurrencies, filters]);

  // Prepare chart data combining historical and predicted data
  const chartData = useMemo((): ChartDataPoint[] => {
    if (!historicalData?.prices || !batchPredictionsData?.results) return [];

    const data: ChartDataPoint[] = [];

    // Helper function to normalize dates to midnight UTC
    const normalizeToMidnightUTC = (dateString: string): Date => {
      const date = new Date(dateString);
      return new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    };

    // Add all historical data
    historicalData.prices.forEach((price) => {
      const date = normalizeToMidnightUTC(price.date);
      const timestamp = date.getTime();
      data.push({
        timestamp,
        date,
        price: price.avg_price,
        confidence: undefined,
        predicted: false,
      });
    });

    // Add predictions from batch response
    const predictionHorizons = [
      { horizon: "1d", days: 1 },
      { horizon: "3d", days: 3 },
      { horizon: "7d", days: 7 },
    ];

    predictionHorizons.forEach(({ horizon, days }) => {
      const prediction = batchPredictionsData.results.find(
        (p) => p.horizon === horizon && p.currency === selectedCurrency
      );
      if (prediction && prediction.timestamp) {
        // Use the prediction timestamp (when it was generated) as the base date
        const predictionGeneratedDate = normalizeToMidnightUTC(prediction.timestamp);
        
        // Add the horizon days to get the actual predicted date
        const predictedDate = new Date(predictionGeneratedDate);
        predictedDate.setUTCDate(predictedDate.getUTCDate() + days);
        const predictedTimestamp = predictedDate.getTime();
        
        data.push({
          timestamp: predictedTimestamp,
          date: predictedDate,
          price: prediction.predicted_price,
          confidence: prediction.confidence,
          predicted: true,
          prediction_lower: prediction.prediction_lower,
          prediction_upper: prediction.prediction_upper,
        });
      }
    });

    return data.sort((a, b) => a.timestamp - b.timestamp);
  }, [historicalData, batchPredictionsData, selectedCurrency]);

  // Filter handlers
  const handlePriceChange = (value: number[]) => {
    setFilters((prev) => ({
      ...prev,
      minPrice: value[0] === 0 ? undefined : value[0],
    }));
  };

  const handleConfidenceChange = (value: number[]) => {
    setFilters((prev) => ({
      ...prev,
      minConfidence: value[0] === 0 ? undefined : value[0] / 100,
    }));
  };

  const clearFilters = () => {
    setFilters({
      search: "",
      minConfidence: undefined,
      minPrice: undefined,
    });
  };

  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (filters.minPrice !== undefined) count++;
    if (filters.minConfidence !== undefined) count++;
    return count;
  }, [filters]);

  // Manual cache clear function
  const clearCache = () => {
    queryClient.invalidateQueries({ queryKey: ["prices"] });
    queryClient.invalidateQueries({ queryKey: ["latest-predictions"] });
  };

  const isLoading = leaguesLoading || latestPredictionsLoading;
  const isFetching = leaguesFetching || latestPredictionsFetching;
  const isChartLoading = historicalLoading || predictionsLoading;
  const isChartFetching = historicalFetching || predictionsFetching;
  // Show loading state during both initial load and refresh
  const showLoading = isLoading || isFetching;
  const showChartLoading = isChartLoading || isChartFetching;

  return (
    <div className="py-8 space-y-8">
      {/* Header with Refresh Button */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Live Prices</h1>
        </div>
        <Button 
          onClick={clearCache}
          variant="outline"
          size="sm"
          className="flex items-center gap-2"
          disabled={isFetching}
        >
          <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Main Layout */}
      <div className="flex gap-6">
        {/* Left Sidebar - Filters */}
        <div className="w-64 flex-shrink-0">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Filters</CardTitle>
                </div>
                {activeFilterCount > 0 && (
                  <Button variant="ghost" size="sm" onClick={clearFilters}>
                    <X className="h-4 w-4 mr-2" />
                    Clear ({activeFilterCount})
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* League Select */}
              <div className="space-y-2">
                <Label htmlFor="league">League</Label>
                <Select value={selectedLeague} onValueChange={setSelectedLeague}>
                  <SelectTrigger id="league">
                    <SelectValue placeholder="Select league" />
                  </SelectTrigger>
                  <SelectContent>
                    {leagues.map((league) => (
                      <SelectItem key={league} value={league}>
                        {league}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Currency Search with Autocomplete */}
              <div className="space-y-2">
                <Label>Search Currency</Label>
                <Popover open={open} onOpenChange={setOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={open}
                      className="w-full justify-between"
                    >
                      {selectedCurrency || "Select currency..."}
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[280px] p-0">
                    <Command>
                      <CommandInput 
                        placeholder="Search currency..." 
                        value={searchValue}
                        onValueChange={setSearchValue}
                      />
                      <CommandList>
                        <CommandEmpty>No currency found.</CommandEmpty>
                        <CommandGroup>
                          {filteredCurrencies.map((currency) => (
                            <CommandItem
                              key={currency.currency}
                              value={currency.currency}
                              onSelect={(currentValue) => {
                                setSelectedCurrency(currentValue === selectedCurrency ? "" : currentValue);
                                setOpen(false);
                                setSearchValue("");
                              }}
                            >
                              <Check
                                className={cn(
                                  "mr-2 h-4 w-4",
                                  selectedCurrency === currency.currency ? "opacity-100" : "opacity-0"
                                )}
                              />
                              {currency.currency}
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
              </div>

              {/* Price Slider */}
              <div className="space-y-2">
                <Label>Min Price</Label>
                <div className="space-y-2">
                  <Slider
                    value={[filters.minPrice || 0]}
                    onValueChange={handlePriceChange}
                    max={1000}
                    step={10}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>0</span>
                    <span className="font-medium">
                      {filters.minPrice ? `${filters.minPrice}c` : "Any"}
                    </span>
                    <span>1000c</span>
                  </div>
                </div>
              </div>

              {/* Confidence Slider */}
              <div className="space-y-2">
                <Label>Min Confidence</Label>
                <div className="space-y-2">
                  <Slider
                    value={[filters.minConfidence ? Math.round(filters.minConfidence * 100) : 0]}
                    onValueChange={handleConfidenceChange}
                    max={100}
                    step={5}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>0%</span>
                    <span className="font-medium">
                      {filters.minConfidence ? `${Math.round(filters.minConfidence * 100)}%` : "Any"}
                    </span>
                    <span>100%</span>
                  </div>
                </div>
              </div>

              {/* Results count */}
              <div className="pt-4 border-t">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">
                    {filteredCurrencies.length} currencies
                  </Badge>
                  {activeFilterCount > 0 && (
                    <span className="text-sm text-muted-foreground">
                      ({availableCurrencies.length} total)
                    </span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Content - Interactive Chart */}
        <div className="flex-1">
          <PriceChart
            data={showChartLoading ? [] : chartData}
            currencyName={selectedCurrency}
            timeRange="30d"
            showPredictionBands={true}
          />
        </div>
      </div>
    </div>
  );
}
