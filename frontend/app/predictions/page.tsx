"use client";

/**
 * Predictions Page - Main investment table with currency predictions
 */

import { useState, useMemo, useEffect } from "react";
import { CurrencyTable } from "@/components/currency/currency-table";
import { CurrencyTableSkeleton } from "@/components/currency/currency-table-skeleton";
import { PriceChart } from "@/components/charts/price-chart";
import { PriceChartSkeleton } from "@/components/charts/price-chart-skeleton";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Search, X } from "lucide-react";
import { useCurrencies, useLeagues, useBatchPredictions, useLivePrices } from "@/lib/hooks";
import type {
  CurrencyWithPredictions,
  PredictionRequest,
  ChartDataPoint,
  CurrencyFilters,
} from "@/types";
import { filterCurrencies, countActiveFilters } from "@/lib/utils";

export default function PredictionsPage() {
  // State
  const [selectedCurrency, setSelectedCurrency] = useState<CurrencyWithPredictions | null>(null);
  const [selectedLeague, setSelectedLeague] = useState<string>("");
  const [filters, setFilters] = useState<CurrencyFilters>({
    search: "",
    minConfidence: undefined,
    minProfit: undefined,
  });

  // Fetch data
  const { data: currenciesData, isLoading: currenciesLoading } = useCurrencies();
  const { data: leaguesData, isLoading: leaguesLoading } = useLeagues();

  // Get available leagues
  const leagues = useMemo(() => {
    if (!leaguesData) return [];
    return Object.keys(leaguesData.leagues);
  }, [leaguesData]);

  // Set default league
  useEffect(() => {
    if (leagues.length > 0 && !selectedLeague) {
      setSelectedLeague(leagues[0]);
    }
  }, [leagues, selectedLeague]);

  // Prepare batch prediction requests
  const predictionRequests = useMemo((): PredictionRequest[] => {
    if (!currenciesData || !selectedLeague) return [];

    const currencies = Object.keys(currenciesData.currencies);
    const requests: PredictionRequest[] = [];

    currencies.forEach((currency) => {
      ["1d", "3d", "7d"].forEach((horizon) => {
        requests.push({
          currency,
          league: selectedLeague,
          horizon: horizon as "1d" | "3d" | "7d",
        });
      });
    });

    return requests;
  }, [currenciesData, selectedLeague]);

  // Fetch batch predictions
  const { data: predictionsData, isLoading: predictionsLoading } = useBatchPredictions(
    { requests: predictionRequests },
    predictionRequests.length > 0
  );

  // Fetch price history for selected currency
  const { data: priceData, isLoading: priceLoading } = useLivePrices(
    {
      currency: selectedCurrency?.currency,
      league: selectedLeague,
      hours: 168, // 7 days
      limit: 500,
    },
    !!selectedCurrency
  );

  // Transform predictions into currency data
  const currenciesWithPredictions = useMemo((): CurrencyWithPredictions[] => {
    if (!predictionsData || !selectedLeague) return [];

    const currencyMap = new Map<string, CurrencyWithPredictions>();

    predictionsData.results.forEach((pred) => {
      const key = pred.currency;
      
      if (!currencyMap.has(key)) {
        currencyMap.set(key, {
          currency: pred.currency,
          league: pred.league,
          current_price: pred.current_price,
          predictions: {},
          average_confidence: 0,
        });
      }

      const currencyData = currencyMap.get(key)!;
      currencyData.predictions[pred.horizon] = {
        predicted_price: pred.predicted_price,
        price_change_percent: pred.price_change_percent,
        confidence: pred.confidence,
        prediction_lower: pred.metadata?.prediction_lower,
        prediction_upper: pred.metadata?.prediction_upper,
        horizon: pred.horizon,
      };
    });

    // Calculate average confidence
    currencyMap.forEach((currency) => {
      const predictions = Object.values(currency.predictions);
      if (predictions.length > 0) {
        currency.average_confidence =
          predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
      }
    });

    return Array.from(currencyMap.values());
  }, [predictionsData, selectedLeague]);

  // Apply filters
  const filteredCurrencies = useMemo(() => {
    return filterCurrencies(currenciesWithPredictions, filters);
  }, [currenciesWithPredictions, filters]);

  // Transform price data for chart
  const chartData = useMemo((): ChartDataPoint[] => {
    if (!priceData || !selectedCurrency) return [];

    const data: ChartDataPoint[] = priceData.prices.map((price) => ({
      timestamp: price.timestamp,
      date: new Date(price.timestamp * 1000),
      price: price.price,
      confidence: price.confidence,
      predicted: false,
    }));

    // Add prediction points
    if (selectedCurrency.predictions["1d"]) {
      const lastPrice = data[data.length - 1];
      if (lastPrice) {
        data.push({
          timestamp: lastPrice.timestamp + 86400, // +1 day
          date: new Date((lastPrice.timestamp + 86400) * 1000),
          price: selectedCurrency.predictions["1d"].predicted_price,
          predicted: true,
          prediction_lower: selectedCurrency.predictions["1d"].prediction_lower,
          prediction_upper: selectedCurrency.predictions["1d"].prediction_upper,
        });
      }
    }

    return data;
  }, [priceData, selectedCurrency]);

  // Handle filter changes
  const handleSearchChange = (value: string) => {
    setFilters((prev) => ({ ...prev, search: value }));
  };

  const handleConfidenceChange = (value: string) => {
    setFilters((prev) => ({
      ...prev,
      minConfidence: value === "all" ? undefined : parseFloat(value),
    }));
  };

  const handleProfitChange = (value: string) => {
    setFilters((prev) => ({
      ...prev,
      minProfit: value === "all" ? undefined : parseFloat(value),
    }));
  };

  const clearFilters = () => {
    setFilters({
      search: "",
      minConfidence: undefined,
      minProfit: undefined,
    });
  };

  const activeFilterCount = countActiveFilters(filters);

  const isLoading = currenciesLoading || leaguesLoading || predictionsLoading;

  return (
    <div className="py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Currency Predictions</h1>
        <p className="text-muted-foreground mt-2">
          View price predictions and profit opportunities for all currencies
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Filters</CardTitle>
              <CardDescription>Refine your search</CardDescription>
            </div>
            {activeFilterCount > 0 && (
              <Button variant="ghost" size="sm" onClick={clearFilters}>
                <X className="h-4 w-4 mr-2" />
                Clear ({activeFilterCount})
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
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

            {/* Search */}
            <div className="space-y-2">
              <Label htmlFor="search">Search Currency</Label>
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="search"
                  placeholder="Search..."
                  value={filters.search || ""}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>

            {/* Min Confidence */}
            <div className="space-y-2">
              <Label htmlFor="confidence">Min Confidence</Label>
              <Select
                value={filters.minConfidence?.toString() || "all"}
                onValueChange={handleConfidenceChange}
              >
                <SelectTrigger id="confidence">
                  <SelectValue placeholder="Any" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any</SelectItem>
                  <SelectItem value="0.9">90%+</SelectItem>
                  <SelectItem value="0.8">80%+</SelectItem>
                  <SelectItem value="0.7">70%+</SelectItem>
                  <SelectItem value="0.6">60%+</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Min Profit */}
            <div className="space-y-2">
              <Label htmlFor="profit">Min Profit</Label>
              <Select
                value={filters.minProfit?.toString() || "all"}
                onValueChange={handleProfitChange}
              >
                <SelectTrigger id="profit">
                  <SelectValue placeholder="Any" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any</SelectItem>
                  <SelectItem value="10">10%+</SelectItem>
                  <SelectItem value="5">5%+</SelectItem>
                  <SelectItem value="2">2%+</SelectItem>
                  <SelectItem value="1">1%+</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Results count */}
          <div className="mt-4 flex items-center gap-2">
            <Badge variant="secondary">
              {filteredCurrencies.length} currencies found
            </Badge>
            {activeFilterCount > 0 && (
              <span className="text-sm text-muted-foreground">
                ({currenciesWithPredictions.length} total)
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Main Table */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Investment Opportunities</h2>
        {isLoading ? (
          <CurrencyTableSkeleton />
        ) : (
          <CurrencyTable
            currencies={filteredCurrencies}
            onSelectCurrency={setSelectedCurrency}
            selectedCurrency={selectedCurrency?.currency}
          />
        )}
      </div>

      {/* Price Chart */}
      {selectedCurrency && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">Price History</h2>
          {priceLoading ? (
            <PriceChartSkeleton />
          ) : (
            <PriceChart
              data={chartData}
              currencyName={selectedCurrency.currency}
              timeRange="7d"
              showPredictionBands={true}
            />
          )}
        </div>
      )}
    </div>
  );
}

