"use client";

/**
 * Investments Page - Investment opportunities ranked by profitability
 */

import { useState, useMemo, useEffect } from "react";
import { InvestmentCurrencyTable } from "@/components/currency/investment-currency-table";
import { CurrencyTableSkeleton } from "@/components/currency/currency-table-skeleton";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TrendingUp, Target, Clock, Search, X, RefreshCw } from "lucide-react";
import { useCurrencies, useLeagues, useLatestPredictions } from "@/lib/hooks";
import { useQueryClient } from "@tanstack/react-query";
import type { CurrencyWithPredictions, CurrencyFilters } from "@/types";
import { filterCurrencies, countActiveFilters } from "@/lib/utils";
import { preloadAllCurrencyIcons, preloadVisibleIcons } from "@/lib/utils/icon-preloader";

export default function InvestmentsPage() {
  const queryClient = useQueryClient();
  const [selectedLeague, setSelectedLeague] = useState<string>("");
  const [selectedTab, setSelectedTab] = useState<string>("short");
  const [filters, setFilters] = useState<CurrencyFilters>({
    search: "",
    minConfidence: undefined,
    minProfit: undefined,
    minPrice: undefined,
  });

  // Fetch data
  const { data: currenciesData, isLoading: currenciesLoading } = useCurrencies();
  const { data: leaguesData, isLoading: leaguesLoading } = useLeagues();

  // Preload all currency icons when data is available
  useEffect(() => {
    if (currenciesData && !currenciesLoading) {
      preloadAllCurrencyIcons(currenciesData.currencies).catch(console.warn);
    }
  }, [currenciesData, currenciesLoading]);

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

  // Fetch latest predictions using the optimized endpoint
  // Load all currencies with all horizons for comprehensive investment analysis
  const { data: predictionsData, isLoading: predictionsLoading } = useLatestPredictions({
    league: selectedLeague || undefined,
    horizons: ["1d", "3d", "7d"], // All horizons for investment analysis
    limit: 500, // Increase limit to get more currencies
    enabled: !!selectedLeague, // Only fetch when we have a selected league
  });

  // Manual cache clear function
  const clearCache = () => {
    queryClient.invalidateQueries({ queryKey: ["latest-predictions"] });
  };

  // Transform predictions into currency data
  const currenciesWithPredictions = useMemo((): CurrencyWithPredictions[] => {
    if (!predictionsData || !selectedLeague || !currenciesData || !predictionsData.predictions) return [];

    const currencies: CurrencyWithPredictions[] = [];

    // Convert latest predictions data to investment format
    Object.entries(predictionsData.predictions).forEach(([currency, horizonData]) => {
      // Get icon URL from currency metadata
      const currencyMetadata = currenciesData.currencies[currency]?.[selectedLeague];
      const iconUrl = currencyMetadata?.icon_url;
      
      // Get current price from 1d prediction (most recent)
      const currentPrice = horizonData["1d"]?.current_price || 0;
      
      const currencyData: CurrencyWithPredictions = {
        currency,
        league: selectedLeague,
        current_price: currentPrice,
        icon_url: iconUrl,
        predictions: {},
        average_confidence: 0,
      };

      // Process each horizon
      Object.entries(horizonData).forEach(([horizon, prediction]) => {
        if (prediction) {
          currencyData.predictions[horizon as "1d" | "3d" | "7d"] = {
            predicted_price: prediction.predicted_price,
            price_change_percent: prediction.price_change_percent,
            confidence: prediction.confidence,
            horizon: horizon as "1d" | "3d" | "7d",
            prediction_lower: prediction.prediction_lower,
            prediction_upper: prediction.prediction_upper,
          };
        }
      });

      // Calculate average confidence
      const predictions = Object.values(currencyData.predictions);
      if (predictions.length > 0) {
        currencyData.average_confidence =
          predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
      }

      currencies.push(currencyData);
    });
    
    // Preload icons for visible currencies
    if (currencies.length > 0) {
      preloadVisibleIcons(currencies).catch(console.warn);
    }
    
    return currencies;
  }, [predictionsData, selectedLeague, currenciesData]);

  // Handle filter changes
  const handleSearchChange = (value: string) => {
    setFilters((prev) => ({ ...prev, search: value }));
  };

  const handleConfidenceChange = (value: number[]) => {
    setFilters((prev) => ({
      ...prev,
      minConfidence: value[0] === 0 ? undefined : value[0] / 100,
    }));
  };

  const handleProfitChange = (value: number[]) => {
    setFilters((prev) => ({
      ...prev,
      minProfit: value[0] === -50 ? undefined : value[0],
    }));
  };

  const handlePriceChange = (value: number[]) => {
    setFilters((prev) => ({
      ...prev,
      minPrice: value[0] === 0 ? undefined : value[0],
    }));
  };

  const clearFilters = () => {
    setFilters({
      search: "",
      minConfidence: undefined,
      minProfit: undefined,
      minPrice: undefined,
    });
  };

  const activeFilterCount = countActiveFilters(filters);

  // Filter by timeframe (include all currencies with predictions), then apply additional filters
  const filteredCurrencies = useMemo(() => {
    // First filter by timeframe - include all currencies with predictions (both positive and negative profit)
    const timeframeFiltered = currenciesWithPredictions.filter((currency) => {
      const horizon = selectedTab === "short" ? "1d" : selectedTab === "medium" ? "3d" : "7d";
      const prediction = currency.predictions[horizon];
      return prediction; // Include all currencies with predictions, regardless of profit direction
    });

    // Then apply additional filters
    const filtered = filterCurrencies(timeframeFiltered, filters);

    // Sort by profit percentage (highest to lowest, so negative profits appear at the bottom)
    return filtered.sort((a, b) => {
      const horizon: "1d" | "3d" | "7d" = selectedTab === "short" ? "1d" : selectedTab === "medium" ? "3d" : "7d";
      const aProfit = a.predictions[horizon]?.price_change_percent || 0;
      const bProfit = b.predictions[horizon]?.price_change_percent || 0;
      return bProfit - aProfit;
    });
  }, [currenciesWithPredictions, selectedTab, filters]);

  const isLoading = currenciesLoading || leaguesLoading || predictionsLoading;

  return (
    <div className="py-8 space-y-8">
      {/* Header with Refresh Button */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Investment Opportunities</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Analyze all currency predictions including profitable and declining opportunities
          </p>
        </div>
        <Button 
          onClick={clearCache}
          variant="outline"
          size="sm"
          className="flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Short-Term (1d)</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="text-2xl font-bold">-</div>
            ) : (
              <div className="text-2xl font-bold">
                {currenciesWithPredictions.filter((c) => c.predictions["1d"]).length}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Total opportunities
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Medium-Term (3d)</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="text-2xl font-bold">-</div>
            ) : (
              <div className="text-2xl font-bold">
                {currenciesWithPredictions.filter((c) => c.predictions["3d"]).length}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Total opportunities
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Long-Term (7d)</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="text-2xl font-bold">-</div>
            ) : (
              <div className="text-2xl font-bold">
                {currenciesWithPredictions.filter((c) => c.predictions["7d"]).length}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Total opportunities
            </p>
          </CardContent>
        </Card>
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

              {/* Price Slider */}
              <div className="space-y-2">
                <Label>Price</Label>
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
                      {filters.minPrice ? `${filters.minPrice}` : "Any"}
                    </span>
                    <span>1000</span>
                  </div>
                </div>
              </div>

              {/* Profit Slider */}
              <div className="space-y-2">
                <Label>Profit</Label>
                <div className="space-y-2">
                  <Slider
                    value={[filters.minProfit || -50]}
                    onValueChange={handleProfitChange}
                    min={-50}
                    max={200}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>-50%</span>
                    <span className="font-medium">
                      {filters.minProfit !== undefined ? `${filters.minProfit}%` : "Any"}
                    </span>
                    <span>200%</span>
                  </div>
                </div>
              </div>

              {/* Confidence Slider */}
              <div className="space-y-2">
                <Label>Confidence</Label>
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
                    {filteredCurrencies.length} opportunities found
                  </Badge>
                  {activeFilterCount > 0 && (
                    <span className="text-sm text-muted-foreground">
                      ({currenciesWithPredictions.filter((c) => {
                        const horizon = selectedTab === "short" ? "1d" : selectedTab === "medium" ? "3d" : "7d";
                        const prediction = c.predictions[horizon];
                        return prediction;
                      }).length} total)
                    </span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Content */}
        <div className="flex-1 space-y-8">
          {/* Investment Opportunities by Timeframe */}
          <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full md:w-[400px] grid-cols-3">
          <TabsTrigger value="short">1 day</TabsTrigger>
          <TabsTrigger value="medium">3 days</TabsTrigger>
          <TabsTrigger value="long">7 days</TabsTrigger>
        </TabsList>

        <TabsContent value="short" className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold">Short-Term Investments</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Investment opportunities for 1-day timeframe (includes both profitable and declining currencies)
              </p>
            </div>
            <Badge variant="secondary">
              {filteredCurrencies.length} opportunities
            </Badge>
          </div>
          {isLoading ? (
            <CurrencyTableSkeleton />
          ) : (
            <InvestmentCurrencyTable currencies={filteredCurrencies} timeframe="1d" />
          )}
        </TabsContent>

        <TabsContent value="medium" className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold">Medium-Term Investments</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Investment opportunities for 3-day timeframe (includes both profitable and declining currencies)
              </p>
            </div>
            <Badge variant="secondary">
              {filteredCurrencies.length} opportunities
            </Badge>
          </div>
          {isLoading ? (
            <CurrencyTableSkeleton />
          ) : (
            <InvestmentCurrencyTable currencies={filteredCurrencies} timeframe="3d" />
          )}
        </TabsContent>

        <TabsContent value="long" className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold">Long-Term Investments</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Investment opportunities for 7-day timeframe (includes both profitable and declining currencies)
              </p>
            </div>
            <Badge variant="secondary">
              {filteredCurrencies.length} opportunities
            </Badge>
          </div>
          {isLoading ? (
            <CurrencyTableSkeleton />
          ) : (
            <InvestmentCurrencyTable currencies={filteredCurrencies} timeframe="7d" />
          )}
        </TabsContent>
        </Tabs>
        </div>
      </div>
    </div>
  );
}

