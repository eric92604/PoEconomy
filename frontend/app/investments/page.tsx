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
import { TrendingUp, Target, Clock, Search, X } from "lucide-react";
import { useCurrencies, useLeagues, useBatchPredictions } from "@/lib/hooks";
import type { PredictionRequest, CurrencyWithPredictions, CurrencyFilters } from "@/types";
import { filterCurrencies, countActiveFilters } from "@/lib/utils";
import { preloadAllCurrencyIcons, preloadVisibleIcons } from "@/lib/utils/icon-preloader";

export default function InvestmentsPage() {
  const [selectedLeague, setSelectedLeague] = useState<string>("");
  const [selectedTab, setSelectedTab] = useState<string>("short");
  const [filters, setFilters] = useState<CurrencyFilters>({
    search: "",
    minConfidence: undefined,
    minProfit: undefined,
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

  // Transform predictions into currency data
  const currenciesWithPredictions = useMemo((): CurrencyWithPredictions[] => {
    if (!predictionsData || !selectedLeague || !currenciesData) return [];

    const currencyMap = new Map<string, CurrencyWithPredictions>();

    predictionsData.results.forEach((pred) => {
      // Use currency as key since we're filtering by selectedLeague
      const key = pred.currency;
      
      if (!currencyMap.has(key)) {
        // Get icon URL from currency metadata
        const currencyMetadata = currenciesData.currencies[pred.currency]?.[pred.league];
        const iconUrl = currencyMetadata?.icon_url;
        
        currencyMap.set(key, {
          currency: pred.currency,
          league: pred.league,
          current_price: pred.current_price,
          icon_url: iconUrl,
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

    const currencies = Array.from(currencyMap.values());
    
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
      minProfit: value[0] === 0 ? undefined : value[0],
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

  // Filter by timeframe and profit, then apply additional filters
  const filteredCurrencies = useMemo(() => {
    // First filter by timeframe and profit
    const timeframeFiltered = currenciesWithPredictions.filter((currency) => {
      const horizon = selectedTab === "short" ? "1d" : selectedTab === "medium" ? "3d" : "7d";
      const prediction = currency.predictions[horizon];
      return prediction && prediction.price_change_percent > 0;
    });

    // Then apply additional filters
    const filtered = filterCurrencies(timeframeFiltered, filters);

    // Sort by profit percentage
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
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Investment Opportunities</h1>
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
                {currenciesWithPredictions.filter((c) => (c.predictions["1d"]?.price_change_percent ?? 0) > 0).length}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Profitable opportunities
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
                {currenciesWithPredictions.filter((c) => (c.predictions["3d"]?.price_change_percent ?? 0) > 0).length}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Profitable opportunities
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
                {currenciesWithPredictions.filter((c) => (c.predictions["7d"]?.price_change_percent ?? 0) > 0).length}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Profitable opportunities
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

              {/* Min Confidence Slider */}
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

              {/* Min Profit Slider */}
              <div className="space-y-2">
                <Label>Min Profit</Label>
                <div className="space-y-2">
                  <Slider
                    value={[filters.minProfit || 0]}
                    onValueChange={handleProfitChange}
                    max={200}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>0%</span>
                    <span className="font-medium">
                      {filters.minProfit ? `${filters.minProfit}%` : "Any"}
                    </span>
                    <span>200%</span>
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
                        return prediction && prediction.price_change_percent > 0;
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
                Best opportunities for 1-day profit
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
                Best opportunities for 3-day profit
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
                Best opportunities for 7-day profit
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

