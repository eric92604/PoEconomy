"use client";

/**
 * Investments Page - Investment opportunities ranked by profitability
 */

import { useState, useMemo, useEffect } from "react";
import { CurrencyTable } from "@/components/currency/currency-table";
import { CurrencyTableSkeleton } from "@/components/currency/currency-table-skeleton";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TrendingUp, Target, Clock } from "lucide-react";
import { useCurrencies, useLeagues, useBatchPredictions } from "@/lib/hooks";
import type { PredictionRequest, CurrencyWithPredictions } from "@/types";

export default function InvestmentsPage() {
  const [selectedLeague, setSelectedLeague] = useState<string>("");
  const [selectedTab, setSelectedTab] = useState<string>("short");

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

  // Filter by timeframe and profit
  const filteredCurrencies = useMemo(() => {
    const filtered = currenciesWithPredictions.filter((currency) => {
      const horizon = selectedTab === "short" ? "1d" : selectedTab === "medium" ? "3d" : "7d";
      const prediction = currency.predictions[horizon];
      return prediction && prediction.price_change_percent > 0;
    });

    // Sort by profit percentage
    return filtered.sort((a, b) => {
      const horizon: "1d" | "3d" | "7d" = selectedTab === "short" ? "1d" : selectedTab === "medium" ? "3d" : "7d";
      const aProfit = a.predictions[horizon]?.price_change_percent || 0;
      const bProfit = b.predictions[horizon]?.price_change_percent || 0;
      return bProfit - aProfit;
    });
  }, [currenciesWithPredictions, selectedTab]);

  const isLoading = currenciesLoading || leaguesLoading || predictionsLoading;

  return (
    <div className="py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Investment Opportunities</h1>
        <p className="text-muted-foreground mt-2">
          Discover the most profitable currency investments
        </p>
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

      {/* League Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select League</CardTitle>
          <CardDescription>Choose which league to analyze</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Label htmlFor="league">League</Label>
            <Select value={selectedLeague} onValueChange={setSelectedLeague}>
              <SelectTrigger id="league" className="w-full md:w-[300px]">
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
        </CardContent>
      </Card>

      {/* Investment Opportunities by Timeframe */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full md:w-[400px] grid-cols-3">
          <TabsTrigger value="short">Short-Term (1d)</TabsTrigger>
          <TabsTrigger value="medium">Medium-Term (3d)</TabsTrigger>
          <TabsTrigger value="long">Long-Term (7d)</TabsTrigger>
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
            <CurrencyTable currencies={filteredCurrencies} />
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
            <CurrencyTable currencies={filteredCurrencies} />
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
            <CurrencyTable currencies={filteredCurrencies} />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

