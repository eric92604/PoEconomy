"use client";

/**
 * Dashboard Page - Market overview and key metrics
 */

import { useMemo, useEffect } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  ArrowRight,
  BarChart3,
} from "lucide-react";
import { useCurrencies, useLeagues, useLatestPredictions } from "@/lib/hooks";
import { formatPrice, formatPercentage, formatConfidence } from "@/lib/utils";
import { CurrencyIcon } from "@/components/currency/currency-icon";
import { preloadAllCurrencyIcons } from "@/lib/utils/icon-preloader";
import type { CurrencyWithPredictions } from "@/types";

export default function DashboardPage() {
  // Fetch data
  const { data: currenciesData, isLoading: currenciesLoading } = useCurrencies();
  const { data: leaguesData, isLoading: leaguesLoading } = useLeagues();

  // Preload all currency icons when data is available
  useEffect(() => {
    if (currenciesData && !currenciesLoading) {
      preloadAllCurrencyIcons(currenciesData.currencies).catch(console.warn);
    }
  }, [currenciesData, currenciesLoading]);

  // Get first league for initial data
  const firstLeague = useMemo(() => {
    if (!leaguesData) return null;
    return Object.keys(leaguesData.leagues)[0];
  }, [leaguesData]);

  // Fetch latest predictions using the optimized endpoint
  // Load all currencies for complete statistics
  const { data: predictionsData, isLoading: predictionsLoading } = useLatestPredictions({
    league: firstLeague || undefined,
    horizons: ["1d"], // Only 1d horizon for dashboard
    limit: 500, // Increase limit to get more currencies
  });

  // Calculate stats
  const stats = useMemo(() => {
    if (!predictionsData || !leaguesData || !currenciesData || !predictionsData.predictions) {
      return {
        totalCurrencies: 0,
        totalLeagues: 0,
        topGainers: [],
        topLosers: [],
      };
    }

    // Convert latest predictions data to dashboard format
    const currencies: CurrencyWithPredictions[] = Object.entries(predictionsData.predictions).map(([currency, horizonData]) => {
      const prediction = horizonData["1d"]; // Get 1d prediction
      if (!prediction) return null;

      // Get icon URL from currency metadata
      const currencyMetadata = currenciesData.currencies[currency]?.[prediction.league];
      const iconUrl = currencyMetadata?.icon_url;
      
      return {
        currency: prediction.currency,
        league: prediction.league,
        current_price: prediction.current_price,
        icon_url: iconUrl,
        predictions: {
          "1d": {
            predicted_price: prediction.predicted_price,
            price_change_percent: prediction.price_change_percent,
            confidence: prediction.confidence,
            horizon: "1d",
          },
        },
        average_confidence: prediction.confidence,
      };
    }).filter(Boolean) as CurrencyWithPredictions[];

    // Data is already sorted by profit from the API (highest first)
    const sortedByProfit = [...currencies].sort(
      (a, b) =>
        (b.predictions["1d"]?.price_change_percent || 0) -
        (a.predictions["1d"]?.price_change_percent || 0)
    );

    // Get top gainers (first 5)
    const topGainers = sortedByProfit.slice(0, 5);
    
    // Get top losers (worst 5 performers from our dataset)
    const topLosers = sortedByProfit.slice(-5).reverse(); // Reverse to show worst first

    return {
      totalCurrencies: currencies.length, // Use actual returned count
      totalLeagues: Object.keys(leaguesData.leagues).length,
      topGainers,
      topLosers,
    };
  }, [predictionsData, leaguesData, currenciesData]);

  const isLoading = currenciesLoading || leaguesLoading || predictionsLoading;
  const isInitialLoad = currenciesLoading || leaguesLoading;

  return (
    <div className="py-8 space-y-8">
      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Currencies</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isInitialLoad ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{stats.totalCurrencies}</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Available for trading
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Leagues</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{stats.totalLeagues}</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              With price data
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Top Gainer</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : stats.topGainers.length > 0 ? (
              <>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {formatPercentage(stats.topGainers[0].predictions["1d"]?.price_change_percent || 0)}
                </div>
                <p className="text-xs text-muted-foreground mt-1 truncate">
                  {stats.topGainers[0].currency}
                </p>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Top Loser</CardTitle>
            <TrendingDown className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : stats.topLosers.length > 0 ? (
              <>
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {formatPercentage(stats.topLosers[0].predictions["1d"]?.price_change_percent || 0)}
                </div>
                <p className="text-xs text-muted-foreground mt-1 truncate">
                  {stats.topLosers[0].currency}
                </p>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Top Gainers */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Top Gainers (24h)</CardTitle>
              <CardDescription>Currencies with highest predicted profit</CardDescription>
            </div>
            <Button asChild variant="ghost" size="sm">
              <Link href="/investments">
                View All
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="flex items-center justify-between">
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-4 w-24" />
                </div>
              ))}
            </div>
          ) : stats.topGainers.length > 0 ? (
            <div className="space-y-4">
              {stats.topGainers.map((currency, index) => (
                <div
                  key={currency.currency}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <Badge variant="outline">{index + 1}</Badge>
                    <CurrencyIcon 
                      iconUrl={currency.icon_url} 
                      currency={currency.currency} 
                      size="md" 
                    />
                    <div>
                      <p className="font-medium">{currency.currency}</p>
                      <p className="text-sm text-muted-foreground">
                        {formatPrice(currency.current_price)}c →{" "}
                        {formatPrice(currency.predictions["1d"]?.predicted_price || 0)}c
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-green-600 dark:text-green-400">
                      {formatPercentage(currency.predictions["1d"]?.price_change_percent || 0)}
                    </p>
                    <Badge variant="secondary" className="mt-1">
                      {formatConfidence(currency.average_confidence)}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No data available</p>
          )}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="hover:bg-muted/50 transition-colors cursor-pointer">
          <Link href="/investments">
            <CardHeader>
              <BarChart3 className="h-8 w-8 mb-2 text-primary" />
              <CardTitle>Investment Opportunities</CardTitle>
              <CardDescription>
                Discover the most profitable currency investments
              </CardDescription>
            </CardHeader>
          </Link>
        </Card>

        <Card className="hover:bg-muted/50 transition-colors cursor-pointer">
          <Link href="/prices">
            <CardHeader>
              <Activity className="h-8 w-8 mb-2 text-primary" />
              <CardTitle>Live Prices</CardTitle>
              <CardDescription>
                Track real-time currency prices and trends
              </CardDescription>
            </CardHeader>
          </Link>
        </Card>
      </div>
    </div>
  );
}