"use client";

/**
 * Prices Page - Live currency prices with auto-refresh
 */

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Search, RefreshCw, TrendingUp, TrendingDown } from "lucide-react";
import { useLeagues, useLivePricesWithRefresh } from "@/lib/hooks";
import { formatPrice, formatRelativeTime } from "@/lib/utils";
import { cn } from "@/lib/utils";

export default function PricesPage() {
  const [selectedLeague, setSelectedLeague] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState("");
  const [timeRange, setTimeRange] = useState("24");

  // Fetch leagues
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

  // Fetch live prices with auto-refresh
  const { data: pricesData, isLoading: pricesLoading } = useLivePricesWithRefresh(
    {
      league: selectedLeague,
      hours: parseInt(timeRange),
      limit: 1000,
    },
    5 * 60 * 1000 // Refresh every 5 minutes
  );

  // Group prices by currency and get latest
  const latestPrices = useMemo(() => {
    if (!pricesData) return [];

    const priceMap = new Map<string, { price: number; timestamp: number; confidence: number }>();

    pricesData.prices.forEach((price) => {
      const existing = priceMap.get(price.currency);
      if (!existing || price.timestamp > existing.timestamp) {
        priceMap.set(price.currency, {
          price: price.price,
          timestamp: price.timestamp,
          confidence: price.confidence,
        });
      }
    });

    return Array.from(priceMap.entries())
      .map(([currency, data]) => ({
        currency,
        ...data,
      }))
      .sort((a, b) => a.currency.localeCompare(b.currency));
  }, [pricesData]);

  // Filter by search term
  const filteredPrices = useMemo(() => {
    if (!searchTerm) return latestPrices;
    return latestPrices.filter((price) =>
      price.currency.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [latestPrices, searchTerm]);

  // Calculate price changes (if we have historical data)
  const pricesWithChanges = useMemo(() => {
    if (!pricesData) return filteredPrices.map(p => ({ ...p, priceChange: 0, priceChangePercent: 0 }));

    return filteredPrices.map((latest) => {
      // Find older price for comparison
      const historicalPrices = pricesData.prices
        .filter((p) => p.currency === latest.currency)
        .sort((a, b) => a.timestamp - b.timestamp);

      if (historicalPrices.length < 2) {
        return { ...latest, priceChange: 0, priceChangePercent: 0 };
      }

      const oldPrice = historicalPrices[0].price;
      const priceChange = latest.price - oldPrice;
      const priceChangePercent = (priceChange / oldPrice) * 100;

      return { ...latest, priceChange, priceChangePercent };
    });
  }, [filteredPrices, pricesData]);

  const isLoading = leaguesLoading || pricesLoading;

  return (
    <div className="py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Live Prices</h1>
        <p className="text-muted-foreground mt-2">
          Real-time currency prices with automatic updates
        </p>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Currencies</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{latestPrices.length}</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              With price data
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Updated</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : pricesData?.metadata.last_updated ? (
              <div className="text-sm font-medium">
                {formatRelativeTime(new Date(pricesData.metadata.last_updated))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">-</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Auto-refreshes every 5 min
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Points</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">
                {pricesData?.metadata.total_count || 0}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              In selected timeframe
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardDescription>Customize your view</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
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

            {/* Time Range */}
            <div className="space-y-2">
              <Label htmlFor="timerange">Time Range</Label>
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger id="timerange">
                  <SelectValue placeholder="Select time range" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="24">Last 24 hours</SelectItem>
                  <SelectItem value="72">Last 3 days</SelectItem>
                  <SelectItem value="168">Last 7 days</SelectItem>
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
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>
          </div>

          <div className="mt-4">
            <Badge variant="secondary">
              {filteredPrices.length} currencies shown
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Prices Table */}
      <Card>
        <CardHeader>
          <CardTitle>Current Prices</CardTitle>
          <CardDescription>Latest prices for all currencies</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              {Array.from({ length: 10 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : pricesWithChanges.length > 0 ? (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Currency</TableHead>
                    <TableHead className="text-right">Current Price</TableHead>
                    <TableHead className="text-right">Change</TableHead>
                    <TableHead className="text-right">Confidence</TableHead>
                    <TableHead className="text-right">Last Updated</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pricesWithChanges.map((price) => (
                    <TableRow key={price.currency}>
                      <TableCell className="font-medium">{price.currency}</TableCell>
                      <TableCell className="text-right font-mono">
                        {formatPrice(price.price)}c
                      </TableCell>
                      <TableCell className="text-right">
                        {price.priceChangePercent !== 0 ? (
                          <div
                            className={cn(
                              "flex items-center justify-end gap-1",
                              price.priceChangePercent > 0
                                ? "text-green-600 dark:text-green-400"
                                : "text-red-600 dark:text-red-400"
                            )}
                          >
                            {price.priceChangePercent > 0 ? (
                              <TrendingUp className="h-3 w-3" />
                            ) : (
                              <TrendingDown className="h-3 w-3" />
                            )}
                            <span className="font-semibold">
                              {price.priceChangePercent > 0 ? "+" : ""}
                              {price.priceChangePercent.toFixed(2)}%
                            </span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge
                          variant={
                            price.confidence >= 0.8
                              ? "default"
                              : price.confidence >= 0.6
                              ? "secondary"
                              : "destructive"
                          }
                        >
                          {(price.confidence * 100).toFixed(0)}%
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right text-sm text-muted-foreground">
                        {formatRelativeTime(price.timestamp)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-64">
              <p className="text-muted-foreground">No prices found</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

