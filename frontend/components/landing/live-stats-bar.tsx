"use client";

/**
 * Live Stats Bar - Shows real-time statistics from the API
 */

import { useCurrencies, useLeagues, useLatestPredictions } from "@/lib/hooks";
import { TrendingUp, Clock, BarChart3, Target } from "lucide-react";
import { formatPercentage } from "@/lib/utils";
import { useMemo } from "react";
import { CurrencyIcon } from "@/components/currency/currency-icon";
import { LATEST_PREDICTIONS_HORIZONS } from "@/lib/constants/predictions";

export function LiveStatsBar() {
  const { data: currenciesData, isLoading: currenciesLoading } = useCurrencies();
  const { data: leaguesData, isLoading: leaguesLoading } = useLeagues();
  
  // Match dashboard: first league from API (seasonal-first sort from backend)
  const preferredLeague = useMemo(() => {
    if (!leaguesData) return null;
    const leagues = Object.keys(leaguesData.leagues);
    return leagues[0] ?? null;
  }, [leaguesData]);

  const { data: predictionsData, isLoading: predictionsLoading } = useLatestPredictions({
    league: preferredLeague || undefined,
    horizons: [...LATEST_PREDICTIONS_HORIZONS],
    limit: 100,
    enabled: !!preferredLeague,
  });

  const stats = useMemo(() => {
    if (!predictionsData?.predictions || !currenciesData) {
      return {
        totalCurrencies: 0,
        topGainer: null,
        topGainerPercent: 0,
        topGainerIcon: null,
        avgConfidence: 0,
      };
    }

    const predictions = Object.entries(predictionsData.predictions);
    let topGainer = "";
    let topGainerPercent = -Infinity;
    let topGainerIcon: string | null = null;
    let totalConfidence = 0;
    let count = 0;

    predictions.forEach(([currency, horizonData]) => {
      const pred = horizonData["1d"];
      if (pred) {
        count++;
        totalConfidence += pred.confidence || 0;
        if (pred.price_change_percent > topGainerPercent) {
          topGainerPercent = pred.price_change_percent;
          topGainer = currency;
          topGainerIcon = currenciesData.currencies[currency]?.[pred.league]?.icon_url || null;
        }
      }
    });

    return {
      totalCurrencies: Object.keys(currenciesData.currencies).length,
      topGainer,
      topGainerPercent,
      topGainerIcon,
      avgConfidence: count > 0 ? Math.round((totalConfidence / count) * 100) : 0,
    };
  }, [predictionsData, currenciesData]);

  const isLoading = currenciesLoading || leaguesLoading || predictionsLoading;

  return (
    <div className="border-y border-border/40 bg-card/30 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-10 py-4 text-sm">
          {/* Total Currencies */}
          <div className="flex items-center gap-2 text-muted-foreground">
            <BarChart3 className="h-4 w-4 text-poe-gold" />
            <span>
              <span className="font-semibold text-foreground">
                {isLoading ? "..." : stats.totalCurrencies}
              </span>{" "}
              currencies tracked
            </span>
          </div>

          {/* Divider */}
          <div className="hidden sm:block h-4 w-px bg-border" />

          {/* Top Gainer */}
          <div className="flex items-center gap-2 text-muted-foreground">
            <TrendingUp className="h-4 w-4 text-green-500" />
            <span>Top gainer:</span>
            {isLoading ? (
              <span className="text-foreground">...</span>
            ) : stats.topGainer ? (
              <span className="flex items-center gap-1.5">
                {stats.topGainerIcon && (
                  <CurrencyIcon
                    iconUrl={stats.topGainerIcon}
                    currency={stats.topGainer}
                    size="sm"
                  />
                )}
                <span className="font-medium text-foreground truncate max-w-[120px]">
                  {stats.topGainer}
                </span>
                <span className="font-semibold text-green-500">
                  {formatPercentage(stats.topGainerPercent)}
                </span>
              </span>
            ) : (
              <span className="text-foreground">N/A</span>
            )}
          </div>

          {/* Divider */}
          <div className="hidden sm:block h-4 w-px bg-border" />

          {/* Average Confidence */}
          <div className="flex items-center gap-2 text-muted-foreground">
            <Target className="h-4 w-4 text-poe-gold" />
            <span>
              <span className="font-semibold text-foreground">
                {isLoading ? "..." : `${stats.avgConfidence}%`}
              </span>{" "}
              avg confidence
            </span>
          </div>

          {/* Divider */}
          <div className="hidden lg:block h-4 w-px bg-border" />

          {/* League indicator */}
          <div className="hidden lg:flex items-center gap-2 text-muted-foreground">
            <Clock className="h-4 w-4 text-poe-gold" />
            <span>
              <span className="font-medium text-foreground">
                {preferredLeague || "Loading..."}
              </span>{" "}
              League
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
