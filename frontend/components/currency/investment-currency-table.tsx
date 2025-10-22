"use client";

/**
 * Investment Currency Table - Shows only relevant profit column for selected timeframe
 */

import { useState, useMemo, useCallback, memo } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpDown, ArrowUp, ArrowDown, TrendingUp, TrendingDown } from "lucide-react";
import type {
  CurrencyWithPredictions,
  CurrencySortField,
  SortDirection,
  CurrencyFilters,
  CurrencyCategory,
} from "@/types";
import {
  formatPrice,
  formatPercentage,
  formatConfidence,
  sortCurrencies,
  filterCurrencies,
} from "@/lib/utils";
import { cn } from "@/lib/utils";
import { CurrencyIcon } from "@/components/currency/currency-icon";

interface InvestmentCurrencyTableProps {
  currencies: CurrencyWithPredictions[];
  onSelectCurrency?: (currency: CurrencyWithPredictions) => void;
  selectedCurrency?: string;
  timeframe: "1d" | "3d" | "7d";
}

export const InvestmentCurrencyTable = memo(function InvestmentCurrencyTable({
  currencies,
  onSelectCurrency,
  selectedCurrency,
  timeframe,
}: InvestmentCurrencyTableProps) {
  const [sortField, setSortField] = useState<CurrencySortField>("profit_1d");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [filters, setFilters] = useState<CurrencyFilters>({});

  // Get available leagues and categories for filters
  const availableLeagues = useMemo(() => {
    return Array.from(new Set(currencies.map(c => c.league))).sort();
  }, [currencies]);

  const availableCategories = useMemo(() => {
    return Array.from(new Set(currencies.map(c => c.category).filter(Boolean))) as CurrencyCategory[];
  }, [currencies]);

  // Filter and sort currencies
  const filteredAndSortedCurrencies = useMemo(() => {
    const filtered = filterCurrencies(currencies, filters);
    return sortCurrencies(filtered, sortField, sortDirection);
  }, [currencies, filters, sortField, sortDirection]);

  // Handle sort
  const handleSort = useCallback((field: CurrencySortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  }, [sortField, sortDirection]);

  // Get sort icon
  const getSortIcon = useCallback((field: CurrencySortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="ml-2 h-4 w-4" />;
    }
    return sortDirection === "asc" ? (
      <ArrowUp className="ml-2 h-4 w-4" />
    ) : (
      <ArrowDown className="ml-2 h-4 w-4" />
    );
  }, [sortField, sortDirection]);

  // Get profit color
  const getProfitColor = useCallback((profit: number) => {
    if (profit > 0) return "text-green-600";
    if (profit < 0) return "text-red-600";
    return "text-muted-foreground";
  }, []);

  // Get confidence variant
  const getConfidenceVariant = useCallback((confidence: number): "default" | "secondary" | "destructive" => {
    if (confidence >= 0.8) return "default";
    if (confidence >= 0.6) return "secondary";
    return "destructive";
  }, []);

  if (filteredAndSortedCurrencies.length === 0) {
    return (
      <div className="flex items-center justify-center py-8">
        <p className="text-muted-foreground">No currencies found</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Currency Table */}
      <div className="rounded-md border">
        <Table>
        <TableHeader>
          <TableRow>
            <TableHead>
              <Button
                variant="ghost"
                onClick={() => handleSort("currency")}
                className="h-8 px-2"
              >
                Currency
                {getSortIcon("currency")}
              </Button>
            </TableHead>
            <TableHead className="text-right">
              <Button
                variant="ghost"
                onClick={() => handleSort("current_price")}
                className="h-8 px-2"
              >
                Current Price
                {getSortIcon("current_price")}
              </Button>
            </TableHead>
            <TableHead className="text-right">
              <Button
                variant="ghost"
                onClick={() => handleSort(`profit_${timeframe}` as CurrencySortField)}
                className="h-8 px-2"
              >
                {timeframe === "1d" ? "1d" : timeframe === "3d" ? "3d" : "7d"} Profit
                {getSortIcon(`profit_${timeframe}` as CurrencySortField)}
              </Button>
            </TableHead>
            <TableHead className="text-right">Price Range</TableHead>
            <TableHead className="text-right">
              <Button
                variant="ghost"
                onClick={() => handleSort("average_confidence")}
                className="h-8 px-2"
              >
                Avg Confidence
                {getSortIcon("average_confidence")}
              </Button>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filteredAndSortedCurrencies.map((currency: CurrencyWithPredictions) => {
            const prediction = currency.predictions[timeframe];

            return (
              <TableRow
                key={`${currency.currency}-${currency.league}`}
                className={cn(
                  "cursor-pointer hover:bg-muted/50",
                  selectedCurrency === currency.currency && "bg-muted"
                )}
                onClick={() => onSelectCurrency?.(currency)}
              >
                <TableCell>
                  <div className="flex items-center gap-3">
                    <CurrencyIcon
                      iconUrl={currency.icon_url}
                      currency={currency.currency}
                      size="sm"
                    />
                    <div>
                      <div className="font-medium">{currency.currency}</div>
                      <div className="text-sm text-muted-foreground">{currency.league}</div>
                    </div>
                  </div>
                </TableCell>
                <TableCell className="text-right">
                  <span className="font-mono">
                    {formatPrice(currency.current_price)}c
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  {prediction ? (
                    <div className="flex flex-col items-end">
                      <span className="font-mono">
                        {formatPrice(prediction.predicted_price)}c
                      </span>
                      <span
                        className={cn(
                          "text-sm font-semibold flex items-center gap-1",
                          getProfitColor(prediction.price_change_percent)
                        )}
                      >
                        {prediction.price_change_percent > 0 ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : prediction.price_change_percent < 0 ? (
                          <TrendingDown className="h-3 w-3" />
                        ) : null}
                        {formatPercentage(prediction.price_change_percent)}
                      </span>
                    </div>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  {(() => {
                    // Try to find a prediction with price range data (prefer current timeframe, then others)
                    const predictionWithRange = prediction?.prediction_lower !== undefined && prediction?.prediction_upper !== undefined ? prediction :
                                             currency.predictions["1d"]?.prediction_lower !== undefined && currency.predictions["1d"]?.prediction_upper !== undefined ? currency.predictions["1d"] :
                                             currency.predictions["3d"]?.prediction_lower !== undefined && currency.predictions["3d"]?.prediction_upper !== undefined ? currency.predictions["3d"] :
                                             currency.predictions["7d"]?.prediction_lower !== undefined && currency.predictions["7d"]?.prediction_upper !== undefined ? currency.predictions["7d"] :
                                             null;

                    if (predictionWithRange) {
                      return (
                        <span className="text-sm text-muted-foreground font-mono">
                          {formatPrice(predictionWithRange.prediction_lower!)} -{" "}
                          {formatPrice(predictionWithRange.prediction_upper!)}
                        </span>
                      );
                    }

                    return <span className="text-muted-foreground">-</span>;
                  })()}
                </TableCell>
                <TableCell className="text-right">
                  <Badge variant={getConfidenceVariant(currency.average_confidence)}>
                    {formatConfidence(currency.average_confidence)}
                  </Badge>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      </div>
    </div>
  );
});
