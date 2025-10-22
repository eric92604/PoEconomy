"use client";

/**
 * Main Investment Table - Core feature displaying currencies with predictions
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
} from "@/types";
import {
  formatPrice,
  formatPercentage,
  formatConfidence,
  formatChaosPrice,
  sortCurrencies,
} from "@/lib/utils";
import { cn } from "@/lib/utils";
import { CurrencyIcon } from "@/components/currency/currency-icon";
import { ChaosPrice } from "@/components/currency/chaos-price";

interface CurrencyTableProps {
  currencies: CurrencyWithPredictions[];
  onSelectCurrency?: (currency: CurrencyWithPredictions) => void;
  selectedCurrency?: string;
}

export const CurrencyTable = memo(function CurrencyTable({
  currencies,
  onSelectCurrency,
  selectedCurrency,
}: CurrencyTableProps) {
  const [sortField, setSortField] = useState<CurrencySortField>("profit_1d");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  // Sort currencies (filtering is done at parent level if needed)
  const sortedCurrencies = useMemo(() => {
    return sortCurrencies(currencies, sortField, sortDirection);
  }, [currencies, sortField, sortDirection]);

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
  const getProfitColor = useCallback((percent: number) => {
    if (percent >= 5) return "text-green-600 dark:text-green-400";
    if (percent >= 2) return "text-green-500 dark:text-green-500";
    if (percent > 0) return "text-green-400 dark:text-green-600";
    if (percent === 0) return "text-muted-foreground";
    if (percent > -2) return "text-red-400 dark:text-red-600";
    if (percent > -5) return "text-red-500 dark:text-red-500";
    return "text-red-600 dark:text-red-400";
  }, []);

  // Get confidence badge variant
  const getConfidenceVariant = (confidence: number): "default" | "secondary" | "destructive" => {
    if (confidence >= 0.8) return "default";
    if (confidence >= 0.6) return "secondary";
    return "destructive";
  };

  if (currencies.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 border rounded-lg">
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
                onClick={() => handleSort("profit_1d")}
                className="h-8 px-2"
              >
                1d Profit
                {getSortIcon("profit_1d")}
              </Button>
            </TableHead>
            <TableHead className="text-right">
              <Button
                variant="ghost"
                onClick={() => handleSort("profit_3d")}
                className="h-8 px-2"
              >
                3d Profit
                {getSortIcon("profit_3d")}
              </Button>
            </TableHead>
            <TableHead className="text-right">
              <Button
                variant="ghost"
                onClick={() => handleSort("profit_7d")}
                className="h-8 px-2"
              >
                7d Profit
                {getSortIcon("profit_7d")}
              </Button>
            </TableHead>
            <TableHead>Price Range</TableHead>
            <TableHead className="text-right w-24">
              <Button
                variant="ghost"
                onClick={() => handleSort("average_confidence")}
                className="h-8 px-2"
              >
                Confidence
                {getSortIcon("average_confidence")}
              </Button>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedCurrencies.map((currency: CurrencyWithPredictions) => {
            const pred1d = currency.predictions["1d"];
            const pred3d = currency.predictions["3d"];
            const pred7d = currency.predictions["7d"];

            return (
              <TableRow
                key={`${currency.currency}-${currency.league}`}
                className={cn(
                  "cursor-pointer hover:bg-muted/50",
                  selectedCurrency === currency.currency && "bg-muted"
                )}
                onClick={() => onSelectCurrency?.(currency)}
              >
                <TableCell className="font-medium">
                  <div className="flex items-center gap-3">
                    <CurrencyIcon 
                      iconUrl={currency.icon_url} 
                      currency={currency.currency} 
                      size="md" 
                    />
                    <div className="flex flex-col">
                      <span>{currency.currency}</span>
                      <span className="text-xs text-muted-foreground">
                        {currency.league}
                      </span>
                    </div>
                  </div>
                </TableCell>
                <TableCell className="text-right">
                  <ChaosPrice price={currency.current_price} />
                </TableCell>
                <TableCell className="text-right">
                  {pred1d ? (
                    <div className="flex flex-col items-end">
                      <ChaosPrice price={pred1d.predicted_price} />
                      <span
                        className={cn(
                          "text-sm font-semibold flex items-center gap-1",
                          getProfitColor(pred1d.price_change_percent)
                        )}
                      >
                        {pred1d.price_change_percent > 0 ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : pred1d.price_change_percent < 0 ? (
                          <TrendingDown className="h-3 w-3" />
                        ) : null}
                        {formatPercentage(pred1d.price_change_percent)}
                      </span>
                    </div>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  {pred3d ? (
                    <div className="flex flex-col items-end">
                      <ChaosPrice price={pred3d.predicted_price} />
                      <span
                        className={cn(
                          "text-sm font-semibold flex items-center gap-1",
                          getProfitColor(pred3d.price_change_percent)
                        )}
                      >
                        {pred3d.price_change_percent > 0 ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : pred3d.price_change_percent < 0 ? (
                          <TrendingDown className="h-3 w-3" />
                        ) : null}
                        {formatPercentage(pred3d.price_change_percent)}
                      </span>
                    </div>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  {pred7d ? (
                    <div className="flex flex-col items-end">
                      <ChaosPrice price={pred7d.predicted_price} />
                      <span
                        className={cn(
                          "text-sm font-semibold flex items-center gap-1",
                          getProfitColor(pred7d.price_change_percent)
                        )}
                      >
                        {pred7d.price_change_percent > 0 ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : pred7d.price_change_percent < 0 ? (
                          <TrendingDown className="h-3 w-3" />
                        ) : null}
                        {formatPercentage(pred7d.price_change_percent)}
                      </span>
                    </div>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  {(() => {
                    // Try to find a prediction with price range data (prefer 1d, then 3d, then 7d)
                    const predictionWithRange = pred1d?.prediction_lower !== undefined && pred1d?.prediction_upper !== undefined ? pred1d :
                                             pred3d?.prediction_lower !== undefined && pred3d?.prediction_upper !== undefined ? pred3d :
                                             pred7d?.prediction_lower !== undefined && pred7d?.prediction_upper !== undefined ? pred7d :
                                             null;

                    if (predictionWithRange) {
                      return (
                        <div className="text-sm text-muted-foreground font-mono flex items-center justify-end gap-1">
                          <span>{formatChaosPrice(predictionWithRange.prediction_lower!)} - {formatChaosPrice(predictionWithRange.prediction_upper!)}</span>
                          <CurrencyIcon 
                            iconUrl="/images/chaos-orb.png" 
                            currency="Chaos Orb" 
                            size="sm" 
                          />
                        </div>
                      );
                    }

                    return <span className="text-muted-foreground">-</span>;
                  })()}
                  {/* Debug info for Chaos Orb */}
                  {currency.currency === "Chaos Orb" && (
                    <div className="text-xs text-red-500 mt-1">
                      Debug: 1d=({pred1d?.prediction_lower}, {pred1d?.prediction_upper}) 
                      3d=({pred3d?.prediction_lower}, {pred3d?.prediction_upper}) 
                      7d=({pred7d?.prediction_lower}, {pred7d?.prediction_upper})
                    </div>
                  )}
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

