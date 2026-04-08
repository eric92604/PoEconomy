"use client";

/**
 * Investment Currency Table - Shows only relevant profit column for selected timeframe
 */

import React, { useState, useMemo, useCallback, useEffect, memo } from "react";
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
  formatPercentage,
  formatConfidence,
  formatChaosPrice,
  sortCurrencies,
} from "@/lib/utils";
import { cn } from "@/lib/utils";
import { CurrencyIcon } from "@/components/currency/currency-icon";
import { ChaosPrice } from "@/components/currency/chaos-price";

interface InvestmentCurrencyTableProps {
  currencies: CurrencyWithPredictions[];
  onSelectCurrency?: (currency: CurrencyWithPredictions) => void;
  selectedCurrency?: string;
  timeframe: "1d" | "3d" | "7d";
  expandedContent?: React.ReactNode;
}

export const InvestmentCurrencyTable = memo(function InvestmentCurrencyTable({
  currencies,
  onSelectCurrency,
  selectedCurrency,
  timeframe,
  expandedContent,
}: InvestmentCurrencyTableProps) {
  const [sortField, setSortField] = useState<CurrencySortField>(() =>
    `profit_${timeframe}` as CurrencySortField
  );
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  useEffect(() => {
    setSortField(`profit_${timeframe}` as CurrencySortField);
    setSortDirection("desc");
  }, [timeframe]);

  // Sort currencies (filtering is done at parent level)
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
  const getProfitColor = useCallback((profit: number) => {
    if (profit > 0) return "text-green-600";
    if (profit < 0) return "text-red-600";
    return "text-muted-foreground";
  }, []);

  // Get confidence variant
  const getConfidenceVariant = useCallback((confidence: number): "success" | "secondary" | "destructive" => {
    if (confidence >= 0.8) return "success";
    if (confidence >= 0.6) return "secondary";
    return "destructive";
  }, []);

  if (sortedCurrencies.length === 0) {
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
            const prediction = currency.predictions[timeframe];
            const isExpanded = selectedCurrency === currency.currency;

            return (
              <React.Fragment key={`${currency.currency}-${currency.league}`}>
                <TableRow
                  className={cn(
                    "cursor-pointer hover:bg-muted/50 transition-colors",
                    isExpanded && "bg-muted"
                  )}
                  onClick={() => onSelectCurrency?.(isExpanded ? null as any : currency)}
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
                    <ChaosPrice price={currency.current_price} />
                  </TableCell>
                  <TableCell className="text-right">
                    {prediction ? (
                      <div className="flex flex-col items-end">
                        <ChaosPrice price={prediction.predicted_price} />
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
                        // Clamp negative values to 0.01
                        const lower = Math.max(0.01, predictionWithRange.prediction_lower ?? 0.01);
                        const upper = Math.max(lower, predictionWithRange.prediction_upper ?? 0);
                        
                        return (
                          <div className="text-sm text-muted-foreground font-mono flex items-center justify-end gap-1">
                            <span>{formatChaosPrice(lower)} - {formatChaosPrice(upper)}</span>
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
                  </TableCell>
                  <TableCell className="text-right">
                    <Badge variant={getConfidenceVariant(currency.average_confidence)}>
                      {formatConfidence(currency.average_confidence)}
                    </Badge>
                  </TableCell>
                </TableRow>
                {isExpanded && expandedContent && (
                  <TableRow>
                    <TableCell colSpan={5} className="p-0">
                      <div className="animate-in slide-in-from-top-2 duration-300">
                        {expandedContent}
                      </div>
                    </TableCell>
                  </TableRow>
                )}
              </React.Fragment>
            );
          })}
        </TableBody>
      </Table>
      </div>
    </div>
  );
});
