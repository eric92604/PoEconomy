"use client";

/**
 * Currency Selector - Select currency, league, and prediction horizon
 */

import { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { PredictionHorizon } from "@/types";
import { STORAGE_KEYS } from "@/lib/constants/config";

interface CurrencySelectorProps {
  currencies: string[];
  leagues: string[];
  selectedCurrency?: string;
  selectedLeague?: string;
  selectedHorizon?: PredictionHorizon;
  onCurrencyChange?: (currency: string) => void;
  onLeagueChange?: (league: string) => void;
  onHorizonChange?: (horizon: PredictionHorizon) => void;
}

export function CurrencySelector({
  currencies,
  leagues,
  selectedCurrency,
  selectedLeague,
  selectedHorizon = "1d",
  onCurrencyChange,
  onLeagueChange,
  onHorizonChange,
}: CurrencySelectorProps) {
  const [searchTerm, setSearchTerm] = useState("");

  // Filter currencies by search term
  const filteredCurrencies = currencies.filter((currency) =>
    currency.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Save preferences to localStorage
  useEffect(() => {
    if (selectedCurrency || selectedLeague || selectedHorizon) {
      const preferences = {
        currency: selectedCurrency,
        league: selectedLeague,
        horizon: selectedHorizon,
      };
      localStorage.setItem(STORAGE_KEYS.PREFERENCES, JSON.stringify(preferences));
    }
  }, [selectedCurrency, selectedLeague, selectedHorizon]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Select Currency</CardTitle>
        <CardDescription>Choose a currency, league, and time horizon</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Currency Search */}
        <div className="space-y-2">
          <Label htmlFor="currency-search">Search Currency</Label>
          <Input
            id="currency-search"
            placeholder="Search currencies..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        {/* Currency Select */}
        <div className="space-y-2">
          <Label htmlFor="currency-select">Currency</Label>
          <Select value={selectedCurrency} onValueChange={onCurrencyChange}>
            <SelectTrigger id="currency-select">
              <SelectValue placeholder="Select a currency" />
            </SelectTrigger>
            <SelectContent>
              {filteredCurrencies.length > 0 ? (
                filteredCurrencies.map((currency) => (
                  <SelectItem key={currency} value={currency}>
                    {currency}
                  </SelectItem>
                ))
              ) : (
                <div className="p-2 text-sm text-muted-foreground">
                  No currencies found
                </div>
              )}
            </SelectContent>
          </Select>
        </div>

        {/* League Select */}
        <div className="space-y-2">
          <Label htmlFor="league-select">League</Label>
          <Select value={selectedLeague} onValueChange={onLeagueChange}>
            <SelectTrigger id="league-select">
              <SelectValue placeholder="Select a league" />
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

        {/* Horizon Select */}
        <div className="space-y-2">
          <Label htmlFor="horizon-select">Prediction Horizon</Label>
          <Select
            value={selectedHorizon}
            onValueChange={(value) => onHorizonChange?.(value as PredictionHorizon)}
          >
            <SelectTrigger id="horizon-select">
              <SelectValue placeholder="Select horizon" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1d">1 Day</SelectItem>
              <SelectItem value="3d">3 Days</SelectItem>
              <SelectItem value="7d">7 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  );
}



