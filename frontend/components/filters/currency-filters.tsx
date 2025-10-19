"use client";

/**
 * Smart Currency Filters Component
 * Provides comprehensive filtering options for the currency table
 */

import { useState, useEffect, useRef } from "react";
import { Search, Filter, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { CurrencyFilters, FilterPreset, CurrencyCategory } from "@/types";
import { DEFAULT_FILTER_PRESETS } from "@/types/filter";
import { getFilterSummary, countActiveFilters } from "@/lib/utils/filtering";

interface CurrencyFiltersProps {
  filters: CurrencyFilters;
  onFiltersChange: (filters: CurrencyFilters) => void;
  availableLeagues: string[];
  availableCategories: CurrencyCategory[];
  totalCount: number;
  filteredCount: number;
}

export function CurrencyFiltersComponent({
  filters,
  onFiltersChange,
  availableLeagues,
  availableCategories,
  totalCount,
  filteredCount,
}: CurrencyFiltersProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchValue, setSearchValue] = useState(filters.search || "");

  // Debounced search handler
  const timeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  
  const handleSearchChange = (value: string) => {
    setSearchValue(value);
    
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(() => {
      onFiltersChange({ ...filters, search: value || undefined });
    }, 300);
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const handleFilterChange = (key: keyof CurrencyFilters, value: unknown) => {
    onFiltersChange({ ...filters, [key]: value });
  };

  const handlePresetSelect = (preset: FilterPreset) => {
    onFiltersChange(preset.filters);
    setSearchValue(preset.filters.search || "");
  };

  const clearAllFilters = () => {
    onFiltersChange({});
    setSearchValue("");
  };

  const activeFilterCount = countActiveFilters(filters);

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4" />
            <CardTitle className="text-sm font-medium">Smart Filters</CardTitle>
            {activeFilterCount > 0 && (
              <Badge variant="secondary" className="text-xs">
                {activeFilterCount}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              {filteredCount} of {totalCount} currencies
            </span>
            {activeFilterCount > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAllFilters}
                className="h-6 px-2 text-xs"
              >
                <X className="h-3 w-3 mr-1" />
                Clear
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Search and Quick Actions */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search currencies..."
              value={searchValue}
              onChange={(e) => handleSearchChange(e.target.value)}
              className="pl-9"
            />
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                Presets
                <ChevronDown className="h-4 w-4 ml-1" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>Quick Filters</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {DEFAULT_FILTER_PRESETS.map((preset) => (
                <DropdownMenuCheckboxItem
                  key={preset.id}
                  onClick={() => handlePresetSelect(preset)}
                  className="flex flex-col items-start"
                >
                  <div className="font-medium">{preset.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {preset.description}
                  </div>
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Filter Summary */}
        {activeFilterCount > 0 && (
          <div className="text-xs text-muted-foreground">
            {getFilterSummary(filters)}
          </div>
        )}

        {/* Advanced Filters Toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full justify-start p-0 h-auto"
        >
          <ChevronDown
            className={`h-4 w-4 mr-2 transition-transform ${
              isExpanded ? "rotate-180" : ""
            }`}
          />
          Advanced Filters
        </Button>

        {/* Advanced Filters */}
        {isExpanded && (
          <div className="space-y-4 pt-2 border-t">
            {/* League Filter */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Leagues</Label>
              <Select
                value={filters.leagues?.[0] || ""}
                onValueChange={(value) =>
                  handleFilterChange("leagues", value ? [value] : undefined)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select league" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Leagues</SelectItem>
                  {availableLeagues.map((league) => (
                    <SelectItem key={league} value={league}>
                      {league}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Category Filter */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Categories</Label>
              <div className="flex flex-wrap gap-1">
                {availableCategories.map((category) => {
                  const isSelected = filters.categories?.includes(category);
                  return (
                    <Button
                      key={category}
                      variant={isSelected ? "default" : "outline"}
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => {
                        const newCategories = isSelected
                          ? filters.categories?.filter((c) => c !== category)
                          : [...(filters.categories || []), category];
                        handleFilterChange(
                          "categories",
                          newCategories && newCategories.length > 0 ? newCategories : undefined
                        );
                      }}
                    >
                      {category}
                    </Button>
                  );
                })}
              </div>
            </div>

            {/* Confidence Threshold */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Min Confidence: {filters.minConfidence ? `${(filters.minConfidence * 100).toFixed(0)}%` : "Any"}
              </Label>
              <Slider
                value={[filters.minConfidence || 0]}
                onValueChange={([value]) =>
                  handleFilterChange("minConfidence", value > 0 ? value : undefined)
                }
                max={1}
                min={0}
                step={0.05}
                className="w-full"
              />
            </div>

            {/* Profit Threshold */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Min Profit: {filters.minProfit ? `${filters.minProfit}%` : "Any"}
              </Label>
              <Slider
                value={[filters.minProfit || 0]}
                onValueChange={([value]) =>
                  handleFilterChange("minProfit", value > 0 ? value : undefined)
                }
                max={50}
                min={0}
                step={1}
                className="w-full"
              />
            </div>

            {/* Price Range */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Price Range</Label>
              <div className="flex gap-2">
                <Input
                  type="number"
                  placeholder="Min price"
                  value={filters.minPrice || ""}
                  onChange={(e) =>
                    handleFilterChange(
                      "minPrice",
                      e.target.value ? Number(e.target.value) : undefined
                    )
                  }
                  className="flex-1"
                />
                <Input
                  type="number"
                  placeholder="Max price"
                  value={filters.maxPrice || ""}
                  onChange={(e) =>
                    handleFilterChange(
                      "maxPrice",
                      e.target.value ? Number(e.target.value) : undefined
                    )
                  }
                  className="flex-1"
                />
              </div>
            </div>

            {/* Additional Options */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Options</Label>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="onlyWithPredictions"
                  checked={filters.onlyWithPredictions || false}
                  onChange={(e) =>
                    handleFilterChange("onlyWithPredictions", e.target.checked || undefined)
                  }
                  className="rounded"
                />
                <Label htmlFor="onlyWithPredictions" className="text-sm">
                  Only currencies with predictions
                </Label>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
