"use client";

/**
 * LeagueSelector - Checkbox list for selecting which previous leagues to overlay
 */

import { LEAGUE_COLORS } from "@/components/charts/league-comparison-chart";

interface LeagueSelectorProps {
  availableLeagues: string[];
  selectedLeagues: string[];
  onToggle: (league: string) => void;
}

export function LeagueSelector({
  availableLeagues,
  selectedLeagues,
  onToggle,
}: LeagueSelectorProps) {
  if (availableLeagues.length === 0) {
    return <p className="text-sm text-muted-foreground">No historical leagues available.</p>;
  }

  return (
    <div className="space-y-2">
      {availableLeagues.map((league, index) => {
        const color = LEAGUE_COLORS[index % LEAGUE_COLORS.length];
        const checked = selectedLeagues.includes(league);
        return (
          <label
            key={league}
            className="flex items-center gap-2 cursor-pointer group"
          >
            <input
              type="checkbox"
              checked={checked}
              onChange={() => onToggle(league)}
              className="h-4 w-4 rounded border-border accent-primary cursor-pointer"
            />
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: color }}
              aria-hidden
            />
            <span className="text-sm leading-none group-hover:text-foreground text-muted-foreground transition-colors">
              {league}
            </span>
          </label>
        );
      })}
    </div>
  );
}
