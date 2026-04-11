import { apiClient } from "./client";
import type { LeagueHistoricalResponse, LeagueHistoricalSeries, HistoricalLeaguesResponse } from "@/types/api";

export interface LeagueHistoricalParams {
  currency: string;
  league: string;
  start_date?: string;
  end_date?: string;
}

export async function getHistoricalLeagues(): Promise<HistoricalLeaguesResponse> {
  return apiClient.get<HistoricalLeaguesResponse>("/prices/leagues");
}

/**
 * Fetch historical prices for a single (currency, league) pair.
 * Called once per selected league so React Query can cache each independently.
 */
export async function getLeagueHistoricalPrices(
  params: LeagueHistoricalParams
): Promise<LeagueHistoricalSeries> {
  const searchParams = new URLSearchParams();
  searchParams.append("currency", params.currency);
  searchParams.append("league", params.league);
  if (params.start_date) searchParams.append("start_date", params.start_date);
  if (params.end_date) searchParams.append("end_date", params.end_date);
  const response = await apiClient.get<LeagueHistoricalResponse>(
    `/prices/league-history?${searchParams}`
  );
  // The backend returns { currency, leagues: { [league]: series } } — unwrap the single entry
  return response.leagues[params.league] ?? { league_start_date: null, count: 0, prices: [] };
}

export const leagueHistoryApi = {
  getHistoricalLeagues,
  getLeagueHistoricalPrices,
};
