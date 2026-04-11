/**
 * Central export for all API functions
 */

export * from "./client";
export * from "./errors";
export * from "./currencies";
export * from "./predictions";
export * from "./prices";
export * from "./health";
export * from "./latest-predictions";
export * from "./league-history";

// Re-export organized API
export { currencyApi } from "./currencies";
export { predictionApi } from "./predictions";
export { priceApi } from "./prices";
export { healthApi } from "./health";
export { latestPredictionsApi } from "./latest-predictions";
export { leagueHistoryApi } from "./league-history";



