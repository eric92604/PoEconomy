/**
 * Horizons for GET /predict/latest. Use the same list on dashboard, investments,
 * prices, and landing stats so React Query shares one cache entry per league.
 */
export const LATEST_PREDICTIONS_HORIZONS = ["1d", "3d", "7d"] as const;
