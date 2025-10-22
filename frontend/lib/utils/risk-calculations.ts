/**
 * Risk assessment calculation utilities
 */

import type { RiskLevel, CurrencyPrediction } from "@/types";

/**
 * Calculate risk score (0-100)
 * Lower confidence and higher volatility = higher risk
 */
export function calculateRiskScore(
  confidence: number,
  volatility: number,
  priceRange?: { lower: number; upper: number; current: number }
): number {
  // Base risk from confidence (inverse)
  const confidenceRisk = (1 - confidence) * 100;

  // Risk from volatility (normalized)
  const volatilityRisk = Math.min(volatility * 100, 100);

  // Risk from price range uncertainty
  let rangeRisk = 0;
  if (priceRange) {
    const range = priceRange.upper - priceRange.lower;
    const rangePercent = (range / priceRange.current) * 100;
    rangeRisk = Math.min(rangePercent, 100);
  }

  // Weighted average
  const weights = priceRange ? [0.4, 0.3, 0.3] : [0.6, 0.4];
  const risks = priceRange
    ? [confidenceRisk, volatilityRisk, rangeRisk]
    : [confidenceRisk, volatilityRisk];

  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  const weightedRisk = risks.reduce((sum, risk, i) => sum + risk * weights[i], 0) / totalWeight;

  return Math.min(Math.max(weightedRisk, 0), 100);
}

/**
 * Classify risk level based on score
 */
export function classifyRiskLevel(riskScore: number): RiskLevel {
  if (riskScore < 25) return "low";
  if (riskScore < 50) return "medium";
  if (riskScore < 75) return "high";
  return "very-high";
}

/**
 * Calculate volatility from price history
 */
export function calculateVolatility(prices: number[]): number {
  if (prices.length < 2) return 0;

  // Calculate returns
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }

  // Calculate standard deviation of returns
  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance =
    returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
  
  return Math.sqrt(variance);
}

/**
 * Calculate prediction uncertainty
 */
export function calculateUncertainty(
  predictedPrice: number,
  predictionLower?: number,
  predictionUpper?: number
): number {
  if (!predictionLower || !predictionUpper) return 0;
  
  const range = predictionUpper - predictionLower;
  return (range / predictedPrice) * 100; // Uncertainty as percentage
}

/**
 * Assess portfolio diversification
 */
export function assessDiversification(
  categories: Record<string, number>,
  riskLevels: Record<RiskLevel, number>
): number {
  // Calculate entropy for categories (higher entropy = better diversification)
  const categoryValues = Object.values(categories);
  const totalCategories = categoryValues.reduce((sum, val) => sum + val, 0);
  
  let categoryEntropy = 0;
  if (totalCategories > 0) {
    for (const count of categoryValues) {
      if (count > 0) {
        const p = count / totalCategories;
        categoryEntropy -= p * Math.log2(p);
      }
    }
  }

  // Calculate entropy for risk levels
  const riskValues = Object.values(riskLevels);
  const totalRisk = riskValues.reduce((sum, val) => sum + val, 0);
  
  let riskEntropy = 0;
  if (totalRisk > 0) {
    for (const count of riskValues) {
      if (count > 0) {
        const p = count / totalRisk;
        riskEntropy -= p * Math.log2(p);
      }
    }
  }

  // Normalize to 0-100 scale
  // Max entropy for 4 categories = log2(4) = 2
  // Max entropy for 4 risk levels = log2(4) = 2
  const maxEntropy = 2;
  const categoryScore = (categoryEntropy / maxEntropy) * 50;
  const riskScore = (riskEntropy / maxEntropy) * 50;

  return Math.min(categoryScore + riskScore, 100);
}

/**
 * Calculate investment recommendation score
 */
export function calculateRecommendationScore(
  profitPercent: number,
  confidence: number,
  riskScore: number
): number {
  // Weighted score: profit (40%), confidence (40%), risk (20% inverse)
  const profitScore = Math.min(profitPercent * 2, 100); // Normalize profit to 0-100
  const confidenceScore = confidence * 100;
  const riskAdjustment = 100 - riskScore;

  return (profitScore * 0.4 + confidenceScore * 0.4 + riskAdjustment * 0.2);
}

/**
 * Determine optimal investment timeframe
 */
export function determineOptimalTimeframe(predictions: {
  "1d"?: CurrencyPrediction;
  "3d"?: CurrencyPrediction;
  "7d"?: CurrencyPrediction;
}): "short" | "medium" | "long" {
  const scores = {
    "1d": predictions["1d"] ? predictions["1d"].price_change_percent * predictions["1d"].confidence : 0,
    "3d": predictions["3d"] ? predictions["3d"].price_change_percent * predictions["3d"].confidence : 0,
    "7d": predictions["7d"] ? predictions["7d"].price_change_percent * predictions["7d"].confidence : 0,
  };

  if (scores["1d"] >= scores["3d"] && scores["1d"] >= scores["7d"]) {
    return "short";
  }
  if (scores["7d"] >= scores["3d"]) {
    return "long";
  }
  return "medium";
}



