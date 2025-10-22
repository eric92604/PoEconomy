/**
 * Profit and ROI calculation utilities
 */

import type { CurrencyPrediction } from "@/types";

/**
 * Calculate profit percentage
 */
export function calculateProfitPercent(currentPrice: number, predictedPrice: number): number {
  if (currentPrice === 0) return 0;
  return ((predictedPrice - currentPrice) / currentPrice) * 100;
}

/**
 * Calculate absolute profit
 */
export function calculateProfit(currentPrice: number, predictedPrice: number): number {
  return predictedPrice - currentPrice;
}

/**
 * Calculate ROI (Return on Investment)
 */
export function calculateROI(
  investmentAmount: number,
  currentPrice: number,
  predictedPrice: number
): number {
  if (currentPrice === 0) return 0;
  const quantity = investmentAmount / currentPrice;
  const futureValue = quantity * predictedPrice;
  return futureValue - investmentAmount;
}

/**
 * Calculate ROI percentage
 */
export function calculateROIPercent(
  investmentAmount: number,
  currentPrice: number,
  predictedPrice: number
): number {
  if (investmentAmount === 0) return 0;
  const roi = calculateROI(investmentAmount, currentPrice, predictedPrice);
  return (roi / investmentAmount) * 100;
}

/**
 * Calculate weighted average of predictions
 */
export function calculateWeightedAverage(predictions: CurrencyPrediction[]): number {
  if (predictions.length === 0) return 0;

  const totalWeight = predictions.reduce((sum, pred) => sum + pred.confidence, 0);
  if (totalWeight === 0) return 0;

  const weightedSum = predictions.reduce(
    (sum, pred) => sum + pred.predicted_price * pred.confidence,
    0
  );

  return weightedSum / totalWeight;
}

/**
 * Calculate average confidence across predictions
 */
export function calculateAverageConfidence(predictions: CurrencyPrediction[]): number {
  if (predictions.length === 0) return 0;
  const sum = predictions.reduce((acc, pred) => acc + pred.confidence, 0);
  return sum / predictions.length;
}

/**
 * Calculate profit potential score (combines profit % and confidence)
 */
export function calculateProfitScore(profitPercent: number, confidence: number): number {
  // Score = profit% * confidence * 100
  // Higher is better
  return profitPercent * confidence * 100;
}

/**
 * Determine if investment is worth it based on thresholds
 */
export function isWorthInvesting(
  profitPercent: number,
  confidence: number,
  minProfit = 2,
  minConfidence = 0.7
): boolean {
  return profitPercent >= minProfit && confidence >= minConfidence;
}

/**
 * Calculate break-even price
 */
export function calculateBreakEven(
  currentPrice: number,
  tradingFee = 0.01 // 1% trading fee
): number {
  return currentPrice * (1 + tradingFee);
}

/**
 * Calculate profit after fees
 */
export function calculateProfitAfterFees(
  profit: number,
  tradingFee = 0.01 // 1% trading fee
): number {
  return profit * (1 - tradingFee);
}

/**
 * Calculate risk-adjusted return (Sharpe-like ratio)
 */
export function calculateRiskAdjustedReturn(
  expectedReturn: number,
  confidence: number,
  riskFreeRate = 0
): number {
  const risk = 1 - confidence; // Higher confidence = lower risk
  if (risk === 0) return expectedReturn;
  return (expectedReturn - riskFreeRate) / risk;
}



