/**
 * Risk assessment types
 */

/**
 * Risk level classification
 */
export type RiskLevel = "low" | "medium" | "high" | "very-high";

/**
 * Risk assessment for a currency
 */
export interface RiskAssessment {
  currency: string;
  league: string;
  riskLevel: RiskLevel;
  riskScore: number; // 0-100
  factors: RiskFactor[];
  recommendation: string;
}

/**
 * Individual risk factor
 */
export interface RiskFactor {
  name: string;
  value: number;
  weight: number;
  impact: "positive" | "negative" | "neutral";
  description: string;
}

/**
 * Portfolio diversification analysis
 */
export interface DiversificationAnalysis {
  score: number; // 0-100
  recommendations: string[];
  categoryDistribution: Record<string, number>;
  riskDistribution: Record<RiskLevel, number>;
}

/**
 * Investment timeline event
 */
export interface TimelineEvent {
  date: Date;
  type: "buy" | "sell" | "hold" | "alert";
  currency: string;
  price: number;
  reason: string;
  confidence: number;
}

/**
 * Investment recommendation
 */
export interface InvestmentRecommendation {
  currency: string;
  league: string;
  action: "buy" | "sell" | "hold";
  timeframe: "short" | "medium" | "long";
  expectedReturn: number;
  riskLevel: RiskLevel;
  confidence: number;
  reasons: string[];
}



