#!/usr/bin/env python3
"""
Currency Investment Report Generator

This script generates comprehensive investment reports with state-of-the-art visualizations
and data analysis to help users make data-driven currency investment decisions.

Features:
- Interactive charts and graphs
- Risk-return analysis
- Investment recommendations
- Market trend analysis
- Portfolio optimization suggestions
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from utils.model_inference import ModelPredictor
from utils.logging_utils import MLLogger
from utils.database import get_db_connection


class InvestmentReportGenerator:
    """Generate comprehensive investment reports with visualizations."""
    
    def __init__(self, output_dir: str = "investment_reports"):
        """Initialize the report generator."""
        self.logger = MLLogger("InvestmentReportGenerator")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize predictor
        self.predictor = None
        self.predictions_data = None
        self.market_data = None
        
    def load_prediction_models(self) -> bool:
        """Load prediction models and generate fresh predictions."""
        try:
            # Get the script directory and find models directory relative to project root
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent  # Go up from ml/scripts to project root
            models_dir = project_root / "ml" / "models" / "currency_production"
            
            # Alternative paths to try if the above doesn't work
            alternative_paths = [
                Path("../models/currency_production"),  # Original relative path
                Path("./models/currency_production"),   # From ml directory
                Path("models/currency_production"),     # Direct path
                script_dir.parent / "models" / "currency_production"  # From ml/scripts to ml/models
            ]
            
            # Try the main path first
            if not models_dir.exists():
                self.logger.warning(f"Primary models directory not found: {models_dir}")
                # Try alternative paths
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        models_dir = alt_path
                        self.logger.info(f"Using alternative models directory: {models_dir}")
                        break
                else:
                    self.logger.error(f"Models directory not found in any of the expected locations")
                    return False
            
            self.predictor = ModelPredictor(models_dir, logger=self.logger)
            available_models = self.predictor.load_available_models()
            
            if not available_models:
                self.logger.error("No models available")
                return False
            
            self.logger.info(f"Loaded {len(available_models)} prediction models")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def generate_predictions(self) -> pd.DataFrame:
        """Generate fresh 1-day predictions for all available currencies."""
        try:
            # Focus on 1-day predictions only
            horizon = 1
            all_predictions = []
            
            predictions = self.predictor.predict_multiple_currencies(
                prediction_horizon_days=horizon
            )
            
            for pred in predictions:
                all_predictions.append({
                    'currency': pred.currency,
                    'current_price': pred.current_price,
                    'predicted_price': pred.predicted_price,
                    'price_change_percent': pred.price_change_percent,
                    'confidence_score': pred.confidence_score,
                    'horizon_days': horizon,
                    'model_type': pred.model_type,
                    'features_used': pred.features_used,
                    'data_points_used': pred.data_points_used,
                    'prediction_timestamp': pred.prediction_timestamp
                })
            
            df = pd.DataFrame(all_predictions)
            self.logger.info(f"Generated {len(df)} 1-day predictions")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to generate predictions: {e}")
            return pd.DataFrame()
    
    def get_market_data(self) -> pd.DataFrame:
        """Get historical market data for analysis."""
        try:
            conn = get_db_connection()
            
            query = """
            SELECT 
                currency_name,
                chaos_equivalent as price,
                sample_time as date,
                league,
                total_change,
                direction,
                value,
                count,
                listing_count,
                confidence_level
            FROM live_currency_prices 
            WHERE league = 'Mercenaries'
            AND sample_time >= NOW() - INTERVAL '7 days'
            ORDER BY currency_name, sample_time
            """
            
            market_data = pd.read_sql(query, conn)
            conn.close()
            
            # Convert date column
            market_data['date'] = pd.to_datetime(market_data['date'])
            
            self.logger.info(f"Retrieved {len(market_data)} market records")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return pd.DataFrame()
    
    def calculate_prediction_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate prediction metrics for each currency based on 1-day predictions."""
        prediction_metrics = []
        
        for currency in df['currency'].unique():
            currency_data = df[df['currency'] == currency]
            
            # Get 1-day prediction (only horizon we have)
            pred_1d = currency_data['price_change_percent'].iloc[0]
            
            # Handle potential NaN or extreme values that could affect bar scaling
            if pd.isna(pred_1d) or abs(pred_1d) > 1000:  # Cap extreme predictions
                pred_1d = 0.0
            
            # Get confidence score and prediction intervals
            confidence_score = currency_data['confidence_score'].iloc[0]
            
            # Handle prediction intervals safely
            if 'prediction_lower' in currency_data.columns:
                prediction_lower = currency_data['prediction_lower'].iloc[0]
            else:
                # Fallback: 20% margin around predicted price
                predicted_price = currency_data['predicted_price'].iloc[0]
                margin = abs(predicted_price) * 0.2
                prediction_lower = predicted_price - margin
                
            if 'prediction_upper' in currency_data.columns:
                prediction_upper = currency_data['prediction_upper'].iloc[0]
            else:
                # Fallback: 20% margin around predicted price
                predicted_price = currency_data['predicted_price'].iloc[0]
                margin = abs(predicted_price) * 0.2
                prediction_upper = predicted_price + margin
                
            if 'interval_width' in currency_data.columns:
                interval_width = currency_data['interval_width'].iloc[0]
            else:
                interval_width = abs(prediction_upper - prediction_lower)
                
            if 'confidence_method' in currency_data.columns:
                confidence_method = currency_data['confidence_method'].iloc[0]
            else:
                confidence_method = 'fallback'
            
            # Get uncertainty components if available
            if 'uncertainty_components' in currency_data.columns:
                uncertainty_components = currency_data['uncertainty_components'].iloc[0]
                if isinstance(uncertainty_components, str):
                    # Handle case where it might be stored as string
                    try:
                        import json
                        uncertainty_components = json.loads(uncertainty_components)
                    except:
                        uncertainty_components = {}
            else:
                uncertainty_components = {}
            
            # Get basic metrics
            data_points = currency_data['data_points_used'].iloc[0]
            current_price = currency_data['current_price'].iloc[0]
            predicted_price = currency_data['predicted_price'].iloc[0]
            
            # Simple quality score based on data points and confidence
            quality_score = min(100, (data_points / 10) * 50 + confidence_score * 50)
            
            # Calculate potential profit/loss in absolute terms
            absolute_profit = predicted_price - current_price
            
            # Investment recommendation based on return and confidence
            investment_recommendation = self._get_investment_recommendation(pred_1d, confidence_score, data_points)
            
            prediction_metrics.append({
                'currency': currency,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'pred_1d': pred_1d,
                'absolute_profit': absolute_profit,
                'prediction_lower': prediction_lower,
                'prediction_upper': prediction_upper,
                'interval_width': interval_width,
                'confidence_score': confidence_score,
                'confidence_method': confidence_method,
                'data_points_used': data_points,
                'quality_score': quality_score,
                'uncertainty_components': uncertainty_components,
                'investment_recommendation': investment_recommendation
            })
        
        result_df = pd.DataFrame(prediction_metrics)
        

        
        return result_df
    
    def _get_investment_recommendation(self, return_pct: float, confidence: float, data_points: int) -> str:
        """Assign investment recommendation based on return, confidence, and data quality."""
        # Data quality factor
        data_quality = min(1.0, data_points / 10)  # Normalize to 0-1 scale
        
        # Combined score considering return, confidence, and data quality
        combined_score = (return_pct * 0.5) + (confidence * 50 * 0.3) + (data_quality * 20 * 0.2)
        
        if return_pct > 30 and confidence >= 0.7 and data_points >= 8:
            return "🚀 STRONG BUY"
        elif return_pct > 15 and confidence >= 0.6 and data_points >= 5:
            return "📈 BUY"
        elif return_pct > 5 and confidence >= 0.5:
            return "👀 WATCH"
        elif return_pct > 0:
            return "⚠️ CAUTION"
        elif return_pct > -10:
            return "🔻 WEAK"
        else:
            return "🚫 AVOID"
    
    def create_profit_prediction_summary(self, prediction_df: pd.DataFrame) -> str:
        """Create HTML summary of profit predictions without charts."""
        # Get top gainers and losers
        top_gainers = prediction_df.nlargest(10, 'pred_1d')
        top_losers = prediction_df.nsmallest(10, 'pred_1d')
        
        html = f"""
        <div style="margin: 20px 0;">
            <h2 style="color: #2c3e50; text-align: center;">📊 Currency Profit Predictions Summary</h2>
            
            <div style="display: flex; justify-content: space-between; margin: 20px 0;">
                
                <div style="width: 48%; background: #d5f4e6; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #27ae60; text-align: center;">🚀 Top 10 Gainers (1-Day)</h3>
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="background-color: #27ae60; color: white;">
                                <th style="padding: 8px; text-align: left;">Currency</th>
                                <th style="padding: 8px; text-align: right;">Return %</th>
                                <th style="padding: 8px; text-align: right;">Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for _, row in top_gainers.iterrows():
            html += f"""
                            <tr style="background-color: white;">
                                <td style="padding: 6px; border: 1px solid #ddd; font-weight: bold;">{row['currency']}</td>
                                <td style="padding: 6px; border: 1px solid #ddd; text-align: right; color: #27ae60; font-weight: bold;">{row['pred_1d']:+.1f}%</td>
                                <td style="padding: 6px; border: 1px solid #ddd; text-align: right;">{row['confidence_score']:.2f}</td>
                            </tr>
            """
        
        html += """
                        </tbody>
                    </table>
                </div>
                
                <div style="width: 48%; background: #fdeaea; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #e74c3c; text-align: center;">🔻 Top 10 Losers (1-Day)</h3>
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="background-color: #e74c3c; color: white;">
                                <th style="padding: 8px; text-align: left;">Currency</th>
                                <th style="padding: 8px; text-align: right;">Return %</th>
                                <th style="padding: 8px; text-align: right;">Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for _, row in top_losers.iterrows():
            html += f"""
                            <tr style="background-color: white;">
                                <td style="padding: 6px; border: 1px solid #ddd; font-weight: bold;">{row['currency']}</td>
                                <td style="padding: 6px; border: 1px solid #ddd; text-align: right; color: #e74c3c; font-weight: bold;">{row['pred_1d']:+.1f}%</td>
                                <td style="padding: 6px; border: 1px solid #ddd; text-align: right;">{row['confidence_score']:.2f}</td>
                            </tr>
            """
        
        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def create_market_trends_summary(self, market_df: pd.DataFrame) -> str:
        """Create HTML summary of market trends without charts."""
        if market_df.empty:
            return "<p>No market data available for trend analysis.</p>"
        
        # Get currencies with sufficient data points and interesting price movements
        currency_stats = market_df.groupby('currency_name').agg({
            'price': ['count', 'std', 'mean', 'min', 'max']
        }).round(2)
        currency_stats.columns = ['data_points', 'volatility', 'avg_price', 'min_price', 'max_price']
        
        # Select currencies with good data and interesting movements
        interesting_currencies = currency_stats[
            (currency_stats['data_points'] >= 5) &  # At least 5 data points
            (currency_stats['volatility'] > 0) &    # Some price movement
            (currency_stats['avg_price'] >= 1.0)    # Focus on more valuable currencies
        ].sort_values(['avg_price', 'volatility'], ascending=[False, False]).head(15)
        
        # Calculate price changes for each currency
        price_changes = []
        for currency in interesting_currencies.index:
            currency_data = market_df[market_df['currency_name'] == currency].sort_values('date')
            if len(currency_data) > 1:
                first_price = currency_data['price'].iloc[0]
                last_price = currency_data['price'].iloc[-1]
                price_change = ((last_price - first_price) / first_price * 100)
                
                price_changes.append({
                    'currency': currency,
                    'first_price': first_price,
                    'last_price': last_price,
                    'price_change': price_change,
                    'avg_price': interesting_currencies.loc[currency, 'avg_price'],
                    'volatility': interesting_currencies.loc[currency, 'volatility'],
                    'data_points': interesting_currencies.loc[currency, 'data_points']
                })
        
        # Sort by price change
        price_changes.sort(key=lambda x: x['price_change'], reverse=True)
        
        html = f"""
        <div style="margin: 20px 0;">
            <h2 style="color: #2c3e50; text-align: center;">📈 Market Trends Summary (Last 7 Days)</h2>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background-color: #34495e; color: white;">
                            <th style="padding: 10px; text-align: left;">Currency</th>
                            <th style="padding: 10px; text-align: right;">Start Price</th>
                            <th style="padding: 10px; text-align: right;">Current Price</th>
                            <th style="padding: 10px; text-align: right;">7-Day Change</th>
                            <th style="padding: 10px; text-align: right;">Avg Price</th>
                            <th style="padding: 10px; text-align: right;">Volatility</th>
                            <th style="padding: 10px; text-align: center;">Data Points</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, change_data in enumerate(price_changes):
            change_color = '#27ae60' if change_data['price_change'] > 0 else '#e74c3c'
            row_bg = '#f8f9fa' if i % 2 == 0 else 'white'
            
            html += f"""
                        <tr style="background-color: {row_bg};">
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{change_data['currency']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{change_data['first_price']:.2f}c</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{change_data['last_price']:.2f}c</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {change_color}; font-weight: bold;">{change_data['price_change']:+.1f}%</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{change_data['avg_price']:.2f}c</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{change_data['volatility']:.2f}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{change_data['data_points']}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <div style="margin-top: 15px; font-size: 12px; color: #7f8c8d; text-align: center;">
                <p><strong>Note:</strong> Trends based on available market data from the last 7 days. 
                Higher volatility indicates more price movement and potential trading opportunities.</p>
            </div>
        </div>
        """
        
        return html
    
    def create_price_distribution_summary(self, prediction_df: pd.DataFrame) -> str:
        """Create HTML summary of price distribution analysis without charts."""
        # Group currencies by price ranges for better visualization
        currencies_data = []
        
        for _, row in prediction_df.iterrows():
            price = row['current_price']
            if price < 1:
                range_label = "< 1c"
            elif price < 5:
                range_label = "1-5c"
            elif price < 20:
                range_label = "5-20c"
            elif price < 100:
                range_label = "20-100c"
            else:
                range_label = "> 100c"
            
            currencies_data.append({
                'currency': row['currency'],
                'price': price,
                'predicted_return': row['pred_1d'],
                'range': range_label
            })
        
        df_viz = pd.DataFrame(currencies_data)
        
        # Define range order for consistent display
        range_order = ["< 1c", "1-5c", "5-20c", "20-100c", "> 100c"]
        
        # Create analysis data
        range_analysis = []
        for range_label in range_order:
            range_data = df_viz[df_viz['range'] == range_label]
            if len(range_data) > 0:
                range_analysis.append({
                    'range': range_label,
                    'count': len(range_data),
                    'avg_return': range_data['predicted_return'].mean(),
                    'max_return': range_data['predicted_return'].max(),
                    'min_return': range_data['predicted_return'].min(),
                    'profitable_count': len(range_data[range_data['predicted_return'] > 0])
                })
        
        html = f"""
        <div style="margin: 20px 0;">
            <h2 style="color: #2c3e50; text-align: center;">💰 Currency Analysis by Price Range</h2>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background-color: #34495e; color: white;">
                            <th style="padding: 10px; text-align: left;">Price Range</th>
                            <th style="padding: 10px; text-align: center;">Currency Count</th>
                            <th style="padding: 10px; text-align: center;">Profitable Count</th>
                            <th style="padding: 10px; text-align: right;">Avg Return %</th>
                            <th style="padding: 10px; text-align: right;">Best Return %</th>
                            <th style="padding: 10px; text-align: right;">Worst Return %</th>
                            <th style="padding: 10px; text-align: center;">Success Rate</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, analysis in enumerate(range_analysis):
            success_rate = (analysis['profitable_count'] / analysis['count']) * 100
            avg_return_color = '#27ae60' if analysis['avg_return'] > 0 else '#e74c3c'
            row_bg = '#f8f9fa' if i % 2 == 0 else 'white'
            
            html += f"""
                        <tr style="background-color: {row_bg};">
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{analysis['range']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{analysis['count']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: #27ae60;">{analysis['profitable_count']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {avg_return_color}; font-weight: bold;">{analysis['avg_return']:+.1f}%</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: #27ae60;">{analysis['max_return']:+.1f}%</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: #e74c3c;">{analysis['min_return']:+.1f}%</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: {'#27ae60' if success_rate >= 50 else '#f39c12' if success_rate >= 30 else '#e74c3c'};">{success_rate:.0f}%</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <div style="margin-top: 15px; font-size: 12px; color: #7f8c8d; text-align: center;">
                <p><strong>Analysis:</strong> Success rate = percentage of currencies with positive predicted returns. 
                Higher-priced currencies often show more stable but lower percentage returns.</p>
            </div>
        </div>
        """
        
        return html
    
    def create_investment_recommendations_table(self, prediction_df: pd.DataFrame) -> str:
        """Create HTML table with investment recommendations including prediction intervals."""
        # Sort by predicted return (best returns first) - include ALL opportunities
        all_investments = prediction_df.sort_values('pred_1d', ascending=False)
        
        html = f"""
        <div style="margin: 20px 0;">
            <h2 style="color: #2c3e50; text-align: center;">🏆 All Investment Opportunities with Prediction Intervals ({len(all_investments)} currencies)</h2>
            <div style="max-height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px;">
                <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 11px;">
                    <thead style="position: sticky; top: 0; z-index: 10;">
                        <tr style="background-color: #34495e; color: white;">
                            <th style="padding: 10px 8px; text-align: left; border: 1px solid #ddd; min-width: 50px;">Rank</th>
                            <th style="padding: 10px 8px; text-align: left; border: 1px solid #ddd; min-width: 180px;">Currency</th>
                            <th style="padding: 10px 8px; text-align: right; border: 1px solid #ddd; min-width: 80px;">Current Price</th>
                            <th style="padding: 10px 8px; text-align: right; border: 1px solid #ddd; min-width: 80px;">Predicted Price</th>
                            <th style="padding: 10px 8px; text-align: right; border: 1px solid #ddd; min-width: 100px;">Prediction Range</th>
                            <th style="padding: 10px 8px; text-align: right; border: 1px solid #ddd; min-width: 80px;">1d Return %</th>
                            <th style="padding: 10px 8px; text-align: right; border: 1px solid #ddd; min-width: 80px;">Absolute Profit</th>
                            <th style="padding: 10px 8px; text-align: right; border: 1px solid #ddd; min-width: 70px;">Confidence</th>
                            <th style="padding: 10px 8px; text-align: center; border: 1px solid #ddd; min-width: 70px;">Data Points</th>
                            <th style="padding: 10px 8px; text-align: center; border: 1px solid #ddd; min-width: 70px;">Quality Score</th>
                            <th style="padding: 10px 8px; text-align: center; border: 1px solid #ddd; min-width: 120px;">Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, (_, row) in enumerate(all_investments.iterrows(), 1):
            # Get recommendation and color
            recommendation = row['investment_recommendation']
            
            # Color coding for recommendations
            rec_colors = {
                "🚀 STRONG BUY": "#27ae60",
                "📈 BUY": "#2ecc71",
                "👀 WATCH": "#f39c12",
                "⚠️ CAUTION": "#e67e22",
                "🔻 WEAK": "#e74c3c",
                "🚫 AVOID": "#c0392b"
            }
            
            rec_color = rec_colors.get(recommendation, "#95a5a6")
            
            # Format prediction range
            pred_lower = row['prediction_lower']
            pred_upper = row['prediction_upper']
            pred_range = f"{pred_lower:.1f} - {pred_upper:.1f}c"
            
            # Format absolute profit with color
            absolute_profit = row['absolute_profit']
            profit_color = '#27ae60' if absolute_profit > 0 else '#e74c3c'
            
            html += f"""
                <tr style="background-color: {'#f8f9fa' if i % 2 == 0 else 'white'}; {'border-left: 4px solid #27ae60;' if row['pred_1d'] > 50 else 'border-left: 4px solid #e74c3c;' if row['pred_1d'] < -20 else ''}">
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; text-align: center;">{i}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; max-width: 180px; word-wrap: break-word;">{row['currency']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{row['current_price']:.2f}c</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {'#27ae60' if row['pred_1d'] > 0 else '#e74c3c'}; font-weight: bold;">{row['predicted_price']:.2f}c</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; font-size: 10px;">{pred_range}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {'#27ae60' if row['pred_1d'] > 0 else '#e74c3c'}; font-weight: bold;">{row['pred_1d']:+.1f}%</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {profit_color}; font-weight: bold;">{absolute_profit:+.2f}c</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{row['confidence_score']:.2f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{row['data_points_used']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: {'#27ae60' if row['quality_score'] > 70 else '#f39c12' if row['quality_score'] > 40 else '#e74c3c'};">{row['quality_score']:.0f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: {rec_color}; font-weight: bold; font-size: 10px;">{recommendation}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #7f8c8d; text-align: center;">
                <p><strong>Legend:</strong> 
                Green left border = High profit potential (>50%) | 
                Red left border = High loss potential (<-20%) | 
                Scroll to view all currencies
                </p>
            </div>
        </div>
        """
        
        return html
    
    def create_portfolio_suggestions(self, prediction_df: pd.DataFrame) -> str:
        """Create portfolio allocation suggestions based on 1-day predictions."""
        # Conservative portfolio (high confidence, positive returns)
        conservative = prediction_df[
            (prediction_df['confidence_score'] >= 0.6) & 
            (prediction_df['pred_1d'] > 0) &
            (prediction_df['data_points_used'] >= 5)
        ].sort_values(['confidence_score', 'quality_score'], ascending=False).head(5)
        
        # Balanced portfolio (good returns with decent confidence)
        balanced = prediction_df[
            (prediction_df['pred_1d'] > 10) &
            (prediction_df['confidence_score'] >= 0.5)
        ].sort_values(['pred_1d', 'confidence_score'], ascending=False).head(5)
        
        # Aggressive portfolio (highest returns regardless of confidence)
        aggressive = prediction_df[
            prediction_df['pred_1d'] > 30
        ].sort_values('pred_1d', ascending=False).head(5)
        
        html = """
        <div style="margin: 30px 0;">
            <h2 style="color: #2c3e50; text-align: center;">💼 Portfolio Allocation Suggestions</h2>
            
            <div style="display: flex; justify-content: space-between; margin: 20px 0;">
                
                <div style="width: 30%; background: #ecf0f1; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #27ae60; text-align: center;">🛡️ Conservative Portfolio</h3>
                    <p style="text-align: center; color: #7f8c8d;">Low Risk • Steady Returns</p>
        """
        
        for _, row in conservative.iterrows():
            html += f"""
                    <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                        <strong>{row['currency']}</strong><br>
                        <span style="color: #27ae60;">1d Return: {row['pred_1d']:+.1f}%</span> | 
                        <span style="color: #3498db;">Confidence: {row['confidence_score']:.2f}</span> |
                        <span style="color: #9b59b6;">Quality: {row['quality_score']:.0f}</span>
                    </div>
            """
        
        html += """
                </div>
                
                <div style="width: 30%; background: #ecf0f1; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #f39c12; text-align: center;">⚖️ Balanced Portfolio</h3>
                    <p style="text-align: center; color: #7f8c8d;">Medium Risk • Good Returns</p>
        """
        
        for _, row in balanced.iterrows():
            html += f"""
                    <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                        <strong>{row['currency']}</strong><br>
                        <span style="color: #27ae60;">1d Return: {row['pred_1d']:+.1f}%</span> | 
                        <span style="color: #3498db;">Confidence: {row['confidence_score']:.2f}</span> |
                        <span style="color: #9b59b6;">Quality: {row['quality_score']:.0f}</span>
                    </div>
            """
        
        html += """
                </div>
                
                <div style="width: 30%; background: #ecf0f1; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #e74c3c; text-align: center;">🚀 Aggressive Portfolio</h3>
                    <p style="text-align: center; color: #7f8c8d;">High Risk • High Returns</p>
        """
        
        for _, row in aggressive.iterrows():
            html += f"""
                    <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                        <strong>{row['currency']}</strong><br>
                        <span style="color: #27ae60;">1d Return: {row['pred_1d']:+.1f}%</span> | 
                        <span style="color: #3498db;">Confidence: {row['confidence_score']:.2f}</span> |
                        <span style="color: #9b59b6;">Quality: {row['quality_score']:.0f}</span>
                    </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
        
        return html
    
    def generate_comprehensive_report(self) -> str:
        """Generate the complete investment report."""
        self.logger.info("🚀 Starting comprehensive investment report generation...")
        
        # Load models and generate predictions
        if not self.load_prediction_models():
            return None
        
        # Generate fresh predictions
        self.predictions_data = self.generate_predictions()
        if self.predictions_data.empty:
            self.logger.error("No predictions generated")
            return None
        
        # Get market data
        self.market_data = self.get_market_data()
        
        # Calculate prediction metrics
        prediction_df = self.calculate_prediction_metrics(self.predictions_data)
        
        # Create analysis summaries
        self.logger.info("📊 Creating analysis summaries...")
        
        # Profit Prediction Summary
        profit_prediction_summary = self.create_profit_prediction_summary(prediction_df)
        
        # Market Trends Summary
        market_trends_summary = self.create_market_trends_summary(self.market_data)
        
        # Price Distribution Summary
        price_distribution_summary = self.create_price_distribution_summary(prediction_df)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PoEconomy Investment Report - {timestamp}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .summary-stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background: #3498db; color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 150px; }}
                .stat-value {{ font-size: 2em; font-weight: bold; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏆 PoEconomy Investment Report</h1>
                <h2>Path of Exile Currency Investment Analysis</h2>
                <p>Generated on {timestamp} | Mercenaries League</p>
            </div>
            
            <div class="section">
                <h2>📈 Executive Summary</h2>
                <div class="summary-stats">
                    <div class="stat-box" style="background: #27ae60;">
                        <div class="stat-value">{len(prediction_df)}</div>
                        <div class="stat-label">Currencies Analyzed</div>
                    </div>
                    <div class="stat-box" style="background: #e74c3c;">
                        <div class="stat-value">{prediction_df['pred_1d'].max():.0f}%</div>
                        <div class="stat-label">Highest 1-Day Return</div>
                    </div>
                    <div class="stat-box" style="background: #f39c12;">
                        <div class="stat-value">{prediction_df['confidence_score'].mean():.2f}</div>
                        <div class="stat-label">Average Confidence</div>
                    </div>
                    <div class="stat-box" style="background: #9b59b6;">
                        <div class="stat-value">{len(prediction_df[prediction_df['pred_1d'] > 0])}</div>
                        <div class="stat-label">Profitable Opportunities</div>
                    </div>
                </div>
            </div>
        """
        
        # Add investment recommendations table
        html_content += f"""
            <div class="section">
                {self.create_investment_recommendations_table(prediction_df)}
            </div>
        """
        
        # Add portfolio suggestions
        html_content += f"""
            <div class="section">
                {self.create_portfolio_suggestions(prediction_df)}
            </div>
        """
        
        # Add analysis summaries
        html_content += f"""
            <div class="section">
                {profit_prediction_summary}
            </div>
            
            <div class="section">
                {market_trends_summary}
            </div>
            
            <div class="section">
                {price_distribution_summary}
            </div>
            
            <div class="section">
                <h2>⚠️ Disclaimer</h2>
                <p style="color: #7f8c8d; font-style: italic;">
                    This report is generated using machine learning models and historical data analysis. 
                    All predictions are estimates and should not be considered as financial advice. 
                    Path of Exile currency markets are volatile and unpredictable. 
                    Always do your own research and invest responsibly.
                </p>
            </div>
            
            <div class="section">
                <h2>📊 Data Sources & Methodology</h2>
                <ul>
                    <li><strong>Prediction Models:</strong> LightGBM regression models trained on historical price data</li>
                    <li><strong>Market Data:</strong> Live currency prices from poe.ninja API</li>
                    <li><strong>Risk Analysis:</strong> Volatility-based risk scoring with confidence weighting</li>
                    <li><strong>Time Horizons:</strong> 1-day predictions focused on short-term opportunities</li>
                    <li><strong>Investment Grades:</strong> Based on return potential, risk score, and model confidence</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_filename = f"investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save data as JSON for API consumption
        data_export = {
            'timestamp': timestamp,
            'predictions': self.predictions_data.to_dict('records'),
            'prediction_analysis': prediction_df.to_dict('records'),
            'summary_stats': {
                'total_currencies': len(prediction_df),
                'profitable_opportunities': len(prediction_df[prediction_df['pred_1d'] > 0]),
                'highest_return': float(prediction_df['pred_1d'].max()),
                'average_confidence': float(prediction_df['confidence_score'].mean()),
                'average_quality_score': float(prediction_df['quality_score'].mean()),
                'top_recommendations': prediction_df.sort_values('pred_1d', ascending=False).head(10).to_dict('records')
            }
        }
        
        json_filename = f"investment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(data_export, f, indent=2, default=str)
        
        self.logger.info(f"✅ Investment report generated successfully!")
        self.logger.info(f"📄 HTML Report: {report_path}")
        self.logger.info(f"📊 Data Export: {json_path}")
        
        return str(report_path)


def main():
    """Main function to generate investment report."""
    print("🚀 PoEconomy Investment Report Generator")
    print("=" * 60)
    
    try:
        generator = InvestmentReportGenerator()
        report_path = generator.generate_comprehensive_report()
        
        if report_path:
            print(f"\n✅ SUCCESS! Investment report generated:")
            print(f"📄 Report: {report_path}")
            print(f"\n💡 Open the HTML file in your browser to view the interactive report!")
        else:
            print("\n❌ Failed to generate investment report")
            
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")


if __name__ == "__main__":
    main() 