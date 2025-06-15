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
import altair as alt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure Altair for better performance and interactivity
alt.data_transformers.enable('json')
alt.renderers.enable('html')

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
        
        # Configure Altair theme for professional appearance
        alt.themes.enable('fivethirtyeight')
        
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
    
    def create_profit_prediction_chart(self, prediction_df: pd.DataFrame) -> alt.Chart:
        """Create horizontal bar chart showing predicted profit percentages using Altair."""
        # Get top 25 currencies by predicted return, including both positive and negative
        top_currencies = prediction_df.nlargest(15, 'pred_1d')  # Top gainers
        bottom_currencies = prediction_df.nsmallest(10, 'pred_1d')  # Biggest losers
        
        # Combine and sort by prediction
        selected_currencies = pd.concat([top_currencies, bottom_currencies]).sort_values('pred_1d', ascending=True)
        
        # Ensure we have clean data for bar scaling
        selected_currencies = selected_currencies.dropna(subset=['pred_1d'])
        selected_currencies = selected_currencies.reset_index(drop=True)
        
        # Add color coding column
        selected_currencies['color'] = selected_currencies['pred_1d'].apply(
            lambda x: '#27ae60' if x > 0 else '#e74c3c'
        )
        
        # Add profit/loss category for better visualization
        selected_currencies['category'] = selected_currencies['pred_1d'].apply(
            lambda x: 'Profit' if x > 0 else 'Loss'
        )
        
        # Create the main bar chart
        bars = alt.Chart(selected_currencies).mark_bar(
            opacity=0.8,
            stroke='white',
            strokeWidth=1
        ).encode(
            x=alt.X(
                'pred_1d:Q',
                title='Predicted Profit/Loss (%)',
                scale=alt.Scale(nice=True),
                axis=alt.Axis(grid=True, tickCount=10, format='.1f')
            ),
            y=alt.Y(
                'currency:N',
                title='Currency',
                sort=alt.SortField(field='pred_1d', order='ascending'),
                axis=alt.Axis(labelLimit=200)
            ),
            color=alt.Color(
                'category:N',
                scale=alt.Scale(
                    domain=['Profit', 'Loss'],
                    range=['#27ae60', '#e74c3c']
                ),
                legend=alt.Legend(title="Category", orient="top-right")
            ),
            tooltip=[
                alt.Tooltip('currency:N', title='Currency'),
                alt.Tooltip('pred_1d:Q', title='Predicted 1-Day Return (%)', format='.1f'),
                alt.Tooltip('current_price:Q', title='Current Price (c)', format='.2f'),
                alt.Tooltip('predicted_price:Q', title='Predicted Price (c)', format='.2f'),
                alt.Tooltip('confidence_score:Q', title='Confidence', format='.2f'),
                alt.Tooltip('data_points_used:Q', title='Data Points')
            ]
        ).properties(
            width=800,
            height=600,
            title=alt.TitleParams(
                text="📊 Currency Profit Predictions (1-Day Horizon)",
                fontSize=18,
                anchor='start',
                offset=20
            )
        )
        
        # Add zero reference line
        zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
            color='gray',
            strokeDash=[5, 5],
            strokeWidth=2,
            opacity=0.7
        ).encode(
            x='x:Q'
        )
        
        # Combine charts
        chart = (bars + zero_line).resolve_scale(
            color='independent'
        ).interactive()
        
        return chart
    
    def create_enhanced_market_trends(self, market_df: pd.DataFrame) -> alt.Chart:
        """Create enhanced line chart tracking currency values through the league using Altair."""
        if market_df.empty:
            return alt.Chart().mark_text(text="No market data available")
        
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
        ].sort_values(['avg_price', 'volatility'], ascending=[False, False]).head(12)
        
        # Filter market data to only include interesting currencies
        filtered_data = market_df[market_df['currency_name'].isin(interesting_currencies.index)].copy()
        
        # Calculate price change percentage for each currency
        price_changes = []
        for currency in interesting_currencies.index:
            currency_data = filtered_data[filtered_data['currency_name'] == currency].sort_values('date')
            if len(currency_data) > 1:
                price_change = ((currency_data['price'].iloc[-1] - currency_data['price'].iloc[0]) / 
                               currency_data['price'].iloc[0] * 100)
                price_changes.append({'currency_name': currency, 'price_change': price_change})
        
        price_change_df = pd.DataFrame(price_changes)
        filtered_data = filtered_data.merge(price_change_df, on='currency_name', how='left')
        
        # Create currency labels with price change
        filtered_data['currency_label'] = filtered_data.apply(
            lambda row: f"{row['currency_name']} ({row['price_change']:+.1f}%)", axis=1
        )
        
        # Create the line chart
        lines = alt.Chart(filtered_data).mark_line(
            point=True,
            strokeWidth=3,
            opacity=0.8
        ).encode(
            x=alt.X(
                'date:T',
                title='Date',
                axis=alt.Axis(format='%m/%d', labelAngle=-45)
            ),
            y=alt.Y(
                'price:Q',
                title='Price (Chaos Orbs)',
                scale=alt.Scale(type='log', nice=True),
                axis=alt.Axis(format='.2f')
            ),
            color=alt.Color(
                'currency_label:N',
                title='Currency (% Change)',
                scale=alt.Scale(scheme='category20'),
                legend=alt.Legend(
                    orient='right',
                    titleLimit=200,
                    labelLimit=200,
                    columns=1
                )
            ),
            tooltip=[
                alt.Tooltip('currency_name:N', title='Currency'),
                alt.Tooltip('date:T', title='Date', format='%Y-%m-%d %H:%M'),
                alt.Tooltip('price:Q', title='Price (c)', format='.2f'),
                alt.Tooltip('listing_count:Q', title='Listings'),
                alt.Tooltip('confidence_level:Q', title='Confidence', format='.2f'),
                alt.Tooltip('price_change:Q', title='Total Change (%)', format='.1f')
            ]
        ).properties(
            width=900,
            height=500,
            title=alt.TitleParams(
                text="📈 Currency Price Trends Throughout the League",
                fontSize=18,
                anchor='start',
                offset=20
            )
        )
        
        # Add interactive selection
        chart = lines.add_selection(
            alt.selection_multi(fields=['currency_label'])
        ).transform_filter(
            alt.datum.price > 0  # Filter out invalid prices
        ).interactive()
        
        return chart
    
    def create_price_distribution_chart(self, prediction_df: pd.DataFrame) -> alt.Chart:
        """Create a chart showing current price distribution of currencies using Altair."""
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
        
        # Create count data
        range_counts = df_viz['range'].value_counts().reset_index()
        range_counts.columns = ['range', 'count']
        range_counts['chart_type'] = 'Currency Count'
        
        # Create average return data
        avg_returns = df_viz.groupby('range')['predicted_return'].mean().reset_index()
        avg_returns.columns = ['range', 'avg_return']
        avg_returns['chart_type'] = 'Average Return'
        avg_returns['return_category'] = avg_returns['avg_return'].apply(
            lambda x: 'Profit' if x > 0 else 'Loss'
        )
        
        # Currency count chart
        count_chart = alt.Chart(range_counts).mark_bar(
            opacity=0.7,
            color='lightblue'
        ).encode(
            x=alt.X(
                'range:N',
                title='Price Range',
                sort=range_order,
                axis=alt.Axis(labelAngle=-45)
            ),
            y=alt.Y(
                'count:Q',
                title='Number of Currencies'
            ),
            tooltip=[
                alt.Tooltip('range:N', title='Price Range'),
                alt.Tooltip('count:Q', title='Currency Count')
            ]
        ).properties(
            width=350,
            height=300,
            title=alt.TitleParams(
                text="Currency Count by Price Range",
                fontSize=14,
                anchor='start'
            )
        )
        
        # Average return chart
        return_chart = alt.Chart(avg_returns).mark_bar(
            opacity=0.8
        ).encode(
            x=alt.X(
                'range:N',
                title='Price Range',
                sort=range_order,
                axis=alt.Axis(labelAngle=-45)
            ),
            y=alt.Y(
                'avg_return:Q',
                title='Average Predicted Return (%)',
                axis=alt.Axis(format='.1f')
            ),
            color=alt.Color(
                'return_category:N',
                scale=alt.Scale(
                    domain=['Profit', 'Loss'],
                    range=['#27ae60', '#e74c3c']
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('range:N', title='Price Range'),
                alt.Tooltip('avg_return:Q', title='Average Return (%)', format='.1f')
            ]
        ).properties(
            width=350,
            height=300,
            title=alt.TitleParams(
                text="Average Return by Price Range",
                fontSize=14,
                anchor='start'
            )
        )
        
        # Add zero reference line to return chart
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
            color='gray',
            strokeDash=[3, 3],
            strokeWidth=1
        ).encode(y='y:Q')
        
        return_chart_with_line = return_chart + zero_line
        
        # Combine charts horizontally
        combined_chart = alt.hconcat(
            count_chart,
            return_chart_with_line,
            spacing=50
        ).resolve_scale(
            color='independent'
        ).properties(
            title=alt.TitleParams(
                text="💰 Currency Analysis by Price Range",
                fontSize=16,
                anchor='start',
                offset=20
            )
        )
        
        return combined_chart
    
    def create_investment_recommendations_table(self, prediction_df: pd.DataFrame) -> str:
        """Create HTML table with investment recommendations including prediction intervals."""
        # Sort by predicted return (best returns first)
        top_investments = prediction_df.sort_values('pred_1d', ascending=False).head(20)
        
        html = """
        <div style="margin: 20px 0;">
            <h2 style="color: #2c3e50; text-align: center;">🏆 Top Investment Opportunities with Prediction Intervals</h2>
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-family: Arial, sans-serif; font-size: 12px;">
                <thead>
                    <tr style="background-color: #34495e; color: white;">
                        <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Rank</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Currency</th>
                        <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Current Price</th>
                        <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Predicted Price</th>
                        <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Prediction Range</th>
                        <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">1d Return %</th>
                        <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Absolute Profit</th>
                        <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Confidence</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Data Points</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Quality Score</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Recommendation</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, (_, row) in enumerate(top_investments.iterrows(), 1):
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
                <tr style="background-color: {'#f8f9fa' if i % 2 == 0 else 'white'};">
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{i}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{row['currency']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{row['current_price']:.2f}c</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {'#27ae60' if row['pred_1d'] > 0 else '#e74c3c'}; font-weight: bold;">{row['predicted_price']:.2f}c</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; font-size: 11px;">{pred_range}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {'#27ae60' if row['pred_1d'] > 0 else '#e74c3c'}; font-weight: bold;">{row['pred_1d']:+.1f}%</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: {profit_color}; font-weight: bold;">{absolute_profit:+.2f}c</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{row['confidence_score']:.2f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{row['data_points_used']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: {'#27ae60' if row['quality_score'] > 70 else '#f39c12' if row['quality_score'] > 40 else '#e74c3c'};">{row['quality_score']:.0f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: {rec_color}; font-weight: bold; font-size: 11px;">{recommendation}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
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
        
        # Create visualizations
        self.logger.info("📊 Creating visualizations...")
        
        # Profit Prediction Chart
        profit_prediction_fig = self.create_profit_prediction_chart(prediction_df)
        
        # Enhanced Market Trends
        trend_fig = self.create_enhanced_market_trends(self.market_data)
        
        # Price Distribution Chart
        price_distribution_fig = self.create_price_distribution_chart(prediction_df)
        
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
                .chart-container {{ margin: 20px 0; text-align: center; }}
                .summary-stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background: #3498db; color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 150px; }}
                .stat-value {{ font-size: 2em; font-weight: bold; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
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
        
        # Add charts
        html_content += f"""
            <div class="section">
                <div class="chart-container" id="profit-prediction-chart"></div>
            </div>
            
            <div class="section">
                <div class="chart-container" id="trend-chart"></div>
            </div>
            
            <div class="section">
                <div class="chart-container" id="price-distribution-chart"></div>
            </div>
            
            <script>
                // Profit Prediction Chart
                var profitPredictionSpec = {profit_prediction_fig.to_json()};
                vegaEmbed('#profit-prediction-chart', profitPredictionSpec, {{"actions": false}});
                
                // Trend Chart
                var trendSpec = {trend_fig.to_json()};
                vegaEmbed('#trend-chart', trendSpec, {{"actions": false}});
                
                // Price Distribution Chart
                var priceDistributionSpec = {price_distribution_fig.to_json()};
                vegaEmbed('#price-distribution-chart', priceDistributionSpec, {{"actions": false}});
            </script>
            
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