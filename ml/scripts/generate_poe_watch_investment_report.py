#!/usr/bin/env python3
"""
POE Watch Currency Investment Report Generator

This script generates comprehensive investment reports using POE Watch API data
with state-of-the-art analysis to help users make data-driven currency investment decisions.

Features:
- POE Watch API data integration
- Market trend analysis
- Investment recommendations
- Portfolio optimization suggestions
- Risk analysis based on price volatility
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
import argparse
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from config.training_config import MLConfig, get_production_config, get_development_config, get_test_config
from utils.logging_utils import MLLogger, setup_ml_logging
from utils.database import get_db_connection
from utils.model_inference import ModelPredictor
from utils.currency_standardizer import CurrencyStandardizer


class PoeWatchInvestmentReportGenerator:
    """Generate comprehensive investment reports using POE Watch data."""
    
    def __init__(
        self, 
        config: Optional[MLConfig] = None,
        logger: Optional[MLLogger] = None
    ):
        """Initialize the report generator."""
        self.config = config or get_production_config()
        
        # Setup logging using config
        if logger is None:
            self.logger = setup_ml_logging(
                name="PoeWatchInvestmentReportGenerator",
                level=self.config.logging.level,
                log_dir=str(self.config.paths.logs_dir),
                experiment_id=self.config.experiment.experiment_id,
                console_output=self.config.logging.console_logging,
                suppress_external=self.config.logging.suppress_lightgbm
            )
        else:
            self.logger = logger
            
        # Use config for output directory
        self.output_dir = self.config.paths.ml_root / "investment_reports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Data holders
        self.market_data = None
        self.analysis_data = None
        self.predictions_data = None
        
        # Initialize predictor and currency standardizer
        self.predictor = None
        self.currency_standardizer = CurrencyStandardizer(logger=self.logger)
        
    def load_prediction_models(self) -> bool:
        """Load prediction models for currency price forecasting."""
        try:
            # Use config to determine models directory
            models_dir = self.config.paths.models_dir
            
            # Check if production models exist
            production_models_dir = models_dir.parent / "currency_production"
            if production_models_dir.exists():
                models_dir = production_models_dir
                self.logger.info(f"Using production models directory: {models_dir}")
            else:
                # Look for latest experiment directory
                models_parent = models_dir.parent
                experiment_dirs = [d for d in models_parent.iterdir() if d.is_dir() and d.name.startswith('currency_exp_')]
                
                if experiment_dirs:
                    # Sort by modification time and use the latest
                    latest_exp_dir = max(experiment_dirs, key=lambda d: d.stat().st_mtime)
                    models_dir = latest_exp_dir
                    self.logger.info(f"Using latest experiment models directory: {models_dir}")
                elif not models_dir.exists():
                    self.logger.error(f"Models directory not found: {models_dir}")
                    return False
            
            self.predictor = ModelPredictor(models_dir, config=self.config, logger=self.logger)
            available_models = self.predictor.load_available_models()
            
            if not available_models:
                self.logger.error("No models available")
                return False
            
            # Log detailed model information
            multi_horizon_count = sum(1 for models in available_models.values() if 'horizons' in models and models['horizons'])
            single_model_count = len(available_models) - multi_horizon_count
            
            self.logger.info(f"Loaded {len(available_models)} currencies: {multi_horizon_count} multi-horizon, {single_model_count} single models")
            
            # Log available currencies and their horizons
            for currency, model_info in available_models.items():
                if 'horizons' in model_info and model_info['horizons']:
                    horizons = sorted(model_info['horizons'])
                    self.logger.info(f"  {currency}: {horizons} day horizons")
                else:
                    self.logger.info(f"  {currency}: single model")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def generate_ml_predictions(self) -> pd.DataFrame:
        """Generate ML predictions for currencies using trained models."""
        try:
            if not self.predictor:
                self.logger.error("Predictor not initialized")
                return pd.DataFrame()
            
            # Generate predictions for different horizons
            all_predictions = []
            horizons = [1, 3, 7]  # 1-day, 3-day, 7-day predictions
            
            for horizon in horizons:
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
            self.logger.info(f"Generated {len(df)} ML predictions across {len(horizons)} horizons")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to generate ML predictions: {e}")
            return pd.DataFrame()

    def get_poe_watch_market_data(self) -> pd.DataFrame:
        """Get POE Watch market data for analysis."""
        try:
            conn = get_db_connection()
            
            query = """
            SELECT
                currency_name,
                mean_price as current_price,
                min_price,
                max_price,
                daily_volume,
                price_change_percent,
                low_confidence,
                league,
                fetch_time as date,
                poe_watch_id,
                category,
                group_name
            FROM live_poe_watch 
            WHERE league = 'Mercenaries'
            AND fetch_time >= NOW() - INTERVAL '7 days'
            AND mean_price > 0
            ORDER BY currency_name, fetch_time
            """
            
            market_data = pd.read_sql(query, conn)
            conn.close()
            
            # Convert date column
            market_data['date'] = pd.to_datetime(market_data['date'])
            
            self.logger.info(f"Retrieved {len(market_data)} POE Watch market records")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get POE Watch market data: {e}")
            return pd.DataFrame()
    
    def analyze_currency_metrics(self, market_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze currency metrics from POE Watch data and ML predictions."""
        if market_df.empty:
            return pd.DataFrame()
        
        analysis_results = []
        
        # Create currency name mapping for fuzzy matching
        poe_watch_currencies = set(market_df['currency_name'].unique())
        
        # Get ML currencies from loaded models, not from predictions (which might be empty)
        ml_currencies = set()
        if self.predictor and hasattr(self.predictor, 'loaded_models'):
            for model_key in self.predictor.loaded_models.keys():
                if '_' in model_key and model_key.split('_')[-1] in ['1d', '3d', '7d']:
                    # Multi-horizon model: extract currency name
                    currency_name = '_'.join(model_key.split('_')[:-1])
                    ml_currencies.add(currency_name)
                else:
                    # Single model: use as-is
                    ml_currencies.add(model_key)
        
        self.logger.info(f"Found {len(ml_currencies)} ML model currencies, {len(poe_watch_currencies)} POE Watch currencies")
        self.logger.info(f"Available ML currencies: {sorted(ml_currencies)}")
        
        # Track matching statistics
        matched_currencies = 0
        skipped_currencies = 0
        
        # Group by currency for analysis
        for currency in market_df['currency_name'].unique():
            currency_data = market_df[market_df['currency_name'] == currency].sort_values('date')
            
            if len(currency_data) < 2:
                continue
            
            # Get latest data point
            latest = currency_data.iloc[-1]
            
            # Calculate price volatility
            price_volatility = currency_data['current_price'].std()
            price_range = currency_data['current_price'].max() - currency_data['current_price'].min()
            
            # Calculate trend (price change over period)
            first_price = currency_data['current_price'].iloc[0]
            last_price = currency_data['current_price'].iloc[-1]
            period_change_percent = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
            
            # Volume analysis
            avg_volume = currency_data['daily_volume'].mean()
            latest_volume = latest['daily_volume']
            volume_trend = 'increasing' if latest_volume > avg_volume else 'decreasing'
            
            liquidity_score = min(100, (latest['daily_volume'] or 0) / 100 * 100)
            
            # Confidence score (inverse of low_confidence flag)
            poe_watch_confidence = 0.3 if latest['low_confidence'] else 0.8
            
            try:
                standardized_currency = self.currency_standardizer.standardize_currency_name(currency)
                if not standardized_currency:
                    self.logger.debug(f"Skipping {currency} - currency not found in database")
                    skipped_currencies += 1
                    continue
                
                # Check if we have an ML model for this standardized currency
                if standardized_currency not in ml_currencies:
                    # No ML model available - use fallback analysis with historical trends
                    matched_ml_currency = None
                else:
                    matched_ml_currency = standardized_currency
            except Exception as e:
                self.logger.warning(f"Error standardizing currency {currency}: {e}")
                skipped_currencies += 1
                continue
            
            # Get ML predictions for this currency using the matched name
            currency_predictions = pd.DataFrame()  # Default to empty
            if matched_ml_currency is not None:
                currency_predictions = predictions_df[predictions_df['currency'] == matched_ml_currency]
                if not currency_predictions.empty and matched_ml_currency != currency:
                    self.logger.debug(f"Matched '{currency}' → '{matched_ml_currency}'")
            
            matched_currencies += 1
            
            # Default ML prediction values
            ml_pred_1d = period_change_percent  # fallback to historical trend
            ml_pred_3d = period_change_percent
            ml_pred_7d = period_change_percent
            ml_confidence = poe_watch_confidence  # fallback to POE Watch confidence
            ml_predicted_price_1d = last_price
            ml_predicted_price_3d = last_price
            ml_predicted_price_7d = last_price
            ml_data_points = len(currency_data)
            
            # Use ML predictions if available
            if not currency_predictions.empty:
                pred_1d = currency_predictions[currency_predictions['horizon_days'] == 1]
                pred_3d = currency_predictions[currency_predictions['horizon_days'] == 3]
                pred_7d = currency_predictions[currency_predictions['horizon_days'] == 7]
                
                if not pred_1d.empty:
                    ml_pred_1d = pred_1d['price_change_percent'].iloc[0]
                    ml_confidence = pred_1d['confidence_score'].iloc[0]
                    ml_predicted_price_1d = pred_1d['predicted_price'].iloc[0]
                    ml_data_points = pred_1d['data_points_used'].iloc[0]
                
                if not pred_3d.empty:
                    ml_pred_3d = pred_3d['price_change_percent'].iloc[0]
                    ml_predicted_price_3d = pred_3d['predicted_price'].iloc[0]
                
                if not pred_7d.empty:
                    ml_pred_7d = pred_7d['price_change_percent'].iloc[0]
                    ml_predicted_price_7d = pred_7d['predicted_price'].iloc[0]
            
            # Combined confidence score (ML + POE Watch)
            combined_confidence = (ml_confidence * 0.7 + poe_watch_confidence * 0.3)
            
            # Quality score based on data points, volume, and confidence
            data_points = len(currency_data)
            quality_score = min(100, 
                (ml_data_points / 10) * 30 + 
                (avg_volume / 100 if avg_volume else 0) * 25 + 
                combined_confidence * 45
            )
            
            # Investment recommendation based on ML predictions
            investment_recommendation = self._get_poe_watch_investment_recommendation(
                ml_pred_1d, combined_confidence, ml_data_points, liquidity_score, price_volatility
            )
            
            # Risk assessment
            risk_level = self._calculate_risk_level(price_volatility, latest['low_confidence'], liquidity_score)
            
            analysis_results.append({
                'currency': currency,
                'ml_currency_name': matched_ml_currency or 'No ML Model',  # Track the matched ML name
                'current_price': last_price,
                'predicted_price_1d': ml_predicted_price_1d,
                'predicted_price_3d': ml_predicted_price_3d,
                'predicted_price_7d': ml_predicted_price_7d,
                'ml_pred_1d': ml_pred_1d,
                'ml_pred_3d': ml_pred_3d,
                'ml_pred_7d': ml_pred_7d,
                'period_change_percent': period_change_percent,  # historical trend
                'min_price': latest['min_price'],
                'max_price': latest['max_price'],
                'exalted_price': 0,  # Not available in live_poe_watch table
                'divine_price': 0,   # Not available in live_poe_watch table
                'price_volatility': price_volatility,
                'price_range': price_range,
                'daily_volume': latest_volume,
                'avg_volume': avg_volume,
                'volume_trend': volume_trend,
                'current_listings': 0,  # Not available in live_poe_watch table
                'accepted_listings': 0, # Not available in live_poe_watch table
                'liquidity_score': liquidity_score,
                'ml_confidence': ml_confidence,
                'poe_watch_confidence': poe_watch_confidence,
                'combined_confidence': combined_confidence,
                'low_confidence': latest['low_confidence'],
                'data_points_used': ml_data_points,
                'quality_score': quality_score,
                'risk_level': risk_level,
                'investment_recommendation': investment_recommendation,
                'category': latest['category'],
                'group_name': latest['group_name'],
                'frame': latest.get('frame', 'normal')  # Default value since not in table
            })
        
        result_df = pd.DataFrame(analysis_results)
        
        # Log matching statistics
        total_processed = matched_currencies + skipped_currencies
        self.logger.info(f"Currency matching complete: {matched_currencies} matched, {skipped_currencies} skipped out of {total_processed} POE Watch currencies")
        self.logger.info(f"Analyzed {len(result_df)} currencies with ML predictions (from {len(ml_currencies)} available ML models)")
        
        return result_df
    
    def _get_poe_watch_investment_recommendation(self, return_pct: float, confidence: float, 
                                               data_points: int, liquidity: float, volatility: float) -> str:
        """Generate investment recommendation based on POE Watch metrics."""
        # Weight factors (removed liquidity dependency)
        return_weight = 0.4
        confidence_weight = 0.35
        data_quality_weight = 0.15
        volatility_weight = 0.1
        
        # Normalize data quality (0-1 scale)
        data_quality = min(1.0, data_points / 20)
        
        # Normalize volatility (lower is better for stability)
        volatility_score = max(0, 1 - (volatility / 100))
        
        # Combined score
        combined_score = (
            (return_pct / 100) * return_weight +
            confidence * confidence_weight +
            data_quality * data_quality_weight +
            volatility_score * volatility_weight
        )
        
        # Thresholds for recommendations
        if combined_score >= 0.7:
            return "🚀 STRONG BUY"
        elif combined_score >= 0.5:
            return "📈 BUY"
        elif combined_score >= 0.4:
            return "👀 WATCH"
        elif combined_score > 0.3:
            return "⚠️ CAUTION"
        elif combined_score > 0.2:
            return "🔻 WEAK"
        else:
            return "🚫 AVOID"
    
    def _calculate_risk_level(self, volatility: float, low_confidence: bool, liquidity: float) -> str:
        """Calculate risk level based on POE Watch metrics."""
        risk_score = 0
        
        # Volatility contributes to risk
        if volatility > 50:
            risk_score += 3
        elif volatility > 20:
            risk_score += 2
        elif volatility > 10:
            risk_score += 1
        
        # Low confidence increases risk
        if low_confidence:
            risk_score += 2
        
        # Low liquidity increases risk
        if liquidity < 20:
            risk_score += 2
        elif liquidity < 50:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 5:
            return "🔴 HIGH"
        elif risk_score >= 3:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"
    
    def create_poe_watch_market_overview(self, analysis_df: pd.DataFrame) -> str:
        """Create market overview section using POE Watch data."""
        if analysis_df.empty:
            return "<p>No POE Watch market data available for analysis.</p>"
        
        # Calculate summary stats
        total_currencies = len(analysis_df)
        profitable_count = len(analysis_df[analysis_df['period_change_percent'] > 0])
        high_volume_count = len(analysis_df[analysis_df['daily_volume'] > 100])
        high_liquidity_count = len(analysis_df[analysis_df['liquidity_score'] > 50])
        
        # Top performers (based on ML predictions)
        top_gainers = analysis_df.nlargest(5, 'ml_pred_1d')
        top_losers = analysis_df.nsmallest(5, 'ml_pred_1d')
        most_liquid = analysis_df.nlargest(5, 'liquidity_score')
        
        html = f"""
        <div style="margin: 20px 0;">
            <h2 style="color: #2c3e50; text-align: center;">🌍 POE Watch Market Overview</h2>
            
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div class="stat-box" style="background: #3498db; color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 120px;">
                    <div style="font-size: 2em; font-weight: bold;">{total_currencies}</div>
                    <div style="font-size: 0.9em;">Total Currencies</div>
                </div>
                <div class="stat-box" style="background: #27ae60; color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 120px;">
                    <div style="font-size: 2em; font-weight: bold;">{profitable_count}</div>
                    <div style="font-size: 0.9em;">Rising Prices</div>
                </div>
                <div class="stat-box" style="background: #f39c12; color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 120px;">
                    <div style="font-size: 2em; font-weight: bold;">{high_volume_count}</div>
                    <div style="font-size: 0.9em;">High Volume</div>
                </div>
                <div class="stat-box" style="background: #9b59b6; color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 120px;">
                    <div style="font-size: 2em; font-weight: bold;">{high_liquidity_count}</div>
                    <div style="font-size: 0.9em;">High Liquidity</div>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin: 30px 0;">
                
                <div style="width: 30%; background: #d5f4e6; padding: 15px; border-radius: 10px;">
                    <h3 style="color: #27ae60; text-align: center;">📈 Top Gainers</h3>
                    <table style="width: 100%; font-size: 11px;">
        """
        
        for _, row in top_gainers.iterrows():
            html += f"""
                        <tr>
                            <td style="font-weight: bold; padding: 3px;">{row['currency']}</td>
                            <td style="text-align: right; color: #27ae60; font-weight: bold; padding: 3px;">{row['ml_pred_1d']:+.1f}%</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div style="width: 30%; background: #fdeaea; padding: 15px; border-radius: 10px;">
                    <h3 style="color: #e74c3c; text-align: center;">📉 Top Losers</h3>
                    <table style="width: 100%; font-size: 11px;">
        """
        
        for _, row in top_losers.iterrows():
            html += f"""
                        <tr>
                            <td style="font-weight: bold; padding: 3px;">{row['currency']}</td>
                            <td style="text-align: right; color: #e74c3c; font-weight: bold; padding: 3px;">{row['ml_pred_1d']:+.1f}%</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div style="width: 30%; background: #e8f4fd; padding: 15px; border-radius: 10px;">
                    <h3 style="color: #3498db; text-align: center;">💧 Most Liquid</h3>
                    <table style="width: 100%; font-size: 11px;">
        """
        
        for _, row in most_liquid.iterrows():
            html += f"""
                        <tr>
                            <td style="font-weight: bold; padding: 3px;">{row['currency']}</td>
                            <td style="text-align: right; color: #3498db; font-weight: bold; padding: 3px;">{row['liquidity_score']:.0f}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def create_investment_opportunities_table(self, analysis_df: pd.DataFrame) -> str:
        """Create comprehensive investment opportunities table with multi-horizon forecasting."""
        if analysis_df.empty:
            return "<h2>💎 Investment Opportunities</h2><p>No investment opportunities available - no data to analyze.</p>"
        
        # Sort by investment potential (combination of ML prediction, confidence, and volume)
        analysis_df['investment_score'] = (
            analysis_df['ml_pred_1d'] * 0.5 +
            analysis_df['combined_confidence'] * 100 * 0.3 +
            (analysis_df['daily_volume'] / analysis_df['daily_volume'].max() * 100) * 0.2
        )
        
        sorted_opportunities = analysis_df.sort_values('investment_score', ascending=False)
        
        html = f"""
        <div style="margin: 20px 0;">
            <h2 style="color: #2c3e50; text-align: center;">💎 Multi-Horizon Investment Opportunities ({len(sorted_opportunities)} currencies)</h2>
            <div style="max-height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px;">
                <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 10px;">
                    <thead style="position: sticky; top: 0; z-index: 10;">
                        <tr style="background-color: #34495e; color: white;">
                            <th style="padding: 6px; text-align: left; border: 1px solid #ddd;">Currency</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">Current Price</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">1d Pred</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">3d Pred</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">7d Pred</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">1d %</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">3d %</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">7d %</th>
                            <th style="padding: 6px; text-align: center; border: 1px solid #ddd;">Confidence</th>
                            <th style="padding: 6px; text-align: right; border: 1px solid #ddd;">Volume</th>
                            <th style="padding: 6px; text-align: center; border: 1px solid #ddd;">Risk</th>
                            <th style="padding: 6px; text-align: center; border: 1px solid #ddd;">Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, (_, row) in enumerate(sorted_opportunities.iterrows(), 1):
            # Color coding for different horizons
            change_color_1d = '#27ae60' if row['ml_pred_1d'] > 0 else '#e74c3c'
            change_color_3d = '#27ae60' if row['ml_pred_3d'] > 0 else '#e74c3c'
            change_color_7d = '#27ae60' if row['ml_pred_7d'] > 0 else '#e74c3c'
            
            # Risk colors
            risk_colors = {
                "🟢 LOW": "#27ae60",
                "🟡 MEDIUM": "#f39c12", 
                "🔴 HIGH": "#e74c3c"
            }
            risk_color = risk_colors.get(row['risk_level'], "#95a5a6")
            
            # Recommendation colors
            rec_colors = {
                "🚀 STRONG BUY": "#27ae60",
                "📈 BUY": "#2ecc71",
                "👀 WATCH": "#f39c12",
                "⚠️ CAUTION": "#e67e22",
                "🔻 WEAK": "#e74c3c",
                "🚫 AVOID": "#c0392b"
            }
            rec_color = rec_colors.get(row['investment_recommendation'], "#95a5a6")
            
            # Calculate price range
            price_range = f"{row['min_price']:.2f} - {row['max_price']:.2f}c"
            
            # Format confidence as percentage
            confidence_pct = f"{row['combined_confidence']*100:.0f}%"
            
            # Confidence color coding
            conf_color = '#27ae60' if row['combined_confidence'] >= 0.7 else '#f39c12' if row['combined_confidence'] >= 0.5 else '#e74c3c'
            
            html += f"""
                <tr style="background-color: {'#f8f9fa' if i % 2 == 0 else 'white'};">
                    <td style="padding: 4px; border: 1px solid #ddd; font-weight: bold; font-size: 9px;">{row['currency']}</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right;">{row['current_price']:.2f}c</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right; font-weight: bold;">{row['predicted_price_1d']:.2f}c</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right; font-weight: bold;">{row['predicted_price_3d']:.2f}c</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right; font-weight: bold;">{row['predicted_price_7d']:.2f}c</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right; color: {change_color_1d}; font-weight: bold;">{row['ml_pred_1d']:+.1f}%</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right; color: {change_color_3d}; font-weight: bold;">{row['ml_pred_3d']:+.1f}%</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right; color: {change_color_7d}; font-weight: bold;">{row['ml_pred_7d']:+.1f}%</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: center; color: {conf_color}; font-weight: bold;">{confidence_pct}</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: right;">{row['daily_volume']:.0f}</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: center; color: {risk_color}; font-weight: bold; font-size: 9px;">{row['risk_level']}</td>
                    <td style="padding: 4px; border: 1px solid #ddd; text-align: center; color: {rec_color}; font-weight: bold; font-size: 8px;">{row['investment_recommendation']}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        return html
    
    def create_portfolio_recommendations(self, analysis_df: pd.DataFrame) -> str:
        """Create portfolio recommendations based on POE Watch analysis."""
        if analysis_df.empty:
            return "<h2>💼 Portfolio Recommendations</h2><p>No portfolio recommendations available - no data to analyze.</p>"
        
        # Conservative: Low risk, positive ML predictions
        conservative = analysis_df[
            (analysis_df['risk_level'] == '🟢 LOW') & 
            (analysis_df['ml_pred_1d'] > 0) &
            (analysis_df['daily_volume'] >= 50) &
            (analysis_df['combined_confidence'] >= 0.6)
        ].sort_values(['ml_pred_1d', 'combined_confidence'], ascending=False).head(5)
        
        # Balanced: Medium risk, good ML predictions
        balanced = analysis_df[
            (analysis_df['ml_pred_1d'] > 5) &
            (analysis_df['daily_volume'] >= 20) &
            (analysis_df['risk_level'].isin(['🟢 LOW', '🟡 MEDIUM'])) &
            (analysis_df['combined_confidence'] >= 0.5)
        ].sort_values('investment_score', ascending=False).head(5)
        
        # Aggressive: High ML predictions, willing to accept higher risk
        aggressive = analysis_df[
            analysis_df['ml_pred_1d'] > 20
        ].sort_values('ml_pred_1d', ascending=False).head(5)
        
        html = """
        <div style="margin: 30px 0;">
            <h2 style="color: #2c3e50; text-align: center;">💼 Portfolio Recommendations</h2>
            
            <div style="display: flex; justify-content: space-between; margin: 20px 0;">
                
                <div style="width: 30%; background: #ecf0f1; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #27ae60; text-align: center;">🛡️ Conservative</h3>
                    <p style="text-align: center; color: #7f8c8d; font-size: 12px;">Low Risk • Steady Growth</p>
        """
        
        for _, row in conservative.iterrows():
            html += f"""
                    <div style="margin: 8px 0; padding: 8px; background: white; border-radius: 5px; font-size: 11px;">
                        <strong>{row['currency']}</strong><br>
                        <span style="color: #27ae60;">Profit: {row['ml_pred_1d']:+.1f}%</span> | 
                        <span style="color: #3498db;">Risk: {row['risk_level']}</span><br>
                        <span style="color: #9b59b6;">Confidence: {row['combined_confidence']*100:.0f}%</span>
                    </div>
            """
        
        html += """
                </div>
                
                <div style="width: 30%; background: #ecf0f1; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #f39c12; text-align: center;">⚖️ Balanced</h3>
                    <p style="text-align: center; color: #7f8c8d; font-size: 12px;">Medium Risk • Good Returns</p>
        """
        
        for _, row in balanced.iterrows():
            html += f"""
                    <div style="margin: 8px 0; padding: 8px; background: white; border-radius: 5px; font-size: 11px;">
                        <strong>{row['currency']}</strong><br>
                        <span style="color: #27ae60;">Profit: {row['ml_pred_1d']:+.1f}%</span> | 
                        <span style="color: #3498db;">Risk: {row['risk_level']}</span><br>
                        <span style="color: #9b59b6;">Score: {row['investment_score']:.0f}</span>
                    </div>
            """
        
        html += """
                </div>
                
                <div style="width: 30%; background: #ecf0f1; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #e74c3c; text-align: center;">🚀 Aggressive</h3>
                    <p style="text-align: center; color: #7f8c8d; font-size: 12px;">High Risk • High Returns</p>
        """
        
        for _, row in aggressive.iterrows():
            html += f"""
                    <div style="margin: 8px 0; padding: 8px; background: white; border-radius: 5px; font-size: 11px;">
                        <strong>{row['currency']}</strong><br>
                        <span style="color: #27ae60;">Profit: {row['ml_pred_1d']:+.1f}%</span> | 
                        <span style="color: #3498db;">Risk: {row['risk_level']}</span><br>
                        <span style="color: #e67e22;">Volume: {row['daily_volume']:.0f}</span>
                    </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
        
        return html
    
    def generate_comprehensive_report(self) -> str:
        """Generate the complete POE Watch investment report."""
        self.logger.info("🚀 Starting POE Watch investment report generation...")
        
        # Load ML prediction models
        if not self.load_prediction_models():
            self.logger.warning("⚠️ ML models not available, using statistical analysis only")
        
        # Get POE Watch market data
        self.market_data = self.get_poe_watch_market_data()
        if self.market_data.empty:
            self.logger.error("No POE Watch market data available")
            return None
        
        # Generate ML predictions
        self.predictions_data = self.generate_ml_predictions()
        if self.predictions_data.empty:
            self.logger.warning("⚠️ No ML predictions generated, using fallback analysis")
            self.predictions_data = pd.DataFrame()  # Empty DataFrame for fallback
        
        # Analyze currency metrics with ML predictions
        self.analysis_data = self.analyze_currency_metrics(self.market_data, self.predictions_data)
        if self.analysis_data.empty:
            self.logger.error("No analysis data generated")
            return None
        
        # Generate report sections
        self.logger.info("📊 Creating investment analysis sections...")
        
        investment_opportunities = self.create_investment_opportunities_table(self.analysis_data)
        portfolio_recommendations = self.create_portfolio_recommendations(self.analysis_data)
        
        # Calculate summary statistics with safe defaults
        total_currencies = len(self.analysis_data)
        
        if total_currencies > 0 and 'ml_pred_1d' in self.analysis_data.columns:
            profitable_count = len(self.analysis_data[self.analysis_data['ml_pred_1d'] > 0])
            profitable_opportunities = len(self.analysis_data[
                (self.analysis_data['ml_pred_1d'] > 5) & 
                (self.analysis_data['combined_confidence'] > 0.5)
            ])
            highest_return = self.analysis_data['ml_pred_1d'].max()
            avg_confidence = self.analysis_data['combined_confidence'].mean()
        else:
            profitable_count = 0
            profitable_opportunities = 0
            highest_return = 0.0
            avg_confidence = 0.0
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>POE Watch Investment Report - {timestamp}</title>
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
                <h1>⚡ PoEconomy Investment Report</h1>
                <h2>AI-Powered Path of Exile Currency Predictions</h2>
                <p>Generated on {timestamp} | Mercenaries League | ML Models + poe.watch API</p>
            </div>
            
            <div class="section">
                <h2>📈 Executive Summary</h2>
                <div class="summary-stats">
                    <div class="stat-box" style="background: #27ae60;">
                        <div class="stat-value">{total_currencies}</div>
                        <div class="stat-label">Currencies Analyzed</div>
                    </div>
                    <div class="stat-box" style="background: #e74c3c;">
                        <div class="stat-value">{highest_return:.0f}%</div>
                        <div class="stat-label">Highest Return</div>
                    </div>
                    <div class="stat-box" style="background: #f39c12;">
                        <div class="stat-value">{avg_confidence:.1f}</div>
                        <div class="stat-label">Average Confidence</div>
                    </div>
                    <div class="stat-box" style="background: #9b59b6;">
                        <div class="stat-value">{profitable_opportunities}</div>
                        <div class="stat-label">Profitable Opportunities</div>
                    </div>
                </div>
            </div>
            

            
            <div class="section">
                {investment_opportunities}
            </div>
            
            <div class="section">
                {portfolio_recommendations}
            </div>
            
            <div class="section">
                <h2>⚠️ Disclaimer</h2>
                <p style="color: #7f8c8d; font-style: italic;">
                    This report uses real-time data from poe.watch API and statistical analysis. 
                    All investment recommendations are algorithmic estimates and should not be considered as financial advice. 
                    Path of Exile currency markets are volatile and subject to manipulation. 
                    Always conduct your own research and invest responsibly.
                </p>
            </div>
            
            <div class="section">
                <h2>📊 Methodology & Data Sources</h2>
                <ul>
                    <li><strong>Data Source:</strong> POE Watch API (poe.watch) - Real-time currency market data</li>
                    <li><strong>ML Models:</strong> LightGBM, XGBoost, RandomForest trained on historical price data</li>
                    <li><strong>Prediction Horizons:</strong> 1-day, 3-day, and 7-day price forecasts</li>
                    <li><strong>Analysis Period:</strong> Last 7 days of trading data for trend analysis</li>
                    <li><strong>Risk Assessment:</strong> Based on price volatility, liquidity, and market confidence</li>
                    <li><strong>Investment Scoring:</strong> ML predictions weighted with quality, liquidity, and confidence</li>
                    <li><strong>Portfolio Allocation:</strong> AI-driven recommendations for different investor profiles</li>
                    <li><strong>Data Quality:</strong> Combines ML confidence scores with POE Watch confidence metrics</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_filename = f"poe_watch_investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save data as JSON (with safe handling for empty data)
        data_export = {
            'timestamp': timestamp,
            'data_source': 'poe.watch API',
            'analysis_results': self.analysis_data.to_dict('records') if not self.analysis_data.empty else [],
            'market_summary': {
                'total_currencies': total_currencies,
                'profitable_count': profitable_count,
                'highest_return': float(highest_return),
                'average_confidence': float(avg_confidence),
                'data_points_analyzed': len(self.market_data),
                'top_opportunities': (
                    self.analysis_data.sort_values('investment_score', ascending=False).head(10).to_dict('records')
                    if not self.analysis_data.empty and 'investment_score' in self.analysis_data.columns
                    else []
                )
            }
        }
        
        json_filename = f"poe_watch_investment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(data_export, f, indent=2, default=str)
        
        self.logger.info(f"✅ POE Watch investment report generated successfully!")
        self.logger.info(f"📄 HTML Report: {report_path}")
        self.logger.info(f"📊 Data Export: {json_path}")
        
        return str(report_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="POE Watch Investment Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production report with full analysis
  python generate_poe_watch_investment_report.py --mode production
  
  # Development report with faster settings
  python generate_poe_watch_investment_report.py --mode development
  
  # Testing with minimal settings
  python generate_poe_watch_investment_report.py --mode test
  
  # Custom configuration
  python generate_poe_watch_investment_report.py --config /path/to/config.json
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['production', 'development', 'test'],
        default='production',
        help='Report generation mode (default: production)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        help='Custom experiment ID'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Experiment description'
    )
    
    parser.add_argument(
        '--tags',
        nargs='*',
        default=[],
        help='Experiment tags'
    )
    
    return parser.parse_args()


def main():
    """Main function to generate POE Watch investment report."""
    args = parse_arguments()
    
    print("⚡ POE Watch Investment Report Generator")
    print("=" * 60)
    
    try:
        # Get configuration based on mode
        if args.config:
            config = MLConfig.from_file(args.config)
        elif args.mode == 'production':
            config = get_production_config()
        elif args.mode == 'development':
            config = get_development_config()
        else:  # test
            config = get_test_config()
        
        # Override configuration with command line arguments
        if args.experiment_id:
            config.experiment.experiment_id = args.experiment_id
        if args.description:
            config.experiment.description = args.description
        if args.tags:
            config.experiment.tags.extend(args.tags)
        
        # Add mode and report generation tags
        config.experiment.tags.append(args.mode)
        config.experiment.tags.append('investment_report')
        
        print(f"Running in {args.mode} mode...")
        print(f"Experiment ID: {config.experiment.experiment_id}")
        
        generator = PoeWatchInvestmentReportGenerator(config=config)
        report_path = generator.generate_comprehensive_report()
        
        if report_path:
            print(f"\n✅ SUCCESS! POE Watch investment report generated:")
            print(f"📄 Report: {report_path}")
            print(f"\n💡 Open the HTML file in your browser to view the report!")
        else:
            print("\n❌ Failed to generate POE Watch investment report")
            
    except KeyboardInterrupt:
        print("\nReport generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 