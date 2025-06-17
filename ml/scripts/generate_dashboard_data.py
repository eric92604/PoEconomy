#!/usr/bin/env python3
"""
Dashboard Data Generator for PoEconomy Live Dashboard

This script generates JSON data specifically for the live web dashboard,
integrating with existing ML models and database infrastructure.
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


class DashboardDataGenerator:
    """Generate JSON data for the live investment dashboard."""
    
    def __init__(self, output_dir: str = "web_dashboard"):
        """Initialize the dashboard data generator."""
        self.logger = MLLogger("DashboardDataGenerator")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize predictor
        self.predictor = None
        
    def load_prediction_models(self) -> bool:
        """Load prediction models."""
        try:
            # Get the script directory and find models directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            models_dir = project_root / "ml" / "models" / "currency_production"
            
            # Alternative paths to try
            alternative_paths = [
                Path("../models/currency_production"),
                Path("./models/currency_production"),
                Path("models/currency_production"),
                script_dir.parent / "models" / "currency_production"
            ]
            
            if not models_dir.exists():
                self.logger.warning(f"Primary models directory not found: {models_dir}")
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        models_dir = alt_path
                        self.logger.info(f"Using alternative models directory: {models_dir}")
                        break
                else:
                    self.logger.error("Models directory not found")
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
    
    def generate_predictions(self) -> List[Dict]:
        """Generate fresh predictions for dashboard."""
        try:
            # Focus on 1-day predictions
            horizon = 1
            predictions = self.predictor.predict_multiple_currencies(
                prediction_horizon_days=horizon
            )
            
            dashboard_data = []
            for pred in predictions:
                # Calculate additional metrics for dashboard
                return_percent = pred.price_change_percent
                current_price = pred.current_price
                predicted_price = pred.predicted_price
                
                # Risk assessment
                risk_level = self._assess_risk(pred.confidence_score, abs(return_percent))
                
                # Investment recommendation
                recommendation = self._get_recommendation(return_percent, pred.confidence_score)
                
                dashboard_data.append({
                    'currency': pred.currency,
                    'current_price': float(current_price),
                    'predicted_price': float(predicted_price),
                    'return_percent': float(return_percent),
                    'confidence': float(pred.confidence_score),
                    'risk_level': risk_level,
                    'recommendation': recommendation,
                    'volume': pred.data_points_used,
                    'model_type': pred.model_type,
                    'timestamp': pred.prediction_timestamp,
                    'features_count': len(pred.features_used) if pred.features_used else 0
                })
            
            self.logger.info(f"Generated {len(dashboard_data)} predictions for dashboard")
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate predictions: {e}")
            return []
    
    def get_market_trends(self, days: int = 7) -> List[Dict]:
        """Get market trends data for charts."""
        try:
            conn = get_db_connection()
            
            query = """
            SELECT 
                DATE(sample_time) as date,
                AVG(chaos_equivalent) as avg_price,
                COUNT(*) as currency_count,
                SUM(total_change) as total_change
            FROM live_currency_prices 
            WHERE league = 'Mercenaries'
            AND sample_time >= NOW() - INTERVAL %s DAY
            GROUP BY DATE(sample_time)
            ORDER BY date
            """
            
            trends_data = pd.read_sql(query, conn, params=[days])
            conn.close()
            
            # Calculate market index (normalized to 100 for first day)
            if not trends_data.empty:
                base_value = trends_data.iloc[0]['avg_price']
                trends_data['market_index'] = (trends_data['avg_price'] / base_value) * 100
                
                trends = []
                for _, row in trends_data.iterrows():
                    trends.append({
                        'date': row['date'].isoformat(),
                        'value': float(row['market_index']),
                        'currency_count': int(row['currency_count']),
                        'total_change': float(row['total_change'] or 0)
                    })
                
                return trends
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get market trends: {e}")
            return self._generate_sample_trends(days)
    
    def _generate_sample_trends(self, days: int = 7) -> List[Dict]:
        """Generate sample trends data when database is unavailable."""
        trends = []
        now = datetime.now()
        
        for i in range(days):
            date = now - timedelta(days=days-1-i)
            # Simulate market movement
            value = 100 + np.sin(i * 0.5) * 5 + np.random.normal(0, 2)
            trends.append({
                'date': date.date().isoformat(),
                'value': float(max(90, min(110, value))),  # Keep within reasonable bounds
                'currency_count': 50 + np.random.randint(-10, 10),
                'total_change': np.random.normal(0, 5)
            })
        
        return trends
    
    def _assess_risk(self, confidence: float, return_magnitude: float) -> str:
        """Assess risk level based on confidence and return magnitude."""
        if confidence > 0.8 and return_magnitude < 30:
            return "LOW"
        elif confidence > 0.6 and return_magnitude < 50:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _get_recommendation(self, return_percent: float, confidence: float) -> str:
        """Get investment recommendation."""
        if return_percent > 50 and confidence > 0.7:
            return "STRONG BUY"
        elif return_percent > 20 and confidence > 0.6:
            return "BUY"
        elif return_percent < -20:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_dashboard_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate key metrics for dashboard."""
        if not predictions:
            return {
                'total_opportunities': 0,
                'best_return': 0,
                'avg_confidence': 0,
                'total_currencies': 0,
                'profitable_count': 0,
                'high_confidence_count': 0
            }
        
        # Filter profitable opportunities (>10% return with >60% confidence)
        profitable_opportunities = [
            p for p in predictions 
            if p['return_percent'] > 10 and p['confidence'] > 0.6
        ]
        
        # Best return
        best_return = max(p['return_percent'] for p in predictions)
        
        # Average confidence
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        # High confidence predictions (>70%)
        high_confidence_count = len([p for p in predictions if p['confidence'] > 0.7])
        
        return {
            'total_opportunities': len(profitable_opportunities),
            'best_return': float(best_return),
            'avg_confidence': float(avg_confidence),
            'total_currencies': len(predictions),
            'profitable_count': len([p for p in predictions if p['return_percent'] > 0]),
            'high_confidence_count': high_confidence_count
        }
    
    def generate_dashboard_json(self) -> Dict:
        """Generate complete dashboard data as JSON."""
        try:
            # Load models
            if not self.load_prediction_models():
                self.logger.error("Failed to load models, using sample data")
                return self._generate_sample_dashboard_data()
            
            # Generate predictions
            predictions = self.generate_predictions()
            
            # Get market trends
            trends = self.get_market_trends()
            
            # Calculate metrics
            metrics = self.calculate_dashboard_metrics(predictions)
            
            # Sort opportunities by return potential
            opportunities = sorted(
                predictions, 
                key=lambda x: x['return_percent'] * x['confidence'], 
                reverse=True
            )
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'opportunities': opportunities[:50],  # Top 50 opportunities
                'trends': trends,
                'status': 'success',
                'data_freshness': {
                    'predictions_generated': len(predictions),
                    'trends_days': len(trends),
                    'last_update': datetime.now().isoformat()
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard data: {e}")
            return self._generate_sample_dashboard_data()
    
    def _generate_sample_dashboard_data(self) -> Dict:
        """Generate sample data when real data is unavailable."""
        sample_currencies = [
            "Valdo's Puzzle Box", "Maven's Chisel of Scarabs", "Sacred Blossom",
            "Otherworldly Scouting Report", "Mortal Rage", "Dedication to the Goddess",
            "Fragment of Constriction", "Baran's Crest", "Tailoring Orb", "Fertile Catalyst"
        ]
        
        opportunities = []
        for i, currency in enumerate(sample_currencies):
            current_price = np.random.uniform(5, 100)
            return_percent = np.random.uniform(-30, 150)
            predicted_price = current_price * (1 + return_percent / 100)
            confidence = np.random.uniform(0.5, 0.95)
            
            opportunities.append({
                'currency': currency,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'return_percent': float(return_percent),
                'confidence': float(confidence),
                'risk_level': self._assess_risk(confidence, abs(return_percent)),
                'recommendation': self._get_recommendation(return_percent, confidence),
                'volume': np.random.randint(100, 5000),
                'model_type': 'sample',
                'timestamp': datetime.now().isoformat(),
                'features_count': np.random.randint(5, 20)
            })
        
        metrics = self.calculate_dashboard_metrics(opportunities)
        trends = self._generate_sample_trends()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'opportunities': opportunities,
            'trends': trends,
            'status': 'sample_data',
            'data_freshness': {
                'predictions_generated': len(opportunities),
                'trends_days': len(trends),
                'last_update': datetime.now().isoformat()
            }
        }
    
    def save_dashboard_data(self, output_file: str = None) -> str:
        """Save dashboard data to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"dashboard_data_{timestamp}.json"
        
        dashboard_data = self.generate_dashboard_json()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Dashboard data saved to: {output_file}")
        return str(output_file)


def main():
    """Main function to generate dashboard data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PoEconomy Dashboard Data")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument("--stdout", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    
    args = parser.parse_args()
    
    # Create generator
    generator = DashboardDataGenerator()
    
    if args.stdout:
        # Output to stdout for API integration
        dashboard_data = generator.generate_dashboard_json()
        if args.pretty:
            print(json.dumps(dashboard_data, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(dashboard_data, ensure_ascii=False))
    else:
        # Save to file
        output_file = generator.save_dashboard_data(args.output)
        print(f"Dashboard data generated: {output_file}")


if __name__ == "__main__":
    main() 