#!/usr/bin/env python3
"""
Full Production Currency-Specific Training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from train_models import ImprovedCurrencyTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run full production training for all currency pairs."""
    data_path = "../../ml/training_data/combined_currency_features_exp_20250611_013044.parquet"
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Create trainer with improved strategies for PRODUCTION
    trainer = ImprovedCurrencyTrainer(data_path, output_dir="ml/models/improved_production")
    
    # Run FULL production training for ALL currency pairs
    logger.info("Starting training for all currency pairs...")
    
    # Load data to show statistics
    data = pd.read_parquet(data_path)
    currency_counts = data['currency_pair'].value_counts()
    eligible_currencies = currency_counts[currency_counts >= trainer.min_samples_required]
    
    logger.info(f"Dataset contains {len(currency_counts)} total currency pairs")
    logger.info(f"{len(eligible_currencies)} pairs have sufficient data (≥{trainer.min_samples_required} samples)")
    logger.info(f"Training models for ALL {len(eligible_currencies)} eligible currency pairs...")
    
    # Run the FULL training (no artificial limitations)
    results = trainer.train_all_currencies()
    
    if results:
        trainer.save_results_summary()
        
        # Display FULL production training summary
        logger.info(f"\n{'='*80}")
        logger.info("FULL PRODUCTION TRAINING RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        
        total_models = len(results)
        mape_under_15 = len([r for r in results if r.mape < 15])
        da_over_65 = len([r for r in results if r.directional_accuracy > 65])
        
        logger.info(f"Total models trained: {total_models}")
        logger.info(f"Models with MAPE < 15%: {mape_under_15}/{total_models} ({mape_under_15/total_models*100:.1f}%)")
        logger.info(f"Models with DA > 65%: {da_over_65}/{total_models} ({da_over_65/total_models*100:.1f}%)")
        
        # Show top 10 performers
        sorted_results = sorted(results, key=lambda x: x.mape)
        logger.info(f"\n🏆 TOP 10 PERFORMERS:")
        for i, result in enumerate(sorted_results[:10]):
            logger.info(f"{i+1:2d}. {result.currency_pair:35s}: MAPE={result.mape:6.2f}%, DA={result.directional_accuracy:5.1f}%")
        
        # Show bottom performers for analysis
        logger.info(f"\nBOTTOM 5 PERFORMERS (need improvement):")
        for i, result in enumerate(sorted_results[-5:]):
            rank = len(sorted_results) - 4 + i
            logger.info(f"{rank:2d}. {result.currency_pair:35s}: MAPE={result.mape:6.2f}%, DA={result.directional_accuracy:5.1f}%")
        
        # Show improvement strategies used
        if results:
            strategies = results[0].improvement_strategies
            logger.info(f"\nIMPROVEMENT STRATEGIES APPLIED:")
            for strategy in strategies:
                logger.info(f"{strategy}")
        
        # Performance statistics
        avg_mape = sum(r.mape for r in results) / len(results)
        median_mape = sorted([r.mape for r in results])[len(results)//2]
        avg_da = sum(r.directional_accuracy for r in results) / len(results)
        
        logger.info(f"\nPERFORMANCE STATISTICS:")
        logger.info(f"Average MAPE: {avg_mape:.2f}%")
        logger.info(f"Median MAPE:  {median_mape:.2f}%")
        logger.info(f"Average DA:   {avg_da:.1f}%")
        
        logger.info(f"\nFULL production training completed! Check ml/models/improved_production/ for ALL {total_models} models.")
        
    else:
        logger.error("No models were trained successfully in full production training")

if __name__ == "__main__":
    main() 