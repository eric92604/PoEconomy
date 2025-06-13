#!/usr/bin/env python3
"""
Feature engineering script.

This script is the entry point for running feature engineering experiments.
It uses the FeatureEngineeringPipeline to process currency price data.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.feature_engineering_pipeline import main

if __name__ == "__main__":
    main() 