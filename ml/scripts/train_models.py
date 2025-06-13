#!/usr/bin/env python3
"""
Currency-Specific Model Training

This script is the entry point for running model training experiments.
It uses the ModelTrainingPipeline to train currency-specific models.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.model_training_pipeline import main

if __name__ == "__main__":
    main() 