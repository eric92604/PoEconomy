# PoEconomy - Currency-Specific Price Prediction

## 🚀 Production-Ready Implementation

This directory contains the **currency-specific machine learning implementation** for Path of Exile market price prediction.

## 🗂️ File Structure

### Core Implementation (Production-Ready)
```
scripts/
├── train_models.py                     # 🎯 MAIN: Currency-specific training pipeline
├── test_training.py                    # ✅ Testing script for validation
└── currency_insights_summary.py         # 📊 Results summary and insights
```

### Analysis & Utilities
```
scripts/
├── feature_engineering.py              # Data preprocessing pipeline
├── identify_target_currencies.py       # Currency selection logic
├── data_exploration.py                 # Dataset exploration tools
├── backup_postgres.py                  # Database backup utility
└── insert_currency_prices.py           # Data insertion pipeline
```

### Data & Models
```
training_data/                           # Training datasets (parquet format)
models/                                  # Model storage (created during training)
```

## 🏃‍♂️ Quick Start

### 1. Generate Training Data
```bash
cd ml/scripts
python feature_engineering.py
```

### 2. Train Currency-Specific Models
```bash
python train_models.py
```

### 3. Test the Implementation
```bash
python test_training.py
```

## 🔧 Key Technical Features

### Currency-Specific Approach
- **Independent models** for each currency pair
- **Hyperparameter optimization** per currency
- **Eliminates cross-currency noise** that destroyed global model accuracy
- **Scalable architecture** for new currencies

### Model Architecture
- **LightGBM + XGBoost** ensemble with optimized hyperparameters
- **Time series cross-validation** with 5-fold splits
- **RobustScaler** for outlier-resistant feature normalization
- **Optuna** for automated hyperparameter tuning (50 trials per currency)

### Data Processing
- **League-phase data extraction** with 60-day league windows
- **Advanced feature engineering** with rolling statistics and momentum indicators
- **Missing value handling** with median/mode imputation strategies
- **Feature selection** using SelectKBest for high-dimensional data
- **Outlier removal** with conservative IQR-based cleaning

## 🚀 Production Deployment

### Model Storage
- Models saved as `.pkl` files in `ml/models/currency_specific/`
- Each currency has its own subdirectory
- Includes both model and scaler files

### API Integration
The currency-specific models are designed for:
- **Real-time price prediction** via API endpoints
- **Currency routing** based on requested currency pair
- **Automated retraining** with new market data
- **Confidence scoring** for prediction reliability

## 🔬 Implementation Details

### Feature Engineering Pipeline
**File**: `feature_engineering.py`
- **League-aware temporal sampling**: Extracts data from comparable league phases
- **Recency weighting**: Recent leagues prioritized via metadata features
- **Multi-horizon targets**: 1-day, 3-day, and 7-day prediction horizons
- **Rolling statistics**: Price mean/std/min/max with multiple time windows
- **Momentum indicators**: Price change and volatility calculations

### Training Architecture
**File**: `train_models.py` → `ImprovedCurrencyTrainer` class
```python
strategies = {
    'log_transform': True,      # Conditional log1p transformation
    'robust_scaling': True,     # RobustScaler for outlier handling
    'feature_selection': True,  # SelectKBest for high-dimensional data
    'ensemble_models': True,    # LightGBM + XGBoost combination
    'outlier_removal': True,    # Conservative IQR-based cleaning
    'advanced_cv': True         # 5-fold TimeSeriesSplit validation
}
```
### Production Features
- **Model persistence**: Joblib serialization with preprocessing metadata
- **Currency routing**: Automatic model selection by currency pair
- **Confidence scoring**: Ensemble agreement and historical performance
- **Error handling**: Robust preprocessing with fallback strategies

---
