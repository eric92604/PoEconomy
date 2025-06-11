# Machine Learning Architecture: PoEconomy Currency Prediction System

## 🏗️ Architecture Overview

The PoEconomy ML system implements a **currency-specific approach** that trains independent models for each currency pair, providing targeted predictions for Path of Exile market price forecasting.

---

## 🔧 Technical Stack

### Core Technologies
```python
# ML Framework
- LightGBM (Primary): Gradient boosting with hyperparameter optimization
- XGBoost (Secondary): Ensemble diversity and regularization
- Optuna: Bayesian hyperparameter optimization
- scikit-learn: Preprocessing and metrics

# Data Processing
- pandas: Data manipulation and feature engineering
- numpy: Numerical computations and array operations
- psycopg2: PostgreSQL database connectivity
- pyarrow: Efficient parquet file operations

# Infrastructure
- joblib: Model serialization and persistence
- pathlib: Cross-platform file operations
- logging: System monitoring and debugging
```

---

## 🏛️ Architecture Components

### 1. Data Layer
```
📁 Database Integration
├── PostgreSQL Database (Historical Data)
├── League-based temporal partitioning  
├── Currency price time series
└── Market confidence indicators

📁 Feature Engineering Pipeline
├── League-phase data extraction
├── Temporal feature creation
└── Currency-specific transformations
```

### 2. Model Layer
```
📁 Currency-Specific Training
├── Independent model per currency pair
├── Hyperparameter optimization (Optuna)
├── Ensemble architecture (LightGBM + XGBoost)
└── Time series cross-validation

📁 Model Persistence
├── ml/models/currency_specific/
│   ├── {currency_pair}/
│   │   ├── ensemble_model.pkl
│   │   ├── scaler.pkl
│   │   └── preprocessing_info.json
└── Experiment tracking and versioning
```

### 3. Prediction Layer
```
📁 Serving Infrastructure
├── Currency routing logic
├── Feature preprocessing pipeline
├── Ensemble prediction combination
└── Confidence scoring system
```

---

## 🔄 Data Flow Architecture

### Training Pipeline
```python
1. Data Collection
   └── get_league_phase_data() → League-aware temporal sampling
   
2. Feature Engineering  
   └── engineer_league_metadata_features() → League context + price features
   
3. Currency-Specific Training
   └── train_currency_model() → Individual model per pair
   
4. Hyperparameter Optimization
   └── optuna.optimize() → 50 trials per currency
   
5. Ensemble Training
   └── train_ensemble_model() → LightGBM + XGBoost
   
6. Model Persistence
   └── joblib.dump() → Serialized models + metadata
```

### Prediction Pipeline
```python
1. Currency Identification
   └── extract_currency_pair(request) → Route to specific model
   
2. Feature Preprocessing
   └── apply_saved_preprocessing() → Consistent transformation
   
3. Model Loading
   └── load_currency_model() → Cached model retrieval
   
4. Ensemble Prediction
   └── predict_ensemble() → Combined LightGBM + XGBoost
   
5. Confidence Scoring
   └── calculate_prediction_confidence() → Uncertainty quantification
```

---

## 🧠 Model Architecture Details

### Currency-Specific Trainer Configuration
```python
class ImprovedCurrencyTrainer:
    # Training strategies
    strategies = {
        'log_transform': True,      # Conditional log1p for wide-range currencies
        'robust_scaling': True,     # RobustScaler for outlier resilience  
        'feature_selection': True,  # SelectKBest for high-dimensional data
        'ensemble_models': True,    # LightGBM + XGBoost combination
        'outlier_removal': True,    # Conservative IQR-based cleaning
        'advanced_cv': True         # 5-fold TimeSeriesSplit validation
    }
    
    # Quality gates
    min_samples_required = 150      # Minimum data points per currency
    test_size = 0.2                # Hold-out validation set
    random_state = 42              # Reproducible results
```

### Feature Engineering Strategy
```python
# League Context Features (Temporal Awareness)
league_features = {
    'league_age_days': 'Days since league start',
    'league_phase_early/mid/late': 'Economic phase indicators',
    'league_recency_rank': 'Relative league age ranking',
    'week_in_league': 'Weekly patterns within league'
}

# Price-Based Features (Currency-Specific)
price_features = {
    'price_mean/std/min/max_{window}d': 'Rolling statistics (1,3,7,14 days)',
    'momentum_{period}d': 'Price momentum indicators (3,7 days)', 
    'volatility_{window}d': 'Normalized volatility (7,14 days)',
    'price_vs_league_mean': 'Relative pricing within league'
}

# Target Variables (Multi-horizon)
target_features = {
    'target_price_{horizon}d': 'Future price prediction (1,3,7 days)',
    'target_change_pct_{horizon}d': 'Percentage change prediction',
    'target_direction_{horizon}d': 'Trend direction (up/down/stable)'
}
```

### Ensemble Architecture
```python
# Primary Model: LightGBM (Optimized Parameters)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 20-150,           # Optuna optimized
    'learning_rate': 0.01-0.3,      # Optuna optimized
    'feature_fraction': 0.5-1.0,    # Optuna optimized
    'bagging_fraction': 0.5-1.0,    # Optuna optimized
    'min_child_samples': 10-100,    # Optuna optimized
    'reg_alpha': 0.0-2.0,           # L1 regularization
    'reg_lambda': 0.0-2.0           # L2 regularization
}

# Secondary Model: XGBoost (Ensemble Diversity)
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200
}

# Ensemble Combination: Simple Average
final_prediction = (lgb_prediction + xgb_prediction) / 2
```

---

## 🏆 Summary

The PoEconomy ML architecture provides a **currency prediction system** with:

1. **Independent Models**: Separate training per currency pair for targeted performance
2. **League-Aware Processing**: Temporal features respecting Path of Exile economic cycles  
3. **Ensemble Architecture**: LightGBM + XGBoost with Bayesian optimization
4. **Scalable Infrastructure**: Model routing, caching, monitoring, and automated retraining capabilities

The system delivers consistent predictions through robust feature engineering, comprehensive validation, and modular deployment architecture. 