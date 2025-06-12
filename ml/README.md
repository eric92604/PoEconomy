This directory contains the implementation of the PoEconomy currency price prediction system, built with industry best practices and enterprise-grade architecture.

## 🏗️ Architecture Overview

The pipeline follows a modular, scalable architecture with clear separation of concerns:

```
ml/
├── config/                     # Centralized configuration management
│   └── training_config.py      # Complete ML configuration system
├── utils/                      # Reusable utility modules
│   ├── logging_utils.py        # Comprehensive logging and monitoring
│   ├── data_processing.py      # Data validation and feature engineering
│   ├── model_training.py       # Model training and hyperparameter optimization
│   ├── database.py             # Database connectivity
│   ├── identify_target_currencies.py  # Currency selection logic
│   ├── backup_postgres.py      # Database backup utilities
│   └── insert_currency_prices.py      # Data insertion utilities
├── scripts/                    # Pipeline scripts
│   ├── feature_engineering.py  # Feature engineering pipeline
│   └── train_models.py         # Model training pipeline
├── models/                     # Trained model artifacts
├── training_data/             # Processed training datasets
└── logs/                      # Comprehensive logging output
```
## 🚀 Quick Start

### 1. Feature Engineering
```bash
# Standard feature engineering
cd ml/scripts
python feature_engineering.py

# With parallel processing (faster)
python feature_engineering.py --parallel

# Custom experiment
python feature_engineering.py --experiment-id my_experiment --description "Testing new features"
```

### 2. Model Training
```bash
# Production training with full optimization
python train_models.py --mode production

# Development training (faster for testing)
python train_models.py --mode development

# Testing mode (minimal for CI/CD)
python train_models.py --mode testing
```

### 3. Configuration Management
```python
from config.training_config import get_production_config, MLConfig

# Use predefined configuration
config = get_production_config()

# Or load from file
config = MLConfig.from_file('my_config.json')

# Customize and save
config.model.n_trials = 100
config.save('custom_config.json')
```

## 📊 Configuration System

### Environment-Specific Configurations

#### Production Configuration
- **200 hyperparameter trials** for optimal performance
- **5-fold cross-validation** for robust evaluation
- **Full logging** with structured output
- **Model artifact saving** enabled

#### Development Configuration
- **50 hyperparameter trials** for faster iteration
- **3-fold cross-validation** for quicker feedback
- **Debug logging** for detailed troubleshooting
- **Reduced data requirements** for testing

#### Testing Configuration
- **10 hyperparameter trials** for CI/CD pipelines
- **2-fold cross-validation** for speed
- **Warning-level logging** for clean output
- **Minimal resource usage**

### Configuration Structure
```python
@dataclass
class MLConfig:
    model: ModelConfig          # Model training parameters
    data: DataConfig           # Data processing parameters
    processing: ProcessingConfig # Feature engineering strategies
    paths: PathConfig          # File and directory paths
    logging: LoggingConfig     # Logging configuration
    experiment: ExperimentConfig # Experiment tracking
```

## 🔧 Utility Modules

### Logging Utilities (`utils/logging_utils.py`)
```python
from utils.logging_utils import setup_ml_logging, MLLogger

# Setup comprehensive logging
logger = setup_ml_logging(
    name="MyExperiment",
    level="INFO",
    log_dir="./logs",
    experiment_id="exp_001"
)

# Structured logging with metadata
logger.info("Training started", extra={"model_type": "ensemble"})

# Operation timing
with logger.log_operation("Model training"):
    # Your training code here
    pass
```

### Data Processing (`utils/data_processing.py`)
```python
from utils.data_processing import DataProcessor, DataValidator

# Comprehensive data validation
validator = DataValidator(config.data)
validation_result = validator.validate_dataframe(df, "chaos_orb")

# Feature engineering pipeline
processor = DataProcessor(config.data, config.processing)
processed_data, metadata = processor.process_currency_data(df, "chaos_orb")
```

### Model Training (`utils/model_training.py`)
```python
from utils.model_training import ModelTrainer, save_model_artifacts

# Model training
trainer = ModelTrainer(config.model, config.processing)
result = trainer.train_single_model(X, y, "chaos_orb")

# Save model artifacts
saved_files = save_model_artifacts(result, output_dir, "chaos_orb")
```

### Currency Selection (`utils/identify_target_currencies.py`)
```python
from utils.identify_target_currencies import generate_target_currency_list

# Get prioritized list of currency pairs for training
target_currencies = generate_target_currency_list()
print(f"Training {len(target_currencies)} currency pairs")
```

### Database Utilities (`utils/database.py`)
```python
from utils.database import get_db_connection

# Database connectivity
conn = get_db_connection()
df = pd.read_sql_query("SELECT * FROM currency_prices", conn)
conn.close()
```

## 📈 Performance Monitoring

### Experiment Tracking
Every experiment generates comprehensive reports:
- **Unique experiment IDs** for reproducibility
- **Configuration snapshots** for exact replication
- **Performance metrics** across all models
- **Resource usage** and timing information
- **Error analysis** and failure modes

### Model Evaluation
- **Cross-validation scores** for robustness assessment
- **Settlers league evaluation** for real-world validation
- **Feature importance** analysis
- **Directional accuracy** for trading relevance
- **MAPE, RMSE, MAE** for comprehensive evaluation

## 🔍 Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Use smaller batch sizes
python train_models.py --mode development

# Or reduce the number of trials
python train_models.py --config reduced_config.json
```

#### Database Connection
```python
# Check database connectivity
from utils.database import get_db_connection
conn = get_db_connection()
print("Database connected successfully!")
```

#### Configuration Errors
```python
# Validate configuration
config = MLConfig.from_file('my_config.json')
print(config.to_dict())  # Check all settings
```

### Performance Optimization

#### Parallel Processing
```bash
# Enable parallel feature engineering
python feature_engineering.py --parallel

# Adjust worker count in configuration
config.processing.max_workers = 4
```

#### Memory Management
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## 📚 Migration Guide

1. **Update imports** to use new utility modules
2. **Replace configuration** with centralized system
3. **Update logging** to use structured logging
4. **Migrate scripts** to use updated versions
5. **Test thoroughly** with development configuration

### Configuration Migration
```python
config = get_production_config()
config.model.min_samples_required = 150
config.model.n_trials = 200
```