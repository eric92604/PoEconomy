# PoEconomy - Path of Exile Currency Prediction Platform

A currency prediction system for Path of Exile using machine learning to analyze historical price data and predict future currency values. The system combines traditional ML models (LightGBM, XGBoost, RandomForest) with comprehensive feature engineering to provide accurate price forecasts.

## Architecture Overview

```
Cloudflare Worker (Edge Caching & Rate Limiting)
AWS Backend - API Gateway + Lambda + DynamoDB + S3 + Fargate
```

### Core Components
1. **AWS Lambda Functions**: Serverless compute for data ingestion, ML inference, and API serving
2. **AWS Fargate Spot**: Cost-optimized containerized ML training (70% cost savings)
3. **DynamoDB**: NoSQL database with TTL for live prices and metadata
4. **S3**: Data lake for historical data and trained ML models
5. **Cloudflare Workers**: Global edge caching and rate limiting

### Model Architecture
- **Ensemble Models**: LightGBM (primary) + XGBoost + RandomForest with Optuna optimization
- **Multi-Horizon Predictions**: Separate models for 1-day, 3-day, and 7-day forecasts
- **Currency-Specific Training**: Independent models per currency pair
- **Feature Engineering**: Advanced time-series features, rolling statistics, and league-aware indicators

## Project Structure

```
PoEconomy/
+-- ml/                     # Machine Learning Pipeline
|   +-- config/            # Training and inference configurations
|   +-- models/            # Trained model artifacts (empty - models stored in S3)
|   +-- pipelines/         # Feature engineering and model training pipelines
|   +-- scripts/           # Training and deployment scripts
|   +-- services/          # Data ingestion services (POE Watch API)
|   +-- utils/             # ML utilities (training, inference, data processing)
+-- aws/                   # AWS Infrastructure
|   +-- cloudformation/    # Modular CloudFormation templates
|   +-- lambda/            # Lambda function handlers
|   +-- scripts/           # Deployment automation scripts
|   +-- feature-engineering/ # Fargate container for feature engineering
|   +-- training/          # Fargate container for model training
+-- cloudflare/            # Edge infrastructure
|   +-- worker/            # Cloudflare Worker for API proxy and caching
+-- training_data/         # Historical CSV data from multiple leagues
+-- docs/                  # Documentation
+-- README.md
```

## Technical Stack

### AWS Backend
- **Runtime**: Python 3.11
- **Compute**: AWS Lambda (256MB-1024MB memory allocation) + Fargate Spot (8 vCPU, 16GB)
- **Database**: DynamoDB with on-demand billing
- **Storage**: S3 with intelligent tiering (data lake + models bucket)
- **API**: API Gateway with regional endpoints
- **Monitoring**: CloudWatch with custom metrics

### Cloudflare Edge Layer
- **Runtime**: V8 JavaScript engine
- **Storage**: KV namespace for caching and rate limiting
- **Features**: CORS handling, request routing, response optimization

### Machine Learning
- **Algorithms**: LightGBM (primary), XGBoost, RandomForest, Optuna hyperparameter optimization
- **Features**: Price lags, moving averages, volatility indicators, league day patterns, rolling statistics
- **Targets**: Multi-horizon predictions (1-day, 3-day, 7-day) with separate models per horizon
- **Training**: Fargate Spot containers with parallel processing and comprehensive experiment tracking
- **Deployment**: Lambda container images with bundled models for fast inference
- **Data Sources**: S3 data lake for training, DynamoDB for live inference

## Quick Start

### Prerequisites
- AWS CLI configured with appropriate permissions
- Docker for container image building
- Python 3.10+ for local ML development
- Git Bash (Windows) or bash (Linux/macOS)

### Environment Setup

The ML pipeline uses both S3 and DynamoDB data sources:
- **S3 Data Lake**: Historical currency data for model training
- **DynamoDB**: Live prices and metadata for inference

After running the infrastructure deployment script, source the generated env file:
```bash
source aws/.env
```

Key environment variables:
- `DATA_LAKE_BUCKET`: S3 bucket with historical currency data
- `MODELS_BUCKET`: S3 bucket for trained model artifacts
- `DYNAMO_CURRENCY_METADATA_TABLE`: per-currency statistics and availability
- `DYNAMO_CURRENCY_PRICES_TABLE`: live price history keyed by currency + league

### 1. Deploy Infrastructure

#### Deploy All Infrastructure
```bash
# Deploy all stacks in the correct order
./aws/scripts/deploy_all.sh production
```

#### Deploy Individual Stacks
```bash
# Deploy only the base infrastructure (data lake bucket)
./aws/scripts/deploy_base.sh production

# Deploy only the lambdas stack (Lambda functions, DynamoDB, API Gateway)
./aws/scripts/deploy_lambdas.sh production

# Deploy only the feature engineering stack (ECS Fargate)
./aws/scripts/deploy_feature_engineering.sh production

# Deploy only the training stack (ECS Fargate + models bucket)
./aws/scripts/deploy_training.sh production
```

### 2. Feature Engineering
```bash
# Standard feature engineering (from repository root)
python ml/scripts/run_feature_engineering.py --mode production

# With parallel processing (faster)
python ml/scripts/run_feature_engineering.py --mode production --parallel

# Custom experiment metadata
python ml/scripts/run_feature_engineering.py --mode development \
  --experiment-id my_experiment --description "Testing new features"
```

### 3. Model Training

#### Quick Pipeline Verification (30 seconds)
```bash
# End-to-end smoke test (feature engineering + training)
python ml/scripts/test_pipeline.py

# Limit to two currencies and reuse an existing dataset
python ml/scripts/test_pipeline.py --max-currencies 2 --skip-feature-engineering \
  --data-path ml/training_data/combined_currency_features_<experiment_id>.parquet
```

#### Production Training
```bash
# Full production training (parallelism auto-configured)
python ml/scripts/train_models.py --mode production

# Pin worker count explicitly
python ml/scripts/train_models.py --mode production --max-workers 4
```

#### Development Training
```bash
# Faster development training
python ml/scripts/train_models.py --mode development

# Focus on a couple of currencies during iteration
python ml/scripts/train_models.py --mode development --currencies "Divine Orb" "Chaos Orb"
```

### 4. Deploy Updated Models

**⚠️ Important**: When models are retrained, Lambda containers must be rebuilt to include the new models.

```bash
# Deploy updated Lambda containers
./aws/scripts/deploy_lambdas.sh production

# Verify deployment
curl https://your-api-url/health
```

## Machine Learning Pipeline

### Feature Engineering
- **Price-based features**: Price lags (1, 3, 5, 7 days), moving averages, volatility indicators
- **Time-based features**: League age, day of week, hour of day, league phase indicators
- **Rolling statistics**: Mean, std, min, max, median across multiple windows
- **Technical indicators**: Momentum, trend strength, volatility measures
- **League-aware features**: League-specific economic patterns and seasonal effects

### Model Training
- **Multi-horizon approach**: Separate models for 1-day, 3-day, and 7-day predictions
- **Ensemble methods**: LightGBM + XGBoost + RandomForest with dynamic weighting
- **Hyperparameter optimization**: Optuna with Bayesian optimization
- **Cross-validation**: Time series cross-validation with league-aware splits
- **Experiment tracking**: Comprehensive logging and model versioning

### Configuration System

#### Environment-Specific Configurations

**Production Configuration**
- **200 hyperparameter trials** for optimal performance
- **5-fold cross-validation** for robust evaluation
- **Full logging** with structured output
- **Model artifact saving** enabled

**Development Configuration**
- **50 hyperparameter trials** for faster iteration
- **3-fold cross-validation** for quicker feedback
- **Debug logging** for detailed troubleshooting
- **Reduced data requirements** for testing

**Test Configuration (Fast Verification)**
- **3 hyperparameter trials** for ultra-fast completion (~30 seconds)
- **2-fold cross-validation** for minimal validation
- **Single currency training** (1 out of 99 available)
- **2 parallel workers** to test parallelization

### Data Sources
- **Training**: S3 data lake with historical CSV files from multiple leagues
- **Inference**: DynamoDB with live price data from POE Watch API
- **Leagues**: Affliction, Ancestor, Necropolis, Phrecia, Settlers, and more

## API Endpoints

- **GET /health**: Health check endpoint
- **GET /predict/currencies**: List available currencies
- **GET /predict/leagues**: List available leagues
- **POST /predict/single**: Single currency prediction
- **POST /predict/batch**: Batch predictions for multiple currencies
- **POST /predict/ml**: On-demand ML prediction with live data
- **GET /prices/live**: Live price data with caching

## AWS Infrastructure

### DynamoDB Tables

#### CurrencyMetadataTable
```yaml
Partition Key: currency (String)
TTL: 30 days
Attributes: currency_id, category, group, frame, influences, icon_url, etc.
Stream: NEW_AND_OLD_IMAGES
```

#### LivePricesTable
```yaml
Partition Key: currency_league (String)  # "Divine Orb#Mercenaries"
Sort Key: timestamp (Number)
TTL: 14 days
GSI: currency-timestamp-index
Stream: NEW_AND_OLD_IMAGES
Attributes: currency, league, pay_currency, price, confidence
```

#### PredictionsTable
```yaml
Partition Key: currency_league (String)   # "Divine Orb#Mercenaries"
TTL: Variable based on confidence score
Attributes: prediction_data, created_at, confidence
```

### Lambda Functions

#### IngestionLambda
- **Trigger**: EventBridge (hourly by default)
- **Memory**: 256MB
- **Timeout**: 60 seconds
- **Purpose**: POE Watch API data ingestion to DynamoDB

#### PredictionRefreshLambda
- **Trigger**: EventBridge (scheduled) + DynamoDB Streams
- **Memory**: 1024MB
- **Timeout**: 300 seconds
- **Purpose**: Generate ML predictions and cache results

#### PredictionApiLambda
- **Trigger**: API Gateway
- **Memory**: 512MB
- **Timeout**: 30 seconds
- **Purpose**: Real-time predictions with cached models

#### LeagueMetadataHandler
- **Trigger**: EventBridge (daily)
- **Memory**: 256MB
- **Timeout**: 60 seconds
- **Purpose**: League metadata ingestion and management

### Fargate Infrastructure

#### FeatureEngineeringCluster
- **Capacity Providers**: Fargate Spot (100%)
- **Compute**: 4 vCPU, 8GB RAM
- **Purpose**: Cost-optimized feature engineering and data processing
- **Data Source**: Historical CSV data from S3
- **Output**: Processed parquet files uploaded to S3 data lake

#### TrainingCluster
- **Capacity Providers**: Fargate Spot (100%)
- **Compute**: 8 vCPU, 16GB RAM
- **Purpose**: Cost-optimized ML model training
- **Data Source**: S3 data lake with processed parquet data
- **Output**: Trained models uploaded to S3 models bucket

## Performance Metrics

### Response Times
- Cloudflare edge cache hit: <100ms
- Lambda warm start: <200ms
- Lambda cold start: <2000ms
- DynamoDB query: <10ms

### Caching Strategy
- Edge cache TTL: 300 seconds (5 minutes)
- Lambda memory cache: Persistent across invocations
- DynamoDB TTL: 14 days for hot data
- S3 lifecycle: Standard → IA (30d) → Glacier (180d) for data lake
- S3 lifecycle: Auto-delete models after 90 days
- Fargate Spot: 70% cost savings vs regular Fargate

## Utility Scripts

### POE Watch Ingestion
```bash
# Ingest live currency data for the default leagues configured in MLConfig
python ml/services/poe_watch_ingestion.py

# Custom leagues with shorter TTL and verbose logging
python ml/services/poe_watch_ingestion.py --leagues Settlers Mercenaries --ttl-days 7 --debug
```

### Prediction Refresh
```bash
# Refresh cached predictions for specific currencies/horizons
python ml/services/prediction_refresh.py \
  --currencies "Divine Orb" "Exalted Orb" \
  --horizons 1d 3d --ttl-hours 12

# Lambda-style invocation (auto-selects top currencies from metadata)
python ml/services/prediction_refresh.py
```

### On-Demand Training (Fargate Spot)
```bash
# Using the training script
./aws/scripts/run_training_task.sh production

# Or manually with AWS CLI
source aws/.env
aws ecs run-task \
  --cluster "$TRAINING_CLUSTER_NAME" \
  --task-definition "$TRAINING_TASK_DEFINITION" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$TRAINING_SUBNET_IDS],securityGroups=[$TRAINING_SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
  --region $AWS_REGION
```

## Performance Monitoring

### Experiment Tracking
Every experiment generates comprehensive reports:
- **Unique experiment IDs** for reproducibility
- **Configuration snapshots** for exact replication
- **Performance metrics** across all models
- **Resource usage** and timing information
- **Error analysis** and failure modes

### Model Evaluation
- **Cross-validation scores** for robustness assessment
- **Multi-league evaluation** for real-world validation
- **Feature importance** analysis
- **Directional accuracy** for trading relevance
- **MAPE, RMSE, MAE** for comprehensive evaluation
- **Multi-horizon predictions** (1d, 3d, 7d) for different trading strategies

## Infrastructure Architecture

The infrastructure is split into four separate CloudFormation stacks:

- **`poeconomy-base.yaml`**: Data lake S3 bucket (shared resource)
- **`poeconomy-lambdas.yaml`**: Lambda functions, DynamoDB tables, API Gateway
- **`poeconomy-feature-engineering.yaml`**: ECS Fargate containers for data processing
- **`poeconomy-training.yaml`**: ECS Fargate containers for model training and models bucket

## Disclaimer

This application is not affiliated with Grinding Gear Games. Path of Exile is a trademark of Grinding Gear Games Ltd. This tool is for educational and informational purposes only.