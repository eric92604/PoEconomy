# PoEconomy - Path of Exile Currency Prediction Platform

A currency prediction system for Path of Exile using machine learning to analyze historical price data and predict future currency values.

### Core Components
- **Frontend**: Next.js 15 static site on Cloudflare Pages
- **Edge**: Cloudflare Worker with KV caching, rate limiting (60 req/min), CORS
- **Backend**: AWS Lambda container images with ML models, API Gateway
- **Data**: DynamoDB (live prices, predictions), S3 (data lake, models)
- **ML**: ECS Fargate Spot for feature engineering and training (70% cost savings)

### Model Architecture
- **Ensemble Models**: LightGBM (primary) + XGBoost + RandomForest with Optuna optimization
- **Multi-Horizon Predictions**: Separate models for 1-day, 3-day, and 7-day forecasts
- **Currency-Specific Training**: Independent models per currency pair

## Project Structure

```
PoEconomy/
├── frontend/          # Next.js 15 app (pages, components, API client)
├── ml/                # ML pipelines (feature engineering, training)
├── aws/               # CloudFormation templates, Lambda functions, Fargate containers
├── cloudflare/worker/ # Edge proxy and caching
├── training_data/     # Historical CSV data
└── docs/              # Documentation
```

## Technical Stack

**Frontend**: Next.js 15, React 18, TypeScript 5, Tailwind CSS 4, shadcn/ui, TanStack Query, Recharts  
**Backend**: Python 3.11, AWS Lambda (containers), API Gateway, DynamoDB, S3  
**ML**: LightGBM 3.3.5, XGBoost 2.1.0, Optuna 3.3.0, pandas, scikit-learn  
**Infrastructure**: CloudFormation, ECS Fargate Spot, Cloudflare Workers/Pages

## Quick Start

### Prerequisites
- AWS CLI configured
- Docker for container images
- Python 3.11+ and Node.js 18+

### 1. Deploy Infrastructure

```bash
# Deploy all stacks in order
./aws/scripts/deploy_base.sh production
./aws/scripts/deploy_ingestion.sh production
./aws/scripts/deploy_prediction.sh production
./aws/scripts/deploy_api.sh production
./aws/scripts/deploy_feature_engineering.sh production
./aws/scripts/deploy_training.sh production

# Source generated environment variables
source aws/.env
```

### 2. Feature Engineering

```bash
python ml/scripts/run_feature_engineering.py --mode production --parallel
```

### 3. Model Training

```bash
python ml/scripts/train_models.py --mode production
```

### 4. Deploy Updated Models

**⚠️ Important**: When models are retrained, Lambda containers must be rebuilt.

```bash
./aws/scripts/deploy_prediction.sh production  # Updates prediction refresh Lambda
./aws/scripts/deploy_api.sh production         # Updates API Lambda
```

### 5. Deploy Frontend & Worker

```bash
# Frontend
cd frontend && npm install && npm run build
# Deploy via Cloudflare Pages dashboard or Git push

# Worker
cd cloudflare/worker && npm install && npx wrangler deploy
```

## Machine Learning Pipeline

### Feature Engineering
- Price lags, moving averages, volatility indicators
- Time-based features (league age, day of week, hour)
- Rolling statistics, technical indicators
- League-aware features

### Model Training
- Multi-horizon: Separate models for 1d, 3d, 7d predictions
- Ensemble: LightGBM + XGBoost + RandomForest
- Hyperparameter optimization with Optuna
- Time series cross-validation

### Configuration
- **Production**: 200 trials, 5-fold CV, full logging

### Model Storage
Models stored in S3: `s3://poeconomy-{env}-datalake/models/currency/{experiment_id}/`

## API Endpoints

All endpoints proxied through Cloudflare Worker with edge caching.

- **GET /health** - Health check
- **GET /predict/currencies** - List currencies (cached 30min)
- **GET /predict/leagues** - List leagues (cached 30min)
- **POST /predict/single** - Single currency prediction
- **POST /predict/batch** - Batch predictions (max 50)
- **GET /predict/currency** - Currency predictions (cached 10min)
- **GET /predict/latest** - Latest predictions with filtering (cached 10min)
- **GET /prices/live** - Live price data (cached 5min)
- **GET /prices/historical** - Historical price data

**Rate Limiting**: 60 requests/minute per IP

## AWS Infrastructure

### DynamoDB Tables
- **CurrencyMetadataTable**: Currency metadata (30-day TTL)
- **LivePricesTable**: Live prices with timestamps (14-day TTL)
- **PredictionsTable**: Cached predictions (variable TTL)
- **LeagueMetadataTable**: League information (30-day TTL)
- **DailyPricesTable**: Daily aggregated prices (90-day TTL)

### Lambda Functions
- **IngestionLambda**: Hourly POE Watch API ingestion (256MB, ZIP)
- **LeagueMetadataHandler**: Daily league metadata (256MB, ZIP)
- **DailyAggregationHandler**: Daily price aggregation (256MB, ZIP)
- **PredictionRefreshLambda**: Generate predictions (1024MB, container image)
- **PredictionApiLambda**: API Gateway handler (512MB, container image)

### Fargate
- **Feature Engineering**: 4 vCPU, 8GB RAM (Fargate Spot)
- **Training**: 16 vCPU, 32GB RAM (Fargate Spot)

### CloudFormation Stacks
1. `poeconomy-base.yaml` - S3 data lake
2. `poeconomy-ingestion.yaml` - DynamoDB tables, ingestion Lambdas
3. `poeconomy-prediction.yaml` - Predictions table, prediction Lambda
4. `poeconomy-api.yaml` - API Gateway, API Lambda
5. `poeconomy-feature-engineering.yaml` - Feature engineering Fargate
6. `poeconomy-training.yaml` - Training Fargate

## Cloudflare Worker

Edge proxy providing:
- **Edge Caching**: KV namespace (30min metadata, 10min predictions, 5min live data)
- **Rate Limiting**: 60 requests/minute per IP
- **CORS Handling**: Automatic headers
- **API Key Forwarding**: Secure forwarding to AWS

**Deployment**: `cd cloudflare/worker && npx wrangler deploy`

## Performance & Caching

### Response Times
- Cloudflare cache hit: <100ms
- Lambda warm start: <200ms
- Lambda cold start: <2000ms
- DynamoDB query: <10ms

### Caching Layers
1. **Cloudflare Worker KV**: Edge caching (30min/10min/5min TTLs)
2. **Lambda Memory**: Model and query caching across invocations
3. **DynamoDB TTL**: Automatic expiration (14-90 days)
4. **S3 Lifecycle**: Standard → IA (30d) → Glacier (180d)

## Environment Variables

### AWS (generated in `aws/.env`)
```bash
DATA_LAKE_BUCKET=poeconomy-production-datalake
DYNAMO_CURRENCY_METADATA_TABLE=poeconomy-production-currency-metadata
DYNAMO_CURRENCY_PRICES_TABLE=poeconomy-production-live-prices
DYNAMO_PREDICTIONS_TABLE=poeconomy-production-predictions
API_BASE_URL=https://x6h57bofoe.execute-api.us-west-2.amazonaws.com/api
```

### Frontend (Cloudflare Pages)
```env
NEXT_PUBLIC_API_URL=https://api.poeconomy.com
NEXT_PUBLIC_API_KEY=your_api_key_here
```

### Cloudflare Worker (`wrangler.toml`)
```toml
AWS_API_GATEWAY_URL=https://x6h57bofoe.execute-api.us-west-2.amazonaws.com/api
AWS_API_KEY=your_api_key_here
CACHE_TTL=1800
RATE_LIMIT_PER_MINUTE=60
```

## Data Flow

1. **Ingestion**: POE Watch API → Ingestion Lambda → DynamoDB
2. **Feature Engineering**: S3 CSV → Feature Engineering Pipeline → S3 Parquet
3. **Training**: S3 Parquet → Training Pipeline → S3 Models
4. **Predictions**: DynamoDB + S3 Models → Prediction Refresh Lambda → DynamoDB
5. **API**: Frontend → Cloudflare Worker → API Gateway → Lambda → DynamoDB

## Utility Scripts

```bash
# POE Watch ingestion
python ml/services/poe_watch_ingestion.py --leagues Settlers Mercenaries

# Prediction refresh
python ml/services/prediction_refresh.py --currencies "Divine Orb" --horizons 1d

# On-demand training
./aws/scripts/run_training_task.sh production
```

## Disclaimer

This application is not affiliated with Grinding Gear Games. Path of Exile is a trademark of Grinding Gear Games Ltd. This tool is for educational and informational purposes only.
