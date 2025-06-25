# PoEconomy - Path of Exile Currency Prediction Platform

A cost-optimized, serverless currency prediction system for Path of Exile, utilizing AWS Lambda, DynamoDB, and Cloudflare Workers for global edge caching and ML-based price forecasting.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ React Frontend  │    │ Cloudflare       │    │ AWS Backend     │
│ (Next.js 14)    │───▶│ Worker           │───▶│ API Gateway +   │
│ Vercel/Pages    │    │ (Edge Caching)   │    │ Lambda + DynamoDB│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components
1. **AWS Lambda Functions**: Serverless compute for data ingestion, ML computation, and API serving
2. **DynamoDB**: NoSQL database with TTL for hot data (14-day retention)
3. **Cloudflare Workers**: Global edge caching and rate limiting
4. **S3**: Data lake for historical data and ML model storage
5. **Lambda Layers**: Zero-cost ML model access optimization

## Project Structure

```
PoEconomy/
├── aws/
│   ├── cloudformation/         # Infrastructure as Code templates
│   ├── lambda/                # Lambda function handlers
│   └── scripts/               # Deployment and layer creation scripts
├── cloudflare/
│   └── worker/                # Edge caching and rate limiting
├── ml/                        # Machine learning pipeline
│   ├── models/               # Trained ML models
│   ├── pipelines/            # Feature engineering and training
│   ├── scripts/              # Training and inference scripts
│   └── utils/                # ML utilities and data processing
├── docs/                     # Technical documentation
└── server/                   # Legacy Node.js server (deprecated)
```

## Technical Stack

### AWS Backend
- **Runtime**: Python 3.11
- **Compute**: AWS Lambda (256MB-1024MB memory allocation)
- **Database**: DynamoDB with on-demand billing
- **Storage**: S3 with intelligent tiering
- **API**: API Gateway with regional endpoints
- **Monitoring**: CloudWatch with custom metrics

### Cloudflare Edge Layer
- **Runtime**: V8 JavaScript engine
- **Storage**: KV namespace for caching and rate limiting
- **Features**: CORS handling, request routing, response optimization
- **Performance**: Sub-100ms global response times

### Machine Learning
- **Algorithms**: LightGBM, XGBoost with MultiOutputRegressor
- **Features**: Price lags, moving averages, volatility indicators
- **Targets**: Multi-horizon predictions (1-day, 3-day, 7-day)
- **Deployment**: Lambda Layers for zero S3 transfer costs

### Frontend (Optional)
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript with strict mode
- **Styling**: Tailwind CSS with Shadcn/ui components
- **State**: Zustand with TanStack Query for server state
- **Testing**: Vitest with Playwright for E2E

## AWS Infrastructure

### DynamoDB Tables

#### CurrencyMetadataTable
```yaml
Partition Key: currency_name (String)
Sort Key: league (String)
TTL: 30 days
Attributes: last_updated, metadata_json
```

#### LivePricesTable
```yaml
Partition Key: currency_league (String)  # "Divine Orb#Sanctum"
Sort Key: timestamp (String)
TTL: 14 days
GSI: league-price-change-index
Stream: NEW_AND_OLD_IMAGES
```

#### PredictionsCacheTable
```yaml
Partition Key: prediction_key (String)   # "Divine Orb#Sanctum#1"
GSI: currency-league-index
TTL: Variable based on confidence score
Attributes: prediction_data, created_at, confidence
```

### Lambda Functions

#### DataIngestionLambda
- **Trigger**: EventBridge (15-minute intervals)
- **Memory**: 256MB
- **Timeout**: 60 seconds
- **Purpose**: POE Watch API data ingestion

#### MLComputationLambda
- **Trigger**: DynamoDB Stream (LivePricesTable)
- **Memory**: 1024MB
- **Timeout**: 300 seconds
- **Purpose**: Generate ML predictions on price changes

#### MLPredictionLambda
- **Trigger**: API Gateway
- **Memory**: 512MB
- **Timeout**: 30 seconds
- **Purpose**: Real-time predictions with layer-cached models

#### ApiServingLambda
- **Trigger**: API Gateway
- **Memory**: 256MB
- **Timeout**: 10 seconds
- **Purpose**: Serve cached predictions and metadata

## API Endpoints

### Prediction Endpoints
- `POST /predict/single` - Single currency prediction
- `POST /predict/batch` - Batch currency predictions
- `POST /predict/ml` - Real-time ML prediction with layers
- `GET /predict/currencies` - Available currencies list
- `GET /predict/leagues` - Available leagues list

### Request Format
```json
{
  "currency": "Divine Orb",
  "league": "Sanctum",
  "prediction_horizon_days": 1
}
```

### Response Format
```json
{
  "currency": "Divine Orb",
  "league": "Sanctum",
  "current_price": 180.5,
  "predicted_price": 185.2,
  "prediction_horizon_days": 1,
  "confidence_score": 0.87,
  "price_change_percent": 2.6,
  "model_type": "ensemble",
  "created_at": "2024-01-15T10:30:00Z"
}
```

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
- S3 lifecycle: Intelligent tiering for cost optimization

## Deployment

### Prerequisites
- AWS CLI configured with appropriate permissions
- Node.js 18+ for Cloudflare Worker deployment
- Python 3.10+ for local ML development

### Quick Deployment
```bash
# Deploy AWS infrastructure
aws cloudformation create-stack \
  --stack-name poeconomy-production \
  --template-body file://aws/cloudformation/poeconomy-infrastructure.yaml \
  --capabilities CAPABILITY_NAMED_IAM

# Create Lambda Layer for cost optimization
./aws/scripts/create_model_layer.sh

# Deploy Cloudflare Worker
cd cloudflare/worker
wrangler deploy --env production
```

### Environment Variables
```bash
# Lambda Functions
METADATA_TABLE=poeconomy-production-metadata
LIVE_PRICES_TABLE=poeconomy-production-live-prices
PREDICTIONS_CACHE_TABLE=poeconomy-production-predictions-cache
DATA_LAKE_BUCKET=poeconomy-production-data-lake
MODELS_BUCKET=poeconomy-production-models

# Cloudflare Worker
AWS_API_GATEWAY_URL=https://api-id.execute-api.us-east-1.amazonaws.com/prod
RATE_LIMIT_PER_MINUTE=60
CACHE_TTL=300
```

## Machine Learning Pipeline

### Feature Engineering
- Price lag features (1, 7, 30 days)
- Moving averages (7, 30 days)
- Volatility indicators (7, 30 days)
- Volume-based features
- Temporal features (league age, season effects)

### Model Architecture
```python
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# Multi-output regression for simultaneous horizon predictions
model = MultiOutputRegressor(LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
))

# Training targets: [1-day, 3-day, 7-day] price changes
targets = prepare_multi_output_targets(data, horizons=[1, 3, 7])
model.fit(features, targets)
```

### Model Performance
- Training data: Historical league data from PostgreSQL
- Validation: Time-series cross-validation
- Metrics: MAE, RMSE, directional accuracy
- Ensemble: LightGBM + XGBoost with weighted averaging

## Security

### AWS Security
- IAM roles with least privilege access
- DynamoDB encryption at rest
- S3 bucket policies with restricted access
- VPC endpoints for private communication

### API Security
- Rate limiting: 60 requests/minute per IP
- Input validation with JSON schema
- CORS configuration for allowed origins
- Request size limits and timeout handling

### Data Privacy
- TTL-based automatic data cleanup
- No PII collection or storage
- Audit logging with CloudTrail
- Compliance with data retention policies

## Monitoring

### CloudWatch Metrics
- Lambda function duration and errors
- DynamoDB read/write capacity utilization
- API Gateway request count and latency
- Custom metrics for prediction accuracy

### Alerting
- Error rate >1% triggers notification
- Response time >5 seconds triggers alert
- Cost budget alerts at 80% threshold
- DynamoDB throttling notifications

## Development

### Local Development
```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r ml/requirements.txt

# Configure database connection
cp server/.env.example server/.env
# Edit .env with PostgreSQL credentials

# Run ML training pipeline
python ml/scripts/train_models.py --test-mode
```

### Testing
```bash
# Test Lambda functions locally
python -m pytest aws/lambda/tests/

# Test ML pipeline
python -m pytest ml/tests/

# Test Cloudflare Worker
cd cloudflare/worker
npm test
```

## License

MIT License - See LICENSE file for details.

## Technical Notes

- All timestamps stored in UTC ISO 8601 format
- Currency names standardized using PostgreSQL lookup table
- Price values stored as strings to avoid floating-point precision issues
- Multi-region deployment supported via CloudFormation parameters
- Horizontal scaling achieved through Lambda concurrency and DynamoDB auto-scaling

## Disclaimer

This application is not affiliated with Grinding Gear Games. Path of Exile is a trademark of Grinding Gear Games Ltd. This tool is for educational and informational purposes only.

## 🎯 Recent Enhancements

### Enhanced ML Training Pipeline

The training pipeline has been significantly improved with advanced techniques for better model performance and prediction accuracy:

#### 🎯 Advanced Cross-Validation
- **Purged Time Series Splits**: Prevents data leakage in financial time series with configurable purge and embargo periods
- **Multi-Objective Evaluation**: Balances RMSE with directional accuracy for better financial predictions
- **Stability Tracking**: Monitors prediction consistency across validation folds

#### 🧠 Enhanced Early Stopping
- **Dynamic Patience**: Adaptive early stopping with configurable patience and minimum improvement thresholds
- **Best Weight Restoration**: Automatically restores optimal model weights after training
- **Validation Monitoring**: Advanced monitoring for gradient boosting models with silent evaluation

#### 🎪 Smart Ensemble Learning
- **Dynamic Weight Learning**: Automatically learns optimal ensemble weights based on validation performance
- **Multi-Objective Weight Training**: Considers both RMSE and directional accuracy when learning weights
- **Scipy-based Optimization**: Uses constrained optimization for robust weight determination

#### 📊 Multi-Objective Hyperparameter Tuning
- **Financial Metrics Integration**: Incorporates directional accuracy into hyperparameter optimization
- **Stability Penalties**: Penalizes models with inconsistent predictions across folds
- **Combined Scoring**: Balances traditional ML metrics with financial prediction quality

#### ⚡ C4d VM Optimizations
Enhanced configurations specifically tuned for Google Cloud C4d instances:
- **AMD EPYC Genoa Optimizations**: Environment variables and threading optimized for AMD architecture
- **Memory-Conscious Processing**: Batch sizes and parallel processing tuned for 31GB memory limit
- **Advanced Feature Integration**: All new features enabled with C4d-specific parameter tuning

### Key Benefits
- **Improved Prediction Accuracy**: Multi-objective tuning leads to better financial predictions
- **Reduced Overfitting**: Advanced cross-validation prevents data leakage and improves generalization
- **Enhanced Stability**: Dynamic early stopping and ensemble learning create more robust models
- **Better ROI Predictions**: Directional accuracy focus improves investment recommendation quality

## 🏗️ Architecture

```
PoEconomy/
├── ml/                     # Machine Learning Pipeline
│   ├── config/            # Enhanced training configurations
│   ├── models/            # Trained model artifacts
│   ├── pipelines/         # Training and inference pipelines
│   ├── scripts/           # Training and deployment scripts
│   ├── services/          # Data ingestion services
│   ├── training_data/     # Feature-engineered datasets
│   └── utils/             # Enhanced ML utilities with advanced algorithms
├── server/                # Backend API (Future)
├── docs/                  # Documentation
└── README.md
```

## 🛠️ Technology Stack

- **Machine Learning**: LightGBM, XGBoost, scikit-learn, Optuna
- **Data Processing**: pandas, NumPy, PostgreSQL
- **Cloud Platform**: Google Cloud (C4d instances for training)
- **Optimization**: Advanced cross-validation, multi-objective tuning
- **Monitoring**: Comprehensive logging and experiment tracking

## 📈 Model Performance

The enhanced pipeline delivers superior performance through:
- **Multi-Objective Optimization**: Balances prediction accuracy with directional correctness
- **Advanced Validation**: Purged time series splits prevent overfitting
- **Dynamic Ensembles**: Learned weights improve ensemble performance
- **Financial Focus**: Metrics specifically designed for trading applications

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PostgreSQL
- Google Cloud SDK (for cloud training)

### Local Development
```bash
# Clone repository
git clone https://github.com/eric92604/PoEconomy.git
cd PoEconomy

# Install dependencies
pip install -r ml/requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your database credentials

# Run training
cd ml
python scripts/train_models.py
```

### Cloud Training (Google Cloud C4d)
```bash
# Enhanced C4d training with advanced features
python scripts/train_models_c4d.py --experiment-id advanced_training
```

## 📊 Training Configuration

The enhanced pipeline supports multiple configuration modes:

- **Development**: Fast training for testing
- **Production**: Full feature training with all enhancements
- **C4d-Optimized**: Specifically tuned for Google Cloud C4d instances
- **High-Value**: Focus on high-value currency pairs
- **All-Currencies**: Comprehensive training across all available currencies

## 🎯 Investment Strategies

PoEconomy supports multiple investment approaches:
- **Short-term Trading**: 1-3 day predictions for quick profits
- **Medium-term Holds**: 7-14 day predictions for stable returns
- **Long-term Investments**: 30+ day predictions for major market shifts
- **Risk-Adjusted Returns**: Balanced approach considering volatility

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the development team.

---

*PoEconomy - Making Path of Exile trading profitable through advanced machine learning.*
