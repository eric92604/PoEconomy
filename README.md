# PoEconomy - Path of Exile Market Analysis and Price Prediction

A sophisticated market analysis and price prediction system for Path of Exile items, leveraging historical data and machine learning to provide accurate market insights.

## Project Overview

This project consists of three main components:
1. **Data Collection System**: Automated collection of Path of Exile market data via the official trade API
2. **Machine Learning Pipeline**: Price prediction models trained on historical league data
3. **API Server**: RESTful API for accessing predictions and historical data

## Directory Structure

```
PoEconomy/
├── data/               # Raw and processed data storage
├── ml/                # Machine learning models and training scripts
├── pg_backup/         # PostgreSQL backup files
└── server/            # API server implementation
    ├── config/        # Server configuration
    ├── controllers/   # API route controllers
    ├── models/        # Database models
    ├── routes/        # API route definitions
    └── scripts/       # Utility scripts
```

## Documentation Index

1. [Data Analyst Guide](docs/data_analyst_guide.md)
   - Data structure overview
   - Database schema
   - Model training workflow
   
2. [Database Documentation](docs/database.md)
   - PostgreSQL setup
   - Schema details
   - Query examples

3. [ML Pipeline Documentation](docs/ml_pipeline.md)
   - Model architecture
   - Training process
   - Evaluation metrics
   - Prediction pipeline

4. [API Documentation](docs/api.md)
   - Endpoints
   - Data formats
   - Authentication

## Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Node.js 16+

## Quick Start for Data Analysts

1. Clone the repository
2. Set up Python environment:
   ```bash
   conda create -n poeconomy python=3.10
   conda activate poeconomy
   pip install -r requirements.txt
   ```
3. Configure database connection in `.env`
4. Follow the [Data Analyst Guide](docs/data_analyst_guide.md)

## License

MIT License - See LICENSE file for details

## Features

- **Real-time Market Data**: Track current currency and item prices across different leagues
- **Price Predictions**: AI-powered predictions for 1-day, 7-day, and 14-day price forecasts
- **Investment Analysis**: Identify the most profitable short and long-term investment opportunities
- **Historical Data**: Comprehensive historical price data from previous leagues
- **Interactive Charts**: Beautiful visualizations of price trends and market analytics
- **League Comparison**: Compare prices and trends across different Path of Exile leagues

## Tech Stack

### Backend
- **Node.js** with Express.js
- **PostgreSQL** database with Sequelize ORM
- **Path of Exile API** integration
- **Cron jobs** for automated data updates
- **Linear regression** for price predictions

### Frontend
- **React** with TypeScript
- **Material-UI** for modern, responsive design
- **Recharts** for data visualization
- **Axios** for API communication
- **React Router** for navigation

## API Endpoints

### Currency Endpoints
- `GET /api/currency` - Get all currencies
- `GET /api/currency/prices/:leagueId` - Get currency prices for a league
- `GET /api/currency/history/:currencyId/:payCurrencyId` - Get price history
- `GET /api/currency/market-overview/:leagueId` - Get market overview
- `GET /api/currency/trending/:leagueId` - Get trending currencies

### League Endpoints
- `GET /api/league` - Get all leagues
- `GET /api/league/active` - Get active league
- `GET /api/league/:id` - Get league by ID

### Prediction Endpoints
- `GET /api/prediction/:leagueId` - Get predictions for a league
- `GET /api/prediction/opportunities/:leagueId` - Get investment opportunities
- `GET /api/prediction/accuracy/:leagueId` - Get prediction accuracy stats

### Data Management
- `POST /api/data/refresh` - Refresh data from PoE API
- `POST /api/data/import-historical` - Import historical data
- `GET /api/data/status` - Get data refresh status

## Disclaimer

This application is not affiliated with Grinding Gear Games. Path of Exile is a trademark of Grinding Gear Games. This tool is for educational and informational purposes only.

---

**Happy Trading, Exile!** 🔥 