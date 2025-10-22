# PoEconomy Frontend

Modern web application for Path of Exile currency price predictions and investment analysis.

## Tech Stack

- **Framework**: Next.js 15+ with App Router
- **Language**: TypeScript 5+
- **Styling**: Tailwind CSS 4+
- **UI Components**: shadcn/ui + Radix UI
- **Data Fetching**: TanStack Query (React Query)
- **State Management**: Zustand
- **Charts**: Recharts
- **Icons**: Lucide React
- **Validation**: Zod

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
├── components/            # React components
│   ├── ui/               # shadcn/ui components
│   ├── layout/           # Layout components
│   ├── charts/           # Chart components
│   ├── currency/         # Currency components
│   ├── filters/          # Filter components
│   ├── risk/             # Risk assessment components
│   └── features/         # Feature components
├── lib/                   # Library code
│   ├── api/              # API client
│   ├── hooks/            # Custom React hooks
│   ├── providers/        # Context providers
│   ├── utils/            # Utility functions
│   └── constants/        # Constants
├── types/                 # TypeScript types
└── public/               # Static assets
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env

# Edit .env with your API URL
```

### Development

```bash
# Run development server
npm run dev

# Open http://localhost:3000
```

### Building

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Features

### Core Features
- **Main Investment Table**: View all currencies with predictions, profit percentages, and confidence scores
- **Dynamic Price Chart**: Interactive charts showing historical prices and predictions

### Enhanced Features
- **Smart Filters & Sorting**: Filter by confidence, profit, league, and category
- **Advanced Charting**: Multi-currency comparison, prediction bands, interactive tooltips
- **Risk Assessment**: Calculate risk scores and investment recommendations
- **Market Intelligence**: Heat maps, trending currencies, volatility indicators
- **Watchlist**: Save favorite currencies and set price alerts

## Development Guidelines

### Code Style
- Use functional components with TypeScript
- Follow React hooks best practices
- Implement proper error boundaries
- Use descriptive variable names with auxiliary verbs

### Performance
- Minimize use of `useEffect` and `useState`
- Favor React Server Components where possible
- Implement code splitting and lazy loading
- Optimize images (WebP format, lazy loading)

### Testing
```bash
# Run type checking
npm run type-check

# Run linting
npm run lint

# Format code
npm run format
```

## Environment Variables

```env
# API Configuration
NEXT_PUBLIC_API_URL=https://api.poeconomy.com
NEXT_PUBLIC_API_KEY=your_api_key_here

# Application
NEXT_PUBLIC_APP_NAME=PoEconomy
NEXT_PUBLIC_APP_VERSION=0.1.0
NEXT_PUBLIC_ENVIRONMENT=production

# Features
NEXT_PUBLIC_ENABLE_DEVTOOLS=true
```

## API Integration

The frontend connects to the PoEconomy backend API:

- **Health**: `GET /health`
- **Currencies**: `GET /predict/currencies`
- **Leagues**: `GET /predict/leagues`
- **Single Prediction**: `POST /predict/single`
- **Batch Predictions**: `POST /predict/batch`
- **Live Prices**: `GET /prices/live`

See [API_REFERENCE.md](../docs/API_REFERENCE.md) for complete API documentation.

## Deployment

### Cloudflare Pages
```bash
# Build for static export
npm run build

# Deploy to Cloudflare Pages
# Follow instructions in deployment documentation
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## License

See LICENSE file in root directory.
