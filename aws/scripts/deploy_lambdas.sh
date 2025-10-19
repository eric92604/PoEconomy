#!/bin/bash

# Deploy PoEconomy All Lambda Infrastructure
# This script deploys all lambda stacks: ingestion, prediction, and API

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"

print_deployment_info

echo "========================================"
echo "Deploying all Lambda infrastructure stacks"
echo "========================================"

# Deploy ingestion stack first (contains DynamoDB tables)
echo "Step 1: Deploying ingestion infrastructure..."
if ! bash "$SCRIPT_DIR/deploy_ingestion.sh"; then
    echo "❌ Ingestion deployment failed"
    exit 1
fi

# Deploy prediction stack second (depends on ingestion tables)
echo "Step 2: Deploying prediction infrastructure..."
if ! bash "$SCRIPT_DIR/deploy_prediction.sh"; then
    echo "❌ Prediction deployment failed"
    exit 1
fi

# Deploy API stack last (depends on both ingestion and prediction)
echo "Step 3: Deploying API infrastructure..."
if ! bash "$SCRIPT_DIR/deploy_api.sh"; then
    echo "❌ API deployment failed"
    exit 1
fi

echo "========================================"
echo "All Lambda infrastructure deployment complete!"
echo "========================================"

echo "✅ All Lambda stacks deployed successfully"
