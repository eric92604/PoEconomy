#!/bin/bash

# Deploy PoEconomy API Infrastructure
# This script deploys the API stack containing API Lambda function and API Gateway

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"
echo "  Models: $DATA_LAKE_BUCKET_NAME/models/"

# Ensure prerequisites
ensure_prerequisites

# Package and upload Lambda function
echo "Packaging and uploading API Lambda function..."
if ! package_and_upload_lambda "api" "prediction_api_handler.py" "api.zip"; then
    echo "❌ Failed to package API Lambda function"
    exit 1
fi

# Verify Lambda package was uploaded
echo "Verifying Lambda package..."
aws s3 ls "s3://$DATA_LAKE_BUCKET_NAME/lambda/"

# Deploy API infrastructure (uses regular Lambda functions, not containers)
echo "Deploying API infrastructure..."
if ! deploy_cloudformation_stack "$API_STACK_NAME" "$API_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "ApiGatewayStageName=$API_STAGE_NAME" \
  "BaseStackName=$BASE_STACK_NAME" \
  "IngestionStackName=$INGESTION_STACK_NAME" \
  "PredictionStackName=$PREDICTION_STACK_NAME"; then
    echo "❌ Failed to deploy API infrastructure"
    exit 1
fi

echo "========================================"
echo "API infrastructure deployment complete!"
echo "========================================"

# Update .env file with current stack outputs
update_env_file

echo "✅ API infrastructure deployed successfully"
