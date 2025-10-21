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

# Generate version timestamp for Lambda deployment
# This ensures CloudFormation always points to the newest zip file
LAMBDA_VERSION=$(date +"%Y%m%d%H%M%S")
echo "Lambda version: $LAMBDA_VERSION"

# Package and upload API Lambda function
echo "Packaging and uploading API Lambda function..."
if ! package_and_upload_lambda "api" "prediction_api_handler.py" "lambda.zip"; then
    echo "❌ Failed to package and upload API Lambda function"
    exit 1
fi

# Rename and upload with versioned filename
echo "Uploading versioned Lambda package..."
if ! aws s3 cp "s3://$DATA_LAKE_BUCKET_NAME/lambda/lambda.zip" "s3://$DATA_LAKE_BUCKET_NAME/lambda/${ENVIRONMENT}-api-${LAMBDA_VERSION}.zip"; then
    echo "❌ Failed to upload versioned Lambda package"
    exit 1
fi

# Deploy API infrastructure with zip package
echo "Deploying API infrastructure with zip package..."
if ! deploy_cloudformation_stack "$API_STACK_NAME" "$API_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "ApiGatewayStageName=$API_STAGE_NAME" \
  "BaseStackName=$BASE_STACK_NAME" \
  "IngestionStackName=$INGESTION_STACK_NAME" \
  "PredictionStackName=$PREDICTION_STACK_NAME" \
  "LambdaS3Bucket=$DATA_LAKE_BUCKET_NAME" \
  "LambdaVersion=$LAMBDA_VERSION"; then
    echo "❌ Failed to deploy API infrastructure"
    exit 1
fi

echo "========================================"
echo "API infrastructure deployment complete!"
echo "========================================"

# Update .env file with current stack outputs
update_env_file

echo "✅ API infrastructure deployed successfully"
