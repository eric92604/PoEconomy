#!/bin/bash

# Deploy PoEconomy Prediction Infrastructure
# This script deploys the prediction stack containing prediction refresh Lambda function and DynamoDB table

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"
source "$SCRIPT_DIR/lib/models.sh"
source "$SCRIPT_DIR/lib/ecr.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"
echo "  Models: $DATA_LAKE_BUCKET_NAME/models/"

# Ensure prerequisites
ensure_prerequisites

# Verify models in S3 for Lambda container
verify_models_in_s3

# Check for local backup and set MODELS_AVAILABLE
echo "Checking for local model backup..."
if verify_local_backup; then
  echo "✅ Local models available"
  export MODELS_AVAILABLE="true"
else
  echo "ℹ️  No local models found, will download from S3"
  export MODELS_AVAILABLE="false"
fi

# Build and push Lambda container image
echo "Building and pushing container images..."
build_and_push_lambda_image

# Deploy prediction infrastructure
# All parameters for poeconomy-prediction.yaml
echo "Deploying prediction infrastructure..."
if ! deploy_cloudformation_stack "$PREDICTION_STACK_NAME" "$PREDICTION_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "LambdaImageUri=$INFERENCE_IMAGE_URI" \
  "BaseStackName=$BASE_STACK_NAME" \
  "IngestionStackName=$INGESTION_STACK_NAME"; then
    echo "❌ Failed to deploy prediction infrastructure"
    exit 1
fi

echo "========================================"
echo "Prediction infrastructure deployment complete!"
echo "========================================"

# Update .env file with current stack outputs
update_env_file

echo "✅ Prediction infrastructure deployed successfully"
