#!/bin/bash

# Deploy PoEconomy Lambdas Infrastructure
# This script deploys the lambdas stack containing Lambda functions, DynamoDB tables, and API Gateway

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"

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

# Deploy lambdas infrastructure
echo "Deploying lambdas infrastructure..."
deploy_cloudformation_stack "$LAMBDAS_STACK_NAME" "$LAMBDAS_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "LambdaImageUri=$INFERENCE_IMAGE_URI" \
  "IngestionScheduleExpression=$INGESTION_CRON" \
  "LeagueMetadataScheduleExpression=$LEAGUE_METADATA_CRON" \
  "ApiGatewayStageName=$API_STAGE_NAME" \
  "BaseStackName=$BASE_STACK_NAME" \
  "ModelsAvailable=${MODELS_AVAILABLE:-false}"

echo "========================================"
echo "Lambdas infrastructure deployment complete!"
echo "========================================"

# Update .env file with current stack outputs
update_env_file

echo "✅ Prediction API deployed successfully"
