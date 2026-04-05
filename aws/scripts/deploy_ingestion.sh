#!/bin/bash

# Deploy PoEconomy Ingestion Infrastructure
# This script deploys the ingestion stack containing ingestion Lambda functions and DynamoDB tables

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"

# Ensure prerequisites
ensure_prerequisites

# Note: Using container-based Lambda functions - no zip packaging needed

# Generate version number based on timestamp to force Lambda updates
LAMBDA_VERSION=$(date +"%Y%m%d%H%M%S")
echo "Using Lambda code version: $LAMBDA_VERSION"

# Build and deploy ingestion container
echo "Building and deploying ingestion container..."

# Configuration
CONTAINER_NAME="poeconomy-ingestion-lambda"
IMAGE_TAG="$LAMBDA_VERSION"
ECR_REPOSITORY_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$CONTAINER_NAME"

echo "Container URI: $ECR_REPOSITORY_URI"

# Ensure ECR repository exists
ensure_ecr_repository "$CONTAINER_NAME"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Build the container image
echo "Building Docker image (linux/amd64)..."
docker build \
  --platform linux/amd64 \
  --tag "$CONTAINER_NAME:$IMAGE_TAG" \
  --file "$ROOT_DIR/aws/lambdas/ingestion/container/Dockerfile" \
  --no-cache \
  "$ROOT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo "✅ Docker image built successfully"

# Tag and push to ECR
echo "Tagging image for ECR..."
docker tag "$CONTAINER_NAME:$IMAGE_TAG" "$ECR_REPOSITORY_URI:$IMAGE_TAG"

echo "Pushing image to ECR..."
docker push "$ECR_REPOSITORY_URI:$IMAGE_TAG"

if [ $? -ne 0 ]; then
    echo "❌ Failed to push container image"
    exit 1
fi

echo "✅ Ingestion container image pushed successfully"
INGESTION_CONTAINER_IMAGE_URI="$ECR_REPOSITORY_URI:$IMAGE_TAG"
echo "Using ingestion container image: $INGESTION_CONTAINER_IMAGE_URI"

# Deploy ingestion infrastructure
# All parameters for poeconomy-ingestion.yaml (defaults match the template; schedules from shared_config)
echo "Deploying ingestion infrastructure..."
# Enable schedules for production; keep disabled for dev/staging to prevent unintended ingestion
SCHEDULES_ENABLED="${SCHEDULES_ENABLED:-false}"
if [[ "$ENVIRONMENT" == "production" ]]; then
  SCHEDULES_ENABLED="true"
fi

if ! deploy_cloudformation_stack "$INGESTION_STACK_NAME" "$INGESTION_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "IngestionScheduleExpression=${INGESTION_SCHEDULE_EXPRESSION:-$INGESTION_CRON}" \
  "LeagueMetadataScheduleExpression=${LEAGUE_METADATA_SCHEDULE_EXPRESSION:-$LEAGUE_METADATA_CRON}" \
  "DailyAggregationScheduleExpression=${DAILY_AGGREGATION_SCHEDULE_EXPRESSION:-$DAILY_AGGREGATION_CRON}" \
  "SchedulesEnabled=$SCHEDULES_ENABLED" \
  "BaseStackName=$BASE_STACK_NAME" \
  "IngestionContainerImageUri=$INGESTION_CONTAINER_IMAGE_URI"; then
  echo "❌ Failed to deploy ingestion infrastructure"
  exit 1
fi

echo "========================================"
echo "Ingestion infrastructure deployment complete!"
echo "========================================"

# Update .env file with current stack outputs
update_env_file

echo "✅ Ingestion infrastructure deployed successfully"
