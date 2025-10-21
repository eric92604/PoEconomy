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
ECR_REPOSITORY_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$CONTAINER_NAME"

echo "Container URI: $ECR_REPOSITORY_URI"

# Build the container image with explicit platform for Lambda compatibility
echo "Building Docker image with explicit platform..."
DOCKER_BUILDKIT=0 docker build \
  --platform linux/amd64 \
  --tag "$CONTAINER_NAME:$IMAGE_TAG" \
  --file "$SCRIPT_DIR/../lambdas/ingestion/container/Dockerfile" \
  --no-cache \
  "$ROOT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo "✅ Docker image built successfully"

# Verify the image architecture
echo "Verifying image architecture..."
docker inspect "$CONTAINER_NAME:$IMAGE_TAG" --format='{{.Architecture}}'

# Tag the image for ECR
echo "Tagging image for ECR..."
docker tag "$CONTAINER_NAME:$IMAGE_TAG" "$ECR_REPOSITORY_URI:$IMAGE_TAG"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REPOSITORY_URI"

# Create ECR repository if it doesn't exist
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "$CONTAINER_NAME" --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "$CONTAINER_NAME" --region "$AWS_REGION"

# Push the image to ECR
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
echo "Deploying ingestion infrastructure..."
if ! deploy_cloudformation_stack "$INGESTION_STACK_NAME" "$INGESTION_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "IngestionScheduleExpression=$INGESTION_CRON" \
  "LeagueMetadataScheduleExpression=$LEAGUE_METADATA_CRON" \
  "DailyAggregationScheduleExpression=$DAILY_AGGREGATION_CRON" \
  "BaseStackName=$BASE_STACK_NAME" \
  "LambdaCodeVersion=$LAMBDA_VERSION" \
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
