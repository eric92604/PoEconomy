#!/bin/bash

# Deploy PoEconomy Feature Engineering Infrastructure
# This script deploys the feature engineering stack containing ECS Fargate containers

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"
echo "  Models: $DATA_LAKE_BUCKET_NAME/models/"

# Ensure prerequisites
ensure_prerequisites

# Build and push feature engineering container image
echo "Building and pushing container images..."
build_and_push_feature_engineering_image

# Upload training data if not already present
echo "Checking training data..."
upload_training_data_if_needed

# Deploy feature engineering infrastructure
echo "Deploying feature engineering infrastructure..."
deploy_cloudformation_stack "$FEATURE_ENGINEERING_STACK_NAME" "$FEATURE_ENGINEERING_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "FeatureEngineeringImageUri=$FEATURE_ENGINEERING_IMAGE_URI" \
  "ScheduleExpression=$FEATURE_ENGINEERING_CRON" \
  "TaskTimeoutMinutes=$TASK_TIMEOUT_MINUTES" \
  "DataLakeBucketName=$DATA_LAKE_BUCKET_NAME" \
  "BaseStackName=$BASE_STACK_NAME"

# Update .env file with current stack outputs
update_env_file

echo "========================================"
echo "Feature engineering infrastructure deployment complete!"
echo "========================================"
