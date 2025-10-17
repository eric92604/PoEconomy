#!/bin/bash

# Deploy PoEconomy Training Infrastructure
# This script deploys the training stack containing ECS Fargate containers and models bucket

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"
echo "  Models: $DATA_LAKE_BUCKET_NAME/models/"

# Ensure prerequisites
ensure_prerequisites

# Build and push training container image
echo "Building and pushing container images..."
build_and_push_training_image

# Deploy training infrastructure
echo "Deploying training infrastructure..."
deploy_cloudformation_stack "$TRAINING_STACK_NAME" "$TRAINING_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "TrainingImageUri=$TRAINING_IMAGE_URI" \
  "TaskTimeoutMinutes=$TASK_TIMEOUT_MINUTES" \
  "MaxCurrenciesToTrain=$MAX_CURRENCIES_TO_TRAIN" \
  "BaseStackName=$BASE_STACK_NAME" \
  "DataLakeBucketName=$DATA_LAKE_BUCKET_NAME"

# Update .env file with current stack outputs
update_env_file

echo "========================================"
echo "Training infrastructure deployment complete!"
echo "========================================"
