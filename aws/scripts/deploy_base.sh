#!/bin/bash

# Deploy PoEconomy Base Infrastructure
# This script deploys the base infrastructure stack containing the data lake bucket

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"
echo "  Models: $DATA_LAKE_BUCKET_NAME/models/"

# Ensure prerequisites
ensure_prerequisites

# Deploy base infrastructure
echo "Deploying base infrastructure..."
deploy_cloudformation_stack "$BASE_STACK_NAME" "$BASE_TEMPLATE" \
  "EnvironmentName=$ENVIRONMENT" \
  "DataLakeBucketName=$DATA_LAKE_BUCKET_NAME" \
  "DataLakeStorageClass=STANDARD"

# Update .env file with current stack outputs
update_env_file

echo "========================================"
echo "Base infrastructure deployment complete!"
echo "========================================"
echo "Data Lake Bucket: $DATA_LAKE_BUCKET_NAME"
