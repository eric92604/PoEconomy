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

# Package and upload Lambda functions
echo "Packaging and uploading Lambda functions..."
if ! package_and_upload_lambda "ingestion" "ingestion_handler.py" "ingestion.zip"; then
    echo "❌ Failed to package ingestion Lambda function"
    exit 1
fi

if ! package_and_upload_lambda "league_metadata" "league_metadata_handler.py" "league_metadata.zip"; then
    echo "❌ Failed to package league metadata Lambda function"
    exit 1
fi

if ! package_and_upload_lambda "daily_aggregation" "daily_aggregation_handler.py" "daily_aggregation.zip"; then
    echo "❌ Failed to package daily aggregation Lambda function"
    exit 1
fi

# Verify Lambda packages were uploaded
echo "Verifying Lambda packages..."
aws s3 ls "s3://$DATA_LAKE_BUCKET_NAME/lambda/"

  # Deploy ingestion infrastructure
  echo "Deploying ingestion infrastructure..."
  if ! deploy_cloudformation_stack "$INGESTION_STACK_NAME" "$INGESTION_TEMPLATE" \
    "EnvironmentName=$ENVIRONMENT" \
    "IngestionScheduleExpression=$INGESTION_CRON" \
    "LeagueMetadataScheduleExpression=$LEAGUE_METADATA_CRON" \
    "DailyAggregationScheduleExpression=$DAILY_AGGREGATION_CRON" \
    "BaseStackName=$BASE_STACK_NAME"; then
    echo "❌ Failed to deploy ingestion infrastructure"
    exit 1
  fi

echo "========================================"
echo "Ingestion infrastructure deployment complete!"
echo "========================================"

# Update .env file with current stack outputs
update_env_file

echo "✅ Ingestion infrastructure deployed successfully"
