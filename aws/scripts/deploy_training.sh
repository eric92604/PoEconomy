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

# Build parameter list - only include parameters if explicitly set
# Required parameters (no defaults in CloudFormation) are always included
PARAMS=(
  "EnvironmentName=$ENVIRONMENT"
  "TrainingImageUri=$TRAINING_IMAGE_URI"
  "BaseStackName=$BASE_STACK_NAME"
  "DataLakeBucketName=$DATA_LAKE_BUCKET_NAME"
)

# Only add optional parameters if environment variables are explicitly set
# This allows CloudFormation template defaults to be used when not specified
[[ -n "${TASK_TIMEOUT_MINUTES:-}" ]] && PARAMS+=("TaskTimeoutMinutes=$TASK_TIMEOUT_MINUTES")
[[ -n "${MAX_CURRENCIES_TO_TRAIN:-}" ]] && PARAMS+=("MaxCurrenciesToTrain=$MAX_CURRENCIES_TO_TRAIN")
[[ -n "${N_HYPERPARAMETER_TRIALS:-}" ]] && PARAMS+=("NHyperparameterTrials=$N_HYPERPARAMETER_TRIALS")
[[ -n "${N_MODEL_TRIALS:-}" ]] && PARAMS+=("NModelTrials=$N_MODEL_TRIALS")
[[ -n "${CV_FOLDS:-}" ]] && PARAMS+=("CVFolds=$CV_FOLDS")
[[ -n "${MAX_DEPTH:-}" ]] && PARAMS+=("MaxDepth=$MAX_DEPTH")
[[ -n "${LEARNING_RATE:-}" ]] && PARAMS+=("LearningRate=$LEARNING_RATE")
[[ -n "${MAX_OPTUNA_WORKERS:-}" ]] && PARAMS+=("MaxOptunaWorkers=$MAX_OPTUNA_WORKERS")
[[ -n "${MODEL_N_JOBS:-}" ]] && PARAMS+=("ModelNJobs=$MODEL_N_JOBS")
[[ -n "${MIN_AVG_VALUE_THRESHOLD:-}" ]] && PARAMS+=("MinAvgValueThreshold=$MIN_AVG_VALUE_THRESHOLD")

deploy_cloudformation_stack "$TRAINING_STACK_NAME" "$TRAINING_TEMPLATE" "${PARAMS[@]}"

# Update .env file with current stack outputs
update_env_file

echo "========================================"
echo "Training infrastructure deployment complete!"
echo "========================================"
