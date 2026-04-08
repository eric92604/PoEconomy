#!/bin/bash

# Deploy PoEconomy Training Infrastructure
# This script deploys the training stack containing ECS Fargate containers and models bucket.

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"
source "$SCRIPT_DIR/lib/ecr.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"
echo "  Models: $DATA_LAKE_BUCKET_NAME/models/"

# Ensure prerequisites
ensure_prerequisites

# Build and push training container image
echo "Building and pushing container images..."
build_and_push_training_image

# Deploy training infrastructure (defaults come from poeconomy-training.yaml via merge_cloudformation_parameter_overrides)
echo "Deploying training infrastructure..."

OVERRIDES=(
  "EnvironmentName=$ENVIRONMENT"
  "BaseStackName=$BASE_STACK_NAME"
  "IngestionStackName=$INGESTION_STACK_NAME"
  "DataLakeBucketName=$DATA_LAKE_BUCKET_NAME"
  "TrainingImageUri=$TRAINING_IMAGE_URI"
)
[[ -n "${TASK_CPU:-}" ]] && OVERRIDES+=("TaskCpu=$TASK_CPU")
[[ -n "${TASK_MEMORY:-}" ]] && OVERRIDES+=("TaskMemory=$TASK_MEMORY")
[[ -n "${EPHEMERAL_STORAGE_GIB:-}" ]] && OVERRIDES+=("EphemeralStorageGiB=$EPHEMERAL_STORAGE_GIB")
[[ -n "${MIN_RECORDS_THRESHOLD:-}" ]] && OVERRIDES+=("MinRecordsThreshold=$MIN_RECORDS_THRESHOLD")
[[ -n "${MAX_CURRENCIES_TO_TRAIN:-}" ]] && OVERRIDES+=("MaxCurrenciesToTrain=$MAX_CURRENCIES_TO_TRAIN")
[[ -n "${MAX_CURRENCY_WORKERS:-}" ]] && OVERRIDES+=("MaxCurrencyWorkers=$MAX_CURRENCY_WORKERS")
[[ -n "${MAX_OPTUNA_WORKERS:-}" ]] && OVERRIDES+=("MaxOptunaWorkers=$MAX_OPTUNA_WORKERS")
[[ -n "${MODEL_N_JOBS:-}" ]] && OVERRIDES+=("ModelNJobs=$MODEL_N_JOBS")
[[ -n "${N_HYPERPARAMETER_TRIALS:-}" ]] && OVERRIDES+=("NHyperparameterTrials=$N_HYPERPARAMETER_TRIALS")
[[ -n "${N_MODEL_TRIALS:-}" ]] && OVERRIDES+=("NModelTrials=$N_MODEL_TRIALS")
[[ -n "${CV_FOLDS:-}" ]] && OVERRIDES+=("CVFolds=$CV_FOLDS")
[[ -n "${EARLY_STOPPING_ROUNDS:-}" ]] && OVERRIDES+=("EarlyStoppingRounds=$EARLY_STOPPING_ROUNDS")
[[ -n "${ENSEMBLE_WEIGHT_OPT_TRIALS:-}" ]] && OVERRIDES+=("EnsembleWeightOptTrials=$ENSEMBLE_WEIGHT_OPT_TRIALS")
[[ -n "${VALIDATION_MAX_DAYS:-}" ]] && OVERRIDES+=("ValidationMaxDays=$VALIDATION_MAX_DAYS")
[[ -n "${MIN_AVG_VALUE_THRESHOLD:-}" ]] && OVERRIDES+=("MinAvgValueThreshold=$MIN_AVG_VALUE_THRESHOLD")
[[ -n "${LOG_LEVEL:-}" ]] && OVERRIDES+=("LogLevel=$LOG_LEVEL")

deploy_cloudformation_stack "$TRAINING_STACK_NAME" "$TRAINING_TEMPLATE" "${OVERRIDES[@]}"

# Update .env file with current stack outputs
update_env_file

echo "========================================"
echo "Training infrastructure deployment complete!"
echo "========================================"
