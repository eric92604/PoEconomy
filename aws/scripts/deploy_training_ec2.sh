#!/bin/bash
#
# Deploy PoEconomy ECS EC2 training capacity.
#
# Prerequisites: the Fargate training stack must exist first (same VPC/subnets/SG exports).
#
# After deployment, start training with the same ECR image as Fargate; only cluster,
# task definition, and launch type differ.
#
# --- Run training on EC2  ---
# Subnets and security group must be the training VPC pair from aws/.env (source deploy_training.sh
# or deploy_training_ec2.sh so update_env_file refreshes them; do not mix with other VPC subnets).
# Wait until ecs list-container-instances shows ACTIVE capacity if ASG scaled from zero.
#   aws ecs run-task --launch-type EC2 \
#     --cluster "$TRAINING_EC2_CLUSTER_NAME" \
#     --task-definition "$TRAINING_EC2_TASK_DEFINITION" \
#     --network-configuration "awsvpcConfiguration={subnets=[${TRAINING_SUBNET_IDS}],securityGroups=[${TRAINING_SECURITY_GROUP_ID}],assignPublicIp=ENABLED}" \
#     --region "$AWS_DEFAULT_REGION"
#
# CloudWatch: EC2 logs under TRAINING_EC2_LOG_GROUP.

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared_config.sh"
source "$SCRIPT_DIR/lib/ecr.sh"

print_deployment_info

echo "Using bucket names:"
echo "  Data Lake: $DATA_LAKE_BUCKET_NAME"
echo "  Models: $DATA_LAKE_BUCKET_NAME/models/"
echo "Fargate training stack (must exist): $TRAINING_STACK_NAME"
echo "EC2 training stack: $TRAINING_EC2_STACK_NAME"
echo "  Default instance (CloudFormation): c6a.8xlarge — 32 vCPU, 64 GiB (override with TRAINING_EC2_INSTANCE_TYPE)"
echo "  Spot: enabled by default (UseSpotInstances; override with TRAINING_EC2_USE_SPOT_INSTANCES=true|false)."

ensure_prerequisites

if ! aws cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "❌ Fargate training stack not found: $TRAINING_STACK_NAME"
  echo "   Deploy it first: aws/scripts/deploy_training.sh $ENVIRONMENT"
  exit 1
fi

echo "Building and pushing training container image..."
build_and_push_training_image

echo "Deploying ECS EC2 training infrastructure..."

# Subnets are imported directly from the Fargate stack in the CloudFormation template.
# Defaults come from poeconomy-training-ec2.yaml via merge_cloudformation_parameter_overrides.
OVERRIDES=(
  "TrainingFargateStackName=$TRAINING_STACK_NAME"
  "IngestionStackName=$INGESTION_STACK_NAME"
  "DataLakeBucketName=$DATA_LAKE_BUCKET_NAME"
  "TrainingImageUri=$TRAINING_IMAGE_URI"
)
[[ -n "${TRAINING_EC2_TASK_CPU:-}" ]] && OVERRIDES+=("TaskCpu=$TRAINING_EC2_TASK_CPU")
[[ -n "${TRAINING_EC2_TASK_MEMORY:-}" ]] && OVERRIDES+=("TaskMemory=$TRAINING_EC2_TASK_MEMORY")
[[ -n "${TRAINING_EC2_INSTANCE_TYPE:-}" ]] && OVERRIDES+=("InstanceType=$TRAINING_EC2_INSTANCE_TYPE")
[[ -n "${TRAINING_EC2_USE_SPOT_INSTANCES:-}" ]] && OVERRIDES+=("UseSpotInstances=$TRAINING_EC2_USE_SPOT_INSTANCES")
[[ -n "${TRAINING_EC2_ASG_MIN:-}" ]] && OVERRIDES+=("AsgMinSize=$TRAINING_EC2_ASG_MIN")
[[ -n "${TRAINING_EC2_ASG_MAX:-}" ]] && OVERRIDES+=("AsgMaxSize=$TRAINING_EC2_ASG_MAX")
[[ -n "${TRAINING_EC2_ASG_DESIRED:-}" ]] && OVERRIDES+=("AsgDesiredCapacity=$TRAINING_EC2_ASG_DESIRED")
[[ -n "${TRAINING_EC2_ROOT_VOLUME_GIB:-}" ]] && OVERRIDES+=("RootVolumeGiB=$TRAINING_EC2_ROOT_VOLUME_GIB")
# EcsOptimizedAmiParameter: defined only in poeconomy-training-ec2.yaml (SSM path Default); not overridden here.
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

deploy_cloudformation_stack "$TRAINING_EC2_STACK_NAME" "$TRAINING_EC2_TEMPLATE" "${OVERRIDES[@]}"

update_env_file

echo "========================================"
echo "ECS EC2 training infrastructure deployment complete!"
echo "========================================"
