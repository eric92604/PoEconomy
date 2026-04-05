#!/bin/bash

# Shared configuration for PoEconomy deployment scripts.
# All scripts must be run from WSL2 (Ubuntu on Windows) — not Git Bash.

set -euo pipefail
set -E -o errtrace 2>/dev/null || set -E 2>/dev/null || true

# WSL2 guard: Git Bash path conversion breaks AWS CLI, Docker, and Python calls.
# Run from WSL2 where everything is native Linux.
if [[ "${OSTYPE:-}" == msys* ]] || [[ "${OSTYPE:-}" == cygwin* ]]; then
  echo "❌ Run deployment scripts from WSL2, not Git Bash." >&2
  echo "   Open a WSL2 terminal: wsl -d Ubuntu" >&2
  echo "   Project path in WSL2: /mnt/c/Workspace/PoEconomy" >&2
  exit 1
fi

# Core configuration
ENVIRONMENT=${1:-production}
REGION=${AWS_DEFAULT_REGION:-us-west-2}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/aws/.env"

# Load .env BEFORE computing derived variables so overrides (ENVIRONMENT, REGION, etc.) take effect.
if [[ -f "$ENV_FILE" ]]; then
  echo "Loading environment variables from $ENV_FILE"
  set -a
  source "$ENV_FILE"
  set +a
fi

# Stack names (computed after .env load so they can be overridden via .env)
BASE_STACK_NAME="poeconomy-${ENVIRONMENT}-base"
INGESTION_STACK_NAME="poeconomy-${ENVIRONMENT}-ingestion"
PREDICTION_STACK_NAME="poeconomy-${ENVIRONMENT}-prediction"
API_STACK_NAME="poeconomy-${ENVIRONMENT}-api"
FEATURE_ENGINEERING_STACK_NAME="poeconomy-${ENVIRONMENT}-feature-engineering"
TRAINING_STACK_NAME="poeconomy-${ENVIRONMENT}-training"
TRAINING_EC2_STACK_NAME="poeconomy-${ENVIRONMENT}-training-ec2"

# Template paths
BASE_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-base.yaml"
INGESTION_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-ingestion.yaml"
PREDICTION_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-prediction.yaml"
API_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-api.yaml"
FEATURE_ENGINEERING_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-feature-engineering.yaml"
TRAINING_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-training.yaml"
TRAINING_EC2_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-training-ec2.yaml"

# Bucket and schedule configuration
DATA_LAKE_BUCKET_NAME="poeconomy-${ENVIRONMENT}-datalake"
INGESTION_CRON="cron(0 * * * ? *)"
LEAGUE_METADATA_CRON="cron(0 0 * * ? *)"
DAILY_AGGREGATION_CRON="cron(0 2 * * ? *)"
FEATURE_ENGINEERING_CRON="cron(0 2 * * ? *)"
API_STAGE_NAME="api"

# AWS account ID (WSL2: no Windows carriage-return stripping needed)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

_po_deploy_err_trap() {
  local ec=$?
  echo "❌ Deployment step failed (exit $ec)." >&2
  echo "   Command: ${BASH_COMMAND:-unknown}" >&2
}

_po_deploy_exit_trap() {
  local ec=$?
  trap - EXIT
  if [[ "$ec" -ne 0 ]] && [[ -z "${CI:-}" ]] && [[ "${DEPLOY_PAUSE_ON_FAILURE:-}" != "0" ]]; then
    if [[ -t 2 ]] || [[ "${DEPLOY_PAUSE_ON_FAILURE:-}" == "1" ]]; then
      read -r -p "Press Enter to exit... " _ </dev/tty 2>/dev/null || read -r -p "Press Enter to exit... " _ || true
    fi
  fi
}

trap '_po_deploy_err_trap' ERR
trap '_po_deploy_exit_trap' EXIT

# Run a command without set -e aborting the script on failure.
po_deploy_run_without_abort() {
  set +e
  "$@"
  local st=$?
  set -e
  return "$st"
}

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

ensure_prerequisites() {
  echo "========================================"
  echo "Ensuring prerequisites..."
  echo "========================================"

  if ! command -v aws >/dev/null 2>&1; then
    echo "❌ AWS CLI not found."
    echo "   Install: curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o awscliv2.zip && unzip awscliv2.zip && sudo ./aws/install"
    exit 1
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found. Ensure Docker Desktop is running with the WSL2 backend enabled."
    exit 1
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3 not found. Install: sudo apt install python3 python3-pip"
    exit 1
  fi

  echo "✅ Prerequisites verified"
}

ensure_ecr_repository() {
  local repository="$1"
  echo "Ensuring ECR repository $repository exists..."
  if ! aws ecr describe-repositories --repository-names "$repository" --region "$REGION" >/dev/null 2>&1; then
    echo "Creating ECR repository $repository"
    aws ecr create-repository --repository-name "$repository" --region "$REGION" >/dev/null
  fi
}

verify_models_in_s3() {
  echo "========================================"
  echo "Verifying models exist in S3..."
  echo "========================================"
  if ! aws s3 ls "s3://$DATA_LAKE_BUCKET_NAME/models/currency/" --region "$REGION" >/dev/null 2>&1; then
    echo "❌ No models found in s3://$DATA_LAKE_BUCKET_NAME/models/currency/"
    echo "   Run the training pipeline to generate models first."
    exit 1
  fi
  echo "✅ Models found in S3"
}

# ---------------------------------------------------------------------------
# CloudFormation deployment
# ---------------------------------------------------------------------------

# Merge template parameter defaults with explicit Key=Value overrides.
# Prints one Key=Value per line. Requires PyYAML: pip install pyyaml
merge_cloudformation_parameter_overrides() {
  local template_path="$1"
  shift
  local merger="$SCRIPT_DIR/lib/cfn_merge_stack_parameters.py"
  if [[ ! -f "$merger" ]]; then
    echo "❌ Missing merger script: $merger" >&2
    return 1
  fi
  python3 "$merger" "$template_path" "$@"
}

deploy_cloudformation_stack() {
  local stack_name="$1"
  local template_path="$2"
  shift 2
  local explicit_overrides=("$@")

  echo "Deploying CloudFormation stack: $stack_name"

  local merged_lines
  if ! merged_lines=$(merge_cloudformation_parameter_overrides "$template_path" "${explicit_overrides[@]}"); then
    echo "❌ Failed to merge CloudFormation parameters for: $template_path" >&2
    return 1
  fi

  local param_overrides=()
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    [[ -z "$line" ]] && continue
    param_overrides+=("$line")
  done <<< "$merged_lines"

  if [[ ${#param_overrides[@]} -eq 0 ]]; then
    echo "❌ No parameters after merge (empty template Parameters?)" >&2
    return 1
  fi

  if ! aws cloudformation deploy \
    --template-file "$template_path" \
    --stack-name "$stack_name" \
    --region "$REGION" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides "${param_overrides[@]}"; then
    echo "❌ CloudFormation deployment failed for stack: $stack_name"
    return 1
  fi

  echo "✅ CloudFormation stack deployed successfully: $stack_name"
}

print_deployment_info() {
  echo "========================================"
  echo "PoEconomy Infrastructure Deployment"
  echo "Environment: $ENVIRONMENT"
  echo "Region: $REGION"
  echo "Account: $ACCOUNT_ID"
  echo "========================================"
}

# ---------------------------------------------------------------------------
# Training data upload
# ---------------------------------------------------------------------------

upload_training_data_if_needed() {
  echo "========================================"
  echo "Checking training data in S3..."
  echo "========================================"
  if aws s3 ls "s3://$DATA_LAKE_BUCKET_NAME/training-data/" --region "$REGION" >/dev/null 2>&1; then
    echo "✅ Training data already exists in S3"
    return 0
  fi
  echo "Uploading training data to S3..."
  upload_training_data
}

upload_training_data() {
  local training_data_dir="$ROOT_DIR/training_data"
  if [[ ! -d "$training_data_dir" ]]; then
    echo "❌ Training data directory not found: $training_data_dir"
    exit 1
  fi
  echo "Uploading $training_data_dir to s3://$DATA_LAKE_BUCKET_NAME/training-data/"
  aws s3 sync "$training_data_dir" "s3://$DATA_LAKE_BUCKET_NAME/training-data/" --region "$REGION"
  echo "✅ Training data uploaded successfully"
}

# ---------------------------------------------------------------------------
# Lambda packaging
# ---------------------------------------------------------------------------

package_and_upload_lambda() {
  local lambda_name=$1
  local handler_file=$2
  local zip_name=$3

  echo "Packaging $lambda_name Lambda function..."

  local python_script="$SCRIPT_DIR/package_lambda.py"
  if [[ ! -f "$python_script" ]]; then
    echo "❌ Package script not found: $python_script" >&2
    return 1
  fi

  local temp_dir
  temp_dir=$(mktemp -d)
  local temp_zip="$temp_dir/$zip_name"

  if ! python3 "$python_script" "$lambda_name" "$handler_file" "$temp_zip"; then
    echo "❌ Failed to package $lambda_name" >&2
    rm -rf "$temp_dir"
    return 1
  fi

  if ! aws s3 cp "$temp_zip" "s3://$DATA_LAKE_BUCKET_NAME/lambda/"; then
    echo "❌ Failed to upload $zip_name to S3" >&2
    rm -rf "$temp_dir"
    return 1
  fi

  rm -rf "$temp_dir"
  echo "✅ $lambda_name packaged and uploaded successfully"
}

# ---------------------------------------------------------------------------
# Subnet resolution
# ---------------------------------------------------------------------------

# Resolve Fargate training stack public subnet IDs. Prefers CloudFormation outputs;
# falls back to EC2 describe with vpc-id + Name tag for older stacks.
# Sets TRAINING_PUBLIC_SUBNET_A_ID and TRAINING_PUBLIC_SUBNET_B_ID.
resolve_training_public_subnet_ids() {
  TRAINING_PUBLIC_SUBNET_A_ID=$(aws cloudformation describe-stacks \
    --stack-name "$TRAINING_STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='TrainingPublicSubnetAId'].OutputValue" --output text)
  TRAINING_PUBLIC_SUBNET_B_ID=$(aws cloudformation describe-stacks \
    --stack-name "$TRAINING_STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='TrainingPublicSubnetBId'].OutputValue" --output text)

  if [[ -n "$TRAINING_PUBLIC_SUBNET_A_ID" && "$TRAINING_PUBLIC_SUBNET_A_ID" != "None" \
     && -n "$TRAINING_PUBLIC_SUBNET_B_ID" && "$TRAINING_PUBLIC_SUBNET_B_ID" != "None" ]]; then
    return 0
  fi

  local training_vpc_id
  training_vpc_id=$(aws cloudformation describe-stacks \
    --stack-name "$TRAINING_STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='TrainingVpcId'].OutputValue" --output text)
  if [[ -z "$training_vpc_id" || "$training_vpc_id" == "None" ]]; then
    echo "❌ Could not resolve TrainingVpcId for training subnet fallback." >&2
    TRAINING_PUBLIC_SUBNET_A_ID=""
    TRAINING_PUBLIC_SUBNET_B_ID=""
    return 1
  fi

  TRAINING_PUBLIC_SUBNET_A_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$training_vpc_id" \
              "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-training-public-subnet-a" \
    --query "Subnets[0].SubnetId" --output text)
  TRAINING_PUBLIC_SUBNET_B_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$training_vpc_id" \
              "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-training-public-subnet-b" \
    --query "Subnets[0].SubnetId" --output text)

  if [[ -z "$TRAINING_PUBLIC_SUBNET_A_ID" || "$TRAINING_PUBLIC_SUBNET_A_ID" == "None" \
     || -z "$TRAINING_PUBLIC_SUBNET_B_ID" || "$TRAINING_PUBLIC_SUBNET_B_ID" == "None" ]]; then
    echo "❌ Could not resolve training public subnet IDs." >&2
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Environment file persistence
# ---------------------------------------------------------------------------

update_env_file() {
  echo "Updating .env file with current stack outputs..."

  cat > "$ENV_FILE" << EOF
# Generated by shared_config.sh - $(date)
AWS_REGION=$REGION
DATA_LAKE_BUCKET=$DATA_LAKE_BUCKET_NAME
MODELS_BUCKET=$DATA_LAKE_BUCKET_NAME
EOF

  if aws cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Base stack outputs..."
    DATA_LAKE_BUCKET_ARN=$(aws cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='DataLakeBucketArn'].OutputValue" --output text)
    TRAINING_DATA_LAKE_ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingDataLakeRoleArn'].OutputValue" --output text)
    FEATURE_ENGINEERING_DATA_LAKE_ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringDataLakeRoleArn'].OutputValue" --output text)
    cat >> "$ENV_FILE" << EOF

# Base Infrastructure Resources
DATA_LAKE_BUCKET_ARN=$DATA_LAKE_BUCKET_ARN
TRAINING_DATA_LAKE_ROLE_ARN=$TRAINING_DATA_LAKE_ROLE_ARN
FEATURE_ENGINEERING_DATA_LAKE_ROLE_ARN=$FEATURE_ENGINEERING_DATA_LAKE_ROLE_ARN
EOF
  fi

  if aws cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Ingestion stack outputs..."
    DYNAMO_CURRENCY_METADATA_TABLE=$(aws cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='CurrencyMetadataTableName'].OutputValue" --output text)
    DYNAMO_CURRENCY_PRICES_TABLE=$(aws cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='LivePricesTableName'].OutputValue" --output text)
    DYNAMO_LEAGUE_METADATA_TABLE=$(aws cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='LeagueMetadataTableName'].OutputValue" --output text)
    INGESTION_LAMBDA_NAME=$(aws cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='IngestionLambdaName'].OutputValue" --output text)
    LEAGUE_METADATA_LAMBDA_NAME=$(aws cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='LeagueMetadataLambdaName'].OutputValue" --output text)
    DAILY_AGGREGATION_LAMBDA_NAME=$(aws cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='DailyAggregationLambdaName'].OutputValue" --output text)
  fi

  if aws cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Prediction stack outputs..."
    DYNAMO_PREDICTIONS_TABLE=$(aws cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='PredictionsTableName'].OutputValue" --output text)
    PREDICTION_REFRESH_LAMBDA_NAME=$(aws cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='PredictionRefreshLambdaName'].OutputValue" --output text)
    LAMBDA_INFERENCE_IMAGE=$(aws cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='LambdaInferenceImage'].OutputValue" --output text)
  fi

  if aws cloudformation describe-stacks --stack-name "$API_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting API stack outputs..."
    API_BASE_URL=$(aws cloudformation describe-stacks --stack-name "$API_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='ApiInvokeUrl'].OutputValue" --output text)
    PREDICTION_API_LAMBDA_NAME=$(aws cloudformation describe-stacks --stack-name "$API_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='PredictionApiLambdaName'].OutputValue" --output text)
  fi

  if [ -n "${DYNAMO_CURRENCY_METADATA_TABLE:-}" ] || [ -n "${DYNAMO_CURRENCY_PRICES_TABLE:-}" ] \
  || [ -n "${DYNAMO_PREDICTIONS_TABLE:-}" ] || [ -n "${DYNAMO_LEAGUE_METADATA_TABLE:-}" ] \
  || [ -n "${API_BASE_URL:-}" ] || [ -n "${LAMBDA_INFERENCE_IMAGE:-}" ] \
  || [ -n "${PREDICTION_API_LAMBDA_NAME:-}" ] || [ -n "${INGESTION_LAMBDA_NAME:-}" ]; then
    cat >> "$ENV_FILE" << EOF

# Lambda Functions and DynamoDB Tables
DYNAMO_CURRENCY_METADATA_TABLE=${DYNAMO_CURRENCY_METADATA_TABLE:-}
DYNAMO_CURRENCY_PRICES_TABLE=${DYNAMO_CURRENCY_PRICES_TABLE:-}
DYNAMO_PREDICTIONS_TABLE=${DYNAMO_PREDICTIONS_TABLE:-}
DYNAMO_LEAGUE_METADATA_TABLE=${DYNAMO_LEAGUE_METADATA_TABLE:-}
API_BASE_URL=${API_BASE_URL:-}
LAMBDA_INFERENCE_IMAGE=${LAMBDA_INFERENCE_IMAGE:-}
PREDICTION_API_LAMBDA_NAME=${PREDICTION_API_LAMBDA_NAME:-}
INGESTION_LAMBDA_NAME=${INGESTION_LAMBDA_NAME:-}
LEAGUE_METADATA_LAMBDA_NAME=${LEAGUE_METADATA_LAMBDA_NAME:-}
DAILY_AGGREGATION_LAMBDA_NAME=${DAILY_AGGREGATION_LAMBDA_NAME:-}
PREDICTION_REFRESH_LAMBDA_NAME=${PREDICTION_REFRESH_LAMBDA_NAME:-}
EOF
  fi

  if aws cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Feature Engineering stack outputs..."
    FEATURE_ENGINEERING_CLUSTER_NAME=$(aws cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringClusterName'].OutputValue" --output text)
    FEATURE_ENGINEERING_TASK_DEFINITION=$(aws cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringTaskDefinitionArn'].OutputValue" --output text)
    FEATURE_ENGINEERING_VPC_ID=$(aws cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringVpcId'].OutputValue" --output text)
    FE_SUBNET_A=$(aws ec2 describe-subnets --region "$REGION" \
      --filters "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-feature-engineering-public-subnet-a" \
      --query "Subnets[0].SubnetId" --output text)
    FE_SUBNET_B=$(aws ec2 describe-subnets --region "$REGION" \
      --filters "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-feature-engineering-public-subnet-b" \
      --query "Subnets[0].SubnetId" --output text)
    FEATURE_ENGINEERING_SUBNET_IDS="${FE_SUBNET_A},${FE_SUBNET_B}"
    FEATURE_ENGINEERING_SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --region "$REGION" \
      --filters "Name=group-name,Values=*feature-engineering*" \
      --query "SecurityGroups[0].GroupId" --output text)
    cat >> "$ENV_FILE" << EOF

# ECS Feature Engineering Resources
FEATURE_ENGINEERING_CLUSTER_NAME=$FEATURE_ENGINEERING_CLUSTER_NAME
FEATURE_ENGINEERING_TASK_DEFINITION=$FEATURE_ENGINEERING_TASK_DEFINITION
FEATURE_ENGINEERING_VPC_ID=$FEATURE_ENGINEERING_VPC_ID
FEATURE_ENGINEERING_SUBNET_IDS=$FEATURE_ENGINEERING_SUBNET_IDS
FEATURE_ENGINEERING_SECURITY_GROUP_ID=$FEATURE_ENGINEERING_SECURITY_GROUP_ID
EOF
  fi

  if aws cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Training stack outputs..."
    TRAINING_CLUSTER_NAME=$(aws cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingClusterName'].OutputValue" --output text)
    TRAINING_TASK_DEFINITION=$(aws cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingTaskDefinitionArn'].OutputValue" --output text)
    TRAINING_VPC_ID=$(aws cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingVpcId'].OutputValue" --output text)
    if ! resolve_training_public_subnet_ids; then
      echo "❌ update_env_file: could not resolve training subnets." >&2
      return 1
    fi
    TRAINING_SUBNET_IDS="${TRAINING_PUBLIC_SUBNET_A_ID},${TRAINING_PUBLIC_SUBNET_B_ID}"
    TRAINING_SECURITY_GROUP_ID=$(aws cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingSecurityGroupId'].OutputValue" --output text)
    cat >> "$ENV_FILE" << EOF

# ECS Training Resources
TRAINING_CLUSTER_NAME=$TRAINING_CLUSTER_NAME
TRAINING_TASK_DEFINITION=$TRAINING_TASK_DEFINITION
TRAINING_VPC_ID=$TRAINING_VPC_ID
TRAINING_SUBNET_IDS=$TRAINING_SUBNET_IDS
TRAINING_SECURITY_GROUP_ID=$TRAINING_SECURITY_GROUP_ID
EOF
  fi

  if aws cloudformation describe-stacks --stack-name "$TRAINING_EC2_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting ECS EC2 Training stack outputs..."
    TRAINING_EC2_CLUSTER_NAME=$(aws cloudformation describe-stacks --stack-name "$TRAINING_EC2_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingEc2ClusterName'].OutputValue" --output text)
    TRAINING_EC2_TASK_DEFINITION=$(aws cloudformation describe-stacks --stack-name "$TRAINING_EC2_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingEc2TaskDefinitionArn'].OutputValue" --output text)
    TRAINING_EC2_LOG_GROUP_NAME=$(aws cloudformation describe-stacks --stack-name "$TRAINING_EC2_STACK_NAME" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='TrainingEc2LogGroupName'].OutputValue" --output text)
    cat >> "$ENV_FILE" << EOF

# ECS Training EC2 Resources (${TRAINING_EC2_STACK_NAME})
TRAINING_EC2_CLUSTER_NAME=$TRAINING_EC2_CLUSTER_NAME
TRAINING_EC2_TASK_DEFINITION=$TRAINING_EC2_TASK_DEFINITION
TRAINING_EC2_LOG_GROUP_NAME=$TRAINING_EC2_LOG_GROUP_NAME
EOF
  fi

  echo "Getting EventBridge schedule information..."
  FEATURE_ENGINEERING_SCHEDULE=$(aws events list-rules --region "$REGION" \
    --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-feature-engineering-schedule'].ScheduleExpression" --output text)
  INGESTION_SCHEDULE=$(aws events list-rules --region "$REGION" \
    --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-ingestion-schedule'].ScheduleExpression" --output text)
  LEAGUE_METADATA_SCHEDULE=$(aws events list-rules --region "$REGION" \
    --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-league-metadata-schedule'].ScheduleExpression" --output text)
  TRAINING_SCHEDULE=$(aws events list-rules --region "$REGION" \
    --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-training-schedule'].ScheduleExpression" --output text)
  PREDICTION_SCHEDULE=$(aws events list-rules --region "$REGION" \
    --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-prediction-schedule'].ScheduleExpression" --output text)

  FEATURE_ENGINEERING_LOG_GROUP="/aws/ecs/${FEATURE_ENGINEERING_STACK_NAME}"
  TRAINING_LOG_GROUP="/aws/ecs/${TRAINING_STACK_NAME}"
  TRAINING_EC2_LOG_GROUP="/aws/ecs/${TRAINING_EC2_STACK_NAME}"
  API_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-api"
  INGESTION_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-ingestion"
  LEAGUE_METADATA_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-league-metadata"
  DAILY_AGGREGATION_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-daily-aggregation"
  PREDICTION_REFRESH_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-prediction-refresh"

  cat >> "$ENV_FILE" << EOF

# EventBridge Schedules
FEATURE_ENGINEERING_SCHEDULE="$FEATURE_ENGINEERING_SCHEDULE"
INGESTION_SCHEDULE="$INGESTION_SCHEDULE"
LEAGUE_METADATA_SCHEDULE="$LEAGUE_METADATA_SCHEDULE"
TRAINING_SCHEDULE="$TRAINING_SCHEDULE"
PREDICTION_SCHEDULE="$PREDICTION_SCHEDULE"

# CloudWatch Log Groups
FEATURE_ENGINEERING_LOG_GROUP=$FEATURE_ENGINEERING_LOG_GROUP
TRAINING_LOG_GROUP=$TRAINING_LOG_GROUP
TRAINING_EC2_LOG_GROUP=$TRAINING_EC2_LOG_GROUP
API_LAMBDA_LOG_GROUP=$API_LAMBDA_LOG_GROUP
INGESTION_LAMBDA_LOG_GROUP=$INGESTION_LAMBDA_LOG_GROUP
LEAGUE_METADATA_LAMBDA_LOG_GROUP=$LEAGUE_METADATA_LAMBDA_LOG_GROUP
DAILY_AGGREGATION_LAMBDA_LOG_GROUP=$DAILY_AGGREGATION_LAMBDA_LOG_GROUP
PREDICTION_REFRESH_LAMBDA_LOG_GROUP=$PREDICTION_REFRESH_LAMBDA_LOG_GROUP

# Latest Container Images (for reference)
LATEST_LAMBDA_IMAGE=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/poeconomy-${ENVIRONMENT}-lambdas:latest
LATEST_FEATURE_ENGINEERING_IMAGE=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/poeconomy-${ENVIRONMENT}-feature-engineering:latest
LATEST_TRAINING_IMAGE=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/poeconomy-${ENVIRONMENT}-training:latest

# Environment Metadata
ENVIRONMENT_NAME=${ENVIRONMENT}
AWS_ACCOUNT_ID=${ACCOUNT_ID}
EOF

  echo "✅ .env file updated successfully"
  echo "Updated .env file contents:"
  cat "$ENV_FILE"
}
