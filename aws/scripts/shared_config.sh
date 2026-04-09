#!/bin/bash

# Shared configuration for PoEconomy deployment scripts
# This file contains common variables and functions used by all deployment scripts

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-production}
REGION=${AWS_DEFAULT_REGION:-us-west-2}

# AWS CLI command selection:
# - In WSL, prefer Linux `aws` to avoid Windows path translation issues in `s3 sync`.
# - Outside WSL, use `aws.exe` if available (Windows), otherwise `aws`.
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  if command -v aws >/dev/null 2>&1; then
    AWS_CMD="aws"
  elif command -v aws.exe >/dev/null 2>&1; then
    AWS_CMD="aws.exe"
  else
    AWS_CMD="aws"
  fi
elif command -v aws.exe >/dev/null 2>&1; then
  AWS_CMD="aws.exe"
else
  AWS_CMD="aws"
fi

# Get account ID and strip Windows line endings (works cross-platform)
# Use bash parameter expansion instead of tr/sed for better Windows compatibility
ACCOUNT_ID_RAW=$($AWS_CMD sts get-caller-identity --query Account --output text)
# Remove carriage returns and newlines using bash parameter expansion
ACCOUNT_ID="${ACCOUNT_ID_RAW//$'\r'/}"
ACCOUNT_ID="${ACCOUNT_ID//$'\n'/}"
# Trim any remaining leading/trailing whitespace (handles spaces, tabs, etc.)
ACCOUNT_ID="${ACCOUNT_ID#"${ACCOUNT_ID%%[![:space:]]*}"}"  # Remove leading whitespace
ACCOUNT_ID="${ACCOUNT_ID%"${ACCOUNT_ID##*[![:space:]]}"}"  # Remove trailing whitespace
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stack names
BASE_STACK_NAME="poeconomy-${ENVIRONMENT}-base"
INGESTION_STACK_NAME="poeconomy-${ENVIRONMENT}-ingestion"
PREDICTION_STACK_NAME="poeconomy-${ENVIRONMENT}-prediction"
API_STACK_NAME="poeconomy-${ENVIRONMENT}-api"
FEATURE_ENGINEERING_STACK_NAME="poeconomy-${ENVIRONMENT}-feature-engineering"
TRAINING_STACK_NAME="poeconomy-${ENVIRONMENT}-training"

# Template paths
BASE_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-base.yaml"
INGESTION_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-ingestion.yaml"
PREDICTION_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-prediction.yaml"
API_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-api.yaml"
FEATURE_ENGINEERING_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-feature-engineering.yaml"
TRAINING_TEMPLATE="$ROOT_DIR/aws/cloudformation/poeconomy-training.yaml"

# Environment file
ENV_FILE="$ROOT_DIR/aws/.env"

# Bucket names
DATA_LAKE_BUCKET_NAME="poeconomy-${ENVIRONMENT}-datalake"
# Models are now stored in the data lake bucket under /models/ prefix

# Schedule expressions
INGESTION_CRON="cron(0 * * * ? *)"
LEAGUE_METADATA_CRON="cron(0 0 * * ? *)"
DAILY_AGGREGATION_CRON="cron(0 2 * * ? *)"
FEATURE_ENGINEERING_CRON="cron(0 2 * * ? *)"

# API configuration
API_STAGE_NAME="api"

# Task configuration

# Load environment variables if file exists
if [[ -f "$ENV_FILE" ]]; then
  echo "Loading environment variables from $ENV_FILE"
  set -a
  source "$ENV_FILE"
  set +a
fi

# Common functions
ensure_prerequisites() {
  echo "========================================"
  echo "Ensuring prerequisites..."
  echo "========================================"
  
  # Check AWS CLI
  if ! command -v "$AWS_CMD" >/dev/null 2>&1; then
    echo "❌ AWS CLI not found. Please install AWS CLI."
    exit 1
  fi
  
  # Check Docker
  if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
  fi
  
  # Check Python (try python3 as fallback)
  if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python not found. Please install Python."
    exit 1
  fi
  
  echo "✅ Prerequisites verified"
}

ensure_ecr_repository() {
  local repository="$1"
  echo "Ensuring ECR repository $repository exists..."
  
  if ! "$AWS_CMD" ecr describe-repositories --repository-names "$repository" --region "$REGION" >/dev/null 2>&1; then
    echo "Creating ECR repository $repository"
    "$AWS_CMD" ecr create-repository --repository-name "$repository" --region "$REGION" >/dev/null
  fi
}

verify_models_in_s3() {
  echo "========================================"
  echo "Verifying models exist in S3..."
  echo "========================================"
  
  # Check if models exist in data lake bucket
  echo "Listing models in s3://$DATA_LAKE_BUCKET_NAME/models/currency"
  if ! "$AWS_CMD" s3 ls "s3://$DATA_LAKE_BUCKET_NAME/models/currency/" --region "$REGION" >/dev/null 2>&1; then
    echo "❌ No models found in s3://$DATA_LAKE_BUCKET_NAME/models/currency/"
    echo "❌ Cannot deploy Lambda functions without models"
    exit 1
  fi
  
  echo "✅ Models found in S3"
}

deploy_cloudformation_stack() {
  local stack_name="$1"
  local template_path="$2"
  local parameters=("${@:3}")
  
  echo "Deploying CloudFormation stack: $stack_name"
  
  # Convert template path for Windows AWS CLI if needed
  local template_file="$template_path"
  if [[ "$AWS_CMD" == *".exe" ]]; then
    template_file=$(wslpath -w "$template_path" 2>/dev/null || echo "$template_path")
  fi
  
  # Build parameter overrides array
  local param_overrides=()
  for param in "${parameters[@]}"; do
    param_overrides+=("${param%%=*}=${param#*=}")
  done
  
  # Deploy stack
  if ! "$AWS_CMD" cloudformation deploy \
    --template-file "$template_file" \
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

build_and_push_lambda_image() {
  local repository="poeconomy-${ENVIRONMENT}-lambdas"
  ensure_ecr_repository "$repository"
  
  echo "Logging in to ECR ($ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com)"
  "$AWS_CMD" ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
  
  echo "Building Lambda container image with models (prediction capabilities enabled)"
  build_lambda_image_with_models
  
  echo "Pushing image to ECR"
  local timestamp=$(date +%Y%m%d%H%M%S)
  docker tag "poeconomy-${ENVIRONMENT}-lambda:latest" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repository:latest"
  docker tag "poeconomy-${ENVIRONMENT}-lambda:latest" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repository:$timestamp"
  
  # Push images to ECR
  echo "Pushing images to ECR..."
  docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repository:latest"
  docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repository:$timestamp"
  
  # Verify the pushed image manifest
  echo "Verifying pushed image manifest..."
  "$AWS_CMD" ecr describe-images --repository-name "$repository" --image-ids imageTag=latest --region "$REGION" --query 'imageDetails[0].{Digest:imageDigest,Size:imageSizeInBytes,PushedAt:imagePushedAt,ImageManifestMediaType:imageManifestMediaType}' --output table
  
  # Resolve immutable image digest and use digest-pinned URI for Lambda deployments.
  # This avoids tag-resolution races during CloudFormation updates.
  local image_digest
  image_digest=$("$AWS_CMD" ecr describe-images \
    --repository-name "$repository" \
    --image-ids imageTag="$timestamp" \
    --region "$REGION" \
    --query 'imageDetails[0].imageDigest' \
    --output text)
  
  if [[ -z "$image_digest" ]] || [[ "$image_digest" == "None" ]]; then
    echo "❌ Failed to resolve image digest for tag: $timestamp"
    exit 1
  fi
  
  # Additional verification - check if the image can be pulled and inspected
  echo "Testing ECR image pull and inspection..."
  docker pull "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repository:$timestamp"
  local pulled_arch
  pulled_arch=$(docker inspect "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repository:$timestamp" --format='{{.Architecture}}')
  echo "Pulled image architecture: $pulled_arch"
  
  if [[ "$pulled_arch" != "amd64" ]]; then
    echo "❌ Pulled image architecture is $pulled_arch, expected amd64"
    exit 1
  fi
  
  echo "✅ ECR image verification successful"
  
  INFERENCE_IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repository@$image_digest"
  echo "Using immutable Lambda image URI: $INFERENCE_IMAGE_URI"
}

find_latest_experiment() {
  local backup_dir="$ROOT_DIR/s3_backup/models/currency"
  local latest_experiment=""
  local latest_timestamp=0
  
  # Find all experiment directories (xp_* pattern)
  for exp_dir in "$backup_dir"/xp_*; do
    if [[ -d "$exp_dir" ]]; then
      local exp_name=$(basename "$exp_dir")
      local exp_models=$(find "$exp_dir" -name "*.pkl" 2>/dev/null | wc -l)
      
      if [[ $exp_models -gt 0 ]]; then
        # Extract timestamp from directory name (xp_YYYYMMDD_HHMMSS)
        local timestamp_str=$(echo "$exp_name" | sed 's/xp_//' | sed 's/_/ /')
        local epoch_time=0
        
        # Try to parse timestamp (format: YYYYMMDD HHMMSS)
        if [[ "$timestamp_str" =~ ^[0-9]{8}\ [0-9]{6}$ ]]; then
          # Convert to epoch (works on Linux/Mac)
          epoch_time=$(date -d "$timestamp_str" +%s 2>/dev/null || date -j -f "%Y%m%d %H%M%S" "$timestamp_str" +%s 2>/dev/null || echo "0")
        fi
        
        # Fallback: use directory name for comparison if date parsing fails
        # Extract digits using bash parameter expansion (more portable than tr)
        if [[ $epoch_time -eq 0 ]]; then
          # Remove all non-digit characters using bash parameter expansion
          digits_only="${exp_name//[^0-9]/}"
          epoch_time="${digits_only:0:14}"
        fi
        
        if [[ $epoch_time -gt $latest_timestamp ]]; then
          latest_timestamp=$epoch_time
          latest_experiment="$exp_name"
        fi
      fi
    fi
  done
  
  if [[ -n "$latest_experiment" ]]; then
    echo "$latest_experiment"
    return 0
  fi
  
  return 1
}

find_latest_experiment_in_s3() {
  local latest_experiment=""
  local experiment_candidates

  # Parse experiment prefixes from `aws s3 ls` output:
  # PRE xp_YYYYMMDD_HHMMSS/
  experiment_candidates=$("$AWS_CMD" s3 ls "s3://$DATA_LAKE_BUCKET_NAME/models/currency/" --region "$REGION" 2>/dev/null \
    | sed -n 's/^[[:space:]]*PRE[[:space:]]\(xp_[0-9]\{8\}_[0-9]\{6\}\)\/[[:space:]]*$/\1/p' \
    | sort)

  latest_experiment=$(echo "$experiment_candidates" | sed '/^$/d' | tail -n 1)
  latest_experiment="${latest_experiment//$'\r'/}"
  latest_experiment="${latest_experiment//$'\n'/}"
  if [[ -n "$latest_experiment" ]]; then
    echo "$latest_experiment"
    return 0
  fi
  
  return 1
}

verify_local_backup() {
  echo "Checking for local model backup..."
  
  local backup_dir="$ROOT_DIR/s3_backup/models/currency"
  
  if [[ ! -d "$backup_dir" ]]; then
    echo "❌ Local backup directory not found: $backup_dir"
    return 1
  fi
  
  echo "✅ Local backup directory found: $backup_dir"
  
  # Check for latest experiment locally and in S3.
  local local_latest_experiment
  local s3_latest_experiment
  local latest_experiment
  local_latest_experiment=$(find_latest_experiment)
  s3_latest_experiment=$(find_latest_experiment_in_s3)

  latest_experiment="$local_latest_experiment"
  if [[ -n "$s3_latest_experiment" ]] && { [[ -z "$latest_experiment" ]] || [[ "$s3_latest_experiment" > "$latest_experiment" ]]; }; then
    latest_experiment="$s3_latest_experiment"
  fi

  # If S3 has a newer experiment than local backup, sync only that experiment directory.
  if [[ -n "$latest_experiment" ]] && [[ "$latest_experiment" == "$s3_latest_experiment" ]] && [[ ! -d "$backup_dir/$latest_experiment" ]]; then
    echo "⬇️  Syncing latest experiment from S3: $latest_experiment"
    if ! "$AWS_CMD" s3 sync \
      "s3://$DATA_LAKE_BUCKET_NAME/models/currency/$latest_experiment/" \
      "$backup_dir/$latest_experiment/" \
      --region "$REGION"; then
      echo "❌ Failed to sync latest experiment from S3: $latest_experiment"
      return 1
    fi
  fi
  
  if [[ -n "$latest_experiment" ]]; then
    local exp_dir="$backup_dir/$latest_experiment"
    local model_count=$(find "$exp_dir" -name "*.pkl" 2>/dev/null | wc -l)
    local metadata_count=$(find "$exp_dir" -name "model_metadata.json" 2>/dev/null | wc -l)
    
    if [[ $model_count -eq 0 ]]; then
      echo "❌ Latest experiment directory has no model files: $exp_dir"
      echo "❌ Refusing to continue with empty latest experiment."
      return 1
    fi
    
    echo "📁 Found latest experiment: $latest_experiment"
    echo "  Model files: $model_count"
    echo "  Metadata files: $metadata_count"
    export LATEST_EXPERIMENT="$latest_experiment"
    return 0
  fi
  
  # Fallback: Check for model files directly (flat structure: {Currency Name}_{horizon}/)
  local model_count
  model_count=$(find "$backup_dir" -mindepth 2 -maxdepth 2 -name "*.pkl" 2>/dev/null | wc -l)
  local metadata_count
  metadata_count=$(find "$backup_dir" -mindepth 2 -maxdepth 2 -name "model_metadata.json" 2>/dev/null | wc -l)
  
  if [[ $model_count -gt 0 ]]; then
    echo "📁 Found models in flat structure:"
    echo "  Model files: $model_count"
    echo "  Metadata files: $metadata_count"
    return 0
  fi
  
  echo "❌ No model files found in local backup"
  return 1
}

build_lambda_image_with_models() {
  local models_source_dir="$ROOT_DIR/s3_backup/models"
  local temp_models_dir=""
  
  # Check if local models are available, otherwise download from S3
  if [[ ! -d "$models_source_dir" ]]; then
    echo "Local models not found, downloading from S3..."
    mkdir -p "$models_source_dir"
    
    echo "Downloading models from s3://$DATA_LAKE_BUCKET_NAME/models/currency/ to $models_source_dir/currency/"
    "$AWS_CMD" s3 sync "s3://$DATA_LAKE_BUCKET_NAME/models/currency/" "$models_source_dir/currency/" --region "$REGION"
    
    if [[ $? -ne 0 ]]; then
      echo "❌ Failed to download models from S3"
      exit 1
    fi
    
    echo "✅ Models downloaded from S3"
  fi
  
  # Verify s3_backup/models exists
  if [[ ! -d "$models_source_dir" ]]; then
    echo "❌ s3_backup/models directory not found"
    exit 1
  fi
  
  # Check if we have a latest experiment to use
  local backup_dir="$ROOT_DIR/s3_backup/models/currency"
  if [[ -n "$LATEST_EXPERIMENT" ]] && [[ -d "$backup_dir/$LATEST_EXPERIMENT" ]]; then
    echo "Using models from latest experiment: $LATEST_EXPERIMENT"
    
    # Always isolate latest experiment to avoid stale nested directories
    # (e.g. previous accidental models/models copies) from leaking into the image.
    echo "Isolating latest experiment for Docker build: $LATEST_EXPERIMENT"
      
      # Create temporary directory with only the latest experiment's models
      temp_models_dir=$(mktemp -d)
      mkdir -p "$temp_models_dir/models/currency"
      
      # Copy only the latest experiment's models
      echo "Copying models from $LATEST_EXPERIMENT to temporary directory..."
      if ! cp -r "$backup_dir/$LATEST_EXPERIMENT" "$temp_models_dir/models/currency/" 2>/dev/null; then
        echo "❌ Failed to copy experiment models to temporary directory"
        echo "   Source: $backup_dir/$LATEST_EXPERIMENT"
        echo "   Destination: $temp_models_dir/models/currency/"
        rm -rf "$temp_models_dir"
        exit 1
      fi
      
      # Update models_source_dir to use temp directory
      models_source_dir="$temp_models_dir/models"
      echo "✅ Using latest experiment models only"
  elif [[ -n "${LATEST_EXPERIMENT:-}" ]]; then
    echo "❌ LATEST_EXPERIMENT is set but directory is missing: $backup_dir/$LATEST_EXPERIMENT"
    echo "❌ Refusing to fall back to all models to avoid oversized Lambda images."
    exit 1
  else
    echo "Using all models from s3_backup directory..."
    models_source_dir="$ROOT_DIR/s3_backup/models"
    temp_models_dir=""
  fi
  
  # Verify models are available
  local model_count
  model_count=$(find "$models_source_dir" -name "*.pkl" 2>/dev/null | wc -l)
  echo "📊 Total model files: $model_count"
  
  if [[ $model_count -eq 0 ]]; then
    echo "❌ No model files found in $models_source_dir"
    echo "   Expected structure: models/currency/{experiment_id}/{currency}_{horizon}/*.pkl"
    [[ -n "$temp_models_dir" ]] && rm -rf "$temp_models_dir"
    exit 1
  fi
  
  # Verify model structure - check for expected pattern
  local metadata_count
  metadata_count=$(find "$models_source_dir" -name "model_metadata.json" 2>/dev/null | wc -l)
  echo "📊 Model metadata files: $metadata_count"
  
  if [[ $metadata_count -eq 0 ]]; then
    echo "❌ No model metadata files found in $models_source_dir"
    echo "   Expected: models/currency/{experiment_id}/{currency}_{horizon}/model_metadata.json"
    echo "   Actual structure:"
    find "$models_source_dir" -type d -maxdepth 3 2>/dev/null | head -10
    [[ -n "$temp_models_dir" ]] && rm -rf "$temp_models_dir"
    exit 1
  fi
  
  # Verify structure matches expected pattern (currency_{horizon} directories)
  local currency_dirs
  currency_dirs=$(find "$models_source_dir" -type d -name "*_*d" 2>/dev/null | wc -l)
  if [[ $currency_dirs -eq 0 ]]; then
    echo "⚠️  Warning: No currency directories found with pattern '*_*d'"
    echo "   This may indicate a structure mismatch"
    echo "   Sample directories found:"
    find "$models_source_dir" -type d -maxdepth 3 2>/dev/null | head -5
  else
    echo "✅ Found $currency_dirs currency directories matching expected pattern"
  fi
  
  # Get models directory size for progress indication (calculate before any copying)
  local models_size_mb=0
  if command -v du >/dev/null 2>&1; then
    if [[ -d "$ROOT_DIR/s3_backup/models" ]]; then
      models_size_mb=$(du -sm "$ROOT_DIR/s3_backup/models" 2>/dev/null | cut -f1 || echo "0")
    elif [[ -d "$models_source_dir" ]]; then
      models_size_mb=$(du -sm "$models_source_dir" 2>/dev/null | cut -f1 || echo "0")
    fi
  fi
  
  # If using temp directory, replace s3_backup/models temporarily
  local original_models_backup=""
  if [[ -n "$temp_models_dir" ]] && [[ -d "$temp_models_dir/models" ]]; then
    # Backup original s3_backup/models if it exists
    if [[ -d "$ROOT_DIR/s3_backup/models" ]]; then
      original_models_backup="$ROOT_DIR/s3_backup/models.backup"
      echo "Backing up existing s3_backup/models..."
      rm -rf "$original_models_backup" 2>/dev/null || true
      # Prefer move for speed/atomicity; fallback to copy+remove.
      if mv "$ROOT_DIR/s3_backup/models" "$original_models_backup" 2>/dev/null; then
        :
      elif cp -a "$ROOT_DIR/s3_backup/models/." "$original_models_backup/" 2>/dev/null; then
        rm -rf "$ROOT_DIR/s3_backup/models" 2>/dev/null || true
      else
        echo "❌ Failed to backup s3_backup/models - permission denied or files in use"
        echo "   Try closing any applications that might be accessing the models directory"
        rm -rf "$temp_models_dir"
        exit 1
      fi
    fi
    # Copy temp models to s3_backup location for Docker build (more reliable than move on Windows/WSL)
    mkdir -p "$ROOT_DIR/s3_backup"
    echo "Copying models to s3_backup/models for Docker build (this may take a few minutes for large directories)..."
    
    # Check if we can write to the destination
    if [[ ! -w "$ROOT_DIR/s3_backup" ]] 2>/dev/null; then
      echo "❌ No write permission to $ROOT_DIR/s3_backup"
      echo "   Please check directory permissions or run with appropriate privileges"
      if [[ -n "$original_models_backup" ]] && [[ -d "$original_models_backup" ]]; then
        rm -rf "$ROOT_DIR/s3_backup/models" 2>/dev/null || true
        mkdir -p "$ROOT_DIR/s3_backup/models"
        cp -a "$original_models_backup/." "$ROOT_DIR/s3_backup/models/" 2>/dev/null || true
        rm -rf "$original_models_backup" 2>/dev/null || true
      fi
      rm -rf "$temp_models_dir"
      exit 1
    fi
    
    # Use rsync if available (faster and more reliable), otherwise use cp
    # Show progress for large directory copies
    echo "Copying ${models_size_mb}MB of models (this may take several minutes)..."
    
    if command -v rsync >/dev/null 2>&1; then
      # Use rsync with progress for large directories
      if rsync -a --info=progress2 "$temp_models_dir/models/" "$ROOT_DIR/s3_backup/models/" 2>/dev/null; then
        rm -rf "$temp_models_dir" 2>/dev/null || true
        echo "✅ Models copied successfully"
      else
        echo "❌ Failed to copy models using rsync - trying cp..."
        # Fallback to cp without progress (copy contents, not parent dir)
        mkdir -p "$ROOT_DIR/s3_backup/models"
        if ! cp -a "$temp_models_dir/models/." "$ROOT_DIR/s3_backup/models/" 2>/dev/null; then
          echo "❌ Failed to copy models to s3_backup/models - permission denied or disk full"
          echo "   Check:"
          echo "   1. Disk space: df -h"
          echo "   2. Directory permissions: ls -ld $ROOT_DIR/s3_backup"
          echo "   3. Close any applications accessing the models directory"
          # Try to restore backup if it existed
          if [[ -n "$original_models_backup" ]] && [[ -d "$original_models_backup" ]]; then
            rm -rf "$ROOT_DIR/s3_backup/models" 2>/dev/null || true
            mkdir -p "$ROOT_DIR/s3_backup/models"
            cp -a "$original_models_backup/." "$ROOT_DIR/s3_backup/models/" 2>/dev/null || true
            rm -rf "$original_models_backup" 2>/dev/null || true
          fi
          rm -rf "$temp_models_dir"
          exit 1
        fi
        rm -rf "$temp_models_dir" 2>/dev/null || true
        echo "✅ Models copied successfully (using cp fallback)"
      fi
    else
      # Fallback to cp if rsync not available
      echo "Using cp to copy models (rsync not available)..."
      mkdir -p "$ROOT_DIR/s3_backup/models"
      if cp -a "$temp_models_dir/models/." "$ROOT_DIR/s3_backup/models/" 2>/dev/null; then
        rm -rf "$temp_models_dir" 2>/dev/null || true
        echo "✅ Models copied successfully"
      else
        echo "❌ Failed to copy models to s3_backup/models - permission denied or disk full"
        echo "   Check:"
        echo "   1. Disk space: df -h"
        echo "   2. Directory permissions: ls -ld $ROOT_DIR/s3_backup"
        echo "   3. Close any applications accessing the models directory"
        # Try to restore backup if it existed
        if [[ -n "$original_models_backup" ]] && [[ -d "$original_models_backup" ]]; then
          rm -rf "$ROOT_DIR/s3_backup/models" 2>/dev/null || true
          mkdir -p "$ROOT_DIR/s3_backup/models"
          cp -a "$original_models_backup/." "$ROOT_DIR/s3_backup/models/" 2>/dev/null || true
          rm -rf "$original_models_backup" 2>/dev/null || true
        fi
        rm -rf "$temp_models_dir"
        exit 1
      fi
    fi
  fi
  
  # Update models directory size after copying (if we copied models)
  if [[ $models_size_mb -eq 0 ]] && command -v du >/dev/null 2>&1; then
    if [[ -d "$ROOT_DIR/s3_backup/models" ]]; then
      models_size_mb=$(du -sm "$ROOT_DIR/s3_backup/models" 2>/dev/null | cut -f1 || echo "0")
    fi
  fi
  
  # Verify model structure before building (after copying to s3_backup/models)
  echo "Verifying model structure before Docker build..."
  local metadata_files
  metadata_files=$(find "$ROOT_DIR/s3_backup/models" -name "model_metadata.json" 2>/dev/null | wc -l)
  if [[ $metadata_files -eq 0 ]]; then
    echo "❌ No model_metadata.json files found in s3_backup/models"
    echo "   Expected structure: s3_backup/models/currency/{experiment_id}/{currency}_{horizon}/model_metadata.json"
    echo "   Actual structure:"
    find "$ROOT_DIR/s3_backup/models" -type d -maxdepth 3 2>/dev/null | head -10
    exit 1
  fi
  echo "✅ Found $metadata_files model metadata files - structure looks correct"
  
  # Verify at least one currency directory exists
  local currency_model_dirs
  currency_model_dirs=$(find "$ROOT_DIR/s3_backup/models" -type d -name "*_*d" 2>/dev/null | wc -l)
  if [[ $currency_model_dirs -eq 0 ]]; then
    echo "❌ No currency model directories found with pattern '*_*d'"
    echo "   This indicates the model structure is incorrect"
    echo "   Expected: s3_backup/models/currency/{experiment_id}/{currency}_{horizon}/"
    echo "   Sample directories found:"
    find "$ROOT_DIR/s3_backup/models" -type d -maxdepth 4 2>/dev/null | head -10
    exit 1
  fi
  echo "✅ Found $currency_model_dirs currency model directories"
  
  # Build image with models using buildx for Lambda-compatible single-platform manifest
  echo "Building Lambda container image with models..."
  echo "Using buildx with --provenance=false for Lambda compatibility..."
  if [[ $models_size_mb -gt 0 ]]; then
    echo "Note: Copying models (${models_size_mb}MB) may take several minutes..."
  else
    echo "Note: Copying models may take several minutes..."
  fi
  
  # Check if buildx is available
  if ! command -v docker >/dev/null 2>&1 || ! docker buildx version >/dev/null 2>&1; then
    echo "⚠️  Warning: docker buildx not available, falling back to standard docker build"
    echo "   Note: This may create an incompatible manifest. Install buildx for Lambda compatibility."
    USE_BUILDX=false
  else
    # Ensure default builder exists
    if ! docker buildx ls | grep -q "default"; then
      echo "Creating default buildx builder..."
      docker buildx create --name default --use 2>/dev/null || true
    fi
    USE_BUILDX=true
  fi
  
  # Backup original .dockerignore and copy the appropriate one for lambda
  if [[ -f "$ROOT_DIR/.dockerignore" ]]; then
    cp "$ROOT_DIR/.dockerignore" "$ROOT_DIR/.dockerignore.backup"
  fi
  cp "$ROOT_DIR/aws/lambdas/prediction/container/.dockerignore" "$ROOT_DIR/.dockerignore" 2>/dev/null || true
  
  # Use cache unless FORCE_REBUILD is set (faster subsequent builds)
  local build_args=""
  if [[ "${FORCE_REBUILD:-false}" == "true" ]]; then
    build_args="--no-cache"
    echo "⚠️  Force rebuild enabled - this will take longer"
  else
    echo "ℹ️  Using Docker layer cache - faster builds"
  fi
  
  # Build with explicit platform for Lambda compatibility
  # Lambda requires single-platform manifest without provenance (BuildKit v0.10+ adds provenance by default)
  echo "Starting Docker build (this may take 10-30 minutes for large model directories)..."
  echo "Progress will be shown below. The build may appear slow during:"
  echo "  - COPY s3_backup/models (copying ${models_size_mb}MB of models)"
  echo "  - pip install (ML library installation)"
  echo ""
  
  if [[ "$USE_BUILDX" == "true" ]]; then
    # Use buildx with --load to ensure single-platform manifest compatible with Lambda
    # --provenance=false is required for Lambda (BuildKit v0.10+ adds provenance by default)
    # --sbom=false also recommended for Lambda compatibility
    if ! docker buildx build \
      --platform linux/amd64 \
      --progress=plain \
      --load \
      --provenance=false \
      --sbom=false \
      --tag "poeconomy-${ENVIRONMENT}-lambda:latest" \
      --file "$ROOT_DIR/aws/lambdas/prediction/container/Dockerfile" \
      $build_args \
      "$ROOT_DIR"; then
      echo "❌ Docker build failed"
      echo "Common issues:"
      echo "  - COPY operation timed out (models too large or network issues)"
      echo "  - Out of disk space (check with: docker system df)"
      echo "  - Permission denied (check s3_backup/models permissions)"
      exit 1
    fi
  else
    # Fallback to standard docker build (may not be Lambda-compatible)
    echo "⚠️  Using standard docker build (buildx not available)"
    echo "   This may create an incompatible manifest for Lambda"
    if ! DOCKER_BUILDKIT=0 docker build \
      --platform linux/amd64 \
      --tag "poeconomy-${ENVIRONMENT}-lambda:latest" \
      --file "$ROOT_DIR/aws/lambdas/prediction/container/Dockerfile" \
      $build_args \
      "$ROOT_DIR"; then
      echo "❌ Docker build failed"
      echo "Common issues:"
      echo "  - COPY operation timed out (models too large or network issues)"
      echo "  - Out of disk space (check with: docker system df)"
      echo "  - Permission denied (check s3_backup/models permissions)"
      exit 1
    fi
  fi
  
  if [[ $? -eq 0 ]]; then
    echo "✅ Docker image built successfully"
    
    # Verify the image architecture
    echo "Verifying image architecture..."
    local image_arch
    image_arch=$(docker inspect "poeconomy-${ENVIRONMENT}-lambda:latest" --format='{{.Architecture}}')
    echo "Image architecture: $image_arch"
    
    if [[ "$image_arch" != "amd64" ]]; then
      echo "❌ Image architecture is $image_arch, expected amd64"
      exit 1
    fi
    
    # Test the image can run (Lambda containers need handler argument, so test Python import instead)
    echo "Testing image functionality..."
    docker run --rm --entrypoint python "poeconomy-${ENVIRONMENT}-lambda:latest" -c "import sys; print(f'Python {sys.version}')"
    if [[ $? -eq 0 ]]; then
      echo "✅ Image test successful"
    else
      echo "❌ Image test failed"
      exit 1
    fi
  else
    echo "❌ Docker image build failed"
    exit 1
  fi
  
  # Restore original .dockerignore if it was backed up
  if [[ -f "$ROOT_DIR/.dockerignore.backup" ]]; then
    mv "$ROOT_DIR/.dockerignore.backup" "$ROOT_DIR/.dockerignore"
  fi
  
  # Restore original s3_backup/models if we used temp directory
  if [[ -n "$original_models_backup" ]] && [[ -d "$original_models_backup" ]]; then
    echo "Restoring original s3_backup/models..."
    # Remove current models directory
    rm -rf "$ROOT_DIR/s3_backup/models" 2>/dev/null || {
      echo "⚠️  Warning: Could not remove current models directory, attempting to restore anyway"
    }
    # Restore backup using copy + remove for better Windows/WSL compatibility
    mkdir -p "$ROOT_DIR/s3_backup/models"
    if cp -a "$original_models_backup/." "$ROOT_DIR/s3_backup/models/" 2>/dev/null; then
      rm -rf "$original_models_backup" 2>/dev/null || {
        echo "⚠️  Warning: Could not remove backup directory, but restoration successful"
      }
      echo "🧹 Restored original s3_backup/models directory"
    else
      echo "⚠️  Warning: Failed to restore original s3_backup/models - permission denied"
      echo "   Backup is still available at: $original_models_backup"
      echo "   You may need to manually restore it later"
    fi
  fi
  
  echo "🧹 Cleaned up temporary files"
}




build_and_push_feature_engineering_image() {
  local repository="poeconomy-${ENVIRONMENT}-feature-engineering"
  local timestamp=$(date +%Y%m%d%H%M%S)
  local ecr_base="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  
  ensure_ecr_repository "$repository"
  
  echo "Logging in to ECR ($ecr_base)"
  "$AWS_CMD" ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ecr_base"
  
  echo "Building feature engineering container image"
  # Backup original .dockerignore and copy the appropriate one for feature engineering
  if [[ -f "$ROOT_DIR/.dockerignore" ]]; then
    cp "$ROOT_DIR/.dockerignore" "$ROOT_DIR/.dockerignore.backup"
  fi
  cp "$ROOT_DIR/aws/feature-engineering/container/.dockerignore" "$ROOT_DIR/.dockerignore"
  
  docker build -t "poeconomy-${ENVIRONMENT}-feature-engineering:latest" \
    -f "$ROOT_DIR/aws/feature-engineering/container/Dockerfile" \
    "$ROOT_DIR"
  
  echo "Pushing feature engineering image to ECR"
  docker tag "poeconomy-${ENVIRONMENT}-feature-engineering:latest" "$ecr_base/$repository:latest"
  docker tag "poeconomy-${ENVIRONMENT}-feature-engineering:latest" "$ecr_base/$repository:$timestamp"
  docker push "$ecr_base/$repository:latest"
  docker push "$ecr_base/$repository:$timestamp"
  
  FEATURE_ENGINEERING_IMAGE_URI="$ecr_base/$repository:$timestamp"
  
  # Restore original .dockerignore if it was backed up
  if [[ -f "$ROOT_DIR/.dockerignore.backup" ]]; then
    mv "$ROOT_DIR/.dockerignore.backup" "$ROOT_DIR/.dockerignore"
  fi
}

build_and_push_training_image() {
  local repository="poeconomy-${ENVIRONMENT}-training"
  local timestamp=$(date +%Y%m%d%H%M%S)
  local ecr_base="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
  
  ensure_ecr_repository "$repository"
  
  echo "Logging in to ECR ($ecr_base)"
  "$AWS_CMD" ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ecr_base"
  
  echo "Building training container image"
  # Backup original .dockerignore and copy the appropriate one for training
  if [[ -f "$ROOT_DIR/.dockerignore" ]]; then
    cp "$ROOT_DIR/.dockerignore" "$ROOT_DIR/.dockerignore.backup"
  fi
  cp "$ROOT_DIR/aws/training/container/.dockerignore" "$ROOT_DIR/.dockerignore"
  
  docker build -t "poeconomy-${ENVIRONMENT}-training:latest" \
    -f "$ROOT_DIR/aws/training/container/Dockerfile" \
    "$ROOT_DIR"
  
  echo "Pushing training image to ECR"
  docker tag "poeconomy-${ENVIRONMENT}-training:latest" "$ecr_base/$repository:latest"
  docker tag "poeconomy-${ENVIRONMENT}-training:latest" "$ecr_base/$repository:$timestamp"
  docker push "$ecr_base/$repository:latest"
  docker push "$ecr_base/$repository:$timestamp"
  
  TRAINING_IMAGE_URI="$ecr_base/$repository:$timestamp"
  
  # Restore original .dockerignore if it was backed up
  if [[ -f "$ROOT_DIR/.dockerignore.backup" ]]; then
    mv "$ROOT_DIR/.dockerignore.backup" "$ROOT_DIR/.dockerignore"
  fi
}

upload_training_data_if_needed() {
  echo "========================================"
  echo "Checking training data in S3..."
  echo "========================================"
  
  # Check if training data already exists
  if "$AWS_CMD" s3 ls "s3://$DATA_LAKE_BUCKET_NAME/training-data/" --region "$REGION" >/dev/null 2>&1; then
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
  
  echo "Uploading training data from $training_data_dir to s3://$DATA_LAKE_BUCKET_NAME/training-data/"
  "$AWS_CMD" s3 sync "$training_data_dir" "s3://$DATA_LAKE_BUCKET_NAME/training-data/" --region "$REGION"
  echo "✅ Training data uploaded successfully"
}

# Function to package and upload Lambda functions
package_and_upload_lambda() {
    local lambda_name=$1
    local handler_file=$2
    local zip_name=$3
    
    echo "Packaging $lambda_name Lambda function..."
    
    # Try using the Python packaging script first (more robust)
    local python_script="$SCRIPT_DIR/package_lambda.py"
    if [[ -f "$python_script" ]]; then
        echo "Using Python packaging script for better reliability..."
        local temp_dir=$(python -c "import tempfile; print(tempfile.mkdtemp())")
        local temp_zip="$temp_dir/$zip_name"
        
        if python "$python_script" "$lambda_name" "$handler_file" "$temp_zip"; then
            echo "Python packaging script succeeded"
            
            # Upload to S3
            echo "Uploading $zip_name to S3..."
            if aws s3 cp "$temp_zip" "s3://$DATA_LAKE_BUCKET_NAME/lambda/"; then
                rm -rf "$temp_dir"
                echo "✅ $lambda_name packaged and uploaded successfully"
                return 0
            else
                echo "Error: Failed to upload $zip_name to S3"
                rm -rf "$temp_dir"
                return 1
            fi
        else
            echo "Python packaging script failed, falling back to shell script method"
            rm -rf "$temp_dir"
        fi
    fi
    
    # Fallback to original shell script method
    echo "Using shell script packaging method..."
    
    # Create temporary directory for packaging (Windows compatible)
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows environment
        local temp_dir=$(python -c "import tempfile; print(tempfile.mkdtemp())")
    else
        # Unix environment
        local temp_dir=$(mktemp -d)
    fi
    local package_dir="$temp_dir/$lambda_name"
    mkdir -p "$package_dir"
    
    # Determine the correct lambda directory based on function type
    local lambda_dir
    if [[ "$lambda_name" == "ingestion" || "$lambda_name" == "league_metadata" || "$lambda_name" == "daily_aggregation" ]]; then
        lambda_dir="$SCRIPT_DIR/../lambdas/ingestion"
    elif [[ "$lambda_name" == "api" ]]; then
        lambda_dir="$SCRIPT_DIR/../lambdas/api"
    elif [[ "$lambda_name" == "prediction_refresh" ]]; then
        lambda_dir="$SCRIPT_DIR/../lambdas/prediction"
    else
        lambda_dir="$SCRIPT_DIR/../lambdas"
    fi
    
    # Validate that required files exist
    if [[ ! -f "$lambda_dir/$handler_file" ]]; then
        echo "Error: Handler file not found: $lambda_dir/$handler_file"
        rm -rf "$temp_dir"
        return 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/../lambdas/config.py" ]]; then
        echo "Error: Config file not found: $SCRIPT_DIR/../lambdas/config.py"
        rm -rf "$temp_dir"
        return 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/../lambdas/__init__.py" ]]; then
        echo "Error: Init file not found: $SCRIPT_DIR/../lambdas/__init__.py"
        rm -rf "$temp_dir"
        return 1
    fi
    
    if [[ ! -f "$lambda_dir/requirements.txt" ]]; then
        echo "Error: Requirements file not found: $lambda_dir/requirements.txt"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Copy Lambda source files
    cp "$lambda_dir/$handler_file" "$package_dir/"
    cp "$SCRIPT_DIR/../lambdas/config.py" "$package_dir/"
    cp "$SCRIPT_DIR/../lambdas/__init__.py" "$package_dir/"
    cp "$lambda_dir/requirements.txt" "$package_dir/"
    
    # Copy ML modules that the Lambda functions depend on
    echo "Copying ML modules..."
    if ! python -c "
import shutil
import os
import sys

try:
    # Handle Git Bash path conversion on Windows
    root_dir_raw = r'$ROOT_DIR'
    package_dir_raw = r'$package_dir'
    
    # Convert Git Bash paths to Windows paths if needed
    if root_dir_raw.startswith('/c/'):
        root_dir = 'C:' + root_dir_raw[2:].replace('/', '\\\\')
    elif root_dir_raw.startswith('/'):
        # Handle other Unix-style paths on Windows
        root_dir = root_dir_raw[1:] + ':'
    else:
        root_dir = root_dir_raw
    
    # Normalize the package directory path
    package_dir = os.path.normpath(package_dir_raw)
    
    ml_source = os.path.join(root_dir, 'ml')
    ml_dest = os.path.join(package_dir, 'ml')
    
    print(f'Looking for ML source at: {ml_source}')
    print(f'Target ML destination: {ml_dest}')
    print(f'Root directory: {root_dir}')
    print(f'Root directory exists: {os.path.exists(root_dir)}')
    
    if os.path.exists(root_dir):
        print(f'Root directory contents: {os.listdir(root_dir)}')
    
    if os.path.exists(ml_source):
        if os.path.exists(ml_dest):
            shutil.rmtree(ml_dest)
        shutil.copytree(ml_source, ml_dest)
        print('ML modules copied successfully')
        
        # Verify the copy was successful
        if os.path.exists(ml_dest):
            print(f'ML destination exists: {os.path.exists(ml_dest)}')
            print(f'ML destination contents: {os.listdir(ml_dest)}')
        else:
            print('Error: ML destination does not exist after copy')
            sys.exit(1)
    else:
        print(f'Error: ML source directory not found: {ml_source}')
        print(f'Root directory exists: {os.path.exists(root_dir)}')
        if os.path.exists(root_dir):
            print(f'Root directory contents: {os.listdir(root_dir)}')
        sys.exit(1)
except Exception as e:
    print(f'Error copying ML modules: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"; then
        echo "Error: Failed to copy ML modules"
        rm -rf "$temp_dir"
        return 1
    fi
    
    echo "Using requirements from $lambda_dir for $lambda_name"
    
    # Try to install dependencies, but continue if it fails
    echo "Installing dependencies for $lambda_name..."
    if pip install -r "$package_dir/requirements.txt" -t "$package_dir/" --quiet 2>/dev/null; then
        echo "Dependencies installed successfully"
    else
        echo "Warning: Some dependencies failed to install, continuing with basic package..."
        # Install only basic dependencies that are likely to work
        pip install boto3 requests -t "$package_dir/" --quiet 2>/dev/null || true
    fi
    
    # Verify ML modules are present in the package
    if [[ ! -d "$package_dir/ml" ]]; then
        echo "Warning: ML modules not found in package directory"
        echo "Package directory contents:"
        ls -la "$package_dir/" || true
        echo "This may cause import errors in the Lambda function"
    else
        echo "ML modules verified in package directory"
        echo "ML directory contents:"
        ls -la "$package_dir/ml/" || true
    fi
    
    # Create zip file using Python (cross-platform compatible)
    echo "Creating zip file..."
    if ! python -c "
import zipfile
import os
import sys

try:
    # Use raw strings and proper path handling for Windows
    package_dir = r'$package_dir'
    zip_path = r'$temp_dir/$zip_name'
    
    # Ensure the zip file directory exists
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_dir)
                # Convert Windows backslashes to forward slashes for zip archive
                arcname = arcname.replace(os.sep, '/')
                zipf.write(file_path, arcname)
    print('Zip file created successfully')
except Exception as e:
    print(f'Error creating zip file: {e}')
    sys.exit(1)
"; then
        echo "Error: Failed to create zip file"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Upload to S3
    echo "Uploading $zip_name to S3..."
    if ! aws s3 cp "$temp_dir/$zip_name" "s3://$DATA_LAKE_BUCKET_NAME/lambda/"; then
        echo "Error: Failed to upload $zip_name to S3"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Clean up
    rm -rf "$temp_dir"
    
    echo "✅ $lambda_name packaged and uploaded successfully"
}

update_env_file() {
  echo "Updating .env file with current stack outputs..."
  
  # Start building new .env file
  cat > "$ENV_FILE" << EOF
# Generated by shared_config.sh - $(date)
AWS_REGION=$REGION
DATA_LAKE_BUCKET=$DATA_LAKE_BUCKET_NAME
MODELS_BUCKET=$DATA_LAKE_BUCKET_NAME
EOF

  # Get Base stack outputs
  if "$AWS_CMD" cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Base stack outputs..."
    
    # Base infrastructure resources
    DATA_LAKE_BUCKET_ARN=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='DataLakeBucketArn'].OutputValue" --output text)
    TRAINING_DATA_LAKE_ROLE_ARN=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='TrainingDataLakeRoleArn'].OutputValue" --output text)
    FEATURE_ENGINEERING_DATA_LAKE_ROLE_ARN=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$BASE_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringDataLakeRoleArn'].OutputValue" --output text)
    
    # Append Base outputs to .env file
    cat >> "$ENV_FILE" << EOF

# Base Infrastructure Resources
DATA_LAKE_BUCKET_ARN=$DATA_LAKE_BUCKET_ARN
TRAINING_DATA_LAKE_ROLE_ARN=$TRAINING_DATA_LAKE_ROLE_ARN
FEATURE_ENGINEERING_DATA_LAKE_ROLE_ARN=$FEATURE_ENGINEERING_DATA_LAKE_ROLE_ARN
EOF
  fi

  # Get Ingestion stack outputs
  if "$AWS_CMD" cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Ingestion stack outputs..."
    
    # DynamoDB tables from ingestion stack
    DYNAMO_CURRENCY_METADATA_TABLE=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='CurrencyMetadataTableName'].OutputValue" --output text)
    DYNAMO_CURRENCY_PRICES_TABLE=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='LivePricesTableName'].OutputValue" --output text)
    DYNAMO_LEAGUE_METADATA_TABLE=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='LeagueMetadataTableName'].OutputValue" --output text)
    
    # Lambda function names from ingestion stack
    INGESTION_LAMBDA_NAME=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='IngestionLambdaName'].OutputValue" --output text)
    LEAGUE_METADATA_LAMBDA_NAME=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='LeagueMetadataLambdaName'].OutputValue" --output text)
    DAILY_AGGREGATION_LAMBDA_NAME=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$INGESTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='DailyAggregationLambdaName'].OutputValue" --output text)
  fi

  # Get Prediction stack outputs
  if "$AWS_CMD" cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Prediction stack outputs..."
    
    # DynamoDB table from prediction stack
    DYNAMO_PREDICTIONS_TABLE=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='PredictionsTableName'].OutputValue" --output text)
    
    # Lambda function name from prediction stack
    PREDICTION_REFRESH_LAMBDA_NAME=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='PredictionRefreshLambdaName'].OutputValue" --output text)
    
    # Lambda inference image from prediction stack
    LAMBDA_INFERENCE_IMAGE=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$PREDICTION_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='LambdaInferenceImage'].OutputValue" --output text)
  fi

  # Get API stack outputs
  if "$AWS_CMD" cloudformation describe-stacks --stack-name "$API_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting API stack outputs..."
    
    # API URL from API stack
    API_BASE_URL=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$API_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='ApiInvokeUrl'].OutputValue" --output text)
    
    # Lambda function name from API stack
    PREDICTION_API_LAMBDA_NAME=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$API_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='PredictionApiLambdaName'].OutputValue" --output text)
  fi

  # Append Lambda outputs to .env file (only if we have values)
  if [ -n "${DYNAMO_CURRENCY_METADATA_TABLE:-}" ] || [ -n "${DYNAMO_CURRENCY_PRICES_TABLE:-}" ] || [ -n "${DYNAMO_PREDICTIONS_TABLE:-}" ] || [ -n "${DYNAMO_LEAGUE_METADATA_TABLE:-}" ] || [ -n "${API_BASE_URL:-}" ] || [ -n "${LAMBDA_INFERENCE_IMAGE:-}" ] || [ -n "${PREDICTION_API_LAMBDA_NAME:-}" ] || [ -n "${INGESTION_LAMBDA_NAME:-}" ] || [ -n "${LEAGUE_METADATA_LAMBDA_NAME:-}" ] || [ -n "${DAILY_AGGREGATION_LAMBDA_NAME:-}" ] || [ -n "${PREDICTION_REFRESH_LAMBDA_NAME:-}" ]; then
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
  
  # Get Feature Engineering stack outputs
  if "$AWS_CMD" cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Feature Engineering stack outputs..."
    
    # Feature Engineering resources
    FEATURE_ENGINEERING_CLUSTER_NAME=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringClusterName'].OutputValue" --output text)
    FEATURE_ENGINEERING_TASK_DEFINITION=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringTaskDefinitionArn'].OutputValue" --output text)
    FEATURE_ENGINEERING_VPC_ID=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$FEATURE_ENGINEERING_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='FeatureEngineeringVpcId'].OutputValue" --output text)
    
    # Get subnet IDs from EC2 for feature engineering
    FE_SUBNET_A=$("$AWS_CMD" ec2 describe-subnets --region "$REGION" --filters "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-feature-engineering-public-subnet-a" --query "Subnets[0].SubnetId" --output text)
    FE_SUBNET_B=$("$AWS_CMD" ec2 describe-subnets --region "$REGION" --filters "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-feature-engineering-public-subnet-b" --query "Subnets[0].SubnetId" --output text)
    FEATURE_ENGINEERING_SUBNET_IDS="${FE_SUBNET_A},${FE_SUBNET_B}"
    
    # Get security group ID for feature engineering
    FEATURE_ENGINEERING_SECURITY_GROUP_ID=$("$AWS_CMD" ec2 describe-security-groups --region "$REGION" --filters "Name=group-name,Values=*feature-engineering*" --query "SecurityGroups[0].GroupId" --output text)
    
    # Append Feature Engineering outputs to .env file
    cat >> "$ENV_FILE" << EOF

# ECS Feature Engineering Resources
FEATURE_ENGINEERING_CLUSTER_NAME=$FEATURE_ENGINEERING_CLUSTER_NAME
FEATURE_ENGINEERING_TASK_DEFINITION=$FEATURE_ENGINEERING_TASK_DEFINITION
FEATURE_ENGINEERING_VPC_ID=$FEATURE_ENGINEERING_VPC_ID
FEATURE_ENGINEERING_SUBNET_IDS=$FEATURE_ENGINEERING_SUBNET_IDS
FEATURE_ENGINEERING_SECURITY_GROUP_ID=$FEATURE_ENGINEERING_SECURITY_GROUP_ID
EOF
  fi
  
  # Get Training stack outputs
  if "$AWS_CMD" cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Getting Training stack outputs..."
    
    # Training resources
    TRAINING_CLUSTER_NAME=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='TrainingClusterName'].OutputValue" --output text)
    TRAINING_TASK_DEFINITION=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='TrainingTaskDefinitionArn'].OutputValue" --output text)
    TRAINING_VPC_ID=$("$AWS_CMD" cloudformation describe-stacks --stack-name "$TRAINING_STACK_NAME" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='TrainingVpcId'].OutputValue" --output text)
    
    # Get subnet IDs from EC2
    SUBNET_A=$("$AWS_CMD" ec2 describe-subnets --region "$REGION" --filters "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-training-public-subnet-a" --query "Subnets[0].SubnetId" --output text)
    SUBNET_B=$("$AWS_CMD" ec2 describe-subnets --region "$REGION" --filters "Name=tag:Name,Values=poeconomy-${ENVIRONMENT}-training-public-subnet-b" --query "Subnets[0].SubnetId" --output text)
    TRAINING_SUBNET_IDS="${SUBNET_A},${SUBNET_B}"
    
    # Get security group ID
    TRAINING_SECURITY_GROUP_ID=$("$AWS_CMD" ec2 describe-security-groups --region "$REGION" --filters "Name=group-name,Values=*training*" --query "SecurityGroups[0].GroupId" --output text)
    
    # Append Training outputs to .env file
    cat >> "$ENV_FILE" << EOF

# ECS Training Resources
TRAINING_CLUSTER_NAME=$TRAINING_CLUSTER_NAME
TRAINING_TASK_DEFINITION=$TRAINING_TASK_DEFINITION
TRAINING_VPC_ID=$TRAINING_VPC_ID
TRAINING_SUBNET_IDS=$TRAINING_SUBNET_IDS
TRAINING_SECURITY_GROUP_ID=$TRAINING_SECURITY_GROUP_ID
EOF
  fi
  
  # Get EventBridge schedule information
  echo "Getting EventBridge schedule information..."
  FEATURE_ENGINEERING_SCHEDULE=$("$AWS_CMD" events list-rules --region "$REGION" --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-feature-engineering-schedule'].ScheduleExpression" --output text)
  INGESTION_SCHEDULE=$("$AWS_CMD" events list-rules --region "$REGION" --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-ingestion-schedule'].ScheduleExpression" --output text)
  LEAGUE_METADATA_SCHEDULE=$("$AWS_CMD" events list-rules --region "$REGION" --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-league-metadata-schedule'].ScheduleExpression" --output text)
  TRAINING_SCHEDULE=$("$AWS_CMD" events list-rules --region "$REGION" --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-training-schedule'].ScheduleExpression" --output text)
  PREDICTION_SCHEDULE=$("$AWS_CMD" events list-rules --region "$REGION" --query "Rules[?Name=='poeconomy-${ENVIRONMENT}-prediction-schedule'].ScheduleExpression" --output text)

  # Get CloudWatch Log Groups
  echo "Getting CloudWatch Log Groups..."
  FEATURE_ENGINEERING_LOG_GROUP="/aws/ecs/${FEATURE_ENGINEERING_STACK_NAME}"
  TRAINING_LOG_GROUP="/aws/ecs/${TRAINING_STACK_NAME}"
  API_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-api"
  INGESTION_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-ingestion"
  LEAGUE_METADATA_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-league-metadata"
  DAILY_AGGREGATION_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-daily-aggregation"
  PREDICTION_REFRESH_LAMBDA_LOG_GROUP="/aws/lambda/poeconomy-${ENVIRONMENT}-prediction-refresh"

  # Add metadata section with latest image information and schedules
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
