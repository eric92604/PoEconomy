#!/bin/bash

# ECR image build and push functions for PoEconomy.
# Source shared_config.sh (and lib/models.sh for lambda image) before this file.

build_and_push_lambda_image() {
  local repository="poeconomy-${ENVIRONMENT}-lambdas"
  ensure_ecr_repository "$repository"

  echo "Logging in to ECR ($ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com)"
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

  echo "Building Lambda container image with models..."
  # Uses LATEST_EXPERIMENT / LATEST_EXPERIMENT_FROM_S3_XP_LISTING when set by verify_local_backup (no second S3 listing).
  build_lambda_image_with_models

  local timestamp
  timestamp=$(date +%Y%m%d%H%M%S)
  local ecr_base="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

  docker tag "poeconomy-${ENVIRONMENT}-lambda:latest" "$ecr_base/$repository:latest"
  docker tag "poeconomy-${ENVIRONMENT}-lambda:latest" "$ecr_base/$repository:$timestamp"

  echo "Pushing images to ECR..."
  docker push "$ecr_base/$repository:latest"
  docker push "$ecr_base/$repository:$timestamp"

  # Verify the pushed image
  echo "Verifying pushed image manifest..."
  aws ecr describe-images \
    --repository-name "$repository" \
    --image-ids imageTag=latest \
    --region "$REGION" \
    --query 'imageDetails[0].{Digest:imageDigest,Size:imageSizeInBytes,PushedAt:imagePushedAt}' \
    --output table

  local pulled_arch
  pulled_arch=$(docker inspect "$ecr_base/$repository:$timestamp" --format='{{.Architecture}}' 2>/dev/null \
    || { docker pull "$ecr_base/$repository:$timestamp" >/dev/null 2>&1; \
         docker inspect "$ecr_base/$repository:$timestamp" --format='{{.Architecture}}'; })
  if [[ "$pulled_arch" != "amd64" ]]; then
    echo "❌ Image architecture is $pulled_arch, expected amd64"
    exit 1
  fi

  echo "✅ ECR image verified (amd64)"
  INFERENCE_IMAGE_URI="$ecr_base/$repository:$timestamp"
}

build_and_push_feature_engineering_image() {
  local repository="poeconomy-${ENVIRONMENT}-feature-engineering"
  local timestamp
  timestamp=$(date +%Y%m%d%H%M%S)
  local ecr_base="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

  ensure_ecr_repository "$repository"

  echo "Logging in to ECR ($ecr_base)"
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ecr_base"

  echo "Building feature engineering container image..."
  [[ -f "$ROOT_DIR/.dockerignore" ]] && cp "$ROOT_DIR/.dockerignore" "$ROOT_DIR/.dockerignore.bak"
  cp "$ROOT_DIR/aws/feature-engineering/container/.dockerignore" "$ROOT_DIR/.dockerignore"

  docker buildx build \
    --platform linux/amd64 \
    --progress=plain \
    --load \
    --provenance=false \
    --sbom=false \
    -t "poeconomy-${ENVIRONMENT}-feature-engineering:latest" \
    -f "$ROOT_DIR/aws/feature-engineering/container/Dockerfile" \
    "$ROOT_DIR"

  [[ -f "$ROOT_DIR/.dockerignore.bak" ]] && mv "$ROOT_DIR/.dockerignore.bak" "$ROOT_DIR/.dockerignore"

  echo "Pushing feature engineering image to ECR..."
  docker tag "poeconomy-${ENVIRONMENT}-feature-engineering:latest" "$ecr_base/$repository:latest"
  docker tag "poeconomy-${ENVIRONMENT}-feature-engineering:latest" "$ecr_base/$repository:$timestamp"
  docker push "$ecr_base/$repository:latest"
  docker push "$ecr_base/$repository:$timestamp"

  FEATURE_ENGINEERING_IMAGE_URI="$ecr_base/$repository:$timestamp"
}

build_and_push_training_image() {
  local repository="poeconomy-${ENVIRONMENT}-training"
  local timestamp
  timestamp=$(date +%Y%m%d%H%M%S)
  local ecr_base="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

  ensure_ecr_repository "$repository"

  echo "Logging in to ECR ($ecr_base)"
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ecr_base"

  echo "Building training container image..."
  [[ -f "$ROOT_DIR/.dockerignore" ]] && cp "$ROOT_DIR/.dockerignore" "$ROOT_DIR/.dockerignore.bak"
  cp "$ROOT_DIR/aws/training/container/.dockerignore" "$ROOT_DIR/.dockerignore"

  docker buildx build \
    --platform linux/amd64 \
    --progress=plain \
    --load \
    --provenance=false \
    --sbom=false \
    -t "poeconomy-${ENVIRONMENT}-training:latest" \
    -f "$ROOT_DIR/aws/training/container/Dockerfile" \
    "$ROOT_DIR"

  [[ -f "$ROOT_DIR/.dockerignore.bak" ]] && mv "$ROOT_DIR/.dockerignore.bak" "$ROOT_DIR/.dockerignore"

  echo "Pushing training image to ECR..."
  docker tag "poeconomy-${ENVIRONMENT}-training:latest" "$ecr_base/$repository:latest"
  docker tag "poeconomy-${ENVIRONMENT}-training:latest" "$ecr_base/$repository:$timestamp"
  docker push "$ecr_base/$repository:latest"
  docker push "$ecr_base/$repository:$timestamp"

  TRAINING_IMAGE_URI="$ecr_base/$repository:$timestamp"
}
