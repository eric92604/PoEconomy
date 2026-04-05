#!/bin/bash

# Model management functions for PoEconomy Lambda image builds.
# Source shared_config.sh before this file.

# Find the most recent experiment directory (xp_YYYYMMDD_HHMMSS) that contains .pkl files.
# Prints the experiment directory name on success, returns 1 if none found.
find_latest_experiment() {
  local backup_dir="$ROOT_DIR/s3_backup/models/currency"
  local latest_experiment=""
  local latest_timestamp=0

  for exp_dir in "$backup_dir"/xp_*; do
    [[ -d "$exp_dir" ]] || continue
    local exp_name
    exp_name=$(basename "$exp_dir")
    local exp_models
    exp_models=$(find "$exp_dir" -name "*.pkl" 2>/dev/null | wc -l)
    [[ "$exp_models" -gt 0 ]] || continue

    # Parse timestamp from xp_YYYYMMDD_HHMMSS
    local timestamp_str="${exp_name#xp_}"
    timestamp_str="${timestamp_str/_/ }"
    local epoch_time=0
    if [[ "$timestamp_str" =~ ^[0-9]{8}\ [0-9]{6}$ ]]; then
      epoch_time=$(date -d "$timestamp_str" +%s 2>/dev/null || echo "0")
    fi
    # Fallback: use digits from directory name for ordering
    if [[ "$epoch_time" -eq 0 ]]; then
      local digits="${exp_name//[^0-9]/}"
      epoch_time="${digits:0:14}"
    fi

    if [[ "$epoch_time" -gt "$latest_timestamp" ]]; then
      latest_timestamp=$epoch_time
      latest_experiment="$exp_name"
    fi
  done

  if [[ -n "$latest_experiment" ]]; then
    echo "$latest_experiment"
    return 0
  fi
  return 1
}

# Verify local model backup exists and set LATEST_EXPERIMENT if an xp_* directory is found.
verify_local_backup() {
  echo "Checking for local model backup..."
  local backup_dir="$ROOT_DIR/s3_backup/models/currency"

  if [[ ! -d "$backup_dir" ]]; then
    echo "❌ Local backup directory not found: $backup_dir"
    return 1
  fi

  echo "✅ Local backup directory found: $backup_dir"

  local latest_experiment
  latest_experiment=$(find_latest_experiment)
  if [[ -n "$latest_experiment" ]]; then
    local exp_dir="$backup_dir/$latest_experiment"
    local model_count metadata_count
    model_count=$(find "$exp_dir" -name "*.pkl" 2>/dev/null | wc -l)
    metadata_count=$(find "$exp_dir" -name "model_metadata.json" 2>/dev/null | wc -l)
    echo "📁 Latest experiment: $latest_experiment ($model_count models, $metadata_count metadata files)"
    export LATEST_EXPERIMENT="$latest_experiment"
    return 0
  fi

  # Fallback: flat model structure (no experiment directories)
  local model_count
  model_count=$(find "$backup_dir" -mindepth 2 -maxdepth 2 -name "*.pkl" 2>/dev/null | wc -l)
  if [[ "$model_count" -gt 0 ]]; then
    echo "📁 Found $model_count models in flat structure"
    return 0
  fi

  echo "❌ No model files found in local backup"
  return 1
}

# Build a Lambda container image with the latest experiment's models baked in.
# Isolates the latest experiment by atomically swapping s3_backup/models so the Docker
# build context contains only that experiment — no rsync, no Windows path workarounds.
build_lambda_image_with_models() {
  local models_dir="$ROOT_DIR/s3_backup/models"
  local models_bak=""  # non-empty if we renamed models_dir for isolation

  # Download models from S3 if not present locally
  if [[ ! -d "$models_dir" ]]; then
    echo "Local models not found, downloading from S3..."
    mkdir -p "$models_dir"
    aws s3 sync "s3://$DATA_LAKE_BUCKET_NAME/models/currency/" "$models_dir/currency/" --region "$REGION"
    echo "✅ Models downloaded from S3"
  fi

  # Isolate the latest experiment if multiple experiments exist
  if [[ -n "${LATEST_EXPERIMENT:-}" ]] && [[ -d "$models_dir/currency/$LATEST_EXPERIMENT" ]]; then
    local n_experiments
    n_experiments=$(find "$models_dir/currency" -mindepth 1 -maxdepth 1 -type d -name "xp_*" 2>/dev/null | wc -l)

    if [[ "$n_experiments" -gt 1 ]]; then
      echo "Multiple experiments found ($n_experiments), isolating $LATEST_EXPERIMENT for Docker build..."

      # Stage isolated copy alongside the original (same filesystem = fast cp, instant mv)
      local staged_dir="$ROOT_DIR/s3_backup/models.staged"
      rm -rf "$staged_dir"
      mkdir -p "$staged_dir/currency"
      cp -r "$models_dir/currency/$LATEST_EXPERIMENT" "$staged_dir/currency/"

      # Atomic rename swap: original → .bak, staged → models
      models_bak="$ROOT_DIR/s3_backup/models.bak"
      rm -rf "$models_bak"
      mv "$models_dir" "$models_bak"
      mv "$staged_dir" "$models_dir"
      echo "✅ Using isolated experiment: $LATEST_EXPERIMENT"
    fi
  fi

  _models_cleanup() {
    if [[ -n "$models_bak" ]] && [[ -d "$models_bak" ]]; then
      rm -rf "$models_dir"
      mv "$models_bak" "$models_dir"
      echo "🧹 Restored original models directory"
    fi
    if [[ -f "$ROOT_DIR/.dockerignore.bak" ]]; then
      mv "$ROOT_DIR/.dockerignore.bak" "$ROOT_DIR/.dockerignore"
    fi
  }

  # Verify models before building
  local model_count metadata_count
  model_count=$(find "$models_dir" -name "*.pkl" 2>/dev/null | wc -l)
  metadata_count=$(find "$models_dir" -name "model_metadata.json" 2>/dev/null | wc -l)
  echo "📊 Model files: $model_count, Metadata files: $metadata_count"

  if [[ "$model_count" -eq 0 ]]; then
    echo "❌ No model files found in $models_dir"
    _models_cleanup
    exit 1
  fi
  if [[ "$metadata_count" -eq 0 ]]; then
    echo "❌ No model_metadata.json files found in $models_dir"
    _models_cleanup
    exit 1
  fi

  # Swap .dockerignore to the lambda-specific one
  [[ -f "$ROOT_DIR/.dockerignore" ]] && cp "$ROOT_DIR/.dockerignore" "$ROOT_DIR/.dockerignore.bak"
  cp "$ROOT_DIR/aws/lambdas/prediction/container/.dockerignore" "$ROOT_DIR/.dockerignore"

  local build_args=()
  [[ "${FORCE_REBUILD:-false}" == "true" ]] && build_args+=(--no-cache)

  local models_size_mb=0
  models_size_mb=$(du -sm "$models_dir" 2>/dev/null | cut -f1 || echo "0")
  echo "Building Lambda container image (linux/amd64, ~${models_size_mb}MB models)..."

  if ! docker buildx build \
    --platform linux/amd64 \
    --progress=plain \
    --load \
    --provenance=false \
    --sbom=false \
    --tag "poeconomy-${ENVIRONMENT}-lambda:latest" \
    --file "$ROOT_DIR/aws/lambdas/prediction/container/Dockerfile" \
    "${build_args[@]}" \
    "$ROOT_DIR"; then
    echo "❌ Docker build failed"
    _models_cleanup
    exit 1
  fi

  echo "✅ Docker image built successfully"

  # Smoke-test the image
  docker run --rm --entrypoint python "poeconomy-${ENVIRONMENT}-lambda:latest" \
    -c "import sys; print(f'Python {sys.version}')" \
  || { _models_cleanup; echo "❌ Image smoke test failed"; exit 1; }

  _models_cleanup
}
