#!/bin/bash

# Model management functions for PoEconomy Lambda image builds.
# Source shared_config.sh before this file.

# Pick the newest experiment name by xp_YYYYMMDD_HHMMSS sort key (same ordering as training folders).
# Args: one or more directory names (e.g. xp_20260406_004633). Prints the winner; returns 1 if none valid.
pick_latest_experiment_name() {
  local latest_experiment="" latest_sort_key=0
  local exp_name sort_key rest digits
  for exp_name in "$@"; do
    sort_key=0
    rest="${exp_name#xp_}"
    if [[ "$rest" =~ ^([0-9]{8})_([0-9]{6})$ ]]; then
      sort_key="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    else
      digits="${exp_name//[^0-9]/}"
      [[ -n "$digits" ]] && sort_key="${digits:0:14}"
    fi
    [[ "$sort_key" =~ ^[0-9]+$ ]] || sort_key=0
    [[ "$sort_key" -gt "$latest_sort_key" ]] || continue
    latest_sort_key=$sort_key
    latest_experiment="$exp_name"
  done
  [[ -n "$latest_experiment" ]] || return 1
  echo "$latest_experiment"
}

# List s3://$DATA_LAKE_BUCKET_NAME/models/currency/ for xp_* prefixes only (no object download).
# Prints the newest experiment folder name.
# Returns 0 on success, 1 if no xp_* prefixes (legacy layout or empty), 2 if the AWS API call failed.
find_latest_experiment_in_s3() {
  local prefixes prefix base names=()
  if ! prefixes=$(aws s3api list-objects-v2 \
    --bucket "$DATA_LAKE_BUCKET_NAME" \
    --prefix "models/currency/" \
    --delimiter "/" \
    --region "$REGION" \
    --query 'CommonPrefixes[].Prefix' \
    --output text); then
    echo "❌ Failed to list S3 prefixes under models/currency/ (check AWS credentials and bucket)." >&2
    return 2
  fi
  [[ -z "$prefixes" ]] && return 1
  for prefix in $prefixes; do
    [[ -z "$prefix" ]] && continue
    base=$(basename "${prefix%/}")
    [[ "$base" == xp_* ]] || continue
    names+=("$base")
  done
  [[ ${#names[@]} -eq 0 ]] && return 1
  pick_latest_experiment_name "${names[@]}"
}

# Find the most recent experiment directory under local backup (xp_YYYYMMDD_HHMMSS) that contains .pkl files.
# Prints the experiment directory name on success, returns 1 if none found.
find_latest_experiment() {
  local backup_dir="$ROOT_DIR/s3_backup/models/currency"
  local names=() exp_dir exp_name exp_models

  for exp_dir in "$backup_dir"/xp_*; do
    [[ -d "$exp_dir" ]] || continue
    exp_name=$(basename "$exp_dir")
    exp_models=$(find "$exp_dir" -name "*.pkl" 2>/dev/null | wc -l)
    [[ "$exp_models" -gt 0 ]] || continue
    names+=("$exp_name")
  done
  [[ ${#names[@]} -eq 0 ]] && return 1
  pick_latest_experiment_name "${names[@]}"
}

# Verify local model backup: when S3 uses xp_* folders, S3 listing is authoritative — local must
# contain that latest folder with .pkl files or it is treated as stale (build will sync that prefix).
verify_local_backup() {
  echo "Checking for local model backup..."
  unset LATEST_EXPERIMENT LATEST_EXPERIMENT_FROM_S3_XP_LISTING
  local backup_dir="$ROOT_DIR/s3_backup/models/currency"

  if [[ ! -d "$backup_dir" ]]; then
    echo "❌ Local backup directory not found: $backup_dir"
    return 1
  fi

  echo "✅ Local backup directory found: $backup_dir"

  local s3_latest local_latest model_count metadata_count exp_dir s3_rc

  s3_latest=$(find_latest_experiment_in_s3)
  s3_rc=$?
  if [[ "$s3_rc" -eq 0 ]]; then
    export LATEST_EXPERIMENT="$s3_latest"
    export LATEST_EXPERIMENT_FROM_S3_XP_LISTING=1
    echo "📡 Latest experiment in S3: $s3_latest"
    exp_dir="$backup_dir/$s3_latest"
    model_count=0
    metadata_count=0
    if [[ -d "$exp_dir" ]]; then
      model_count=$(find "$exp_dir" -name "*.pkl" 2>/dev/null | wc -l)
      metadata_count=$(find "$exp_dir" -name "model_metadata.json" 2>/dev/null | wc -l)
    fi
    if [[ -d "$exp_dir" ]] && [[ "$model_count" -gt 0 ]]; then
      echo "📁 Local copy matches S3 latest: $s3_latest ($model_count models, $metadata_count metadata files)"
      return 0
    fi
    if local_latest=$(find_latest_experiment); then
      echo "⚠️  Local newest experiment is $local_latest but S3 newest is $s3_latest — local tree is stale or incomplete."
    else
      echo "⚠️  S3 latest $s3_latest is missing locally or has no .pkl files."
    fi
    return 1
  fi
  if [[ "$s3_rc" -eq 2 ]]; then
    echo "❌ Cannot validate local models against S3 (S3 listing failed)." >&2
    return 1
  fi

  echo "ℹ️  No xp_* experiment prefixes found in S3; validating local layout only."

  local latest_experiment
  latest_experiment=$(find_latest_experiment)
  if [[ -n "$latest_experiment" ]]; then
    exp_dir="$backup_dir/$latest_experiment"
    model_count=$(find "$exp_dir" -name "*.pkl" 2>/dev/null | wc -l)
    metadata_count=$(find "$exp_dir" -name "model_metadata.json" 2>/dev/null | wc -l)
    echo "📁 Latest experiment (local only): $latest_experiment ($model_count models, $metadata_count metadata files)"
    export LATEST_EXPERIMENT="$latest_experiment"
    export LATEST_EXPERIMENT_FROM_S3_XP_LISTING=0
    return 0
  fi

  # Fallback: flat model structure (no experiment directories)
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
#
# When deploy_prediction.sh has already run verify_local_backup, LATEST_EXPERIMENT and
# LATEST_EXPERIMENT_FROM_S3_XP_LISTING are set — this function skips another S3 listing
# and uses those values. Standalone invocation still discovers the experiment here.
build_lambda_image_with_models() {
  local models_dir="$ROOT_DIR/s3_backup/models"
  local models_bak=""  # non-empty if we renamed models_dir for isolation
  local s3_latest s3_rc resolved_experiment target_dir n_pkls

  mkdir -p "$models_dir/currency"

  resolved_experiment="${LATEST_EXPERIMENT:-}"
  if [[ -n "$resolved_experiment" ]]; then
    echo "Using latest experiment from deployment prep: $resolved_experiment"
  else
    # S3-first: latest experiment folder name from listing only; sync that prefix if missing locally.
    s3_latest=$(find_latest_experiment_in_s3)
    s3_rc=$?
    case $s3_rc in
      0)
        resolved_experiment="$s3_latest"
        export LATEST_EXPERIMENT="$resolved_experiment"
        export LATEST_EXPERIMENT_FROM_S3_XP_LISTING=1
        ;;
      2)
        echo "❌ S3 listing failed; cannot ensure latest models for the image build." >&2
        exit 1
        ;;
      *)
        echo "ℹ️  No xp_* prefixes in S3 (or listing empty); using full currency sync if local cache is empty."
        if [[ ! -d "$models_dir/currency" ]] || [[ -z "$(ls -A "$models_dir/currency" 2>/dev/null)" ]]; then
          echo "Local currency tree empty, downloading from S3..."
          if ! aws s3 sync "s3://$DATA_LAKE_BUCKET_NAME/models/currency/" "$models_dir/currency/" --region "$REGION"; then
            echo "❌ Failed to sync models/currency/ from S3" >&2
            exit 1
          fi
          echo "✅ Models downloaded from S3"
        fi
        if resolved_experiment=$(find_latest_experiment); then
          export LATEST_EXPERIMENT="$resolved_experiment"
          export LATEST_EXPERIMENT_FROM_S3_XP_LISTING=0
          echo "📁 Latest experiment (from local tree): $LATEST_EXPERIMENT"
        fi
        ;;
    esac
  fi

  # Per-experiment S3 prefix sync only when S3 uses xp_* prefixes for this name (see verify_local_backup).
  if [[ -n "${resolved_experiment:-}" ]] && [[ "${LATEST_EXPERIMENT_FROM_S3_XP_LISTING:-1}" == "1" ]]; then
    target_dir="$models_dir/currency/$resolved_experiment"
    # With set -o pipefail, skip find until the directory exists (stale local cache).
    n_pkls=0
    if [[ -d "$target_dir" ]]; then
      n_pkls=$(find "$target_dir" -name "*.pkl" 2>/dev/null | wc -l)
    fi
    if [[ ! -d "$target_dir" ]] || [[ "$n_pkls" -eq 0 ]]; then
      echo "Syncing latest experiment from S3: $resolved_experiment"
      mkdir -p "$target_dir"
      if ! aws s3 sync "s3://$DATA_LAKE_BUCKET_NAME/models/currency/$resolved_experiment/" "$target_dir/" --region "$REGION"; then
        echo "❌ Failed to sync s3://$DATA_LAKE_BUCKET_NAME/models/currency/$resolved_experiment/" >&2
        exit 1
      fi
      echo "✅ Models synced for $resolved_experiment"
    else
      echo "✅ Local models match S3 latest ($resolved_experiment); skipping download"
    fi
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
