#!/usr/bin/env bash
set -euo pipefail

# Set project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Env paths
export TRAIN_DATA_PATH="$ROOT_DIR/data"
export TRAIN_CKPT_PATH="$ROOT_DIR/checkpoints"
export TRAIN_LOG_PATH="$ROOT_DIR/logs"
export TRAIN_TF_EVENTS_PATH="$ROOT_DIR/tf_events"
export USER_CACHE_PATH="$ROOT_DIR/cache"

mkdir -p \
  "$TRAIN_DATA_PATH" \
  "$TRAIN_CKPT_PATH" \
  "$TRAIN_LOG_PATH" \
  "$TRAIN_TF_EVENTS_PATH" \
  "$USER_CACHE_PATH"

# 1) Preprocess seq.jsonl
python -u get_stat.py

# 2) Train model
python -u main.py