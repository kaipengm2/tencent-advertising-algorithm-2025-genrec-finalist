#!/usr/bin/env bash
set -euo pipefail

# Inference entrypoint for Top-K retrieval

# Project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# First arg: directory containing a single .pt checkpoint
MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Usage: $0 MODEL_DIR [TOPK]" >&2
  exit 1
fi

# Second arg: K for Top-K (default: 10)
TOPK="${2:-10}"

export EVAL_DATA_PATH="${EVAL_DATA_PATH:-$ROOT_DIR/eval}"
export MODEL_OUTPUT_PATH="$MODEL_DIR"

python -u infer.py --topk "$TOPK"
