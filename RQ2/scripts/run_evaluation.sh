#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=cached
DEVICE=cuda
BATCH_SIZE=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$STAGE" != "cached" && "$STAGE" != "evaluate" && "$STAGE" != "all" ]]; then
  echo "--stage must be cached, evaluate, or all" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$STAGE" == "all" ]]; then
  python "$ROOT/scripts/evaluate_rq1.py" \
    --device "$DEVICE" --batch-size "$BATCH_SIZE"
fi

python "$ROOT/scripts/evaluate_codebert_distance.py"

if [[ "$STAGE" == "evaluate" || "$STAGE" == "all" ]]; then
  python "$ROOT/scripts/evaluate_model_aware_ood.py" \
    --device "$DEVICE" --batch-size "$BATCH_SIZE"
  python "$ROOT/scripts/evaluate_prediction_sensitivity.py" \
    --device "$DEVICE" --batch-size "$BATCH_SIZE"
fi

python "$ROOT/scripts/build_pca.py"

echo "RQ2 evaluation complete: $ROOT/results"
