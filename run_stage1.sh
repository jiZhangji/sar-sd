#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/sar-sd}"
DATASET_ROOT="${DATASET_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset}"
PREPARED_ROOT="${PREPARED_ROOT:-${DATASET_ROOT}/stage1_prepared}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/stage1_opt2sar_pretrain.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/stage1_opt2sar_pretrain}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"

cd "${PROJECT_ROOT}"

python tools/prepare_stage1_data.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${PREPARED_ROOT}" \
  --limit-per-dataset "${LIMIT_PER_DATASET}"

python train_stage1.py \
  --config "${CONFIG}" \
  --manifest "${PREPARED_ROOT}/manifest.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
