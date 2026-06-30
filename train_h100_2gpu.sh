#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/sar-sd}"
MANIFEST="${MANIFEST:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared/manifest.jsonl}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/stage1_opt2sar_pretrain.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/stage1_opt2sar_pretrain}"
HF_HOME="${HF_HOME:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/huggingface}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Set OFFLINE=0 only when the model cache is incomplete and network access is required.
OFFLINE="${OFFLINE:-1}"
export HF_HOME OMP_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF
if [[ "${OFFLINE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  unset HF_HUB_OFFLINE || true
  unset TRANSFORMERS_OFFLINE || true
fi

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

echo "[launch] project=${PROJECT_ROOT}"
echo "[launch] manifest=${MANIFEST}"
echo "[launch] output=${OUTPUT_DIR}"
echo "[launch] GPUs=${CUDA_DEVICES}, per_gpu_batch=${BATCH_SIZE}, epochs=${EPOCHS}"
echo "[launch] global_batch=$((BATCH_SIZE * NPROC_PER_NODE)), offline=${OFFLINE}"
echo "[launch] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_stage1.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  "$@"
