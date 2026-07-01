#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

FULL_MANIFEST="${FULL_MANIFEST:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/quick_diagnostics_$(date +%Y%m%d_%H%M%S)}"
GPU_BASE="${GPU_BASE:-0}"
GPU_PAIRED="${GPU_PAIRED:-1}"
OVERFIT_EPOCHS="${OVERFIT_EPOCHS:-200}"
OVERFIT_TRAIN_SAMPLES="${OVERFIT_TRAIN_SAMPLES:-32}"
OVERFIT_VAL_SAMPLES="${OVERFIT_VAL_SAMPLES:-4}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"
RUN_SINGLE_DOMAIN="${RUN_SINGLE_DOMAIN:-0}"
SINGLE_DOMAIN_DATASET="${SINGLE_DOMAIN_DATASET:-SAR2Opt}"
SINGLE_DOMAIN_EPOCHS="${SINGLE_DOMAIN_EPOCHS:-80}"
SINGLE_DOMAIN_TRAIN_SAMPLES="${SINGLE_DOMAIN_TRAIN_SAMPLES:-2048}"
SINGLE_DOMAIN_VAL_SAMPLES="${SINGLE_DOMAIN_VAL_SAMPLES:-64}"
SINGLE_DOMAIN_TEST_SAMPLES="${SINGLE_DOMAIN_TEST_SAMPLES:-64}"

mkdir -p "${OUTPUT_ROOT}/logs"

echo "[quick] project=${PROJECT_ROOT}"
echo "[quick] full_manifest=${FULL_MANIFEST}"
echo "[quick] output_root=${OUTPUT_ROOT}"
echo "[quick] base_gpu=${GPU_BASE}, paired_gpu=${GPU_PAIRED}, parallel=${RUN_PARALLEL}"
echo "[quick] overfit_samples=train:${OVERFIT_TRAIN_SAMPLES},val:${OVERFIT_VAL_SAMPLES}, epochs=${OVERFIT_EPOCHS}"

python tools/make_debug_manifest.py \
  --input "${FULL_MANIFEST}" \
  --output data/debug_overfit_32/manifest.jsonl \
  --train-samples "${OVERFIT_TRAIN_SAMPLES}" \
  --val-samples "${OVERFIT_VAL_SAMPLES}" \
  --seed 42 | tee "${OUTPUT_ROOT}/logs/make_overfit32_manifest.log"

run_experiment() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local epochs="$4"
  local log="${OUTPUT_ROOT}/logs/${name}.log"
  echo "[quick] launch ${name} on GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train_stage1.py \
    --config "${config}" \
    --output-dir "${OUTPUT_ROOT}/${name}" \
    --epochs "${epochs}" \
    > "${log}" 2>&1
  echo "[quick] done ${name}"
}

if [[ "${RUN_PARALLEL}" == "1" && "${GPU_BASE}" != "${GPU_PAIRED}" ]]; then
  run_experiment overfit32_base "${GPU_BASE}" configs/debug_overfit32_base.yaml "${OVERFIT_EPOCHS}" &
  pid_base=$!
  run_experiment overfit32_paired "${GPU_PAIRED}" configs/debug_overfit32_paired.yaml "${OVERFIT_EPOCHS}" &
  pid_paired=$!
  wait "${pid_base}"
  wait "${pid_paired}"
else
  run_experiment overfit32_base "${GPU_BASE}" configs/debug_overfit32_base.yaml "${OVERFIT_EPOCHS}"
  run_experiment overfit32_paired "${GPU_PAIRED}" configs/debug_overfit32_paired.yaml "${OVERFIT_EPOCHS}"
fi

if [[ "${RUN_SINGLE_DOMAIN}" == "1" ]]; then
  echo "[quick] prepare single-domain manifest: ${SINGLE_DOMAIN_DATASET}"
  python tools/make_debug_manifest.py \
    --input "${FULL_MANIFEST}" \
    --output data/debug_single_domain2k/manifest.jsonl \
    --dataset "${SINGLE_DOMAIN_DATASET}" \
    --train-samples "${SINGLE_DOMAIN_TRAIN_SAMPLES}" \
    --val-samples "${SINGLE_DOMAIN_VAL_SAMPLES}" \
    --test-samples "${SINGLE_DOMAIN_TEST_SAMPLES}" \
    --seed 42 | tee "${OUTPUT_ROOT}/logs/make_single_domain_manifest.log"
  run_experiment single_domain2k_paired "${GPU_BASE}" configs/debug_single_domain2k_paired.yaml "${SINGLE_DOMAIN_EPOCHS}"
fi

python tools/summarize_quick_diagnostics.py \
  --output-root "${OUTPUT_ROOT}" \
  --max-panels 4 | tee "${OUTPUT_ROOT}/logs/summary.log"

echo "[quick] finished"
echo "[quick] summary: ${OUTPUT_ROOT}/SUMMARY.md"
echo "[quick] latest panels copied under: ${OUTPUT_ROOT}/latest_panels"
