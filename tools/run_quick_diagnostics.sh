#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

FULL_MANIFEST="${FULL_MANIFEST:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/quick_diagnostics_$(date +%Y%m%d_%H%M%S)}"
GPU_BASE="${GPU_BASE:-0}"
GPU_PAIRED="${GPU_PAIRED:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-${GPU_BASE},${GPU_PAIRED}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RUN_MODE="${RUN_MODE:-split}"
OVERFIT_EPOCHS="${OVERFIT_EPOCHS:-200}"
OVERFIT_TRAIN_SAMPLES="${OVERFIT_TRAIN_SAMPLES:-32}"
OVERFIT_VAL_SAMPLES="${OVERFIT_VAL_SAMPLES:-4}"
PER_GPU_BATCH="${PER_GPU_BATCH:-4}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-20}"
VALIDATION_EVERY_EPOCHS="${VALIDATION_EVERY_EPOCHS:-5}"
VALIDATION_INFERENCE_STEPS="${VALIDATION_INFERENCE_STEPS:-20}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-5}"
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
echo "[quick] mode=${RUN_MODE}, base_gpu=${GPU_BASE}, paired_gpu=${GPU_PAIRED}, cuda_devices=${CUDA_DEVICES}, nproc=${NPROC_PER_NODE}"
echo "[quick] overfit_samples=train:${OVERFIT_TRAIN_SAMPLES},val:${OVERFIT_VAL_SAMPLES}, epochs=${OVERFIT_EPOCHS}"
echo "[quick] per_gpu_batch=${PER_GPU_BATCH}, val_every=${VALIDATION_EVERY_EPOCHS}, val_steps=${VALIDATION_INFERENCE_STEPS}, eval_every=${EVAL_EVERY_EPOCHS}"

python tools/make_debug_manifest.py \
  --input "${FULL_MANIFEST}" \
  --output data/debug_overfit_32/manifest.jsonl \
  --train-samples "${OVERFIT_TRAIN_SAMPLES}" \
  --val-samples "${OVERFIT_VAL_SAMPLES}" \
  --seed 42 | tee "${OUTPUT_ROOT}/logs/make_overfit32_manifest.log"

make_runtime_config() {
  local src="$1"
  local dst="$2"
  local batch_size="$3"
  python - "$src" "$dst" "$batch_size" "$SAVE_EVERY_EPOCHS" "$VALIDATION_EVERY_EPOCHS" "$VALIDATION_INFERENCE_STEPS" "$EVAL_EVERY_EPOCHS" <<'PY'
import sys
import yaml

src, dst, batch_size, save_every, val_every, val_steps, eval_every = sys.argv[1:]
with open(src, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
cfg["data"]["batch_size"] = int(batch_size)
cfg["train"]["save_every_epochs"] = int(save_every)
cfg["train"]["validation_every_epochs"] = int(val_every)
cfg["train"]["validation_inference_steps"] = int(val_steps)
cfg["train"]["eval_every_epochs"] = int(eval_every)
cfg["train"]["eval_batch_size"] = int(batch_size)
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
print(f"wrote runtime config: {dst}")
PY
}

mkdir -p "${OUTPUT_ROOT}/configs"
BASE_CONFIG="${OUTPUT_ROOT}/configs/debug_overfit32_base.runtime.yaml"
PAIRED_CONFIG="${OUTPUT_ROOT}/configs/debug_overfit32_paired.runtime.yaml"
SINGLE_CONFIG="${OUTPUT_ROOT}/configs/debug_single_domain2k_paired.runtime.yaml"
make_runtime_config configs/debug_overfit32_base.yaml "${BASE_CONFIG}" "${PER_GPU_BATCH}" | tee "${OUTPUT_ROOT}/logs/make_base_config.log"
make_runtime_config configs/debug_overfit32_paired.yaml "${PAIRED_CONFIG}" "${PER_GPU_BATCH}" | tee "${OUTPUT_ROOT}/logs/make_paired_config.log"
make_runtime_config configs/debug_single_domain2k_paired.yaml "${SINGLE_CONFIG}" "${PER_GPU_BATCH}" | tee "${OUTPUT_ROOT}/logs/make_single_config.log"

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

run_experiment_ddp() {
  local name="$1"
  local config="$2"
  local epochs="$3"
  local log="${OUTPUT_ROOT}/logs/${name}.log"
  echo "[quick] launch ${name} with DDP devices=${CUDA_DEVICES}; log=${log}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train_stage1.py \
    --config "${config}" \
    --output-dir "${OUTPUT_ROOT}/${name}" \
    --epochs "${epochs}" \
    > "${log}" 2>&1
  echo "[quick] done ${name}"
}

if [[ "${RUN_MODE}" == "ddp" ]]; then
  run_experiment_ddp overfit32_base "${BASE_CONFIG}" "${OVERFIT_EPOCHS}"
  run_experiment_ddp overfit32_paired "${PAIRED_CONFIG}" "${OVERFIT_EPOCHS}"
elif [[ "${RUN_PARALLEL}" == "1" && "${GPU_BASE}" != "${GPU_PAIRED}" ]]; then
  run_experiment overfit32_base "${GPU_BASE}" "${BASE_CONFIG}" "${OVERFIT_EPOCHS}" &
  pid_base=$!
  run_experiment overfit32_paired "${GPU_PAIRED}" "${PAIRED_CONFIG}" "${OVERFIT_EPOCHS}" &
  pid_paired=$!
  wait "${pid_base}"
  wait "${pid_paired}"
else
  run_experiment overfit32_base "${GPU_BASE}" "${BASE_CONFIG}" "${OVERFIT_EPOCHS}"
  run_experiment overfit32_paired "${GPU_PAIRED}" "${PAIRED_CONFIG}" "${OVERFIT_EPOCHS}"
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
  if [[ "${RUN_MODE}" == "ddp" ]]; then
    run_experiment_ddp single_domain2k_paired "${SINGLE_CONFIG}" "${SINGLE_DOMAIN_EPOCHS}"
  else
    run_experiment single_domain2k_paired "${GPU_BASE}" "${SINGLE_CONFIG}" "${SINGLE_DOMAIN_EPOCHS}"
  fi
fi

python tools/summarize_quick_diagnostics.py \
  --output-root "${OUTPUT_ROOT}" \
  --max-panels 4 | tee "${OUTPUT_ROOT}/logs/summary.log"

echo "[quick] finished"
echo "[quick] summary: ${OUTPUT_ROOT}/SUMMARY.md"
echo "[quick] latest panels copied under: ${OUTPUT_ROOT}/latest_panels"
