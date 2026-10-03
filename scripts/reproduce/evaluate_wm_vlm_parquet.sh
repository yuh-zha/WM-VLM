#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 MODEL OUTPUT_DIR NAME=PARQUET [NAME=PARQUET ...]" >&2
  exit 2
fi

MODEL="$1"
OUTPUT_DIR="$2"
shift 2
PARQUET_SPLITS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
FLOW_STEPS="${FLOW_STEPS:-1}"
FLOW_SOLVER="${FLOW_SOLVER:-euler}"
FLOW_SEED="${FLOW_SEED:-0}"
FLOW_NOISE_DEVICE="${FLOW_NOISE_DEVICE:-cpu}"
FLOW_STATE_DTYPE="${FLOW_STATE_DTYPE:-bfloat16}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Missing checkpoint config: ${MODEL}/config.json" >&2
  exit 2
fi
if (( NUM_SHARDS < 1 )); then
  echo "NUM_SHARDS must be positive" >&2
  exit 2
fi

SPLIT_NAMES=()
PARQUET_ARGS=()
for specification in "${PARQUET_SPLITS[@]}"; do
  name="${specification%%=*}"
  path="${specification#*=}"
  if [[ "${name}" == "${specification}" || -z "${name}" || -z "${path}" ]]; then
    echo "Expected NAME=PARQUET, got: ${specification}" >&2
    exit 2
  fi
  if [[ ! -f "${path}" ]]; then
    echo "Missing Parquet split ${name}: ${path}" >&2
    exit 2
  fi
  SPLIT_NAMES+=("${name}")
  PARQUET_ARGS+=(--parquet-split "${name}=${path}")
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DATASETS_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

pids=()
labels=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  shard_dir="${OUTPUT_DIR}/shard_${shard}"
  metrics_path="${shard_dir}/metrics.json"
  mkdir -p "${shard_dir}"
  if [[ -s "${metrics_path}" ]]; then
    echo "[wm_vlm-eval] shard=${shard} already has metrics; reusing it"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" \
    scripts/eval/evaluate_wm_vlm_tetris.py \
    --model "${MODEL}" \
    --output-dir "${shard_dir}" \
    "${PARQUET_ARGS[@]}" \
    --splits "${SPLIT_NAMES[@]}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-latent-steps "${FLOW_STEPS}" \
    --flow-solver "${FLOW_SOLVER}" \
    --flow-seed "${FLOW_SEED}" \
    --flow-noise-device "${FLOW_NOISE_DEVICE}" \
    --flow-state-dtype "${FLOW_STATE_DTYPE}" \
    --dtype "${MODEL_DTYPE}" \
    --device cuda:0 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --log-every 10 \
    >"${shard_dir}/run.log" 2>&1 &
  pids+=("$!")
  labels+=("${shard}")
done

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    echo "Shard ${labels[index]} failed; inspect its run.log" >&2
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  exit 1
fi

metric_paths=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  metrics_path="${OUTPUT_DIR}/shard_${shard}/metrics.json"
  if [[ ! -s "${metrics_path}" ]]; then
    echo "Missing shard metrics: ${metrics_path}" >&2
    exit 1
  fi
  metric_paths+=("${metrics_path}")
done

"${PYTHON_BIN}" scripts/eval/aggregate_wm_vlm_tetris_shards.py \
  --output "${OUTPUT_DIR}/aggregated_metrics.json" \
  "${metric_paths[@]}"

