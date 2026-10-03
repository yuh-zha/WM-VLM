#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 {tetris-2d|tetris-3d} OUTPUT_DIR [MODEL] [extra trainer args...]" >&2
  exit 2
fi

DATASET="$1"
OUTPUT_DIR="$2"
MODEL="${3:-Qwen/Qwen2.5-VL-7B-Instruct}"
shift "$(( $# >= 3 ? 3 : 2 ))"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

case "${DATASET}" in
  tetris-2d)
    TASK=tetris_2d
    SUPERVISION=answer_only
    EXAMPLES=4000
    ;;
  tetris-3d)
    TASK=tetris_3d
    SUPERVISION=interleaved_image_placeholder
    EXAMPLES=16000
    ;;
  *) echo "Unknown dataset: ${DATASET}" >&2; exit 2 ;;
esac

WORLD_SIZE="${WORLD_SIZE:-8}"
EPOCHS="${EPOCHS:-115}"
MAX_STEPS="${MAX_STEPS:-$((EPOCHS * EXAMPLES / WORLD_SIZE))}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec torchrun --standalone --nnodes=1 --nproc_per_node="${WORLD_SIZE}" \
  scripts/train/train_qwen25vl_sft.py \
  --task "${TASK}" \
  --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --supervision-mode "${SUPERVISION}" \
  --max-steps "${MAX_STEPS}" \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --logging-steps 10 \
  --save-strategy steps \
  --save-steps "${SAVE_STEPS:-$((5 * EXAMPLES / WORLD_SIZE))}" \
  --save-total-limit 3 \
  --dataloader-num-workers "${NUM_WORKERS:-4}" \
  --attn-implementation sdpa \
  --bf16 \
  --report-to "${REPORT_TO:-none}" \
  "$@"
