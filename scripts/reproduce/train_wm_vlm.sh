#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 {tetris-2d|tetris-3d} {stage1|stage2|joint} OUTPUT_DIR [MODEL_OR_CHECKPOINT] [extra trainer args...]" >&2
  exit 2
fi

DATASET="$1"
STAGE="$2"
OUTPUT_DIR="$3"
MODEL="${4:-Qwen/Qwen2.5-VL-7B-Instruct}"
shift "$(( $# >= 4 ? 4 : 3 ))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

case "${DATASET}" in
  tetris-2d)
    TASK=tetris_2d
    DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/Tetris-2D}"
    MANIFEST="${MANIFEST:-${DATA_ROOT}/manifest_wm_vlm_train.jsonl}"
    EXAMPLES=4000
    SOURCE_FORMAT=tetris_2d
    HELPER_ROLE=rotated_query
    ;;
  tetris-3d)
    TASK=tetris_3d
    DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/Tetris-3D}"
    MANIFEST="${MANIFEST:-${DATA_ROOT}/manifest_wm_vlm_train.jsonl}"
    EXAMPLES=16000
    SOURCE_FORMAT=tetris_3d
    HELPER_ROLE=rotation_state
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

WORLD_SIZE="${WORLD_SIZE:-8}"
if (( EXAMPLES % WORLD_SIZE != 0 )); then
  echo "Dataset size ${EXAMPLES} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi
STEPS_PER_EPOCH="$((EXAMPLES / WORLD_SIZE))"

case "${STAGE}" in
  stage1)
    EPOCHS="${EPOCHS:-100}"
    LEARNING_RATE="${LEARNING_RATE:-1e-4}"
    FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-1.0}"
    CE_WEIGHT="${CE_WEIGHT:-0.0}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
    LR_SCHEDULER="${LR_SCHEDULER:-constant}"
    TRAINABILITY_ARGS=(--freeze-vlm-backbone --no-freeze-generation-branch)
    ;;
  stage2)
    EPOCHS="${EPOCHS:-15}"
    LEARNING_RATE="${LEARNING_RATE:-1e-5}"
    FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.0}"
    CE_WEIGHT="${CE_WEIGHT:-1.0}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
    LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
    TRAINABILITY_ARGS=(--no-freeze-vlm-backbone --no-freeze-generation-branch)
    ;;
  joint)
    EPOCHS="${EPOCHS:-30}"
    LEARNING_RATE="${LEARNING_RATE:-1e-5}"
    FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.1}"
    CE_WEIGHT="${CE_WEIGHT:-0.9}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
    LR_SCHEDULER="${LR_SCHEDULER:-constant}"
    TRAINABILITY_ARGS=(--no-freeze-vlm-backbone --no-freeze-generation-branch)
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

MAX_STEPS="${MAX_STEPS:-$((EPOCHS * STEPS_PER_EPOCH))}"
SAVE_STEPS="${SAVE_STEPS:-$((5 * STEPS_PER_EPOCH))}"
INIT_ARGS=(--initialize-generation-from-text)
if [[ "${INIT_FROM_TEXT:-1}" == "0" ]]; then
  INIT_ARGS=(--no-initialize-generation-from-text)
fi
FLOW_ARGS=(--fixed-flow-noise-seed 0 --fixed-flow-timestep 0.0)
if [[ "${FIXED_FLOW:-1}" == "0" ]]; then
  FLOW_ARGS=()
fi
LATENT_SIZE="${VISUAL_TOKENS:-361}"
RESOLUTION_ARGS=()
if [[ -n "${VISUAL_TOKENS+x}" ]]; then
  case "${LATENT_SIZE}" in
    1) GRID_SIDE=1 ;;
    4) GRID_SIDE=2 ;;
    25) GRID_SIDE=5 ;;
    100) GRID_SIDE=10 ;;
    225) GRID_SIDE=15 ;;
    361) GRID_SIDE=19 ;;
    *)
      echo "VISUAL_TOKENS must be one of 1, 4, 25, 100, 225, or 361" >&2
      exit 2
      ;;
  esac
  HELPER_SIDE="$((GRID_SIDE * 28))"
  HELPER_PIXELS="$((HELPER_SIDE * HELPER_SIDE))"
  RESOLUTION_ARGS=(
    --problem-min-pixels 262144
    --problem-max-pixels 262144
    --helper-min-pixels "${HELPER_PIXELS}"
    --helper-max-pixels "${HELPER_PIXELS}"
  )
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${WORLD_SIZE}" \
  scripts/train/train_wm_vlm.py \
  --task "${TASK}" \
  --model "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --expected-train-examples "${EXAMPLES}" \
  --minimum-input-images 1 \
  --exact-input-images 1 \
  --expected-source-format "${SOURCE_FORMAT}" \
  --helper-image-role "${HELPER_ROLE}" \
  --image-size 512 \
  --wm-vlm-num-layers "${WM_VLM_NUM_LAYERS:-4}" \
  --wm-vlm-layer-placement "${WM_VLM_LAYER_PLACEMENT:-middle}" \
  --latent-size "${LATENT_SIZE}" \
  "${RESOLUTION_ARGS[@]}" \
  --flow-loss-weight "${FLOW_LOSS_WEIGHT}" \
  --pixel-loss-weight "${PIXEL_LOSS_WEIGHT:-0.0}" \
  --ce-weight "${CE_WEIGHT}" \
  --ce-consumer-source "${CE_CONSUMER_SOURCE:-generated_endpoint}" \
  --flow-noise-scale 1.0 \
  --flow-source-mode gaussian_noise \
  "${INIT_ARGS[@]}" \
  "${TRAINABILITY_ARGS[@]}" \
  --freeze-vision-encoder \
  --prompt-style wm_vlm \
  --token-style wm_vlm \
  --no-processor-use-fast \
  --no-qwen-vl-utils-preprocess \
  --padding longest \
  --max-length 8192 \
  "${FLOW_ARGS[@]}" \
  --max-steps "${MAX_STEPS}" \
  --num-train-epochs "${EPOCHS}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --lr-scheduler-type "${LR_SCHEDULER}" \
  --optim adamw_torch_fused \
  --logging-steps 10 \
  --save-strategy steps \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 3 \
  --dataloader-num-workers "${NUM_WORKERS:-4}" \
  --dataloader-persistent-workers \
  --gradient-checkpointing \
  --attn-implementation sdpa \
  --bf16 \
  --report-to "${REPORT_TO:-none}" \
  "$@"
