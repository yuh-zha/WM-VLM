#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/converted_checkpoint_repro}"
RUNNER="scripts/reproduce/evaluate_wm_vlm_parquet.sh"

export NUM_SHARDS="${NUM_SHARDS:-8}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export FLOW_STEPS="${FLOW_STEPS:-1}"
export FLOW_SOLVER="${FLOW_SOLVER:-euler}"
export FLOW_SEED="${FLOW_SEED:-0}"
export FLOW_NOISE_DEVICE="${FLOW_NOISE_DEVICE:-cpu}"
export FLOW_STATE_DTYPE="${FLOW_STATE_DTYPE:-bfloat16}"
export MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

"${RUNNER}" \
  checkpoints/wm_vlm_tetris_2d_middle_checkpoint_7500 \
  "${OUTPUT_ROOT}/tetris_2d" \
  id=datasets/Tetris-2D-ID/data/eval-00000-of-00001.parquet \
  ood=datasets/Tetris-2D-OOD/data/held_out-00000-of-00001.parquet

"${RUNNER}" \
  checkpoints/wm_vlm_tetris_3d_bottom_checkpoint_30000 \
  "${OUTPUT_ROOT}/tetris_3d" \
  sc_id=datasets/Tetris-3D-SC-ID/data/eval-00000-of-00001.parquet \
  c_id=datasets/Tetris-3D-C-ID/data/eval-00000-of-00001.parquet \
  ood=datasets/Tetris-3D-OOD/data/held_out-00000-of-00001.parquet
