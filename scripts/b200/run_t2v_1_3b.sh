#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${TURBODIFFUSION_VENV:-${PROJECT_ROOT}/.venv-b200}"

if [[ ! -f "${VENV_PATH}/activate-turbodiffusion.sh" ]]; then
    echo "Missing B200 environment: ${VENV_PATH}" >&2
    echo "Run scripts/b200/install.sh first." >&2
    exit 1
fi

: "${DIT_PATH:?Set DIT_PATH to the unquantized TurboWan2.1 1.3B checkpoint}"
: "${VAE_PATH:?Set VAE_PATH to Wan2.1_VAE.pth}"
: "${TEXT_ENCODER_PATH:?Set TEXT_ENCODER_PATH to models_t5_umt5-xxl-enc-bf16.pth}"
: "${PROMPT:?Set PROMPT to the generation prompt}"

# shellcheck disable=SC1091
source "${VENV_PATH}/activate-turbodiffusion.sh"
cd "$PROJECT_ROOT"

python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model Wan2.1-1.3B \
    --dit_path "$DIT_PATH" \
    --vae_path "$VAE_PATH" \
    --text_encoder_path "$TEXT_ENCODER_PATH" \
    --prompt "$PROMPT" \
    --resolution "${RESOLUTION:-480p}" \
    --aspect_ratio "${ASPECT_RATIO:-16:9}" \
    --num_frames "${NUM_FRAMES:-81}" \
    --num_samples "${NUM_SAMPLES:-1}" \
    --num_steps "${NUM_STEPS:-4}" \
    --seed "${SEED:-0}" \
    --attention_type "${ATTENTION_TYPE:-sla}" \
    --sla_topk "${SLA_TOPK:-0.1}" \
    --save_path "${SAVE_PATH:-output/b200_t2v_1_3b.mp4}"
