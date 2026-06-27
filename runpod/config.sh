#!/usr/bin/env bash
# Shared paths/config sourced by the other runpod scripts.
# Override any of these by exporting them before running a script.
set -euo pipefail

# Everything lives on the pod's ephemeral disk under /workspace.
export WORKSPACE="${WORKSPACE:-/workspace}"
export MUSUBI="${MUSUBI:-$WORKSPACE/musubi-tuner}"
export MODELS_DIR="${MODELS_DIR:-$WORKSPACE/models}"
export DATASET_DIR="${DATASET_DIR:-$WORKSPACE/dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/output}"
export DATASET_CONFIG="${DATASET_CONFIG:-$WORKSPACE/dataset.toml}"

# Model files (musubi-tuner expects these exact formats)
export DIT_LOW="${DIT_LOW:-$MODELS_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors}"
export T5="${T5:-$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth}"
export VAE="${VAE:-$MODELS_DIR/wan_2.1_vae.safetensors}"

# Output LoRA name (a v-suffix so iterations don't clobber)
export OUTPUT_NAME="${OUTPUT_NAME:-k3llydw_lownoise_v1}"
