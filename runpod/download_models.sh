#!/usr/bin/env bash
# Download the 3 model files musubi-tuner needs for Wan 2.2 I2V LoRA training.
# Only the LOW-noise expert is needed (we train identity on it). No CLIP — Wan 2.2
# I2V does not require the CLIP vision encoder (only Wan 2.1 did).
# Total ~40GB. Uses aria2c (parallel + resume).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

dl() {  # url dest
    if [ -f "$2" ] && [ -s "$2" ] && [ ! -f "$2.aria2" ]; then
        echo "SKIP  $(basename "$2") (exists)"; return 0
    fi
    echo "GET   $(basename "$2")"
    aria2c -x 8 -s 8 --console-log-level=warn --summary-interval=30 \
        -d "$(dirname "$2")" -o "$(basename "$2")" "$1"
}

mkdir -p "$MODELS_DIR"

# DiT — Comfy-Org repackaged low-noise I2V 14B (~28GB). musubi accepts this format.
dl "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
   "$DIT_LOW"

# T5 text encoder — musubi's own .pth format from the Wan-AI repo (NOT the ComfyUI
# fp8 scaled safetensors). T5 is identical between Wan 2.1 and 2.2. (~11GB)
dl "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth" \
   "$T5"

# VAE — Wan 2.1 VAE (same for 2.2 14B). ComfyUI repackaged safetensors is accepted. (~254MB)
dl "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
   "$VAE"

echo "=== model download complete ==="
ls -lh "$MODELS_DIR"
