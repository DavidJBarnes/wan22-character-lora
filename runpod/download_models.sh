#!/usr/bin/env bash
# Download the 3 model files musubi-tuner needs for Wan 2.2 I2V LoRA training.
# Only the LOW-noise expert is needed (we train identity on it). No CLIP — Wan 2.2
# I2V does not require the CLIP vision encoder (only Wan 2.1 did). Total ~40GB.
#
# Uses huggingface_hub's native downloader (NOT aria2): HF now serves large weights
# from Xet storage with short-lived presigned URLs that aria2's multi-connection mode
# fails to refresh (403s mid-download). huggingface-cli + hf_xet handle this correctly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

pip install -q -U "huggingface_hub[cli,hf_xet]" >/dev/null 2>&1 || true

DL="$MODELS_DIR/.dl"          # staging dir (same filesystem -> moves are instant)
mkdir -p "$MODELS_DIR" "$DL"

fetch() {  # repo  remote_path  final_name
    local repo="$1" path="$2" name="$3"
    if [ -f "$MODELS_DIR/$name" ] && [ -s "$MODELS_DIR/$name" ]; then
        echo "SKIP  $name (exists)"; return 0
    fi
    echo "GET   $name  <-  $repo/$path"
    hf download "$repo" "$path" --local-dir "$DL"
    mv -f "$DL/$path" "$MODELS_DIR/$name"
    rm -rf "$DL/.cache" "$DL/split_files"   # reclaim staging space
}

# DiT — low-noise I2V 14B (~27GB). musubi accepts the Comfy-Org repackaged format.
fetch "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
      "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
      "wan2.2_i2v_low_noise_14B_fp16.safetensors"

# T5 text encoder — musubi's own .pth from Wan-AI repo (NOT the ComfyUI fp8 one).
# T5 is identical between Wan 2.1 and 2.2. (~11GB)
fetch "Wan-AI/Wan2.1-I2V-14B-720P" \
      "models_t5_umt5-xxl-enc-bf16.pth" \
      "models_t5_umt5-xxl-enc-bf16.pth"

# VAE — Wan 2.1 VAE (same for 2.2 14B). (~254MB)
fetch "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
      "split_files/vae/wan_2.1_vae.safetensors" \
      "wan_2.1_vae.safetensors"

rm -rf "$DL"
echo "=== model download complete ==="
ls -lh "$MODELS_DIR"
