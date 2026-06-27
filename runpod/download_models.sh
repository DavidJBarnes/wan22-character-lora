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

# Only add the Xet accelerator — do NOT bump huggingface_hub itself: musubi's
# transformers pins huggingface-hub<1.0, and upgrading it breaks the cache/train step.
pip install -q hf_xet >/dev/null 2>&1 || true

fetch() {  # repo  remote_path  final_name
    local repo="$1" path="$2" name="$3"
    if [ -f "$MODELS_DIR/$name" ] && [ -s "$MODELS_DIR/$name" ]; then
        echo "SKIP  $name (exists)"; return 0
    fi
    echo "GET   $name  <-  $repo/$path"
    # Python API is stable across huggingface_hub versions (CLI name changed in 1.x);
    # Xet-aware, so it refreshes presigned URLs (no aria2 403s on large weights).
    MODELS_DIR="$MODELS_DIR" python - "$repo" "$path" "$name" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, path, name = sys.argv[1], sys.argv[2], sys.argv[3]
src = hf_hub_download(repo_id=repo, filename=path)
dst = os.path.join(os.environ["MODELS_DIR"], name)
shutil.copy(src, dst)   # copy out of the hub cache, then it can be GC'd
print("saved", dst)
PY
}

mkdir -p "$MODELS_DIR"

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

echo "=== model download complete ==="
ls -lh "$MODELS_DIR"
