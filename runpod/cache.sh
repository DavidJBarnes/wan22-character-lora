#!/usr/bin/env bash
# Pre-cache VAE latents + T5 text-encoder outputs (mandatory before training).
# Re-run after changing the dataset or dataset.toml.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

# Copy the committed dataset.toml into place if the workspace one isn't present.
[ -f "$DATASET_CONFIG" ] || cp "$HERE/dataset.toml" "$DATASET_CONFIG"

echo "=== caching VAE latents (--i2v; no --clip needed for Wan 2.2) ==="
python "$MUSUBI/src/musubi_tuner/wan_cache_latents.py" \
    --dataset_config "$DATASET_CONFIG" \
    --vae "$VAE" \
    --i2v

echo "=== caching T5 text-encoder outputs ==="
python "$MUSUBI/src/musubi_tuner/wan_cache_text_encoder_outputs.py" \
    --dataset_config "$DATASET_CONFIG" \
    --t5 "$T5" \
    --batch_size 16

echo "=== caching complete ==="
