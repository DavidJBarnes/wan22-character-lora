#!/usr/bin/env bash
# Install musubi-tuner + tools on a fresh RunPod pod (PyTorch 2.x template).
# Idempotent: safe to re-run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

echo "=== apt deps (aria2 for fast downloads) ==="
apt-get update -y && apt-get install -y aria2 git

echo "=== clone musubi-tuner -> $MUSUBI ==="
if [ ! -d "$MUSUBI/.git" ]; then
    git clone https://github.com/kohya-ss/musubi-tuner.git "$MUSUBI"
else
    git -C "$MUSUBI" pull --ff-only || true
fi

echo "=== install musubi-tuner (editable) ==="
# The RunPod PyTorch template already ships a CUDA-matched torch; -e . reuses it.
# If pip tries to downgrade torch, install with --no-deps then add the rest from
# musubi's requirements instead.
cd "$MUSUBI"
pip install -e .

mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"
echo "=== setup complete ==="
echo "Next: bash $HERE/download_models.sh"
