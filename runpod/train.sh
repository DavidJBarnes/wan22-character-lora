#!/usr/bin/env bash
# Train the Kelly character LoRA on the Wan 2.2 LOW-NOISE I2V expert.
#
# Why these choices (see ../GUIDELINES.md):
#   --task i2v-A14B        Wanly runs I2V only; I2V-trained LoRAs give the best I2V results.
#   --dit <low-noise>      Identity lives in the low-noise expert. Training it (not the
#                          high-noise/motion expert) is the fix for the past "static" failure.
#   --min/--max_timestep   0..900 = the low-noise band (per musubi Wan docs).
#   --preserve_distribution_shape   Wan 2.2 recommended for stable timestep sampling.
#   --discrete_flow_shift 5.0       Official I2V inference shift.
#   dim 16 / alpha 16      alpha == dim gives better Wan results than the alpha=1 default.
#
# Tuning knobs if v1 is off:
#   - Undertrained (weak likeness): raise --learning_rate to 2e-4, or num_repeats in dataset.toml.
#   - Overtrained (plasticky/static): pick an earlier epoch checkpoint, or lower steps.
#   - OOM on 24GB: add `--blocks_to_swap 16` (trades a little speed for VRAM).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

mkdir -p "$OUTPUT_DIR"

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    "$MUSUBI/src/musubi_tuner/wan_train_network.py" \
    --task i2v-A14B \
    --dit "$DIT_LOW" \
    --vae "$VAE" \
    --t5 "$T5" \
    --dataset_config "$DATASET_CONFIG" \
    --network_module networks.lora_wan \
    --network_dim 16 --network_alpha 16 \
    --learning_rate 1e-4 \
    --optimizer_type adamw8bit \
    --timestep_sampling shift --discrete_flow_shift 5.0 \
    --min_timestep 0 --max_timestep 900 \
    --preserve_distribution_shape \
    --max_train_epochs 10 \
    --save_every_n_epochs 1 \
    --mixed_precision bf16 --fp8_base \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME"

echo "=== training complete -> $OUTPUT_DIR/$OUTPUT_NAME*.safetensors ==="
ls -lh "$OUTPUT_DIR"
