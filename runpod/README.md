# RunPod runbook — Wan 2.2 character LoRA (pure musubi-tuner)

Ephemeral 4090 pod, CLI-only (no UI). Trains the **low-noise I2V expert** for one
character. ~$1–3 and ~2–3 hours end to end. See `../GUIDELINES.md` for the why.

## 0. Launch the pod
- GPU: **RTX 4090 (24GB)** — A5000/L4 24GB also fine.
- Template: **RunPod PyTorch 2.x** (CUDA 12.x). 
- Disk: **~80GB container/volume disk** (models ~40GB + caches + output). Ephemeral is fine.
- Expose a web terminal / SSH.

> Ephemeral pod: models are re-downloaded each fresh pod. For repeat iterations,
> a persistent network volume avoids that — but you chose ephemeral, so just keep
> the pod alive between iterations rather than terminating it.

## 1. Get these scripts onto the pod
```bash
cd /workspace
git clone <this-repo-url> wan22-character-lora   # or: runpodctl receive ...
cd wan22-character-lora/runpod
```

## 2. Upload the dataset (local machine -> pod)
The captioned dataset is at `~/projects/loras/wan22/kelly-driveway/dataset` (18 imgs + .txt).
From your **local** machine:
```bash
runpodctl send ~/projects/loras/wan22/kelly-driveway/dataset
```
On the **pod**, run the printed `runpodctl receive <code>` command, then put it in place:
```bash
mkdir -p /workspace/dataset && mv dataset/* /workspace/dataset/   # adjust if needed
ls /workspace/dataset | wc -l   # expect 36 (18 png + 18 txt)
```
(Alternatively scp the folder to `/workspace/dataset`.)

## 3. Install + download models
```bash
bash setup.sh            # musubi-tuner + aria2  (~5 min)
bash download_models.sh  # low-noise DiT + T5 + VAE  (~15 min, ~40GB)
```

## 4. Cache + train
```bash
bash cache.sh   # VAE latents + T5 outputs  (~2-5 min)
bash train.sh   # ~1800 steps, 10 epoch-checkpoints  (~1.5-3 hr)
```
Checkpoints land in `/workspace/output/` as `k3llydw_lownoise_v1-000001.safetensors` …
`-000010.safetensors` (one per epoch).

## 5. Pull the LoRA back
From your **local** machine, grab the best epoch (start by eyeballing 6–9):
```bash
# on the pod:
runpodctl send /workspace/output/k3llydw_lownoise_v1-000008.safetensors
# locally: runpodctl receive <code>
```

## 6. Use it in Wanly
Register in the LoRA library: `low_file` = this checkpoint, `high_file` = empty,
`default_low_weight` ≈ 0.7, `default_high_weight` = 0. Trigger token: **`k3llydw`**.
Generate with the character LoRA on the low expert and your motion LoRA on the high
expert — no faceswap.

## Picking the best checkpoint
Earlier epochs = looser likeness but more motion/flexibility; later = stronger likeness
but risk of stiffness. Test ~epoch 6, 8, 10 in a real I2V job and pick the one that holds
identity without going static. That comparison *is* the iteration loop.
