# Wan 2.2 Character LoRA — Training Guidelines

Research-backed guidelines for training character-identity LoRAs for Wan 2.2 I2V 14B,
to replace post-hoc faceswap (which flattens facial expression). Written 2026-06-27.

## TL;DR

- **You do NOT need video clips.** Image-only training is fully supported and is the
  *recommended* path for character identity. Motion comes from the base model + your
  motion LoRA — not the character LoRA.
- **Why a past attempt went "washboard / static":** a character LoRA trained/applied on
  the **high-noise (motion) expert** overfits motion to static training stills and freezes
  movement. The fix is to keep the character LoRA on the **low-noise (identity) expert**
  and off the motion expert.
- This maps 1:1 onto the high/low LoRA-weight system the Wanly daemon already uses:
  motion LoRA = high-weighted; character LoRA = low-weighted. They live on different
  experts → identity + motion coexist, faceswap retired.

## Wan 2.2 dual-expert recap (why this works)

Wan 2.2 I2V 14B is a 2-expert MoE:

- **High-noise expert** (early/noisy timesteps): composition, **motion**, camera.
- **Low-noise expert** (late/clean timesteps): fine detail, **identity**, texture.

Character identity lives in the low-noise expert. Training identity into the high-noise
expert is what kills motion.

## Dataset (images only)

- **12–40 images** of the character; sweet spot **~15–30**. Below ~15 the LoRA conflates
  identity with photo artifacts.
- **Pose / angle / expression diversity > location diversity.** Include varied
  **expressions** (this is what buys you expressive faces), angles, framing
  (portrait / medium / full), and balanced lighting/exposure.
- **≥512 px on the short side** (the trainer buckets/resizes; train at 512, it generalizes
  to 1024 at inference).
- **Regularization images** (optional but recommended for anti-stiffness): 1–3× per
  training image, same class (e.g. "woman"), captioned *without* the trigger token. They
  "give the LoRA permission to keep the base's range while holding identity" — i.e.
  preserve motion.

### Shot list (collect against this)

Target **~20 images** for one character. The counts below sum to 20; scale them if you go
higher. **Variety beats volume** — 20 varied shots beat 40 near-duplicates. Every image
should be the *same person*, *sharp focus on the face*, *good lighting*, and *one subject
only* (no other faces in frame).

| # | Framing | Count | What it teaches | Notes |
|---|---------|-------|-----------------|-------|
| 1 | **Close-up — face, neutral** | 3 | Core identity (eyes, nose, jaw, skin) | Front-on, eyes to chin fills frame. The identity backbone. |
| 2 | **Close-up — expressions** | 4 | Expressive range (smile, laugh, serious, surprised) | One expression each. This is what buys you *expressive* video faces. |
| 3 | **Close-up — angles** | 3 | 3D head shape | ¾ left, ¾ right, slight up/down tilt. NOT pure profile-only. |
| 4 | **Medium — head + shoulders/torso** | 4 | Identity in context, hair, neck, build | Vary clothing/background a little; mix 1–2 expressions in. |
| 5 | **Full / ¾ body** | 3 | Proportions, posture, build | Standing, sitting; helps full-frame video shots. |
| 6 | **Lighting variety** | 3 | Robustness to scene light | One bright/even, one soft window light, one warmer/dimmer. Avoid harsh shadow across the face. |

**Dimensions:** ≥**512 px on the short side** is the floor; **1024²** is ideal (extra
resolution = sharper supervision, downscaled to the 512 train bucket — never wasted). All
of these are fine and can be **mixed in one folder** — the trainer's aspect-ratio buckets
keep each image's native ratio:

| Source size | Verdict | Use for |
|-------------|---------|---------|
| **1024×1024** | Best | Anything — prefer it when you can get it |
| **512×1024 / 768×1024** (portrait) | Good | Full / ¾ body, head-to-torso shots |
| **512×512** | Fine | Tight face close-ups (already at train res, no headroom) |

Rules that matter more than the exact size:
- **Never go below 512 on the short side** (trainer would upscale → lost detail).
- **Don't distort to force a ratio** — crop to what frames the shot naturally
  (square for tight faces, portrait for bodies). Mixed ratios in one set is expected.
- **Maximize face share of the frame.** A 512² tight crop teaches more than a 1024² shot
  where the face is a speck — it's pixels-on-face that count, not total resolution.

### Avoid (these poison the LoRA)

- Heavy faceswap/AI-smoothed faces (you'd be training on the artifacts you're trying to escape).
- Sunglasses, hands/objects/hair covering the face, heavy motion blur, low light, tiny faces.
- Two near-identical shots (wasted slots — diversity is the whole game).
- Other people's faces anywhere in frame (the LoRA will blend identities).
- Extreme stylization/filters if you want a photoreal result.

## Captioning

- Per-image same-name `.txt` (`0001.jpg` + `0001.txt`).
- **Short unique trigger token** + minimal class/context:
  `ch_m4k0t0, denim jacket, window light, medium close-up`.
- **Do NOT describe facial details** in captions — it competes with the trigger, weakens
  identity binding, and can flatten expression.
- Auto-caption with Florence2 / Qwen-VL, then trim facial descriptors.

Use **`caption.py`** in this repo to write the `.txt` sidecars:

```bash
# baseline — every caption is "<trigger>, <class>" (safe, usually enough)
python caption.py ./dataset --trigger ch_m4k0t0 --class woman

# auto scene-context via a local Ollama vision model (facial terms auto-stripped)
python caption.py ./dataset --trigger ch_m4k0t0 --class woman --vlm --ollama-model llava

# preview without writing
python caption.py ./dataset --trigger ch_m4k0t0 --class woman --vlm --dry-run
```

The script omits facial details by design and, in `--vlm` mode, strips them from the
model's output. **Always eyeball the generated `.txt` files before training.**

## Training (musubi-tuner) — LOW-noise expert first

Train against the **low-noise** Wan 2.2 I2V model only, restricted to low-noise timesteps:

- `--min_timestep 0 --max_timestep 875` (low-noise band; high band is 875–1000)
- Rank/dim **16–32**, alpha **16** (start 16/16 for one character; 32 for tougher likeness)
- LR **5e-5** (Wan punishes high LR with plasticky skin; keep it low for identity)
- ~**1,200–2,000 steps** (12–20 imgs) or ~**10 epochs**; save every epoch, pick by validation
- Resolution **512**, batch **1**, grad-accum **4**, fp8 base, adamw
- VRAM ~12 GB (the 3090's 24 GB is ample)

Pre-cache first (mandatory):

```
wan_cache_latents.py --dataset_config dataset.toml --vae <wan2.2_vae>
wan_cache_text_encoder_outputs.py --dataset_config dataset.toml --t5 <umt5_xxl_enc> --clip <xlm_roberta_clip>
```

`musubi-tuner-ui` should wrap most of this.

### Sequencing (risk-managed)

1. Train **low-noise only** first. Test. This is the safe, identity-without-static path.
2. Only if identity still drifts under heavy motion, optionally add a *weak* high-noise
   LoRA (same dataset, `--min_timestep 875 --max_timestep 1000`) and apply it at low
   strength.

## Inference / apply weights (maps to the daemon's high/low system)

- **Character LoRA:** low_weight **0.6–0.8** (start 0.7), high_weight **0** (or ≤0.4 only
  if identity fades). For clips >5 s, push low to **0.8–0.9** to prevent identity fade-out.
- **Motion LoRA (existing):** high_weight ~**0.8**, low_weight ~**0.2** (validated finding).
- In the Wanly LoRA library: register the character LoRA with `low_file` = the trained
  low-noise LoRA, `high_file` empty (or the weak high one), `default_low_weight` 0.7,
  `default_high_weight` 0.
- Net: motion from the high expert, identity + expression from the low expert,
  **no faceswap**.

## Caveats

- LoRA training is empirical — these are strong starting points; expect **2–3 iterations**
  (dataset balance + step count are the usual knobs).
- "Train both experts ON" is the *general* advice; for an identity LoRA whose failure mode
  is static, **low-noise-first** is the safer call. Revisit if needed.
- The daemon already supports 3 simultaneous user LoRAs with per-expert weights, so no
  pipeline changes are needed to *apply* this. Possible later win: a "character LoRA" preset.

## Sources

- HF Wan 2.2 LoRA training doc — images-only, hyperparams, cache commands
- Civitai "My WAN2.2 LoRA training workflow TLDR" — timestep ranges, dim/alpha/LR
- wan27.org Wan 2.2 LoRA training guide — image count, low-noise-for-identity, inference strengths
- WaveSpeed Wan 2.2 LoRA settings — LR/steps, reg images, trigger words
- kohya-ss/musubi-tuner — trainer, dual-expert training
- RunComfy Wan 2.2 I2V character-consistency trainer
