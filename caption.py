#!/usr/bin/env python3
"""
Caption images for a Wan 2.2 character LoRA.

Writes a same-name `.txt` sidecar next to each image (0001.jpg -> 0001.txt) in the
form the trainer expects. The captioning philosophy for an *identity* LoRA:

    <trigger>, <class>[, <scene context>]

  - The trigger token binds identity.  KEEP IT SHORT AND UNIQUE (e.g. ch_m4k0t0).
  - The class anchors the LoRA to the base model's prior (e.g. woman, man).
  - Scene context (clothing, framing, lighting, setting) is OK and helps generalize.
  - FACIAL DETAILS ARE DELIBERATELY OMITTED.  Describing eyes/nose/smile/skin competes
    with the trigger, weakens identity binding, and can flatten expression in the
    generated video.  This script strips those words automatically in --vlm mode.

Two modes:
  1. Baseline (default): every caption is `<trigger>, <class>` plus any --context you
     pass.  Deterministic, safe, and honestly enough for most single-character LoRAs.
  2. --vlm: auto-describe each image with a local Ollama vision model, strip facial
     descriptors, and append the cleaned scene context after the trigger+class.

Examples:
    # baseline — fastest, safest
    python caption.py ./dataset --trigger ch_m4k0t0 --class woman

    # add a fixed context fragment to every image
    python caption.py ./dataset --trigger ch_m4k0t0 --class woman --context "studio portrait"

    # auto-context via Ollama (must have a vision model pulled, e.g. `ollama pull llava`)
    python caption.py ./dataset --trigger ch_m4k0t0 --class woman --vlm --ollama-model llava

    # preview without writing files
    python caption.py ./dataset --trigger ch_m4k0t0 --class woman --vlm --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Words/phrases describing the FACE — stripped from VLM output so they don't compete
# with the trigger token. Matched case-insensitively as whole words.
FACIAL_TERMS = {
    "face", "facial", "eye", "eyes", "eyebrow", "eyebrows", "eyelash", "eyelashes",
    "nose", "lip", "lips", "mouth", "teeth", "smile", "smiling", "smirk", "frown",
    "grin", "grinning", "cheek", "cheeks", "cheekbone", "cheekbones", "jaw", "jawline",
    "chin", "freckle", "freckles", "skin", "complexion", "wrinkle", "wrinkles",
    "expression", "gaze", "staring", "looking", "eyeshadow", "lipstick", "makeup",
    "beautiful", "pretty", "attractive", "gorgeous", "handsome", "cute",
    "young", "old", "age", "aged",  # let the trigger own age/identity
}

# Filler the VLM loves to emit that adds no training signal.
FILLER_PHRASES = [
    "the image shows", "this image shows", "this is an image of", "an image of",
    "a photo of", "a picture of", "the photo shows", "this photo shows",
    "a portrait of", "a close-up of", "a closeup of", "there is", "we can see",
    "the picture shows", "depicts", "featuring",
]

OLLAMA_PROMPT = (
    "Describe only the clothing, body pose/framing, lighting, and background/setting of "
    "the single person in this image. Be brief (one short phrase, comma-separated). "
    "Do NOT describe the face, eyes, expression, age, or attractiveness."
)


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def ollama_caption(image_path: Path, model: str, host: str) -> str:
    """Query a local Ollama vision model for a scene description."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "prompt": OLLAMA_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "").strip()


def clean_context(raw: str) -> str:
    """Lowercase, strip filler + facial terms, dedupe, return a comma-joined fragment."""
    text = raw.lower().replace("\n", " ")
    for filler in FILLER_PHRASES:
        text = text.replace(filler, " ")
    # Split into candidate phrases on commas / sentence punctuation.
    parts = re.split(r"[,.;]+", text)
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        words = [w for w in re.findall(r"[a-z0-9\-]+", part) if w not in FACIAL_TERMS]
        # drop empties and tiny stopword-only fragments
        phrase = " ".join(words).strip()
        phrase = re.sub(r"^(a|an|the|of|with|wearing|in|on)\s+", "", phrase).strip()
        if len(phrase) < 3 or phrase in seen:
            continue
        seen.add(phrase)
        cleaned.append(phrase)
    return ", ".join(cleaned[:6])  # cap context length


def build_caption(trigger: str, klass: str, context: str) -> str:
    pieces = [trigger.strip(), klass.strip()]
    if context.strip():
        pieces.append(context.strip())
    return ", ".join(p for p in pieces if p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="Directory of images to caption")
    ap.add_argument("--trigger", required=True, help="Unique trigger token, e.g. ch_m4k0t0")
    ap.add_argument("--class", dest="klass", required=True, help="Class word, e.g. woman / man")
    ap.add_argument("--context", default="", help="Fixed context fragment appended to every caption")
    ap.add_argument("--vlm", action="store_true", help="Auto-describe scene via Ollama vision model")
    ap.add_argument("--ollama-model", default="llava", help="Ollama vision model (default: llava)")
    ap.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API host")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    ap.add_argument("--dry-run", action="store_true", help="Print captions, write nothing")
    args = ap.parse_args()

    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a directory", file=sys.stderr)
        return 1

    images = find_images(args.folder)
    if not images:
        print(f"error: no images ({', '.join(sorted(IMAGE_EXTS))}) in {args.folder}", file=sys.stderr)
        return 1

    print(f"Captioning {len(images)} image(s) in {args.folder}")
    print(f"  trigger={args.trigger!r}  class={args.klass!r}  vlm={args.vlm}\n")

    written = skipped = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if txt.exists() and not args.overwrite and not args.dry_run:
            print(f"  skip   {img.name}  (txt exists; use --overwrite)")
            skipped += 1
            continue

        context = args.context
        if args.vlm:
            try:
                raw = ollama_caption(img, args.ollama_model, args.ollama_host)
                vlm_ctx = clean_context(raw)
                context = ", ".join(p for p in [args.context.strip(), vlm_ctx] if p)
            except Exception as e:  # noqa: BLE001 - report and fall back to baseline
                print(f"  warn   {img.name}  VLM failed ({e}); using baseline")

        caption = build_caption(args.trigger, args.klass, context)
        if args.dry_run:
            print(f"  [dry]  {img.name} -> {caption}")
        else:
            txt.write_text(caption + "\n", encoding="utf-8")
            print(f"  write  {img.name} -> {caption}")
            written += 1

    print(f"\nDone. wrote={written} skipped={skipped} dry_run={args.dry_run}")
    print("Review the .txt files before training — trim anything that slipped through.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
