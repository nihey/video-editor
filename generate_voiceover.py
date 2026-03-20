#!/usr/bin/env python3
"""
Generate AI voiceover narration for violence highlight segments.
Uses Qwen3-TTS to create dramatic narration based on segment scores and labels.
Produces WAV files that can be layered over the video in Remotion.
"""

import os
import sys
import json
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"


# ── Narration templates ─────────────────────────────────────────────────────

def generate_narration_text(segments: list[dict]) -> list[dict]:
    """
    Generate dramatic narration lines for each segment based on score.
    Returns list of {label, text, score} dicts.
    """
    narrations = []

    for seg in segments:
        score = seg["score"]
        label = seg["label"]
        duration = seg["duration"]

        # Skip very short segments — no time for narration
        if duration < 2.0:
            continue

        # Pick narration style based on score intensity
        if score > 0.55:
            # Peak violence — short, intense callouts
            lines = [
                "Here we go.",
                "This is it.",
                "Watch this.",
                "Absolute chaos.",
                "No mercy.",
                "Total destruction.",
                "They never saw it coming.",
                "And just like that, it's over.",
            ]
        elif score > 0.4:
            # Mid-high — building tension
            lines = [
                "Things are heating up.",
                "Trouble ahead.",
                "This gets intense.",
                "Stay sharp.",
                "It's about to go down.",
                "The action picks up.",
            ]
        else:
            # Lower — scene setting
            lines = [
                "And then...",
                "Meanwhile...",
                "The calm before the storm.",
                "Something's not right.",
            ]

        # Deterministic selection based on label to keep it consistent
        idx = sum(ord(c) for c in label) % len(lines)
        narrations.append({
            "label": label,
            "text": lines[idx],
            "score": score,
        })

    return narrations


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate AI voiceover for violence highlights")
    parser.add_argument("manifest", help="Manifest JSON from condense_violence.py")
    parser.add_argument("-o", "--output-dir", help="Output directory for WAV files")
    parser.add_argument("--speaker", default="Ryan", help="TTS speaker voice (default: Ryan)")
    parser.add_argument("--dry-run", action="store_true", help="Show narration text without generating audio")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                        help="TTS model name (default: 0.6B for 8GB GPU)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    segments = manifest["segments"]
    narrations = generate_narration_text(segments)

    if not narrations:
        print("No narrations to generate (all segments too short)")
        return

    print(f"Generated {len(narrations)} narration lines:")
    for n in narrations:
        print(f"  [{n['label']}] (score={n['score']:.3f}) \"{n['text']}\"")

    if args.dry_run:
        return

    # Output directory
    output_dir = args.output_dir or str(Path(args.manifest).parent / "voiceover")
    os.makedirs(output_dir, exist_ok=True)

    # Load TTS model
    print(f"\nLoading TTS model: {args.model}...")
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    # Generate voiceover for each narration
    for i, n in enumerate(narrations):
        print(f"  [{n['label']}] Generating: \"{n['text']}\"...", end=" ", flush=True)

        wavs, sr = model.generate_custom_voice(
            text=n["text"],
            language="English",
            speaker=args.speaker,
            instruct="Speak with a deep, dramatic action movie narrator voice. Short and punchy.",
        )

        out_path = os.path.join(output_dir, f"vo_{n['label']}.wav")
        sf.write(out_path, wavs[0], sr)
        print(f"saved ({os.path.getsize(out_path) / 1024:.0f}KB)")

    # Update manifest with voiceover paths
    vo_manifest = {
        "voiceover_dir": os.path.abspath(output_dir),
        "speaker": args.speaker,
        "narrations": narrations,
        "files": {n["label"]: f"vo_{n['label']}.wav" for n in narrations},
    }

    vo_manifest_path = os.path.join(output_dir, "voiceover.json")
    with open(vo_manifest_path, "w") as f:
        json.dump(vo_manifest, f, indent=2)

    print(f"\nVoiceover manifest: {vo_manifest_path}")
    print(f"Generated {len(narrations)} clips in {output_dir}")
    print(f"\nTo use with Remotion, copy WAV files to renderer/public/voiceover/")


if __name__ == "__main__":
    main()
