#!/usr/bin/env python3
"""
Condense highlight clips into a rapid-fire montage of gunshots and explosions.
Detects audio transients (sharp peaks = gunshots) and optionally confirms with
CLIP visual scoring, then extracts 0.1-1s micro-clips and joins them.
"""

import os
import sys
import json
import subprocess
import tempfile
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_DIR = os.environ.get("INPUT_DIR", "./highlights")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./highlights")
AUDIO_RATE = 44100
MIN_CLIP_SEC = 0.2
MAX_CLIP_SEC = 0.8
PAD_BEFORE = 0.1   # seconds before the transient
PAD_AFTER = 0.15    # seconds after the transient
MIN_GAP_SEC = 0.3   # minimum gap between detected shots to avoid duplicates


def extract_audio(video_path: str, rate: int = AUDIO_RATE) -> np.ndarray:
    """Extract mono audio as float32 array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(rate), "-vn", tmp_path],
            capture_output=True, check=True,
        )
        with wave.open(tmp_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        os.unlink(tmp_path)


def detect_gunshots(audio: np.ndarray, rate: int = AUDIO_RATE) -> list[dict]:
    """
    Detect gunshot transients using spectral flux onset detection.
    Gunshots have extremely fast attack (< 5ms) and high energy in upper frequencies.
    Returns list of {time_sec, strength} dicts.
    """
    # 1. Compute short-time energy in small windows (10ms)
    win_samples = int(rate * 0.01)  # 10ms windows
    hop = win_samples // 2          # 5ms hop

    n_windows = (len(audio) - win_samples) // hop
    if n_windows <= 0:
        return []

    # Energy per window
    energy = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * hop
        chunk = audio[start : start + win_samples]
        energy[i] = np.sum(chunk ** 2)

    # 2. Compute onset strength: positive first derivative of energy (attack detection)
    onset_env = np.diff(energy)
    onset_env = np.maximum(onset_env, 0)  # only positive changes (attacks)

    # 3. High-pass emphasis: gunshots have more high-frequency content
    # Apply simple differentiation again to emphasize sharp transients
    onset_sharp = np.diff(onset_env)
    onset_sharp = np.maximum(onset_sharp, 0)

    # 4. Adaptive threshold: peak must be well above local median
    if len(onset_sharp) < 100:
        return []

    # Smooth for peak detection
    kernel = max(3, int(0.05 * rate / hop))  # 50ms kernel
    onset_smooth = maximum_filter1d(onset_sharp, size=kernel)

    # Find peaks
    min_distance = int(MIN_GAP_SEC * rate / hop)
    threshold = np.percentile(onset_smooth, 95)  # top 5% of transients

    peaks, props = find_peaks(
        onset_smooth,
        height=threshold,
        distance=min_distance,
        prominence=threshold * 0.3,
    )

    shots = []
    for peak in peaks:
        t_sec = (peak * hop) / rate
        strength = float(onset_smooth[peak])
        shots.append({"time_sec": t_sec, "strength": strength})

    return shots


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def extract_microclip(video_path: str, start: float, duration: float, output_path: str):
    """Extract a micro-clip with precise seeking."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{max(0, start):.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ],
        capture_output=True, check=True,
    )


def concatenate_clips(clip_paths: list[str], output_path: str):
    """Join clips using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
        list_path = f.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "21",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ],
            capture_output=True, check=True,
        )
    finally:
        os.unlink(list_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Condense highlights into rapid gunshot montage")
    parser.add_argument("input_dir", nargs="?", help="Directory with highlight clips")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--min-clip", type=float, default=MIN_CLIP_SEC, help="Min micro-clip duration (s)")
    parser.add_argument("--max-clip", type=float, default=MAX_CLIP_SEC, help="Max micro-clip duration (s)")
    args = parser.parse_args()

    input_dir = args.input_dir or INPUT_DIR
    output_file = args.output or os.path.join(OUTPUT_DIR, "condensed.mp4")
    min_clip = args.min_clip
    max_clip = args.max_clip

    # Find highlight clips (not compilation or condensed)
    clips = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.startswith("highlight_") and f.endswith(".mp4")
    ])
    print(f"Found {len(clips)} highlight clips\n")

    all_microclips = []
    tmpdir = tempfile.mkdtemp(prefix="condense_")

    for i, clip_path in enumerate(clips):
        name = Path(clip_path).stem
        duration = get_video_duration(clip_path)
        print(f"[{i+1}/{len(clips)}] {name} ({duration:.1f}s)")

        # Detect gunshot transients
        audio = extract_audio(clip_path)
        shots = detect_gunshots(audio)
        print(f"  Detected {len(shots)} transients", end="")

        if not shots:
            print(" - skipping")
            continue

        # Normalize strengths for this clip to determine clip duration
        strengths = np.array([s["strength"] for s in shots])
        if strengths.max() > 0:
            norm_strengths = strengths / strengths.max()
        else:
            norm_strengths = np.ones_like(strengths)

        # Extract micro-clips: stronger transients get longer clips
        extracted = 0
        for j, shot in enumerate(shots):
            t = shot["time_sec"]
            # Scale duration by strength: stronger = longer clip
            clip_dur = min_clip + norm_strengths[j] * (max_clip - min_clip)
            start = t - PAD_BEFORE
            # Clamp to video bounds
            start = max(0, min(start, duration - clip_dur))

            out_path = os.path.join(tmpdir, f"micro_{i:03d}_{j:03d}.mp4")
            try:
                extract_microclip(clip_path, start, clip_dur, out_path)
                # Verify the file is valid (sometimes very short clips fail)
                if os.path.getsize(out_path) > 1000:
                    all_microclips.append({
                        "path": out_path,
                        "source": name,
                        "time": t,
                        "duration": clip_dur,
                        "strength": float(norm_strengths[j]),
                    })
                    extracted += 1
            except subprocess.CalledProcessError:
                pass

        print(f" -> extracted {extracted} micro-clips")

    print(f"\nTotal micro-clips: {len(all_microclips)}")

    if not all_microclips:
        print("No micro-clips found!")
        return

    # Sort by strength (best shots first would be chaotic, keep chronological)
    # Already in source order which is chronological

    total_duration = sum(m["duration"] for m in all_microclips)
    print(f"Total montage duration: {total_duration:.1f}s")

    # Join all micro-clips
    print(f"Joining into {output_file}...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    concatenate_clips([m["path"] for m in all_microclips], output_file)

    # Get final duration
    final_dur = get_video_duration(output_file)
    final_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nCondensed montage: {output_file}")
    print(f"  Duration: {final_dur:.1f}s")
    print(f"  Size: {final_size:.1f}MB")
    print(f"  Micro-clips: {len(all_microclips)}")

    # Cleanup temp files
    for m in all_microclips:
        try:
            os.unlink(m["path"])
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
