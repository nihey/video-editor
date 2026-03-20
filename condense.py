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
from scipy.ndimage import uniform_filter1d, median_filter
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
    Detect gunshot transients via onset detection with adaptive thresholding.
    Returns list of {time_sec, strength} dicts with no overlapping regions.
    """
    # 1. Short-time energy in 10ms windows, 5ms hop
    win_samples = int(rate * 0.01)
    hop = win_samples // 2

    n_windows = (len(audio) - win_samples) // hop
    if n_windows <= 0:
        return []

    energy = np.zeros(n_windows)
    for i in range(n_windows):
        s = i * hop
        energy[i] = np.sum(audio[s : s + win_samples] ** 2)

    # 2. Onset strength: positive energy derivative (attack)
    onset_env = np.maximum(np.diff(energy), 0)
    if len(onset_env) < 100:
        return []

    # 3. Adaptive threshold: subtract local median to ignore sustained loud sections
    #    (music, ambient noise). Only sharp spikes above local background remain.
    median_win = int(0.5 * rate / hop)  # 500ms median window
    local_median = median_filter(onset_env, size=median_win)
    onset_adapted = np.maximum(onset_env - local_median, 0)

    # 4. Light smoothing (NOT max-filter which creates plateaus)
    smooth_win = max(3, int(0.02 * rate / hop))  # 20ms gaussian-like
    onset_smooth = uniform_filter1d(onset_adapted, size=smooth_win)

    # 5. Find peaks with strong constraints
    min_distance = int(MIN_GAP_SEC * rate / hop)
    threshold = np.percentile(onset_smooth[onset_smooth > 0], 85) if np.any(onset_smooth > 0) else 0
    if threshold <= 0:
        return []

    peaks, props = find_peaks(
        onset_smooth,
        height=threshold,
        distance=min_distance,
        prominence=threshold * 0.5,
    )

    shots = []
    for peak in peaks:
        t_sec = (peak * hop) / rate
        shots.append({"time_sec": t_sec, "strength": float(onset_smooth[peak])})

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

        # Build time ranges and merge overlapping ones
        ranges = []
        for j, shot in enumerate(shots):
            t = shot["time_sec"]
            clip_dur = min_clip + norm_strengths[j] * (max_clip - min_clip)
            start = max(0, t - PAD_BEFORE)
            end = min(duration - 0.1, start + clip_dur)  # stay 100ms from video end
            start = max(0, end - clip_dur)  # re-adjust start if end was clamped
            strength = float(norm_strengths[j])
            ranges.append((start, end, strength))

        # Sort by start time and merge overlapping ranges
        ranges.sort(key=lambda r: r[0])
        merged = [ranges[0]]
        for start, end, strength in ranges[1:]:
            prev_start, prev_end, prev_strength = merged[-1]
            if start < prev_end:
                # Overlapping: merge, keep the stronger strength
                merged[-1] = (prev_start, max(prev_end, end), max(prev_strength, strength))
            else:
                merged.append((start, end, strength))

        # Skip clips that start in the last 0.5s (too close to end, causes glitches)
        merged = [(s, e, st) for s, e, st in merged if s < duration - 0.5]

        # Extract micro-clips from merged ranges
        extracted = 0
        for j, (start, end, strength) in enumerate(merged):
            clip_dur = end - start
            out_path = os.path.join(tmpdir, f"micro_{i:03d}_{j:03d}.mp4")
            try:
                extract_microclip(clip_path, start, clip_dur, out_path)
                if os.path.getsize(out_path) > 1000:
                    all_microclips.append({
                        "path": out_path,
                        "source": name,
                        "time": start,
                        "duration": clip_dur,
                        "strength": strength,
                    })
                    extracted += 1
            except subprocess.CalledProcessError:
                pass

        print(f" -> {len(merged)} merged ranges -> extracted {extracted} micro-clips")

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
