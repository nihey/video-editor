#!/usr/bin/env python3
"""
Remove RDR2 radial wheel menu segments from gameplay videos.
Detects menu by color saturation — the wheel menu desaturates the screen
to near-grayscale, which is a strong and reliable signal.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import cv2
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLE_FPS = 5          # frames per second to sample for detection
SAT_THRESHOLD = 35      # mean saturation below this = menu open (0-255 scale)
MIN_MENU_SEC = 0.5      # ignore menu detections shorter than this
PAD_SEC = 0.15          # trim extra around menu boundaries to avoid transition frames


def get_video_info(video_path: str) -> dict:
    """Get video duration and fps."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries",
         "format=duration:stream=r_frame_rate,width,height",
         "-select_streams", "v:0", "-of", "json", video_path],
        capture_output=True, text=True,
    )
    import json
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    stream = info["streams"][0]
    r_num, r_den = map(int, stream["r_frame_rate"].split("/"))
    fps = r_num / r_den
    return {"duration": duration, "fps": fps, "width": stream["width"], "height": stream["height"]}


def detect_menu_frames(video_path: str, sample_fps: int = SAMPLE_FPS,
                       sat_threshold: float = SAT_THRESHOLD,
                       min_menu_sec: float = MIN_MENU_SEC) -> list[tuple[float, float]]:
    """
    Detect segments where the wheel menu is open by checking color saturation.
    Returns list of (start_sec, end_sec) tuples for menu segments.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps
    frame_interval = max(1, int(native_fps / sample_fps))

    print(f"  Video: {duration:.1f}s, {native_fps:.0f}fps, sampling every {frame_interval} frames")

    saturations = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Downscale for speed
            small = cv2.resize(frame, (320, 180))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mean_sat = float(np.mean(hsv[:, :, 1]))
            saturations.append(mean_sat)
            timestamps.append(frame_idx / native_fps)

        frame_idx += 1

    cap.release()

    if not saturations:
        return []

    saturations = np.array(saturations)
    timestamps = np.array(timestamps)

    print(f"  Saturation stats: min={saturations.min():.1f}, max={saturations.max():.1f}, "
          f"mean={saturations.mean():.1f}, median={np.median(saturations):.1f}")

    # Detect menu frames: low saturation
    is_menu = saturations < sat_threshold

    # Find contiguous menu segments
    segments = []
    in_menu = False
    start_t = 0.0

    for i, (t, menu) in enumerate(zip(timestamps, is_menu)):
        if menu and not in_menu:
            start_t = t
            in_menu = True
        elif not menu and in_menu:
            segments.append((start_t, t))
            in_menu = False

    if in_menu:
        segments.append((start_t, timestamps[-1]))

    # Filter out very short detections (likely false positives)
    segments = [(s, e) for s, e in segments if (e - s) >= min_menu_sec]

    # Add padding to trim transition frames
    segments = [(max(0, s - PAD_SEC), min(duration, e + PAD_SEC)) for s, e in segments]

    return segments


def build_keep_segments(menu_segments: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Invert menu segments to get the parts we want to keep."""
    if not menu_segments:
        return [(0, duration)]

    keep = []
    cursor = 0.0

    for menu_start, menu_end in sorted(menu_segments):
        if cursor < menu_start:
            keep.append((cursor, menu_start))
        cursor = menu_end

    if cursor < duration:
        keep.append((cursor, duration))

    # Filter out tiny remnants
    keep = [(s, e) for s, e in keep if (e - s) > 0.1]

    return keep


def extract_and_join(video_path: str, keep_segments: list[tuple[float, float]], output_path: str):
    """Extract kept segments and concatenate them."""
    tmpdir = tempfile.mkdtemp(prefix="remove_menu_")
    clip_paths = []

    for i, (start, end) in enumerate(keep_segments):
        dur = end - start
        out = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "21",
                "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                out,
            ],
            capture_output=True, check=True,
        )
        if os.path.getsize(out) > 1000:
            clip_paths.append(out)

    if not clip_paths:
        print("ERROR: No segments extracted!")
        return

    # Concatenate
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

    # Cleanup
    for p in clip_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remove RDR2 wheel menu segments from video")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output file path (default: input_no_menu.mp4)")
    parser.add_argument("--threshold", type=float, default=SAT_THRESHOLD,
                        help=f"Saturation threshold for menu detection (default: {SAT_THRESHOLD})")
    parser.add_argument("--min-menu", type=float, default=MIN_MENU_SEC,
                        help=f"Minimum menu segment duration in seconds (default: {MIN_MENU_SEC})")
    parser.add_argument("--dry-run", action="store_true", help="Detect menu segments without cutting")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).with_name(f"{stem}_no_menu.mp4"))

    sat_threshold = args.threshold
    min_menu = args.min_menu

    info = get_video_info(input_path)
    print(f"Input: {input_path}")
    print(f"  Duration: {info['duration']:.1f}s, FPS: {info['fps']:.0f}")

    print(f"\nDetecting wheel menu (saturation < {sat_threshold})...")
    menu_segments = detect_menu_frames(input_path, sat_threshold=sat_threshold, min_menu_sec=min_menu)

    if not menu_segments:
        print("\nNo menu segments detected! Video is clean.")
        return

    total_menu = sum(e - s for s, e in menu_segments)
    print(f"\nFound {len(menu_segments)} menu segment(s), total {total_menu:.1f}s:")
    for i, (s, e) in enumerate(menu_segments):
        print(f"  [{i+1}] {s:.1f}s - {e:.1f}s ({e-s:.1f}s)")

    if args.dry_run:
        print(f"\n[DRY RUN] Would remove {total_menu:.1f}s of menu, "
              f"keeping {info['duration'] - total_menu:.1f}s")
        return

    keep = build_keep_segments(menu_segments, info["duration"])
    total_keep = sum(e - s for s, e in keep)
    print(f"\nKeeping {len(keep)} segment(s), total {total_keep:.1f}s")

    print(f"Extracting and joining...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    extract_and_join(input_path, keep, output_path)

    final_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput: {output_path}")
    print(f"  Size: {final_size:.1f}MB")
    print(f"  Removed: {total_menu:.1f}s of menu UI")


if __name__ == "__main__":
    main()
