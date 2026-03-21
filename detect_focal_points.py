#!/usr/bin/env python3
"""
Detect focal points (action center) per segment for smart vertical reframing.
Splits each frame into a horizontal grid and scores each column using optical flow
magnitude — the column with most motion is where the action is.
Outputs focal_x (0.0=left, 0.5=center, 1.0=right) per segment into the manifest.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import cv2
import numpy as np


SAMPLE_FPS = 4  # frames per second to analyze
GRID_COLS = 5   # split frame into 5 horizontal columns
SMOOTH_WINDOW = 3  # smooth focal point over N samples


def analyze_segment_focal(video_path: str, start: float, end: float) -> float:
    """
    Analyze a segment and return the focal X position (0.0 to 1.0).
    Uses optical flow to find where motion is concentrated horizontally.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start * native_fps)
    end_frame = int(end * native_fps)
    frame_step = max(1, int(native_fps / SAMPLE_FPS))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_gray = None
    column_scores = np.zeros(GRID_COLS)
    n_samples = 0

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_step == 0:
            # Downscale for speed
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=2, winsize=11,
                    iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

                # Split into columns and sum motion in each
                col_width = mag.shape[1] // GRID_COLS
                for c in range(GRID_COLS):
                    col_slice = mag[:, c * col_width:(c + 1) * col_width]
                    column_scores[c] += np.mean(col_slice)

                n_samples += 1

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if n_samples == 0:
        return 0.5

    # Average scores
    column_scores /= n_samples

    # Weighted average to get focal X position (0.0 to 1.0)
    col_positions = np.linspace(0, 1, GRID_COLS)
    total_weight = column_scores.sum()
    if total_weight < 1e-6:
        return 0.5

    focal_x = float(np.average(col_positions, weights=column_scores))

    # Clamp to avoid extreme edges
    focal_x = max(0.2, min(0.8, focal_x))

    return round(focal_x, 3)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect focal points per segment")
    parser.add_argument("manifest", help="Manifest JSON from condense_violence.py")
    parser.add_argument("--dry-run", action="store_true", help="Show results without modifying manifest")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    source = manifest["source"]
    segments = manifest["segments"]

    if not os.path.exists(source):
        print(f"ERROR: Source video not found: {source}")
        sys.exit(1)

    print(f"Analyzing {len(segments)} segments for focal points...")

    for seg in segments:
        focal_x = analyze_segment_focal(source, seg["start"], seg["end"])
        seg["focal_x"] = focal_x
        print(f"  [{seg['label']}] {seg['start']:.1f}-{seg['end']:.1f}s -> focal_x={focal_x:.3f}"
              f" ({'left' if focal_x < 0.4 else 'right' if focal_x > 0.6 else 'center'})")

    if args.dry_run:
        return

    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nUpdated manifest with focal_x values: {args.manifest}")


if __name__ == "__main__":
    main()
