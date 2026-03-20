#!/usr/bin/env python3
"""
Vision-Language Model (VLM) violence/highlight detector.
Samples keyframes and sends them to Claude for semantic rating.
Most accurate but slowest approach — uses actual scene understanding.

Requires: ANTHROPIC_API_KEY environment variable.
"""

import os
import sys
import json
import base64
import subprocess
import tempfile
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import cv2
import numpy as np
from decord import VideoReader, cpu

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_DIR = os.environ.get("VIDEO_DIR", "./videos")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./highlights_vlm")
SAMPLE_INTERVAL = 5  # sample one frame every N seconds (lower = more API calls)
BATCH_FRAMES = 4     # number of frames to send per API call (as a grid)
TOP_N = int(os.environ.get("TOP_N", "20"))
WINDOW_SEC = int(os.environ.get("WINDOW_SEC", "15"))
MODEL = "claude-sonnet-4-20250514"

# What to look for — customize this prompt for different games/criteria
RATING_PROMPT = """Rate each of these 4 sequential gameplay frames (from a Red Dead Redemption 2 session)
on a scale of 0-10 for VIOLENCE INTENSITY. Consider:
- Gunfire, muzzle flash, shooting (high score)
- Fistfights, melee combat, beatings (high score)
- Explosions, destruction (high score)
- Knife/weapon attacks (high score)
- Dead bodies, aftermath of violence (medium score)
- Tense standoffs before violence (medium score)
- Calm scenes, walking, riding peacefully (low score)
- Menus, loading screens, UI (score 0)

Respond ONLY with a JSON object like: {"scores": [7, 8, 3, 1], "description": "brief description of what's happening"}
No other text."""


def sample_keyframes(video_path: str, interval_sec: int) -> list[tuple[float, np.ndarray]]:
    """Sample frames at regular intervals. Returns list of (timestamp_sec, frame)."""
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total = len(vr)
    step = max(1, int(fps * interval_sec))

    results = []
    for idx in range(0, total, step):
        frame = vr[idx].asnumpy()
        t_sec = idx / fps
        results.append((t_sec, frame))
    del vr
    return results


def frames_to_grid(frames: list[np.ndarray], cols: int = 2) -> bytes:
    """Arrange frames into a grid image and return as JPEG bytes."""
    rows_needed = (len(frames) + cols - 1) // cols
    # Resize all to same size
    target_h, target_w = 360, 640
    resized = []
    for f in frames:
        r = cv2.resize(f, (target_w, target_h))
        resized.append(r)

    # Pad if needed
    while len(resized) < rows_needed * cols:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    # Build grid
    grid_rows = []
    for r in range(rows_needed):
        row = np.hstack(resized[r * cols : (r + 1) * cols])
        grid_rows.append(row)
    grid = np.vstack(grid_rows)

    # Encode as JPEG
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def rate_frames_vlm(grid_bytes: bytes, api_key: str) -> dict:
    """Send frame grid to Claude and get violence ratings."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    b64 = base64.b64encode(grid_bytes).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                },
                {"type": "text", "text": RATING_PROMPT},
            ],
        }],
    )

    text = response.content[0].text.strip()
    # Parse JSON from response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"scores": [0, 0, 0, 0], "description": "parse error"}


def extract_clip(video_path: str, start_sec: float, duration: float, output_path: str):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(max(0, start_sec)),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ],
        capture_output=True, check=True,
    )


def concatenate_clips(clip_paths: list[str], output_path: str):
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
    parser = argparse.ArgumentParser(description="VLM-based violence detection in gameplay videos")
    parser.add_argument("video_dir", nargs="?", help="Directory containing video files")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-n", "--top-n", type=int, help="Number of top moments to extract")
    parser.add_argument("-w", "--window", type=int, help="Clip duration in seconds")
    parser.add_argument("--interval", type=int, default=SAMPLE_INTERVAL, help="Seconds between sampled frames")
    parser.add_argument("--dry-run", action="store_true", help="Show frame count and estimated API cost without calling API")
    args = parser.parse_args()

    video_dir = args.video_dir or VIDEO_DIR
    output_dir = args.output or OUTPUT_DIR
    top_n = args.top_n or TOP_N
    window_sec = args.window or WINDOW_SEC
    interval = args.interval

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith(".mp4")
    ])
    print(f"Found {len(video_files)} videos")

    # Estimate cost
    total_frames = 0
    for vpath in video_files:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", vpath],
            capture_output=True, text=True,
        )
        dur = float(result.stdout.strip())
        total_frames += int(dur / interval)

    api_calls = (total_frames + BATCH_FRAMES - 1) // BATCH_FRAMES
    # Rough cost: ~$0.003 per image call with Sonnet
    est_cost = api_calls * 0.004
    print(f"Total frames to analyze: {total_frames}")
    print(f"API calls needed: {api_calls} (batches of {BATCH_FRAMES})")
    print(f"Estimated cost: ~${est_cost:.2f}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without making API calls.")
        return

    print(f"\nProcessing...\n")

    all_segments = []

    for vi, vpath in enumerate(video_files):
        name = Path(vpath).stem
        print(f"[{vi+1}/{len(video_files)}] {name}")

        keyframes = sample_keyframes(vpath, interval)
        print(f"  {len(keyframes)} keyframes")

        # Process in batches of BATCH_FRAMES
        for bi in range(0, len(keyframes), BATCH_FRAMES):
            batch = keyframes[bi : bi + BATCH_FRAMES]
            if len(batch) < 2:
                continue  # skip tiny batches

            timestamps = [t for t, _ in batch]
            frames = [f for _, f in batch]

            # Pad to BATCH_FRAMES if needed
            while len(frames) < BATCH_FRAMES:
                frames.append(frames[-1])
                timestamps.append(timestamps[-1])

            grid = frames_to_grid(frames[:BATCH_FRAMES])
            result = rate_frames_vlm(grid, api_key)
            scores = result.get("scores", [0] * BATCH_FRAMES)
            desc = result.get("description", "")

            for fi in range(min(len(batch), BATCH_FRAMES)):
                if scores[fi] >= 5:  # only keep scores >= 5
                    all_segments.append({
                        "video": vpath,
                        "video_name": name,
                        "time_sec": timestamps[fi],
                        "score": scores[fi] / 10.0,
                        "vlm_score": scores[fi],
                        "description": desc,
                    })

            print(f"  batch {bi//BATCH_FRAMES + 1}: scores={scores} - {desc}")

    # Rank and extract
    all_segments.sort(key=lambda x: x["score"], reverse=True)
    top = all_segments[:top_n]

    print(f"\n{'='*70}")
    print(f"Top {len(top)} violent moments (VLM-rated):")
    print(f"{'='*70}")

    clip_paths = []
    for i, seg in enumerate(top):
        start = max(0, seg["time_sec"] - window_sec / 2)
        print(f"  #{i+1}: {seg['video_name']} @ {seg['time_sec']:.0f}s "
              f"(vlm={seg['vlm_score']}/10) {seg['description']}")

        output_file = os.path.join(output_dir, f"vlm_highlight_{i+1:02d}.mp4")
        extract_clip(seg["video"], start, window_sec, output_file)
        clip_paths.append(output_file)

    # Save metadata
    with open(os.path.join(output_dir, "vlm_highlights.json"), "w") as f:
        json.dump(top, f, indent=2)

    # Compilation
    if clip_paths:
        comp_path = os.path.join(output_dir, "vlm_compilation.mp4")
        print(f"\nJoining {len(clip_paths)} clips...")
        concatenate_clips(clip_paths, comp_path)
        print(f"Compilation: {comp_path}")


if __name__ == "__main__":
    main()
