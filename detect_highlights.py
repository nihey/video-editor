#!/usr/bin/env python3
"""
Hybrid gameplay highlight detector.
Combines CLIP semantic scoring + audio energy analysis to find intense gunfight moments.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from decord import VideoReader, cpu

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_DIR = os.environ.get("VIDEO_DIR", "./videos")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./highlights")
SAMPLE_FPS = 2  # frames per second to analyze
CLIP_MODEL = "openai/clip-vit-large-patch14"
WINDOW_SEC = int(os.environ.get("WINDOW_SEC", "15"))
TOP_N = int(os.environ.get("TOP_N", "20"))
AUDIO_WEIGHT = 0.5
CLIP_WEIGHT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text prompts for CLIP scoring
POSITIVE_PROMPTS = [
    "a gunfight with muzzle flash and smoke from a gun barrel",
    "shooting a revolver or rifle at enemies with bullet impact",
    "an intense shootout with multiple gunshots being fired",
    "aiming down sights and firing a weapon in a western game",
    "a gun battle with enemies being shot and falling down",
    "rapid gunfire with muzzle flash lighting up the scene",
]

NEGATIVE_PROMPTS = [
    "a pause menu or settings screen",
    "a loading screen",
    "a character walking calmly in nature",
    "a map or inventory screen",
    "a cutscene with characters talking quietly",
    "riding a horse peacefully through the countryside",
    "standing still looking at scenery",
]


def extract_audio_energy(video_path: str, sample_fps: int) -> np.ndarray:
    """Extract audio RMS energy at `sample_fps` resolution using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", "16000", "-vn",
                tmp_path,
            ],
            capture_output=True, check=True,
        )

        # Read raw PCM from wav
        import wave
        with wave.open(tmp_path, "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate()
            raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio /= 32768.0  # normalize to [-1, 1]

        # Compute RMS energy per window aligned with sample_fps
        samples_per_window = rate // sample_fps
        n_windows = len(audio) // samples_per_window
        if n_windows == 0:
            return np.array([0.0])

        audio = audio[: n_windows * samples_per_window]
        chunks = audio.reshape(n_windows, samples_per_window)
        rms = np.sqrt(np.mean(chunks ** 2, axis=1))
        return rms
    finally:
        os.unlink(tmp_path)


def sample_frames(video_path: str, sample_fps: int) -> list[np.ndarray]:
    """Sample frames from video at given fps using decord (memory-efficient)."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / sample_fps))
    indices = list(range(0, total_frames, step))

    # Batch in chunks of 64 to avoid OOM on long videos
    frames = []
    for i in range(0, len(indices), 64):
        chunk = indices[i : i + 64]
        batch = vr.get_batch(chunk).asnumpy()
        frames.extend([batch[j] for j in range(len(batch))])
        del batch
    return frames


def score_frames_clip(
    frames: list[np.ndarray],
    model: CLIPModel,
    processor: CLIPProcessor,
    pos_features: torch.Tensor,
    neg_features: torch.Tensor,
    batch_size: int = 32,
) -> np.ndarray:
    """Score frames using CLIP against pre-encoded text prompts."""
    all_scores = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        images = [Image.fromarray(f) for f in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            img_out = model.get_image_features(**inputs)
            img_features = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            # Cosine similarity with positive and negative prompts
            pos_sim = (img_features @ pos_features.T).mean(dim=-1)
            neg_sim = (img_features @ neg_features.T).mean(dim=-1)

            # Score = positive similarity - negative similarity
            score = pos_sim - neg_sim
            all_scores.append(score.cpu().numpy())

    return np.concatenate(all_scores)


def extract_clip(video_path: str, start_sec: float, duration: float, output_path: str):
    """Extract a clip from video using ffmpeg."""
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
    """Join multiple clips into a single video using ffmpeg concat demuxer."""
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


def process_video(
    video_path: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    pos_features: torch.Tensor,
    neg_features: torch.Tensor,
) -> list[dict]:
    """Process a single video and return scored segments."""
    name = Path(video_path).stem
    print(f"  Sampling frames...", end=" ", flush=True)
    frames = sample_frames(video_path, SAMPLE_FPS)
    print(f"{len(frames)} frames")

    print(f"  CLIP scoring...", end=" ", flush=True)
    clip_scores = score_frames_clip(frames, model, processor, pos_features, neg_features)
    print("done")

    print(f"  Audio energy...", end=" ", flush=True)
    audio_energy = extract_audio_energy(video_path, SAMPLE_FPS)
    print("done")

    # Align lengths (may differ by 1-2 samples)
    min_len = min(len(clip_scores), len(audio_energy))
    clip_scores = clip_scores[:min_len]
    audio_energy = audio_energy[:min_len]

    # Normalize each signal to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    clip_norm = normalize(clip_scores)
    audio_norm = normalize(audio_energy)

    # Hybrid score
    hybrid = CLIP_WEIGHT * clip_norm + AUDIO_WEIGHT * audio_norm

    # Smooth with a window
    smooth_window = max(3, SAMPLE_FPS * 3)  # 3 second smoothing
    hybrid_smooth = uniform_filter1d(hybrid, size=smooth_window)

    # Find peaks
    peak_distance = SAMPLE_FPS * WINDOW_SEC  # minimum distance between peaks
    peaks, properties = find_peaks(
        hybrid_smooth,
        distance=peak_distance,
        height=np.percentile(hybrid_smooth, 75),  # top 25%
    )

    segments = []
    for peak in peaks:
        t_sec = peak / SAMPLE_FPS
        segments.append({
            "video": video_path,
            "video_name": name,
            "time_sec": float(t_sec),
            "score": float(hybrid_smooth[peak]),
            "clip_score": float(clip_norm[peak]),
            "audio_score": float(audio_norm[peak]),
        })

    return segments


def main(compile_only=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not compile_only:
        # Find all videos
        video_files = sorted([
            os.path.join(VIDEO_DIR, f)
            for f in os.listdir(VIDEO_DIR)
            if f.endswith(".mp4")
        ])
        print(f"Found {len(video_files)} videos\n")

        # Load CLIP model
        print(f"Loading CLIP model ({CLIP_MODEL})...")
        model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        model.eval()

        # Pre-encode text prompts
        print("Encoding text prompts...")
        with torch.no_grad():
            pos_inputs = processor(text=POSITIVE_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
            pos_out = model.get_text_features(**pos_inputs)
            pos_features = pos_out if isinstance(pos_out, torch.Tensor) else pos_out.pooler_output
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)

            neg_inputs = processor(text=NEGATIVE_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
            neg_out = model.get_text_features(**neg_inputs)
            neg_features = neg_out if isinstance(neg_out, torch.Tensor) else neg_out.pooler_output
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

        print(f"Model loaded on {DEVICE}\n")

        # Process all videos
        all_segments = []
        for i, vpath in enumerate(video_files):
            print(f"[{i+1}/{len(video_files)}] {Path(vpath).name}")
            try:
                segs = process_video(vpath, model, processor, pos_features, neg_features)
                all_segments.extend(segs)
                print(f"  Found {len(segs)} candidate moments\n")
            except Exception as e:
                print(f"  ERROR: {e}\n")

        # Rank by hybrid score and take top N
        all_segments.sort(key=lambda x: x["score"], reverse=True)
        top = all_segments[:TOP_N]

        print(f"\n{'='*60}")
        print(f"Top {len(top)} highlights:")
        print(f"{'='*60}")

        clip_paths = []
        for i, seg in enumerate(top):
            start = max(0, seg["time_sec"] - WINDOW_SEC / 2)
            print(
                f"  #{i+1}: {seg['video_name']} @ {seg['time_sec']:.1f}s "
                f"(score={seg['score']:.3f}, clip={seg['clip_score']:.3f}, audio={seg['audio_score']:.3f})"
            )

            output_file = os.path.join(OUTPUT_DIR, f"highlight_{i+1:02d}.mp4")
            print(f"       Extracting {WINDOW_SEC}s clip -> {output_file}")
            extract_clip(seg["video"], start, WINDOW_SEC, output_file)
            clip_paths.append(output_file)

        # Save metadata
        meta_path = os.path.join(OUTPUT_DIR, "highlights.json")
        with open(meta_path, "w") as f:
            json.dump(top, f, indent=2)
        print(f"\nMetadata saved to {meta_path}")
        print(f"Clips saved to {OUTPUT_DIR}/")
    else:
        # Compile-only mode: find existing clips
        clip_paths = sorted([
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith("highlight_") and f.endswith(".mp4")
        ])
        print(f"Found {len(clip_paths)} existing highlight clips")

    # Concatenate all clips into a compilation
    if clip_paths:
        # Sort clips chronologically by source video timestamp for a natural flow
        meta_path = os.path.join(OUTPUT_DIR, "highlights.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            # Re-sort clips by source video name + timestamp for chronological order
            indexed = []
            for i, seg in enumerate(meta):
                clip_file = os.path.join(OUTPUT_DIR, f"highlight_{i+1:02d}.mp4")
                if os.path.exists(clip_file):
                    indexed.append((seg["video_name"], seg["time_sec"], clip_file))
            indexed.sort(key=lambda x: (x[0], x[1]))
            clip_paths = [x[2] for x in indexed]

        compilation_path = os.path.join(OUTPUT_DIR, "compilation.mp4")
        print(f"\nJoining {len(clip_paths)} clips into compilation...")
        concatenate_clips(clip_paths, compilation_path)
        total_dur = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", compilation_path],
            capture_output=True, text=True,
        )
        dur = float(total_dur.stdout.strip())
        print(f"Compilation saved: {compilation_path} ({dur:.0f}s / {dur/60:.1f}min)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect best moments in gameplay videos")
    parser.add_argument("video_dir", nargs="?", help="Directory containing video files")
    parser.add_argument("-o", "--output", help="Output directory for highlight clips")
    parser.add_argument("-n", "--top-n", type=int, help="Number of top highlights to extract")
    parser.add_argument("-w", "--window", type=int, help="Duration of each highlight clip in seconds")
    parser.add_argument("--compile", action="store_true", help="Only join existing clips into compilation (skip detection)")
    args = parser.parse_args()

    if args.video_dir:
        VIDEO_DIR = args.video_dir
    if args.output:
        OUTPUT_DIR = args.output
    if args.top_n:
        TOP_N = args.top_n
    if args.window:
        WINDOW_SEC = args.window

    main(compile_only=args.compile)
