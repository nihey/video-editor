#!/usr/bin/env python3
"""
Violence detector for gameplay videos.
Combines PANNs audio classification + CLIP visual scoring + optical flow
+ VideoMAE action recognition to detect gunshots, beatings, explosions,
and other intense action.
"""

import os
import sys
import json
import subprocess
import tempfile
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, VideoMAEForVideoClassification, VideoMAEImageProcessor
from decord import VideoReader, cpu
from panns_inference import AudioTagging

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_DIR = os.environ.get("VIDEO_DIR", "./videos")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./highlights")
SAMPLE_FPS = 2
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
WINDOW_SEC = int(os.environ.get("WINDOW_SEC", "15"))
TOP_N = int(os.environ.get("TOP_N", "20"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Signal weights (4 signals)
PANNS_WEIGHT = 0.35
CLIP_WEIGHT = 0.25
FLOW_WEIGHT = 0.15
VIDEOMAE_WEIGHT = 0.25

# VideoMAE config
VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
VIDEOMAE_VIOLENCE_LABELS = {
    105: "drop kicking",
    150: "headbutting",
    152: "high kick",
    259: "punching person (boxing)",
    302: "side kick",
    314: "slapping",
    345: "sword fighting",
    395: "wrestling",
}

# PANNs violence-related AudioSet label indices
VIOLENCE_AUDIO_LABELS = {
    427: "Gunshot, gunfire",
    428: "Machine gun",
    430: "Artillery fire",
    426: "Explosion",
    436: "Boom",
    466: "Bang",
    467: "Slap, smack",
    468: "Whack, thwack",
    469: "Smash, crash",
    470: "Breaking",
    472: "Whip",
    14:  "Screaming",
    8:   "Shout",
    11:  "Yell",
    12:  "Battle cry",
    38:  "Groan",
    298: "Fire",
    458: "Arrow",
}

# CLIP text prompts
POSITIVE_PROMPTS = [
    "a gunfight with muzzle flash and smoke from a gun barrel",
    "shooting a revolver or rifle at enemies with bullet impact",
    "an intense shootout with multiple gunshots being fired",
    "a fistfight with punching and kicking",
    "a violent melee brawl with people fighting",
    "an explosion with fire and debris flying",
    "a person being beaten or attacked violently",
    "a knife fight or stabbing in a video game",
]

NEGATIVE_PROMPTS = [
    "a pause menu or settings screen",
    "a loading screen",
    "a character walking calmly in nature",
    "a map or inventory screen",
    "a cutscene with characters talking quietly",
    "riding a horse peacefully through the countryside",
    "standing still looking at scenery",
    "a shop or store interface",
]


# ── Audio: PANNs Classification ─────────────────────────────────────────────

def classify_audio_panns(video_path: str, at_model: AudioTagging, sample_fps: int) -> np.ndarray:
    """
    Classify audio in chunks using PANNs and return a violence score
    per time step at `sample_fps` resolution.
    """
    # Extract audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "32000", "-vn", tmp_path],
            capture_output=True, check=True,
        )
        with wave.open(tmp_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        os.unlink(tmp_path)

    # Process in chunks aligned to sample_fps (each chunk = 1/sample_fps seconds)
    chunk_samples = rate // sample_fps
    n_chunks = len(audio) // chunk_samples
    if n_chunks == 0:
        return np.array([0.0])

    # PANNs expects ~1-2 second windows for good classification.
    # Use a 2-second sliding window, step by 1/sample_fps
    window_samples = rate * 2  # 2 second window
    violence_indices = list(VIOLENCE_AUDIO_LABELS.keys())

    scores = np.zeros(n_chunks)
    for i in range(n_chunks):
        center = i * chunk_samples + chunk_samples // 2
        start = max(0, center - window_samples // 2)
        end = min(len(audio), start + window_samples)
        chunk = audio[start:end]

        # Pad if too short
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))

        # PANNs inference
        chunk_tensor = chunk[np.newaxis, :]  # (1, samples)
        clipwise_output, _ = at_model.inference(chunk_tensor)
        probs = clipwise_output[0]  # (527,)

        # Sum probabilities of violence-related labels
        violence_score = sum(probs[idx] for idx in violence_indices)
        scores[i] = violence_score

    return scores


# ── Visual: CLIP Scoring ────────────────────────────────────────────────────

def sample_frames(video_path: str, sample_fps: int) -> list[np.ndarray]:
    """Sample frames from video at given fps using decord."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / sample_fps))
    indices = list(range(0, total_frames, step))

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
    """Score frames using CLIP against text prompts."""
    all_scores = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        images = [Image.fromarray(f) for f in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            img_out = model.get_image_features(**inputs)
            img_features = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            pos_sim = (img_features @ pos_features.T).mean(dim=-1)
            neg_sim = (img_features @ neg_features.T).mean(dim=-1)
            score = pos_sim - neg_sim
            all_scores.append(score.cpu().numpy())
    return np.concatenate(all_scores)


# ── Motion: Optical Flow ────────────────────────────────────────────────────

def compute_optical_flow(video_path: str, sample_fps: int) -> np.ndarray:
    """Compute optical flow magnitude by reading frames directly (low memory)."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / sample_fps))
    indices = list(range(0, total_frames, step))

    if len(indices) < 2:
        return np.zeros(len(indices))

    # Read first frame
    first = vr[indices[0]].asnumpy()
    prev_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 240.0 / prev_gray.shape[1])  # downscale to 240px wide
    if scale < 1.0:
        prev_gray = cv2.resize(prev_gray, None, fx=scale, fy=scale)
    del first

    flow_mags = [0.0]
    for i in range(1, len(indices)):
        frame = vr[indices[i]].asnumpy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        del frame

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_mags.append(float(np.mean(mag)))
        prev_gray = gray

    del vr
    return np.array(flow_mags)


# ── Video: Action Recognition (VideoMAE) ────────────────────────────────────

def classify_video_actions(
    video_path: str,
    mae_model: VideoMAEForVideoClassification,
    mae_processor: VideoMAEImageProcessor,
    sample_fps: int,
) -> np.ndarray:
    """
    Classify video segments using VideoMAE and return a violence score
    per time step at `sample_fps` resolution.
    VideoMAE needs 16 frames per classification window.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / sample_fps))
    n_samples = len(range(0, total_frames, step))

    # VideoMAE expects 16 frames per window
    mae_window = 16
    # Step through video in windows of 16 sampled frames, stride of 8
    mae_stride = 8
    violence_indices = list(VIDEOMAE_VIOLENCE_LABELS.keys())

    scores = np.zeros(n_samples)
    sample_indices = list(range(0, total_frames, step))

    for win_start in range(0, len(sample_indices), mae_stride):
        win_end = min(win_start + mae_window, len(sample_indices))
        frame_indices = sample_indices[win_start:win_end]

        if len(frame_indices) < 4:
            continue

        # Pad to 16 if needed by repeating last frame
        while len(frame_indices) < mae_window:
            frame_indices.append(frame_indices[-1])

        frames = vr.get_batch(frame_indices[:mae_window]).asnumpy()
        frame_list = [frames[j] for j in range(len(frames))]
        del frames

        inputs = mae_processor(frame_list, return_tensors="pt").to(DEVICE)
        del frame_list

        with torch.no_grad():
            outputs = mae_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        violence_score = sum(probs[idx] for idx in violence_indices)

        # Assign score to all samples in this window
        actual_end = min(win_start + mae_window, n_samples)
        for si in range(win_start, actual_end):
            scores[si] = max(scores[si], violence_score)

    del vr
    return scores


# ── Pipeline ────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def process_video(
    video_path: str,
    at_model: AudioTagging,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    pos_features: torch.Tensor,
    neg_features: torch.Tensor,
    mae_model: VideoMAEForVideoClassification,
    mae_processor: VideoMAEImageProcessor,
) -> list[dict]:
    """Process a single video with all four signals."""
    name = Path(video_path).stem

    # 1. Sample frames
    print(f"  Frames...", end=" ", flush=True)
    frames = sample_frames(video_path, SAMPLE_FPS)
    print(f"{len(frames)}", end=" | ", flush=True)

    # 2. CLIP scoring
    print(f"CLIP...", end=" ", flush=True)
    clip_scores = score_frames_clip(frames, clip_model, clip_processor, pos_features, neg_features)
    del frames  # free memory
    print("ok", end=" | ", flush=True)

    # 3. PANNs audio classification
    print(f"PANNs...", end=" ", flush=True)
    panns_scores = classify_audio_panns(video_path, at_model, SAMPLE_FPS)
    print("ok", end=" | ", flush=True)

    # 4. VideoMAE action recognition
    print(f"MAE...", end=" ", flush=True)
    mae_scores = classify_video_actions(video_path, mae_model, mae_processor, SAMPLE_FPS)
    print("ok", end=" | ", flush=True)

    # 5. Optical flow (reads frames independently, low memory)
    print(f"Flow...", end=" ", flush=True)
    flow_scores = compute_optical_flow(video_path, SAMPLE_FPS)
    print("ok")

    # Align lengths
    min_len = min(len(clip_scores), len(panns_scores), len(flow_scores), len(mae_scores))
    clip_scores = clip_scores[:min_len]
    panns_scores = panns_scores[:min_len]
    flow_scores = flow_scores[:min_len]
    mae_scores = mae_scores[:min_len]

    # Normalize each
    clip_norm = normalize(clip_scores)
    panns_norm = normalize(panns_scores)
    flow_norm = normalize(flow_scores)
    mae_norm = normalize(mae_scores)

    # Hybrid score
    hybrid = (PANNS_WEIGHT * panns_norm + CLIP_WEIGHT * clip_norm +
              FLOW_WEIGHT * flow_norm + VIDEOMAE_WEIGHT * mae_norm)

    # Smooth
    smooth_window = max(3, SAMPLE_FPS * 3)
    hybrid_smooth = uniform_filter1d(hybrid, size=smooth_window)

    # Find peaks
    peak_distance = SAMPLE_FPS * WINDOW_SEC
    peaks, _ = find_peaks(
        hybrid_smooth,
        distance=peak_distance,
        height=np.percentile(hybrid_smooth, 75),
    )

    segments = []
    for peak in peaks:
        t_sec = peak / SAMPLE_FPS
        segments.append({
            "video": video_path,
            "video_name": name,
            "time_sec": float(t_sec),
            "score": float(hybrid_smooth[peak]),
            "panns_score": float(panns_norm[peak]),
            "clip_score": float(clip_norm[peak]),
            "flow_score": float(flow_norm[peak]),
            "mae_score": float(mae_norm[peak]),
        })

    return segments


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


def main(compile_only=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not compile_only:
        video_files = sorted([
            os.path.join(VIDEO_DIR, f)
            for f in os.listdir(VIDEO_DIR)
            if f.endswith(".mp4")
        ])
        print(f"Found {len(video_files)} videos\n")

        # Load models
        print("Loading PANNs audio classifier...")
        at_model = AudioTagging(checkpoint_path=None, device=DEVICE)

        print(f"Loading CLIP ({CLIP_MODEL_NAME})...")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model.eval()

        print(f"Loading VideoMAE ({VIDEOMAE_MODEL_NAME})...")
        mae_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL_NAME).to(DEVICE)
        mae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL_NAME)
        mae_model.eval()

        with torch.no_grad():
            pos_inputs = clip_processor(text=POSITIVE_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
            pos_out = clip_model.get_text_features(**pos_inputs)
            pos_features = pos_out if isinstance(pos_out, torch.Tensor) else pos_out.pooler_output
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)

            neg_inputs = clip_processor(text=NEGATIVE_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
            neg_out = clip_model.get_text_features(**neg_inputs)
            neg_features = neg_out if isinstance(neg_out, torch.Tensor) else neg_out.pooler_output
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

        print(f"Models loaded on {DEVICE}\n")

        # Process all videos
        all_segments = []
        for i, vpath in enumerate(video_files):
            print(f"[{i+1}/{len(video_files)}] {Path(vpath).name}")
            try:
                segs = process_video(vpath, at_model, clip_model, clip_processor, pos_features, neg_features, mae_model, mae_processor)
                all_segments.extend(segs)
                print(f"  -> {len(segs)} candidates\n")
            except Exception as e:
                print(f"  ERROR: {e}\n")

        # Rank and extract top N
        all_segments.sort(key=lambda x: x["score"], reverse=True)
        top = all_segments[:TOP_N]

        print(f"\n{'='*70}")
        print(f"Top {len(top)} violent moments:")
        print(f"{'='*70}")

        clip_paths = []
        for i, seg in enumerate(top):
            start = max(0, seg["time_sec"] - WINDOW_SEC / 2)
            print(
                f"  #{i+1}: {seg['video_name']} @ {seg['time_sec']:.1f}s "
                f"(total={seg['score']:.3f} panns={seg['panns_score']:.3f} "
                f"clip={seg['clip_score']:.3f} flow={seg['flow_score']:.3f} "
                f"mae={seg['mae_score']:.3f})"
            )
            output_file = os.path.join(OUTPUT_DIR, f"violence_{i+1:02d}.mp4")
            print(f"       Extracting {WINDOW_SEC}s -> {output_file}")
            extract_clip(seg["video"], start, WINDOW_SEC, output_file)
            clip_paths.append(output_file)

        meta_path = os.path.join(OUTPUT_DIR, "violence.json")
        with open(meta_path, "w") as f:
            json.dump(top, f, indent=2)
        print(f"\nMetadata: {meta_path}")
    else:
        clip_paths = sorted([
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith("violence_") and f.endswith(".mp4")
        ])
        print(f"Found {len(clip_paths)} existing violence clips")

    # Compilation
    if clip_paths:
        meta_path = os.path.join(OUTPUT_DIR, "violence.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            indexed = []
            for i, seg in enumerate(meta):
                clip_file = os.path.join(OUTPUT_DIR, f"violence_{i+1:02d}.mp4")
                if os.path.exists(clip_file):
                    indexed.append((seg["video_name"], seg["time_sec"], clip_file))
            indexed.sort(key=lambda x: (x[0], x[1]))
            clip_paths = [x[2] for x in indexed]

        comp_path = os.path.join(OUTPUT_DIR, "violence_compilation.mp4")
        print(f"\nJoining {len(clip_paths)} clips...")
        concatenate_clips(clip_paths, comp_path)
        dur = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", comp_path],
            capture_output=True, text=True,
        )
        d = float(dur.stdout.strip())
        print(f"Compilation: {comp_path} ({d:.0f}s / {d/60:.1f}min)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect violent moments in gameplay videos")
    parser.add_argument("video_dir", nargs="?", help="Directory containing video files")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-n", "--top-n", type=int, help="Number of top moments to extract")
    parser.add_argument("-w", "--window", type=int, help="Clip duration in seconds")
    parser.add_argument("--compile", action="store_true", help="Only join existing clips")
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
