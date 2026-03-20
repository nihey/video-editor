#!/usr/bin/env python3
"""
Condense a video into its most violent moments.
Uses CLIP + PANNs + optical flow to score every moment, then extracts
the highest-scoring 0.2-2s clips into a ~60s highlight reel.
"""

import os
import sys
import subprocess
import tempfile
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import gc
import cv2
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from decord import VideoReader, cpu
from panns_inference import AudioTagging

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLE_FPS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_DURATION = 90.0     # target output length in seconds
MIN_CLIP_SEC = 0.5
MAX_CLIP_SEC = 5.0
PAD_BEFORE = 0.5           # seconds before peak
PAD_AFTER = 0.5            # seconds after peak
MIN_GAP_SEC = 0.3          # minimum gap between peaks
MERGE_GAP_SEC = 3.0        # merge clips within this gap (same scene)
SCORE_PERCENTILE = 40      # include peaks above this percentile

# Signal weights
PANNS_WEIGHT = 0.40
CLIP_WEIGHT = 0.35
FLOW_WEIGHT = 0.25

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
    429: "Cap gun",
    431: "Fusillade",
    471: "Crushing",
    473: "Thud",
    143: "Horse gallop",
}

POSITIVE_PROMPTS = [
    # Gunfights
    "a gunfight with muzzle flash and smoke from a gun barrel",
    "shooting a revolver or rifle at enemies with bullet impact",
    "an intense shootout with multiple gunshots being fired",
    "a Dead Eye slow motion shooting sequence in a western game",
    # Melee combat
    "a fistfight with punching and kicking in a western game",
    "a violent melee brawl with people fighting on the ground",
    "a lasso takedown of an enemy being pulled off a horse",
    "tackling someone to the ground in a violent struggle",
    # Weapons and explosions
    "an explosion with fire and debris flying through the air",
    "throwing dynamite that explodes with a large blast",
    "a knife fight or stabbing attack in close combat",
    # Chases and action
    "a high speed horse chase with gunfire and pursuit",
    "a dramatic shootout from behind cover with enemies",
    "a train robbery with intense action and shooting",
    # Dramatic moments
    "a person being beaten or attacked violently",
    "a dramatic standoff or duel in a western setting",
]

NEGATIVE_PROMPTS = [
    "a pause menu or settings screen",
    "a loading screen or black screen",
    "a character walking calmly in nature",
    "a map or inventory or radial wheel screen",
    "a cutscene with characters talking quietly",
    "riding a horse peacefully through the countryside",
    "standing still looking at scenery or landscape",
    "a shop or store or catalog interface",
    "a character crafting or cooking at a campfire",
    "fishing by a river or lake peacefully",
    "a grayscale desaturated menu overlay",
]


# ── Audio: PANNs Classification ─────────────────────────────────────────────

def classify_audio_panns(video_path: str, at_model: AudioTagging, sample_fps: int) -> np.ndarray:
    """Classify audio using PANNs, return violence score per time step."""
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

    chunk_samples = rate // sample_fps
    n_chunks = len(audio) // chunk_samples
    if n_chunks == 0:
        return np.array([0.0])

    window_samples = rate * 2
    violence_indices = list(VIOLENCE_AUDIO_LABELS.keys())

    scores = np.zeros(n_chunks)
    for i in range(n_chunks):
        center = i * chunk_samples + chunk_samples // 2
        start = max(0, center - window_samples // 2)
        end = min(len(audio), start + window_samples)
        chunk = audio[start:end]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        chunk_tensor = chunk[np.newaxis, :]
        clipwise_output, _ = at_model.inference(chunk_tensor)
        probs = clipwise_output[0]
        scores[i] = sum(probs[idx] for idx in violence_indices)

    return scores


# ── Visual: CLIP Scoring ────────────────────────────────────────────────────

def score_frames_clip(
    frames: list[np.ndarray],
    model: CLIPModel,
    processor: CLIPProcessor,
    pos_features: torch.Tensor,
    neg_features: torch.Tensor,
    batch_size: int = 32,
) -> np.ndarray:
    """Score frames using CLIP against violence text prompts."""
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
    """Compute optical flow magnitude, reading frames one at a time."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / sample_fps))
    indices = list(range(0, total_frames, step))

    if len(indices) < 2:
        return np.zeros(len(indices))

    first = vr[indices[0]].asnumpy()
    prev_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 240.0 / prev_gray.shape[1])
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def segment_label(index: int) -> str:
    """Generate short labels: A, B, C, ..., Z, AA, AB, ..."""
    if index < 26:
        return chr(65 + index)
    return chr(65 + index // 26 - 1) + chr(65 + index % 26)


def extract_clip(video_path: str, start: float, duration: float, output_path: str,
                 label: str | None = None):
    vf_filters = []
    if label:
        # Burn label into top-left corner with semi-transparent background
        vf_filters.append(
            f"drawtext=text='{label}'"
            f":fontsize=36:fontcolor=white"
            f":borderw=2:bordercolor=black"
            f":x=20:y=20"
        )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0, start):.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
    ]
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


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


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Condense video into most violent moments")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--target", type=float, default=TARGET_DURATION,
                        help=f"Target output duration in seconds (default: {TARGET_DURATION})")
    parser.add_argument("--min-clip", type=float, default=MIN_CLIP_SEC,
                        help=f"Min clip duration (default: {MIN_CLIP_SEC})")
    parser.add_argument("--max-clip", type=float, default=MAX_CLIP_SEC,
                        help=f"Max clip duration (default: {MAX_CLIP_SEC})")
    parser.add_argument("--merge-gap", type=float, default=MERGE_GAP_SEC,
                        help=f"Merge clips within this gap in seconds (default: {MERGE_GAP_SEC})")
    parser.add_argument("--percentile", type=float, default=SCORE_PERCENTILE,
                        help=f"Include peaks above this score percentile (default: {SCORE_PERCENTILE})")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).with_name(f"{stem}_condensed.mp4"))

    target_dur = args.target
    min_clip = args.min_clip
    max_clip = args.max_clip
    merge_gap = args.merge_gap
    score_pct = args.percentile

    video_duration = get_video_duration(input_path)
    print(f"Input: {input_path} ({video_duration:.1f}s)")
    print(f"Target: ~{target_dur:.0f}s output\n")

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Pre-compute text features
    pos_inputs = clip_processor(text=POSITIVE_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    neg_inputs = clip_processor(text=NEGATIVE_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        pos_out = clip_model.get_text_features(**pos_inputs)
        pos_features = pos_out if isinstance(pos_out, torch.Tensor) else pos_out.pooler_output
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_out = clip_model.get_text_features(**neg_inputs)
        neg_features = neg_out if isinstance(neg_out, torch.Tensor) else neg_out.pooler_output
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

    print("Loading PANNs model...")
    at_model = AudioTagging(checkpoint_path=None, device=DEVICE)

    # ── Score video ──────────────────────────────────────────────────────────
    print(f"\nScoring video at {SAMPLE_FPS}fps...")

    # CLIP
    print("  CLIP visual scoring...", end=" ", flush=True)
    vr = VideoReader(input_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / SAMPLE_FPS))
    all_indices = list(range(0, len(vr), step))

    clip_scores_list = []
    for ci in range(0, len(all_indices), 64):
        chunk_idx = all_indices[ci : ci + 64]
        chunk_frames = [vr[idx].asnumpy() for idx in chunk_idx]
        chunk_scores = score_frames_clip(chunk_frames, clip_model, clip_processor, pos_features, neg_features)
        clip_scores_list.append(chunk_scores)
        del chunk_frames
    clip_scores = np.concatenate(clip_scores_list)
    del vr
    gc.collect()
    print("done")

    # Free CLIP from GPU before PANNs
    del clip_model, clip_processor, pos_features, neg_features, pos_inputs, neg_inputs
    torch.cuda.empty_cache()
    gc.collect()

    # PANNs
    print("  PANNs audio scoring...", end=" ", flush=True)
    panns_scores = classify_audio_panns(input_path, at_model, SAMPLE_FPS)
    print("done")

    del at_model
    torch.cuda.empty_cache()
    gc.collect()

    # Optical flow
    print("  Optical flow...", end=" ", flush=True)
    flow_scores = compute_optical_flow(input_path, SAMPLE_FPS)
    print("done")

    # ── Combine scores ───────────────────────────────────────────────────────
    min_len = min(len(clip_scores), len(panns_scores), len(flow_scores))
    clip_scores = clip_scores[:min_len]
    panns_scores = panns_scores[:min_len]
    flow_scores = flow_scores[:min_len]

    clip_norm = normalize(clip_scores)
    panns_norm = normalize(panns_scores)
    flow_norm = normalize(flow_scores)

    hybrid = (PANNS_WEIGHT * panns_norm + CLIP_WEIGHT * clip_norm + FLOW_WEIGHT * flow_norm)

    # Light smoothing
    smooth_window = max(3, SAMPLE_FPS * 2)
    hybrid_smooth = uniform_filter1d(hybrid, size=smooth_window)

    print(f"\n  Hybrid score stats: min={hybrid_smooth.min():.3f}, "
          f"max={hybrid_smooth.max():.3f}, mean={hybrid_smooth.mean():.3f}")

    # ── Find peaks and build clips ───────────────────────────────────────────
    min_distance = max(1, int(MIN_GAP_SEC * SAMPLE_FPS))

    # Score threshold: include everything above the given percentile
    score_threshold = np.percentile(hybrid_smooth, score_pct)
    print(f"  Score threshold (p{score_pct:.0f}): {score_threshold:.3f}")

    peaks, props = find_peaks(
        hybrid_smooth,
        distance=min_distance,
        height=score_threshold,
    )

    if len(peaks) == 0:
        print("No peaks found!")
        return

    peak_scores = hybrid_smooth[peaks]

    # Build a clip around each peak, duration proportional to score
    ranges = []
    for i, peak in enumerate(peaks):
        score = peak_scores[i]
        t_sec = peak / SAMPLE_FPS

        norm_score = (score - peak_scores.min()) / (peak_scores.max() - peak_scores.min() + 1e-8)
        clip_dur = min_clip + norm_score * (max_clip - min_clip)

        start = max(0, t_sec - PAD_BEFORE - clip_dur * 0.3)
        end = min(video_duration - 0.05, start + clip_dur)
        start = max(0, end - clip_dur)

        ranges.append((start, end, score))

    # Sort by start time and merge overlapping + nearby clips (same scene)
    ranges.sort(key=lambda r: r[0])
    merged = [ranges[0]]
    for start, end, score in ranges[1:]:
        prev_start, prev_end, prev_score = merged[-1]
        if start <= prev_end + merge_gap:
            # Nearby or overlapping: merge into one continuous segment
            merged[-1] = (prev_start, max(prev_end, end), max(prev_score, score))
        else:
            merged.append((start, end, score))

    # If we're way over target, drop the lowest-scoring merged segments
    total_dur = sum(e - s for s, e, _ in merged)
    if total_dur > target_dur * 2:
        # Sort by score, keep best until we hit ~target
        by_score = sorted(merged, key=lambda r: r[2], reverse=True)
        kept = []
        kept_dur = 0.0
        for seg in by_score:
            kept.append(seg)
            kept_dur += seg[1] - seg[0]
            if kept_dur >= target_dur:
                break
        merged = sorted(kept, key=lambda r: r[0])
        total_dur = sum(e - s for s, e, _ in merged)

    # Assign labels
    segments = []
    for i, (s, e, sc) in enumerate(merged):
        label = segment_label(i)
        segments.append({"label": label, "start": round(s, 2), "end": round(e, 2),
                         "duration": round(e - s, 2), "score": round(sc, 3)})

    print(f"\nSelected {len(segments)} segments, total {total_dur:.1f}s:")
    for seg in segments:
        print(f"  [{seg['label']}] {seg['start']:.1f}s - {seg['end']:.1f}s "
              f"({seg['duration']:.1f}s, score={seg['score']:.3f})")

    # Save manifest
    manifest_path = str(Path(output_path).with_suffix(".json"))
    manifest = {
        "source": os.path.abspath(input_path),
        "segments": segments,
    }
    import json
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    # ── Extract and join ─────────────────────────────────────────────────────
    print(f"Extracting clips...")
    tmpdir = tempfile.mkdtemp(prefix="condense_violence_")
    clip_paths = []

    for seg in segments:
        dur = seg["duration"]
        out = os.path.join(tmpdir, f"clip_{seg['label']}.mp4")
        try:
            extract_clip(input_path, seg["start"], dur, out, label=seg["label"])
            if os.path.getsize(out) > 1000:
                clip_paths.append(out)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Failed to extract clip {seg['label']}: {e}")

    if not clip_paths:
        print("ERROR: No clips extracted!")
        return

    print(f"Joining {len(clip_paths)} clips...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    concatenate_clips(clip_paths, output_path)

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

    final_dur = get_video_duration(output_path)
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput: {output_path}")
    print(f"  Duration: {final_dur:.1f}s")
    print(f"  Size: {final_size:.1f}MB")
    print(f"  Segments: {len(clip_paths)}")


def reassemble():
    """Re-assemble video from a manifest, with optional include/exclude/trim edits."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Re-assemble video from manifest with edits")
    parser.add_argument("manifest", help="Manifest JSON file from condense_violence.py")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--include", help="Only include these segments (comma-separated, e.g. A,C,F)")
    parser.add_argument("--exclude", help="Exclude these segments (comma-separated, e.g. B,D)")
    parser.add_argument("--trim", action="append", default=[],
                        help="Trim a segment: LABEL:START:END (seconds relative to segment start, "
                             "e.g. D:0:3 keeps first 3s of D, F:2: trims first 2s of F)")
    parser.add_argument("--no-labels", action="store_true", help="Don't burn labels into video")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    source = manifest["source"]
    segments = manifest["segments"]

    if not os.path.exists(source):
        print(f"ERROR: Source video not found: {source}")
        sys.exit(1)

    # Parse trim directives
    trims = {}
    for t in args.trim:
        parts = t.split(":")
        label = parts[0].upper()
        trim_start = float(parts[1]) if len(parts) > 1 and parts[1] else None
        trim_end = float(parts[2]) if len(parts) > 2 and parts[2] else None
        trims[label] = (trim_start, trim_end)

    # Filter segments
    if args.include:
        include_set = {s.strip().upper() for s in args.include.split(",")}
        segments = [s for s in segments if s["label"] in include_set]
    if args.exclude:
        exclude_set = {s.strip().upper() for s in args.exclude.split(",")}
        segments = [s for s in segments if s["label"] not in exclude_set]

    if not segments:
        print("ERROR: No segments remaining after filtering!")
        sys.exit(1)

    # Apply trims
    for seg in segments:
        if seg["label"] in trims:
            trim_start, trim_end = trims[seg["label"]]
            orig_start = seg["start"]
            orig_end = seg["end"]
            if trim_start is not None:
                seg["start"] = orig_start + trim_start
            if trim_end is not None:
                seg["end"] = orig_start + trim_end
            seg["start"] = max(orig_start, min(seg["start"], orig_end))
            seg["end"] = max(seg["start"] + 0.1, min(seg["end"], orig_end))
            seg["duration"] = round(seg["end"] - seg["start"], 2)

    output_path = args.output
    if not output_path:
        manifest_stem = Path(args.manifest).stem
        output_path = str(Path(args.manifest).with_name(f"{manifest_stem}_edited.mp4"))

    total_dur = sum(s["duration"] for s in segments)
    print(f"Source: {source}")
    print(f"Segments ({len(segments)}, {total_dur:.1f}s):")
    for seg in segments:
        print(f"  [{seg['label']}] {seg['start']:.1f}s - {seg['end']:.1f}s ({seg['duration']:.1f}s)")

    print(f"\nExtracting clips...")
    tmpdir = tempfile.mkdtemp(prefix="reassemble_")
    clip_paths = []
    burn_labels = not args.no_labels

    for seg in segments:
        out = os.path.join(tmpdir, f"clip_{seg['label']}.mp4")
        try:
            extract_clip(source, seg["start"], seg["duration"], out,
                         label=seg["label"] if burn_labels else None)
            if os.path.getsize(out) > 1000:
                clip_paths.append(out)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Failed to extract {seg['label']}: {e}")

    if not clip_paths:
        print("ERROR: No clips extracted!")
        return

    print(f"Joining {len(clip_paths)} clips...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    concatenate_clips(clip_paths, output_path)

    for p in clip_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    final_dur = get_video_duration(output_path)
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput: {output_path}")
    print(f"  Duration: {final_dur:.1f}s")
    print(f"  Size: {final_size:.1f}MB")


if __name__ == "__main__":
    # If first arg is "reassemble", run that mode
    if len(sys.argv) > 1 and sys.argv[1] == "reassemble":
        sys.argv.pop(1)
        reassemble()
    else:
        main()
