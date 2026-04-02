#!/usr/bin/env python3
"""
Query videos with natural language using Qwen2.5-VL.

Two-phase system:
  1. index  — Extract keyframes, generate VLM descriptions, save searchable index
  2. query  — Search descriptions, re-rank with VLM, build compilation

Requires:
  pip install transformers bitsandbytes accelerate qwen-vl-utils

Hardware: RTX 4060 8GB (Qwen2.5-VL-7B-Instruct with 4-bit quantization)
"""

import os
import sys
import json
import gc
import re
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_INDEX_FPS = 0.5    # frames per second for indexing
FRAME_WIDTH = 640          # saved frame width
FRAME_HEIGHT = 360         # saved frame height
FRAME_QUALITY = 85         # JPEG quality

# Query defaults
DEFAULT_TOP_N = 20         # number of clips in compilation
PAD_BEFORE = 1.5           # seconds before matched frame
PAD_AFTER = 2.5            # seconds after matched frame
MIN_CLIP_SEC = 2.0         # minimum clip duration
MAX_CLIP_SEC = 10.0        # maximum clip duration
MERGE_GAP_SEC = 3.0        # merge segments within this gap (seconds)
RERANK_CANDIDATES = 100    # how many text matches to re-rank with VLM

DESCRIBE_PROMPT = (
    "Describe this video game screenshot in one detailed paragraph. "
    "Include: what actions are being performed, any weapons or tools visible, "
    "characters and their poses, the environment/setting, any violence or combat, "
    "and the overall mood of the scene. Be specific about details like headshots, "
    "explosions, melee attacks, stealth kills, chases, or any notable events."
)

SCORE_PROMPT_TEMPLATE = (
    "Rate how relevant this video game scene is to the following description. "
    "Respond with ONLY a single number from 0 to 10, nothing else.\n\n"
    'Description: "{query}"'
)


# ── Model ───────────────────────────────────────────────────────────────────

def load_model(model_name=DEFAULT_MODEL, use_4bit=True):
    """Load Qwen2.5-VL with optional 4-bit quantization for 8GB GPUs."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading {model_name}{'  (4-bit)' if use_4bit else ''}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model loaded.")
    return model, processor


def vlm_generate(model, processor, image_path, prompt, max_tokens=256):
    """Send an image + prompt to the VLM and return the response text."""
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # Free intermediate tensors
    del inputs, generated_ids, trimmed
    return result


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def segment_label(index: int) -> str:
    if index < 26:
        return chr(65 + index)
    return chr(65 + index // 26 - 1) + chr(65 + index % 26)


def extract_clip(video_path: str, start: float, duration: float,
                 output_path: str, label: str | None = None):
    vf = []
    if label:
        vf.append(
            f"drawtext=text='{label}':fontsize=36:fontcolor=white"
            f":borderw=2:bordercolor=black:x=20:y=20"
        )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0, start):.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
    ]
    if vf:
        cmd += ["-vf", ",".join(vf)]
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
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "21",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ], capture_output=True, check=True)
    finally:
        os.unlink(list_path)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ── Indexing ────────────────────────────────────────────────────────────────

def extract_and_save_frames(video_path: str, output_dir: str,
                            fps: float) -> list[tuple[float, str]]:
    """Extract keyframes from video, save as JPEG.
    Returns list of (timestamp_sec, frame_path).
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    total_frames = len(vr)
    step = max(1, int(native_fps / fps))
    indices = list(range(0, total_frames, step))

    video_name = Path(video_path).stem
    frames = []

    for idx in indices:
        t_sec = round(idx / native_fps, 2)
        frame_filename = f"{video_name}_{t_sec:.2f}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        if not os.path.exists(frame_path):
            frame = vr[idx].asnumpy()
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
            del frame, frame_resized, frame_bgr

        frames.append((t_sec, frame_path))

    del vr
    return frames


def index_videos(video_dir: str, output_dir: str, fps: float = DEFAULT_INDEX_FPS,
                 model_name: str = DEFAULT_MODEL, use_4bit: bool = True):
    """Index all videos in a directory with VLM descriptions."""
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    index_path = os.path.join(output_dir, "index.json")

    # Load existing index for incremental updates
    existing_videos: dict[str, dict] = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = json.load(f)
        for v in data.get("videos", []):
            existing_videos[v["path"]] = v
        print(f"Existing index: {len(existing_videos)} videos")

    # Find videos
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    videos = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if Path(f).suffix.lower() in video_exts
    ])

    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    # Check which need indexing
    to_index = []
    for v in videos:
        abs_path = os.path.abspath(v)
        if abs_path in existing_videos:
            print(f"  Skip (indexed): {Path(v).name}")
        else:
            to_index.append(v)

    print(f"\nFound {len(videos)} videos, {len(to_index)} need indexing")

    if not to_index:
        print("All videos already indexed.")
        return

    # Estimate total frames
    total_est_frames = 0
    for v in to_index:
        dur = get_video_duration(v)
        total_est_frames += int(dur * fps)
    print(f"Estimated frames to describe: ~{total_est_frames}")

    # Load model
    model, processor = load_model(model_name, use_4bit=use_4bit)

    # Index each video
    all_video_data = list(existing_videos.values())
    global_frame_count = 0
    global_t0 = time.time()

    for vi, video_path in enumerate(to_index):
        abs_path = os.path.abspath(video_path)
        name = Path(video_path).stem
        duration = get_video_duration(video_path)
        print(f"\n[{vi+1}/{len(to_index)}] {name} ({duration:.1f}s)")

        # Extract frames
        print(f"  Extracting frames at {fps} fps...", end=" ", flush=True)
        frames = extract_and_save_frames(video_path, frames_dir, fps)
        print(f"{len(frames)} frames")

        # Generate descriptions
        frame_data = []
        video_t0 = time.time()

        for fi, (timestamp, frame_path) in enumerate(frames):
            try:
                desc = vlm_generate(
                    model, processor, frame_path, DESCRIBE_PROMPT, max_tokens=200,
                )
            except Exception as e:
                desc = f"[error: {e}]"
                print(f"  WARNING: VLM error at {timestamp:.1f}s: {e}")

            frame_data.append({
                "timestamp": timestamp,
                "frame": os.path.relpath(frame_path, output_dir),
                "description": desc,
            })

            global_frame_count += 1
            elapsed = time.time() - global_t0
            rate = global_frame_count / elapsed if elapsed > 0 else 0
            remaining_frames = total_est_frames - global_frame_count
            eta = remaining_frames / rate if rate > 0 else 0

            # Truncate description for display
            desc_short = desc[:70].replace("\n", " ")
            print(f"  [{fi+1}/{len(frames)}] {timestamp:.1f}s — {desc_short}... "
                  f"({rate:.2f} fr/s, ETA {format_time(eta)})")

        video_data = {
            "path": abs_path,
            "name": name,
            "duration": round(duration, 1),
            "frames_indexed": len(frame_data),
            "fps": fps,
            "indexed_at": datetime.now().isoformat(),
            "frames": frame_data,
        }
        all_video_data.append(video_data)

        # Save after each video (resume-safe)
        index = {
            "model": model_name,
            "fps": fps,
            "updated": datetime.now().isoformat(),
            "videos": all_video_data,
        }
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        video_elapsed = time.time() - video_t0
        print(f"  Done in {format_time(video_elapsed)} "
              f"({len(all_video_data)} videos indexed total)")

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    total_frames = sum(v["frames_indexed"] for v in all_video_data)
    total_elapsed = time.time() - global_t0
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"  Videos: {len(all_video_data)}")
    print(f"  Frames: {total_frames}")
    print(f"  Time: {format_time(total_elapsed)}")
    print(f"  Index: {index_path}")


# ── Text Matching ───────────────────────────────────────────────────────────

def text_relevance(query: str, description: str) -> float:
    """Score description relevance to query using text matching.
    Returns a score between 0.0 and 1.0.
    """
    q = query.lower()
    d = description.lower()

    # Exact substring match gets highest score
    if q in d:
        return 1.0

    # Tokenize into words (3+ chars to skip articles/prepositions)
    q_words = set(re.findall(r"\w{3,}", q))
    d_words = set(re.findall(r"\w{3,}", d))

    if not q_words:
        return 0.0

    # Direct word overlap
    overlap = q_words & d_words
    word_score = len(overlap) / len(q_words)

    # Partial/stem matches: query word prefix matches description word prefix
    partial = 0
    unmatched = q_words - overlap
    for qw in unmatched:
        stem = qw[:min(5, len(qw))]
        for dw in d_words:
            if dw.startswith(stem) or stem in dw:
                partial += 0.5
                break
    partial_score = partial / len(q_words)

    # Bigram overlap for phrase matching
    q_tokens = q.split()
    d_tokens = d.split()
    if len(q_tokens) >= 2:
        q_bigrams = set(zip(q_tokens, q_tokens[1:]))
        d_bigrams = set(zip(d_tokens, d_tokens[1:]))
        bigram_hit = len(q_bigrams & d_bigrams)
        bigram_score = bigram_hit / len(q_bigrams)
    else:
        bigram_score = 0.0

    return 0.5 * word_score + 0.3 * partial_score + 0.2 * bigram_score


# ── Querying ────────────────────────────────────────────────────────────────

def query_index(index_dir: str, query: str, output_path: str,
                top_n: int = DEFAULT_TOP_N, fast: bool = False,
                model_name: str = DEFAULT_MODEL, no_labels: bool = False,
                use_4bit: bool = True):
    """Search indexed videos and build a compilation."""
    index_path = os.path.join(index_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"ERROR: Index not found at {index_path}")
        print(f"Run: python query_videos.py index <video_dir> -o {index_dir}")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    total_frames = sum(len(v["frames"]) for v in index["videos"])
    print(f"Index: {len(index['videos'])} videos, {total_frames} frames")
    print(f"Query: \"{query}\"\n")

    # ── Stage 1: Text matching on descriptions ──────────────────────────────
    print("Stage 1: Text matching...")
    candidates = []
    for video in index["videos"]:
        for frame in video["frames"]:
            score = text_relevance(query, frame["description"])
            if score > 0.05:
                candidates.append({
                    "video_path": video["path"],
                    "video_name": video["name"],
                    "video_duration": video["duration"],
                    "timestamp": frame["timestamp"],
                    "frame_rel": frame["frame"],
                    "description": frame["description"],
                    "text_score": score,
                })

    candidates.sort(key=lambda x: -x["text_score"])
    print(f"  {len(candidates)} candidates above threshold")

    if not candidates:
        print("\nNo text matches found. Try a broader query.")
        return

    # Show top text matches
    print(f"\n  Top text matches:")
    for c in candidates[:5]:
        desc_short = c["description"][:80].replace("\n", " ")
        print(f"    [{c['text_score']:.2f}] {c['video_name']} @ {c['timestamp']:.1f}s "
              f"— {desc_short}...")

    # ── Stage 2: VLM re-ranking ─────────────────────────────────────────────
    if fast:
        print(f"\nStage 2: Skipped (--fast mode)")
        scored = candidates[:top_n * 3]
        for c in scored:
            c["final_score"] = c["text_score"]
    else:
        rerank_count = min(len(candidates), RERANK_CANDIDATES)
        rerank_set = candidates[:rerank_count]
        print(f"\nStage 2: VLM re-ranking top {rerank_count} candidates...")

        model, processor = load_model(model_name, use_4bit=use_4bit)
        score_prompt = SCORE_PROMPT_TEMPLATE.format(query=query)

        t0 = time.time()
        for i, c in enumerate(rerank_set):
            frame_path = os.path.join(index_dir, c["frame_rel"])
            if not os.path.exists(frame_path):
                print(f"  WARNING: Frame not found: {frame_path}")
                c["vlm_score"] = 0
                c["final_score"] = 0.0
                continue

            try:
                response = vlm_generate(
                    model, processor, frame_path, score_prompt, max_tokens=10,
                )
                nums = re.findall(r"\d+", response)
                vlm_score = min(10, int(nums[0])) if nums else 0
            except Exception as e:
                print(f"  WARNING: VLM error: {e}")
                vlm_score = 0

            c["vlm_score"] = vlm_score
            c["final_score"] = vlm_score / 10.0

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (rerank_count - i - 1) / rate if rate > 0 else 0

            if vlm_score >= 5:
                print(f"  [{i+1}/{rerank_count}] {c['video_name']} @ "
                      f"{c['timestamp']:.1f}s — VLM: {vlm_score}/10 "
                      f"(ETA {format_time(eta)})")

        del model, processor
        torch.cuda.empty_cache()
        gc.collect()

        scored = sorted(rerank_set, key=lambda x: -x["final_score"])
        scored = [c for c in scored if c["final_score"] >= 0.3]

        high_scores = [c for c in scored if c["final_score"] >= 0.7]
        med_scores = [c for c in scored if 0.3 <= c["final_score"] < 0.7]
        print(f"\n  VLM results: {len(high_scores)} high (>=7), "
              f"{len(med_scores)} medium (3-6)")

    if not scored:
        print("\nNo relevant scenes found. Try a different query.")
        return

    # ── Stage 3: Build segments from scored frames ──────────────────────────
    print(f"\nStage 3: Building segments...")

    # Group by video
    by_video: dict[str, list[dict]] = {}
    for c in scored:
        key = c["video_path"]
        if key not in by_video:
            by_video[key] = []
        by_video[key].append(c)

    segments = []
    for video_path, frames in by_video.items():
        frames.sort(key=lambda x: x["timestamp"])
        video_dur = frames[0]["video_duration"]

        # Merge nearby frames into segments
        merged: list[dict] = []
        for f in frames:
            t = f["timestamp"]
            start = max(0.0, t - PAD_BEFORE)
            end = min(video_dur, t + PAD_AFTER)

            if merged and start <= merged[-1]["end"] + MERGE_GAP_SEC:
                merged[-1]["end"] = max(merged[-1]["end"], end)
                merged[-1]["score"] = max(merged[-1]["score"], f["final_score"])
                merged[-1]["descriptions"].append(f["description"])
            else:
                merged.append({
                    "source": video_path,
                    "source_name": f["video_name"],
                    "start": start,
                    "end": end,
                    "score": f["final_score"],
                    "descriptions": [f["description"]],
                    "video_duration": video_dur,
                })

        for m in merged:
            dur = m["end"] - m["start"]
            if dur < MIN_CLIP_SEC:
                m["end"] = min(m["video_duration"], m["start"] + MIN_CLIP_SEC)
            if dur > MAX_CLIP_SEC:
                m["end"] = m["start"] + MAX_CLIP_SEC
            m["duration"] = round(m["end"] - m["start"], 2)
            segments.append(m)

    # Sort by score descending and take top N
    segments.sort(key=lambda x: -x["score"])
    segments = segments[:top_n]

    # Assign labels
    for i, seg in enumerate(segments):
        seg["label"] = segment_label(i)
        seg["start"] = round(seg["start"], 2)
        seg["end"] = round(seg["end"], 2)
        seg["score"] = round(seg["score"], 3)

    total_dur = sum(s["duration"] for s in segments)
    print(f"\nSelected {len(segments)} segments, total {total_dur:.1f}s:")
    for seg in segments:
        desc_short = seg["descriptions"][0][:60].replace("\n", " ")
        print(f"  [{seg['label']}] {seg['source_name']} @ {seg['start']:.1f}s - "
              f"{seg['end']:.1f}s ({seg['duration']:.1f}s, score={seg['score']:.3f})")
        print(f"        {desc_short}...")

    # ── Save manifest ───────────────────────────────────────────────────────
    manifest_path = str(Path(output_path).with_suffix(".json"))
    manifest = {
        "query": query,
        "created": datetime.now().isoformat(),
        "segments": [{
            "label": s["label"],
            "source": s["source"],
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
            "score": s["score"],
        } for s in segments],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    # ── Stage 4: Build compilation ──────────────────────────────────────────
    print(f"\nStage 4: Extracting clips...")
    tmpdir = tempfile.mkdtemp(prefix="query_videos_")
    clip_paths = []

    for seg in segments:
        out = os.path.join(tmpdir, f"clip_{seg['label']}.mp4")
        try:
            label_text = seg["label"] if not no_labels else None
            extract_clip(seg["source"], seg["start"], seg["duration"],
                         out, label=label_text)
            if os.path.getsize(out) > 1000:
                clip_paths.append(out)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Failed to extract {seg['label']}: {e}")

    if not clip_paths:
        print("ERROR: No clips extracted!")
        return

    print(f"Joining {len(clip_paths)} clips...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
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
    print(f"\n{'='*60}")
    print(f"COMPILATION READY")
    print(f"  Output: {output_path}")
    print(f"  Duration: {final_dur:.1f}s")
    print(f"  Size: {final_size:.1f}MB")
    print(f"  Segments: {len(clip_paths)}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Query: \"{query}\"")


# ── Reassemble ──────────────────────────────────────────────────────────────

def reassemble(manifest_path: str, output_path: str,
               include: str | None = None, exclude: str | None = None,
               trims: list[str] | None = None, no_labels: bool = False):
    """Re-assemble video from a multi-source manifest with optional edits."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    segments = manifest["segments"]

    # Parse trim directives
    trim_map: dict[str, tuple[float | None, float | None]] = {}
    for t in (trims or []):
        parts = t.split(":")
        label = parts[0].upper()
        trim_start = float(parts[1]) if len(parts) > 1 and parts[1] else None
        trim_end = float(parts[2]) if len(parts) > 2 and parts[2] else None
        trim_map[label] = (trim_start, trim_end)

    # Filter segments
    if include:
        keep = {s.strip().upper() for s in include.split(",")}
        segments = [s for s in segments if s["label"] in keep]
    if exclude:
        drop = {s.strip().upper() for s in exclude.split(",")}
        segments = [s for s in segments if s["label"] not in drop]

    if not segments:
        print("ERROR: No segments remaining after filtering!")
        sys.exit(1)

    # Apply trims
    for seg in segments:
        if seg["label"] in trim_map:
            ts, te = trim_map[seg["label"]]
            orig_start, orig_end = seg["start"], seg["end"]
            if ts is not None:
                seg["start"] = orig_start + ts
            if te is not None:
                seg["end"] = orig_start + te
            seg["start"] = max(orig_start, min(seg["start"], orig_end))
            seg["end"] = max(seg["start"] + 0.1, min(seg["end"], orig_end))
            seg["duration"] = round(seg["end"] - seg["start"], 2)

    total_dur = sum(s["duration"] for s in segments)
    print(f"Segments ({len(segments)}, {total_dur:.1f}s):")
    for seg in segments:
        print(f"  [{seg['label']}] {Path(seg['source']).stem} @ "
              f"{seg['start']:.1f}s - {seg['end']:.1f}s ({seg['duration']:.1f}s)")

    print(f"\nExtracting clips...")
    tmpdir = tempfile.mkdtemp(prefix="reassemble_query_")
    clip_paths = []

    for seg in segments:
        source = seg["source"]
        if not os.path.exists(source):
            print(f"  WARNING: Source not found: {source}")
            continue

        out = os.path.join(tmpdir, f"clip_{seg['label']}.mp4")
        try:
            label_text = seg["label"] if not no_labels else None
            extract_clip(source, seg["start"], seg["duration"], out,
                         label=label_text)
            if os.path.getsize(out) > 1000:
                clip_paths.append(out)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Failed to extract {seg['label']}: {e}")

    if not clip_paths:
        print("ERROR: No clips extracted!")
        return

    print(f"Joining {len(clip_paths)} clips...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
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


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Query videos with natural language using Qwen2.5-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all videos in a directory (one-time)
  python query_videos.py index /path/to/videos -o ./video_index

  # Search for specific scenes and build compilation
  python query_videos.py query ./video_index "Scenes with shots to the head" -o headshots.mp4

  # Fast search (text-only, no VLM re-ranking)
  python query_videos.py query ./video_index "explosions and gunfire" -o booms.mp4 --fast

  # More results, no labels burned in
  python query_videos.py query ./video_index "horse chases" -o chases.mp4 -n 30 --no-labels

  # Re-edit a query result (keep only segments A, C, F)
  python query_videos.py reassemble headshots.json -o headshots_v2.mp4 --include A,C,F

  # Use smaller model for faster indexing
  python query_videos.py index /path/to/videos -o ./index --model Qwen/Qwen2.5-VL-3B-Instruct --no-4bit
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── index ───────────────────────────────────────────────────────────────
    idx = subparsers.add_parser("index", help="Index videos with VLM descriptions")
    idx.add_argument("video_dir", help="Directory containing video files")
    idx.add_argument("-o", "--output", default="./video_index",
                     help="Output directory for index (default: ./video_index)")
    idx.add_argument("--fps", type=float, default=DEFAULT_INDEX_FPS,
                     help=f"Frames per second to sample (default: {DEFAULT_INDEX_FPS})")
    idx.add_argument("--model", default=DEFAULT_MODEL,
                     help=f"Qwen VL model (default: {DEFAULT_MODEL})")
    idx.add_argument("--no-4bit", action="store_true",
                     help="Disable 4-bit quantization (for smaller models or >8GB GPUs)")

    # ── query ───────────────────────────────────────────────────────────────
    q = subparsers.add_parser("query", help="Search indexed videos and build compilation")
    q.add_argument("index_dir", help="Index directory from 'index' command")
    q.add_argument("query", help="Natural language search query")
    q.add_argument("-o", "--output", default="./query_result.mp4",
                   help="Output video path (default: ./query_result.mp4)")
    q.add_argument("-n", "--top-n", type=int, default=DEFAULT_TOP_N,
                   help=f"Max clips in compilation (default: {DEFAULT_TOP_N})")
    q.add_argument("--fast", action="store_true",
                   help="Text matching only, skip VLM re-ranking (instant)")
    q.add_argument("--no-labels", action="store_true",
                   help="Don't burn segment labels into video")
    q.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model for VLM re-ranking (default: {DEFAULT_MODEL})")
    q.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization")

    # ── reassemble ──────────────────────────────────────────────────────────
    r = subparsers.add_parser("reassemble",
                              help="Re-edit a query result manifest")
    r.add_argument("manifest", help="Manifest JSON from a previous query")
    r.add_argument("-o", "--output", help="Output video path")
    r.add_argument("--include",
                   help="Only include these segments (comma-separated, e.g. A,C,F)")
    r.add_argument("--exclude",
                   help="Exclude these segments (comma-separated, e.g. B,D)")
    r.add_argument("--trim", action="append", default=[],
                   help="Trim segment: LABEL:START:END (e.g. D:0:3 keeps first 3s)")
    r.add_argument("--no-labels", action="store_true",
                   help="Don't burn segment labels into video")

    args = parser.parse_args()

    if args.command == "index":
        index_videos(
            args.video_dir, args.output,
            fps=args.fps, model_name=args.model,
            use_4bit=not args.no_4bit,
        )
    elif args.command == "query":
        query_index(
            args.index_dir, args.query, args.output,
            top_n=args.top_n, fast=args.fast,
            model_name=args.model, no_labels=args.no_labels,
            use_4bit=not args.no_4bit,
        )
    elif args.command == "reassemble":
        out = args.output or str(
            Path(args.manifest).with_name(
                Path(args.manifest).stem + "_edited.mp4"
            )
        )
        reassemble(
            args.manifest, out,
            include=args.include, exclude=args.exclude,
            trims=args.trim, no_labels=args.no_labels,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
