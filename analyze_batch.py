#!/usr/bin/env python3
"""
Batch video analysis pipeline.
Analyzes all videos in a directory, generates per-video score reports,
and produces an annotated compilation showing WHY each clip was selected.

Outputs:
  - report.json: full analysis data for all videos
  - report.md: human-readable markdown report
  - annotated_compilation.mp4: video with score overlays explaining selections
"""

import os
import sys
import json
import subprocess
import tempfile
import wave
from pathlib import Path
from datetime import datetime

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
WINDOW_SEC = 15
TOP_N_PER_VIDEO = 5  # top N highlights per video

# Signal weights
PANNS_WEIGHT = 0.40
CLIP_WEIGHT = 0.35
FLOW_WEIGHT = 0.25

# PANNs violence labels
VIOLENCE_AUDIO_LABELS = {
    427: "Gunshot, gunfire", 428: "Machine gun", 430: "Artillery fire",
    426: "Explosion", 436: "Boom", 466: "Bang",
    467: "Slap, smack", 468: "Whack, thwack", 469: "Smash, crash",
    470: "Breaking", 472: "Whip",
    14: "Screaming", 8: "Shout", 11: "Yell", 12: "Battle cry", 38: "Groan",
    298: "Fire", 458: "Arrow", 429: "Cap gun", 431: "Fusillade",
    471: "Crushing", 473: "Thud", 143: "Horse gallop",
}

POSITIVE_PROMPTS = [
    "a gunfight with muzzle flash and smoke from a gun barrel",
    "shooting a revolver or rifle at enemies with bullet impact",
    "an intense shootout with multiple gunshots being fired",
    "a Dead Eye slow motion shooting sequence in a western game",
    "a fistfight with punching and kicking in a western game",
    "a violent melee brawl with people fighting on the ground",
    "a lasso takedown of an enemy being pulled off a horse",
    "tackling someone to the ground in a violent struggle",
    "an explosion with fire and debris flying through the air",
    "throwing dynamite that explodes with a large blast",
    "a knife fight or stabbing attack in close combat",
    "a high speed horse chase with gunfire and pursuit",
    "a dramatic shootout from behind cover with enemies",
    "a train robbery with intense action and shooting",
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


def get_video_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def classify_audio_panns(video_path, at_model, sample_fps):
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
        return np.array([0.0]), {}

    window_samples = rate * 2
    violence_indices = list(VIOLENCE_AUDIO_LABELS.keys())

    scores = np.zeros(n_chunks)
    # Track which audio labels triggered
    label_triggers = {}

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
        violence_score = sum(probs[idx] for idx in violence_indices)
        scores[i] = violence_score

        # Record top audio triggers at peaks
        if violence_score > 0.5:
            for idx in violence_indices:
                if probs[idx] > 0.1:
                    name = VIOLENCE_AUDIO_LABELS[idx]
                    label_triggers[name] = max(label_triggers.get(name, 0), float(probs[idx]))

    return scores, label_triggers


def score_frames_clip(frames, model, processor, pos_features, neg_features, batch_size=32):
    all_scores = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
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


def compute_optical_flow(video_path, sample_fps):
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
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_mags.append(float(np.mean(mag)))
        prev_gray = gray

    del vr
    return np.array(flow_mags)


def analyze_video(video_path, clip_model, clip_processor, pos_features, neg_features, at_model):
    """Analyze a single video and return detailed scoring data."""
    name = Path(video_path).stem
    duration = get_video_duration(video_path)

    print(f"  CLIP...", end=" ", flush=True)
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / SAMPLE_FPS))
    all_indices = list(range(0, len(vr), step))
    n_frames = len(all_indices)

    clip_scores_list = []
    for ci in range(0, len(all_indices), 64):
        chunk_idx = all_indices[ci:ci+64]
        chunk_frames = [vr[idx].asnumpy() for idx in chunk_idx]
        chunk_scores = score_frames_clip(chunk_frames, clip_model, clip_processor, pos_features, neg_features)
        clip_scores_list.append(chunk_scores)
        del chunk_frames
    clip_scores = np.concatenate(clip_scores_list)
    del vr
    gc.collect()
    print("ok", end=" | ", flush=True)

    print(f"PANNs...", end=" ", flush=True)
    panns_scores, audio_triggers = classify_audio_panns(video_path, at_model, SAMPLE_FPS)
    print("ok", end=" | ", flush=True)

    print(f"Flow...", end=" ", flush=True)
    flow_scores = compute_optical_flow(video_path, SAMPLE_FPS)
    print("ok")

    min_len = min(len(clip_scores), len(panns_scores), len(flow_scores))
    clip_scores = clip_scores[:min_len]
    panns_scores = panns_scores[:min_len]
    flow_scores = flow_scores[:min_len]

    clip_norm = normalize(clip_scores)
    panns_norm = normalize(panns_scores)
    flow_norm = normalize(flow_scores)

    hybrid = PANNS_WEIGHT * panns_norm + CLIP_WEIGHT * clip_norm + FLOW_WEIGHT * flow_norm
    smooth_window = max(3, SAMPLE_FPS * 3)
    hybrid_smooth = uniform_filter1d(hybrid, size=smooth_window)

    # Find top peaks
    peak_distance = max(1, SAMPLE_FPS * WINDOW_SEC)
    threshold = np.percentile(hybrid_smooth, 75)
    peaks, _ = find_peaks(hybrid_smooth, distance=peak_distance, height=threshold)

    # Build highlights with signal breakdown
    highlights = []
    for peak in peaks[:TOP_N_PER_VIDEO]:
        t_sec = peak / SAMPLE_FPS
        half_win = WINDOW_SEC / 2
        start = max(0, t_sec - half_win)
        end = min(duration, t_sec + half_win)

        # Get signal values at peak
        clip_val = float(clip_norm[peak]) if peak < len(clip_norm) else 0
        panns_val = float(panns_norm[peak]) if peak < len(panns_norm) else 0
        flow_val = float(flow_norm[peak]) if peak < len(flow_norm) else 0
        hybrid_val = float(hybrid_smooth[peak])

        # Determine primary reason
        signals = {"audio": panns_val, "visual": clip_val, "motion": flow_val}
        primary_signal = max(signals, key=signals.get)
        reasons = []
        if panns_val > 0.5:
            reasons.append("loud audio (gunshots/explosions)")
        if clip_val > 0.5:
            reasons.append("violent visual content")
        if flow_val > 0.5:
            reasons.append("fast motion/action")
        if not reasons:
            reasons.append(f"combined signals ({primary_signal} dominant)")

        highlights.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "peak_time": round(t_sec, 2),
            "hybrid_score": round(hybrid_val, 3),
            "clip_score": round(clip_val, 3),
            "panns_score": round(panns_val, 3),
            "flow_score": round(flow_val, 3),
            "primary_signal": primary_signal,
            "reasons": reasons,
        })

    return {
        "file": str(video_path),
        "name": name,
        "duration": round(duration, 1),
        "frames_analyzed": n_frames,
        "score_stats": {
            "hybrid_mean": round(float(hybrid_smooth.mean()), 3),
            "hybrid_max": round(float(hybrid_smooth.max()), 3),
            "hybrid_min": round(float(hybrid_smooth.min()), 3),
        },
        "audio_triggers": dict(sorted(audio_triggers.items(), key=lambda x: -x[1])[:10]),
        "highlights": highlights,
        "violence_rating": "high" if hybrid_smooth.max() > 0.6 else "medium" if hybrid_smooth.max() > 0.4 else "low",
    }


def generate_report_md(results, output_path):
    """Generate a human-readable markdown report."""
    with open(output_path, "w") as f:
        f.write(f"# RDR2 Video Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Videos analyzed**: {len(results)}\n")
        total_highlights = sum(len(r["highlights"]) for r in results)
        f.write(f"**Total highlights found**: {total_highlights}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| # | Video | Duration | Rating | Max Score | Highlights | Top Audio |\n")
        f.write("|---|-------|----------|--------|-----------|------------|----------|\n")
        for i, r in enumerate(sorted(results, key=lambda x: -x["score_stats"]["hybrid_max"])):
            top_audio = ", ".join(list(r["audio_triggers"].keys())[:3]) or "none"
            f.write(f"| {i+1} | {r['name'][-15:]} | {r['duration']:.0f}s | "
                    f"{r['violence_rating']} | {r['score_stats']['hybrid_max']:.3f} | "
                    f"{len(r['highlights'])} | {top_audio} |\n")

        f.write("\n---\n\n")

        # Per-video details
        f.write("## Per-Video Details\n\n")
        for r in results:
            f.write(f"### {r['name']}\n\n")
            f.write(f"- **Duration**: {r['duration']}s\n")
            f.write(f"- **Violence rating**: {r['violence_rating']}\n")
            f.write(f"- **Score range**: {r['score_stats']['hybrid_min']:.3f} - {r['score_stats']['hybrid_max']:.3f}\n")

            if r["audio_triggers"]:
                f.write(f"- **Audio events detected**: {', '.join(r['audio_triggers'].keys())}\n")

            if r["highlights"]:
                f.write(f"\n**Highlights ({len(r['highlights'])}):**\n\n")
                for j, h in enumerate(r["highlights"]):
                    f.write(f"{j+1}. **{h['start']:.1f}s - {h['end']:.1f}s** "
                            f"(score: {h['hybrid_score']:.3f})\n")
                    f.write(f"   - Primary: {h['primary_signal']} | "
                            f"Audio: {h['panns_score']:.2f} | "
                            f"Visual: {h['clip_score']:.2f} | "
                            f"Motion: {h['flow_score']:.2f}\n")
                    f.write(f"   - Why: {'; '.join(h['reasons'])}\n")
            else:
                f.write(f"\nNo significant highlights found.\n")
            f.write("\n")


def build_annotated_compilation(results, output_dir):
    """Build a compilation with score annotations explaining each clip selection."""
    all_clips = []
    label_idx = 0

    for r in results:
        for h in r["highlights"]:
            label = chr(65 + label_idx // 26) + str(label_idx % 26 + 1) if label_idx >= 26 else chr(65 + label_idx)
            if label_idx < 26:
                label = chr(65 + label_idx)
            else:
                label = chr(65 + label_idx // 26 - 1) + chr(65 + label_idx % 26)

            all_clips.append({
                "label": label,
                "source_file": r["file"],
                "source_name": r["name"],
                "start": h["start"],
                "end": h["end"],
                "duration": h["duration"],
                "score": h["hybrid_score"],
                "clip_score": h["clip_score"],
                "panns_score": h["panns_score"],
                "flow_score": h["flow_score"],
                "primary_signal": h["primary_signal"],
                "reasons": h["reasons"],
            })
            label_idx += 1

    # Sort by score descending
    all_clips.sort(key=lambda x: -x["score"])

    # Extract and annotate each clip with ffmpeg drawtext
    tmpdir = tempfile.mkdtemp(prefix="annotated_")
    clip_paths = []

    for i, clip in enumerate(all_clips):
        out = os.path.join(tmpdir, f"clip_{i:04d}.mp4")
        reason_text = clip["reasons"][0][:40] if clip["reasons"] else "combined signals"
        # Escape special chars for ffmpeg drawtext
        reason_text = reason_text.replace("'", "").replace(":", " -").replace("(", "").replace(")", "")

        annotation = (
            f"[{clip['label']}] Score: {clip['score']:.2f} | "
            f"Audio: {clip['panns_score']:.2f} Visual: {clip['clip_score']:.2f} Motion: {clip['flow_score']:.2f}"
        )
        source_text = f"Source: {clip['source_name'][-20:]}"
        why_text = f"Why: {reason_text}"

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", f"{clip['start']:.3f}",
                "-i", clip["source_file"],
                "-t", f"{clip['duration']:.3f}",
                "-vf", (
                    f"drawtext=text='{annotation}':fontsize=24:fontcolor=white:borderw=2:bordercolor=black:x=10:y=10,"
                    f"drawtext=text='{source_text}':fontsize=20:fontcolor=#aaaaaa:borderw=1:bordercolor=black:x=10:y=45,"
                    f"drawtext=text='{why_text}':fontsize=22:fontcolor=#ffcc00:borderw=2:bordercolor=black:x=10:y=75"
                ),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                out,
            ], capture_output=True, check=True)

            if os.path.getsize(out) > 1000:
                clip_paths.append(out)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Failed to extract {clip['label']}: {e.stderr[-200:] if e.stderr else ''}")

    return clip_paths, all_clips, tmpdir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch analyze videos and generate annotated compilation")
    parser.add_argument("input_dir", help="Directory with video files")
    parser.add_argument("-o", "--output-dir", default="./highlights_batch",
                        help="Output directory (default: ./highlights_batch)")
    parser.add_argument("-n", "--top-n", type=int, default=TOP_N_PER_VIDEO,
                        help=f"Top N highlights per video (default: {TOP_N_PER_VIDEO})")
    parser.add_argument("--skip-annotated", action="store_true",
                        help="Only generate report, skip annotated video")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    global TOP_N_PER_VIDEO
    TOP_N_PER_VIDEO = args.top_n

    # Find video files
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    videos = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in video_exts
    ])

    if not videos:
        print(f"No video files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} videos in {input_dir}\n")

    # Load models
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

    # Analyze each video
    results = []
    for i, video_path in enumerate(videos):
        name = Path(video_path).stem
        print(f"\n[{i+1}/{len(videos)}] {name}")
        try:
            result = analyze_video(
                video_path, clip_model, clip_processor,
                pos_features, neg_features, at_model,
            )
            results.append(result)
            print(f"  Rating: {result['violence_rating']} | "
                  f"Max score: {result['score_stats']['hybrid_max']:.3f} | "
                  f"Highlights: {len(result['highlights'])}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Free GPU
    del clip_model, clip_processor, pos_features, neg_features, at_model
    torch.cuda.empty_cache()
    gc.collect()

    # Save JSON report
    report_json_path = os.path.join(output_dir, "report.json")
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON report: {report_json_path}")

    # Generate markdown report
    report_md_path = os.path.join(output_dir, "report.md")
    generate_report_md(results, report_md_path)
    print(f"Markdown report: {report_md_path}")

    if args.skip_annotated:
        print("\nSkipping annotated compilation (--skip-annotated)")
        return

    # Build annotated compilation
    print(f"\nBuilding annotated compilation...")
    clip_paths, all_clips, tmpdir = build_annotated_compilation(results, output_dir)

    if not clip_paths:
        print("No clips extracted!")
        return

    # Concatenate
    compilation_path = os.path.join(output_dir, "annotated_compilation.mp4")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
        list_path = f.name

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "21",
            "-c:a", "aac", "-b:a", "192k",
            compilation_path,
        ], capture_output=True, check=True)
    finally:
        os.unlink(list_path)

    # Save manifest for Remotion
    manifest = {
        "source": os.path.abspath(compilation_path),
        "segments": [{
            "label": c["label"],
            "start": sum(cl["duration"] for cl in all_clips[:j]),
            "end": sum(cl["duration"] for cl in all_clips[:j+1]),
            "duration": c["duration"],
            "score": c["score"],
        } for j, c in enumerate(all_clips)],
    }
    manifest_path = os.path.join(output_dir, "annotated_compilation.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

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

    comp_dur = get_video_duration(compilation_path)
    comp_size = os.path.getsize(compilation_path) / (1024 * 1024)
    print(f"\nAnnotated compilation: {compilation_path}")
    print(f"  Duration: {comp_dur:.1f}s")
    print(f"  Size: {comp_size:.1f}MB")
    print(f"  Clips: {len(all_clips)}")
    print(f"  Manifest: {manifest_path}")

    # Summary
    total_highlights = sum(len(r["highlights"]) for r in results)
    high_videos = sum(1 for r in results if r["violence_rating"] == "high")
    med_videos = sum(1 for r in results if r["violence_rating"] == "medium")
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS COMPLETE")
    print(f"  Videos: {len(results)}")
    print(f"  High violence: {high_videos} | Medium: {med_videos} | Low: {len(results) - high_videos - med_videos}")
    print(f"  Total highlights: {total_highlights}")
    print(f"  Report: {report_md_path}")
    print(f"  Compilation: {compilation_path}")


if __name__ == "__main__":
    main()
