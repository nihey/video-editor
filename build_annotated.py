#!/usr/bin/env python3
"""
Build annotated compilation from an existing report.json.
Re-uses the analysis data without re-running ML models.
"""
import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"


def get_video_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def segment_label(index):
    if index < 26:
        return chr(65 + index)
    return chr(65 + index // 26 - 1) + chr(65 + index % 26)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build annotated compilation from report.json")
    parser.add_argument("report", help="report.json from analyze_batch.py")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-n", "--max-clips", type=int, default=0,
                        help="Max number of clips (0 = all)")
    args = parser.parse_args()

    with open(args.report) as f:
        results = json.load(f)

    output_dir = os.path.dirname(args.report)
    output_path = args.output or os.path.join(output_dir, "annotated_compilation.mp4")

    # Collect all highlights
    all_clips = []
    for r in results:
        for h in r["highlights"]:
            all_clips.append({
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

    # Sort by score descending
    all_clips.sort(key=lambda x: -x["score"])

    if args.max_clips > 0:
        all_clips = all_clips[:args.max_clips]

    # Assign labels
    for i, clip in enumerate(all_clips):
        clip["label"] = segment_label(i)

    print(f"Building annotated compilation: {len(all_clips)} clips")

    tmpdir = tempfile.mkdtemp(prefix="annotated_")
    clip_paths = []

    for i, clip in enumerate(all_clips):
        out = os.path.join(tmpdir, f"clip_{i:04d}.mp4")
        reason_text = clip["reasons"][0][:50] if clip["reasons"] else "combined signals"

        line1 = (f"[{clip['label']}] Score {clip['score']:.2f} | "
                 f"Audio {clip['panns_score']:.2f}  Visual {clip['clip_score']:.2f}  Motion {clip['flow_score']:.2f}")
        line2 = f"Source {clip['source_name'][-25:]}"
        line3 = f"Why {reason_text}"

        tf1 = os.path.join(tmpdir, f"t1_{i:04d}.txt")
        tf2 = os.path.join(tmpdir, f"t2_{i:04d}.txt")
        tf3 = os.path.join(tmpdir, f"t3_{i:04d}.txt")
        for path, text in [(tf1, line1), (tf2, line2), (tf3, line3)]:
            with open(path, "w") as tf:
                tf.write(text)

        print(f"  [{clip['label']}] {clip['source_name'][-20:]} "
              f"{clip['start']:.1f}-{clip['end']:.1f}s (score={clip['score']:.3f})", end="")

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", f"{clip['start']:.3f}",
                "-i", clip["source_file"],
                "-t", f"{clip['duration']:.3f}",
                "-vf", (
                    f"drawtext=textfile='{tf1}':fontsize=24:fontcolor=white:borderw=2:bordercolor=black:x=10:y=10,"
                    f"drawtext=textfile='{tf2}':fontsize=20:fontcolor=#aaaaaa:borderw=1:bordercolor=black:x=10:y=45,"
                    f"drawtext=textfile='{tf3}':fontsize=22:fontcolor=#ffcc00:borderw=2:bordercolor=black:x=10:y=75"
                ),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                out,
            ], capture_output=True, check=True)

            if os.path.getsize(out) > 1000:
                clip_paths.append(out)
                print(" ok")
            else:
                print(" empty")
        except subprocess.CalledProcessError as e:
            print(f" FAILED: {e.stderr[-100:].decode() if e.stderr else ''}")

    if not clip_paths:
        print("ERROR: No clips extracted!")
        return

    # Concatenate
    print(f"\nJoining {len(clip_paths)} clips...")
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
            output_path,
        ], capture_output=True, check=True)
    finally:
        os.unlink(list_path)

    # Save manifest
    manifest = {
        "source": os.path.abspath(output_path),
        "segments": [],
    }
    cursor = 0.0
    for clip in all_clips[:len(clip_paths)]:
        manifest["segments"].append({
            "label": clip["label"],
            "start": round(cursor, 2),
            "end": round(cursor + clip["duration"], 2),
            "duration": clip["duration"],
            "score": clip["score"],
        })
        cursor += clip["duration"]

    manifest_path = output_path.replace(".mp4", ".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Cleanup
    for p in clip_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    for f in os.listdir(tmpdir):
        try:
            os.unlink(os.path.join(tmpdir, f))
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    dur = get_video_duration(output_path)
    size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput: {output_path}")
    print(f"  Duration: {dur:.1f}s")
    print(f"  Size: {size:.1f}MB")
    print(f"  Clips: {len(clip_paths)}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
