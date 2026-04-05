#!/usr/bin/env python3
"""
Query videos with natural language using multimodal embeddings.

Embeds video chunks and text queries into the same vector space using
Qwen3-VL-Embedding. ChromaDB stores embeddings for instant semantic search.
No text middleman — video pixels are directly comparable to text.

Two-phase system:
  1. index  — Chunk videos, skip still frames, embed chunks, store in ChromaDB
  2. query  — Embed text query, vector search, build compilation (sub-second)

Requires: pip install transformers bitsandbytes accelerate qwen-vl-utils chromadb

Hardware: RTX 4060 8GB (Qwen3-VL-Embedding-2B, ~4GB VRAM)
"""

import os
import sys
import json
import gc
import hashlib
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"

# Chunking
CHUNK_DURATION = 30    # seconds per chunk
CHUNK_OVERLAP = 5      # overlap between consecutive chunks

# Preprocessing (applied to chunks before embedding)
PREPROCESS_FPS = 5     # reduce to 5fps for embedding
PREPROCESS_HEIGHT = 480  # downscale to 480p

# Embedding
EMBED_FPS = 1.0        # fps fed to the model
EMBED_MAX_FRAMES = 32  # max frames per chunk for the model
EMBED_DIMS = 768       # MRL truncation dimensionality

# Still-frame detection
STILL_THRESHOLD = 0.98  # JPEG size ratio above which a chunk is "still"

# Query defaults
DEFAULT_TOP_N = 20
PAD_BEFORE = 2.0
PAD_AFTER = 2.0
MERGE_GAP_SEC = 5.0
MIN_CLIP_SEC = 3.0
MAX_CLIP_SEC = 30.0
CONFIDENCE_THRESHOLD = 0.30

# ChromaDB
COLLECTION_PREFIX = "video_chunks"


# ── Model ───────────────────────────────────────────────────────────────────

def _build_embedding_class():
    """Build Qwen3VLForEmbedding using the correct Qwen3VL base classes.
    Wraps Qwen3VLModel directly to get last_hidden_state for pooling,
    avoiding the causal LM head entirely.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig,
    )

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        config: Qwen3VLConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, pixel_values=None,
                    pixel_values_videos=None, image_grid_thw=None,
                    video_grid_thw=None, cache_position=None, **kwargs):
            return self.model(
                input_ids=input_ids, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                position_ids=position_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                cache_position=cache_position, **kwargs,
            )

    return Qwen3VLForEmbedding


def _pooling_last(hidden_state, attention_mask):
    """Pool at the last non-padded token position."""
    flipped = attention_mask.flip(dims=[1])
    last_pos = flipped.argmax(dim=1)
    col = attention_mask.shape[1] - last_pos - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


def load_model(model_name=DEFAULT_MODEL, use_4bit=True):
    """Load Qwen3-VL-Embedding with optional 4-bit quantization."""
    Qwen3VLForEmbedding = _build_embedding_class()
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    print(f"Loading {model_name}{'  (4-bit)' if use_4bit else ''}...")

    load_kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = Qwen3VLForEmbedding.from_pretrained(model_name, **load_kwargs)
    if not use_4bit:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    processor = Qwen3VLProcessor.from_pretrained(model_name, padding_side="right")
    model.eval()
    print("Model loaded.")
    return model, processor


def embed_video_chunk(model, processor, chunk_path):
    """Embed a video chunk into a 768-dim vector."""
    from qwen_vl_utils import process_vision_info

    conversation = [{
        "role": "system",
        "content": [{"type": "text", "text": "Represent the video for retrieval."}],
    }, {
        "role": "user",
        "content": [{"type": "video", "video": f"file://{os.path.abspath(chunk_path)}",
                      "fps": EMBED_FPS, "max_frames": EMBED_MAX_FRAMES}],
    }]

    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )
    images, video_inputs, video_kwargs = process_vision_info(
        conversation, return_video_metadata=True, return_video_kwargs=True,
    )

    if video_inputs is not None:
        videos, video_metadata = zip(*video_inputs)
        videos, video_metadata = list(videos), list(video_metadata)
    else:
        videos, video_metadata = None, None

    inputs = processor(
        text=[text], images=images, videos=videos, video_metadata=video_metadata,
        padding=True, return_tensors="pt", **video_kwargs,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        emb = _pooling_last(outputs.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)

    # MRL truncation to EMBED_DIMS + L2 normalize
    vec = emb[0][:EMBED_DIMS]
    norm = torch.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    result = vec.cpu().float().tolist()

    del inputs, outputs, emb
    return result


def embed_query(model, processor, query_text):
    """Embed a text query into the same 768-dim vector space."""
    conversation = [{
        "role": "system",
        "content": [{"type": "text", "text": "Retrieve videos relevant to the query."}],
    }, {
        "role": "user",
        "content": [{"type": "text", "text": query_text}],
    }]

    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], padding=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        emb = _pooling_last(outputs.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)

    vec = emb[0][:EMBED_DIMS]
    norm = torch.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    result = vec.cpu().float().tolist()

    del inputs, outputs, emb
    return result


# ── Video Chunking ──────────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def chunk_video(video_path: str, output_dir: str,
                chunk_sec: int = CHUNK_DURATION,
                overlap_sec: int = CHUNK_OVERLAP) -> list[dict]:
    """Split video into overlapping chunks using ffmpeg stream copy.
    Returns list of {path, start, end, source}.
    """
    duration = get_video_duration(video_path)
    step = chunk_sec - overlap_sec
    chunks = []

    for start in range(0, int(duration), step):
        end = min(start + chunk_sec, duration)
        if end - start < 2:  # skip tiny tail chunks
            break

        chunk_name = f"chunk_{start:06d}.mp4"
        chunk_path = os.path.join(output_dir, chunk_name)

        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
            "-t", str(chunk_sec), "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            chunk_path,
        ], capture_output=True, check=True)

        if os.path.getsize(chunk_path) > 1000:
            chunks.append({
                "path": chunk_path,
                "start": float(start),
                "end": float(end),
                "source": os.path.abspath(video_path),
            })

    return chunks


def is_still_frame(chunk_path: str, threshold: float = STILL_THRESHOLD) -> bool:
    """Detect still/static chunks by comparing JPEG sizes of 3 sample frames.
    If min/max size ratio >= threshold, the chunk is effectively frozen.
    """
    try:
        duration = get_video_duration(chunk_path)
    except (ValueError, subprocess.CalledProcessError):
        return True

    if duration < 1:
        return True

    sizes = []
    with tempfile.TemporaryDirectory(prefix="still_") as tmpdir:
        for t in [0.1, duration * 0.5, max(0.2, duration - 0.1)]:
            frame_path = os.path.join(tmpdir, f"f_{t:.2f}.jpg")
            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", chunk_path,
                "-frames:v", "1", "-q:v", "5", frame_path,
            ], capture_output=True)
            if os.path.exists(frame_path):
                sizes.append(os.path.getsize(frame_path))

    if len(sizes) < 2:
        return True

    ratio = min(sizes) / max(sizes) if max(sizes) > 0 else 1.0
    return ratio >= threshold


def preprocess_chunk(input_path: str, output_path: str,
                     fps: int = PREPROCESS_FPS,
                     height: int = PREPROCESS_HEIGHT):
    """Downscale and reduce fps for efficient embedding."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale=-2:{height}",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-an",  # drop audio — embedding is visual only
        output_path,
    ], capture_output=True, check=True)


# ── ChromaDB Store ──────────────────────────────────────────────────────────

def get_collection(db_path: str, model_name: str):
    """Get or create a ChromaDB collection for this model."""
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    # Collection name includes model to avoid mixing embeddings
    safe_model = model_name.replace("/", "_").replace("-", "_").lower()
    name = f"{COLLECTION_PREFIX}_{safe_model}"
    # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
    name = name[:63]
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def chunk_id(source_path: str, start_time: float) -> str:
    """Deterministic chunk ID from source path + start time."""
    key = f"{source_path}:{start_time:.1f}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Helpers ─────────────────────────────────────────────────────────────────

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
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ── Index Command ───────────────────────────────────────────────────────────

def index_videos(video_dir: str, index_dir: str,
                 model_name: str = DEFAULT_MODEL, use_4bit: bool = True,
                 chunk_sec: int = CHUNK_DURATION,
                 overlap_sec: int = CHUNK_OVERLAP):
    """Index all videos: chunk → skip stills → preprocess → embed → store."""
    os.makedirs(index_dir, exist_ok=True)
    db_path = os.path.join(index_dir, "chromadb")

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

    # Load model + store
    model, processor = load_model(model_name, use_4bit=use_4bit)
    collection = get_collection(db_path, model_name)

    # Check already-indexed chunks
    existing_ids = set()
    if collection.count() > 0:
        existing_data = collection.get()
        existing_ids = set(existing_data["ids"])
    print(f"Found {len(videos)} videos, {len(existing_ids)} chunks already indexed\n")

    total_chunks_embedded = 0
    total_chunks_skipped = 0
    total_still = 0
    t0_global = time.time()

    for vi, video_path in enumerate(videos):
        abs_path = os.path.abspath(video_path)
        name = Path(video_path).stem
        duration = get_video_duration(video_path)
        print(f"[{vi+1}/{len(videos)}] {name} ({duration:.1f}s)")

        with tempfile.TemporaryDirectory(prefix="idx_chunks_") as chunk_dir:
            # 1. Chunk the video
            print(f"  Chunking ({chunk_sec}s, {overlap_sec}s overlap)...",
                  end=" ", flush=True)
            chunks = chunk_video(video_path, chunk_dir, chunk_sec, overlap_sec)
            print(f"{len(chunks)} chunks")

            for ci, chunk in enumerate(chunks):
                cid = chunk_id(abs_path, chunk["start"])

                # Skip if already indexed
                if cid in existing_ids:
                    total_chunks_skipped += 1
                    continue

                # 2. Still-frame detection
                if is_still_frame(chunk["path"]):
                    total_still += 1
                    print(f"    [{ci+1}/{len(chunks)}] {chunk['start']:.0f}s "
                          f"— still frame, skipped")
                    continue

                # 3. Preprocess
                preprocessed_path = os.path.join(chunk_dir, f"pp_{ci:04d}.mp4")
                try:
                    preprocess_chunk(chunk["path"], preprocessed_path)
                except subprocess.CalledProcessError:
                    print(f"    [{ci+1}/{len(chunks)}] {chunk['start']:.0f}s "
                          f"— preprocess failed, skipped")
                    continue

                # 4. Embed
                t0 = time.time()
                try:
                    embedding = embed_video_chunk(model, processor, preprocessed_path)
                except Exception as e:
                    print(f"    [{ci+1}/{len(chunks)}] {chunk['start']:.0f}s "
                          f"— embed error: {e}")
                    continue
                embed_time = time.time() - t0

                # 5. Store in ChromaDB
                collection.add(
                    ids=[cid],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": abs_path,
                        "source_name": name,
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "duration": duration,
                        "indexed_at": datetime.now().isoformat(),
                    }],
                )
                existing_ids.add(cid)
                total_chunks_embedded += 1

                elapsed = time.time() - t0_global
                rate = total_chunks_embedded / elapsed if elapsed > 0 else 0
                print(f"    [{ci+1}/{len(chunks)}] {chunk['start']:.0f}s-"
                      f"{chunk['end']:.0f}s — embedded in {embed_time:.1f}s "
                      f"({rate:.2f} chunks/s)")

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    total_elapsed = time.time() - t0_global
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"  Videos: {len(videos)}")
    print(f"  Chunks embedded: {total_chunks_embedded}")
    print(f"  Chunks skipped (already indexed): {total_chunks_skipped}")
    print(f"  Chunks skipped (still frame): {total_still}")
    print(f"  Total in index: {collection.count()}")
    print(f"  Time: {format_time(total_elapsed)}")
    print(f"  Index: {db_path}")


# ── Query Command ───────────────────────────────────────────────────────────

def query_index(index_dir: str, query_text: str, output_path: str,
                top_n: int = DEFAULT_TOP_N,
                model_name: str = DEFAULT_MODEL, no_labels: bool = False,
                use_4bit: bool = True, threshold: float = CONFIDENCE_THRESHOLD):
    """Embed query, search ChromaDB, build compilation."""
    db_path = os.path.join(index_dir, "chromadb")
    if not os.path.exists(db_path):
        print(f"ERROR: Index not found at {db_path}")
        print(f"Run: python query_videos.py index <video_dir> -o {index_dir}")
        sys.exit(1)

    collection = get_collection(db_path, model_name)
    total_chunks = collection.count()
    if total_chunks == 0:
        print("ERROR: Index is empty. Run the index command first.")
        sys.exit(1)

    print(f"Index: {total_chunks} chunks")
    print(f"Query: \"{query_text}\"\n")

    # 1. Embed the text query
    print("Loading model for query embedding...")
    model, processor = load_model(model_name, use_4bit=use_4bit)

    t0 = time.time()
    query_embedding = embed_query(model, processor, query_text)
    print(f"Query embedded in {time.time() - t0:.2f}s")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Search ChromaDB
    # Request more results than needed to allow for merging
    n_results = min(total_chunks, top_n * 5)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Convert cosine distance to similarity score
    scored = []
    for dist, meta in zip(distances, metadatas):
        score = 1.0 - dist  # ChromaDB cosine distance → similarity
        if score >= threshold:
            scored.append({
                "source": meta["source"],
                "source_name": meta["source_name"],
                "start": meta["start"],
                "end": meta["end"],
                "video_duration": meta["duration"],
                "score": round(score, 4),
            })

    print(f"\nSearch results: {len(scored)} chunks above threshold "
          f"({threshold:.0%})")

    if not scored:
        print("No relevant scenes found. Try a different query or lower --threshold.")
        return

    # Show top results
    print(f"\nTop matches:")
    for r in scored[:10]:
        print(f"  [{r['score']:.3f}] {r['source_name']} "
              f"@ {r['start']:.0f}s-{r['end']:.0f}s")

    # 3. Build segments — merge overlapping/nearby chunks from same video
    by_video: dict[str, list[dict]] = {}
    for r in scored:
        key = r["source"]
        if key not in by_video:
            by_video[key] = []
        by_video[key].append(r)

    segments = []
    for source, chunks in by_video.items():
        chunks.sort(key=lambda x: x["start"])
        video_dur = chunks[0]["video_duration"]

        merged: list[dict] = []
        for c in chunks:
            start = max(0.0, c["start"] - PAD_BEFORE)
            end = min(video_dur, c["end"] + PAD_AFTER)

            if merged and start <= merged[-1]["end"] + MERGE_GAP_SEC:
                merged[-1]["end"] = max(merged[-1]["end"], end)
                merged[-1]["score"] = max(merged[-1]["score"], c["score"])
            else:
                merged.append({
                    "source": source,
                    "source_name": c["source_name"],
                    "start": start,
                    "end": end,
                    "score": c["score"],
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

    # Sort by score, take top N
    segments.sort(key=lambda x: -x["score"])
    segments = segments[:top_n]

    # Assign labels
    for i, seg in enumerate(segments):
        seg["label"] = segment_label(i)
        seg["start"] = round(seg["start"], 2)
        seg["end"] = round(seg["end"], 2)

    total_dur = sum(s["duration"] for s in segments)
    print(f"\nSelected {len(segments)} segments, total {total_dur:.1f}s:")
    for seg in segments:
        print(f"  [{seg['label']}] {seg['source_name']} @ {seg['start']:.1f}s - "
              f"{seg['end']:.1f}s ({seg['duration']:.1f}s, score={seg['score']:.4f})")

    # 4. Save manifest
    manifest_path = str(Path(output_path).with_suffix(".json"))
    manifest = {
        "query": query_text,
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

    # 5. Build compilation
    print(f"\nExtracting clips...")
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
    print(f"  Query: \"{query_text}\"")


# ── Reassemble Command ─────────────────────────────────────────────────────

def reassemble(manifest_path: str, output_path: str,
               include: str | None = None, exclude: str | None = None,
               trims: list[str] | None = None, no_labels: bool = False):
    """Re-assemble video from a multi-source manifest with optional edits."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    segments = manifest["segments"]

    # Parse trims
    trim_map: dict[str, tuple[float | None, float | None]] = {}
    for t in (trims or []):
        parts = t.split(":")
        label = parts[0].upper()
        trim_start = float(parts[1]) if len(parts) > 1 and parts[1] else None
        trim_end = float(parts[2]) if len(parts) > 2 and parts[2] else None
        trim_map[label] = (trim_start, trim_end)

    if include:
        keep = {s.strip().upper() for s in include.split(",")}
        segments = [s for s in segments if s["label"] in keep]
    if exclude:
        drop = {s.strip().upper() for s in exclude.split(",")}
        segments = [s for s in segments if s["label"] not in drop]

    if not segments:
        print("ERROR: No segments remaining after filtering!")
        sys.exit(1)

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


# ── Stats Command ───────────────────────────────────────────────────────────

def show_stats(index_dir: str, model_name: str = DEFAULT_MODEL):
    """Show index statistics."""
    db_path = os.path.join(index_dir, "chromadb")
    if not os.path.exists(db_path):
        print(f"No index found at {db_path}")
        return

    collection = get_collection(db_path, model_name)
    total = collection.count()

    if total == 0:
        print("Index is empty.")
        return

    data = collection.get(include=["metadatas"])
    sources = set()
    total_duration = 0.0
    for meta in data["metadatas"]:
        sources.add(meta["source_name"])
        total_duration = max(total_duration,
                             total_duration)  # approximate

    print(f"Index: {db_path}")
    print(f"  Model: {model_name}")
    print(f"  Total chunks: {total}")
    print(f"  Unique videos: {len(sources)}")
    for name in sorted(sources):
        count = sum(1 for m in data["metadatas"] if m["source_name"] == name)
        print(f"    {name}: {count} chunks")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Query videos with natural language using multimodal embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all videos (one-time, ~2s per 30s chunk)
  python query_videos.py index /path/to/videos -o ./video_index

  # Search — sub-second after model loads
  python query_videos.py query ./video_index "explosions and dynamite" -o explosions.mp4
  python query_videos.py query ./video_index "shots to the head" -o headshots.mp4
  python query_videos.py query ./video_index "horse chases with gunfire" -o chases.mp4

  # Re-edit a result
  python query_videos.py reassemble headshots.json -o v2.mp4 --include A,C,F

  # Index stats
  python query_videos.py stats ./video_index
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── index ───────────────────────────────────────────────────────────────
    idx = subparsers.add_parser("index",
                                help="Index videos with multimodal embeddings")
    idx.add_argument("video_dir", help="Directory containing video files")
    idx.add_argument("-o", "--output", default="./video_index",
                     help="Index directory (default: ./video_index)")
    idx.add_argument("--model", default=DEFAULT_MODEL,
                     help=f"Embedding model (default: {DEFAULT_MODEL})")
    idx.add_argument("--no-4bit", action="store_true",
                     help="Disable 4-bit quantization")
    idx.add_argument("--chunk-duration", type=int, default=CHUNK_DURATION,
                     help=f"Chunk duration in seconds (default: {CHUNK_DURATION})")
    idx.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP,
                     help=f"Overlap between chunks in seconds (default: {CHUNK_OVERLAP})")

    # ── query ───────────────────────────────────────────────────────────────
    q = subparsers.add_parser("query",
                              help="Search indexed videos with text")
    q.add_argument("index_dir", help="Index directory from 'index' command")
    q.add_argument("query", help="Natural language search query")
    q.add_argument("-o", "--output", default="./query_result.mp4",
                   help="Output video path (default: ./query_result.mp4)")
    q.add_argument("-n", "--top-n", type=int, default=DEFAULT_TOP_N,
                   help=f"Max clips in compilation (default: {DEFAULT_TOP_N})")
    q.add_argument("--no-labels", action="store_true",
                   help="Don't burn segment labels into video")
    q.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                   help=f"Min similarity score (default: {CONFIDENCE_THRESHOLD})")
    q.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Embedding model (default: {DEFAULT_MODEL})")
    q.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization")

    # ── reassemble ──────────────────────────────────────────────────────────
    r = subparsers.add_parser("reassemble",
                              help="Re-edit a query result manifest")
    r.add_argument("manifest", help="Manifest JSON from a previous query")
    r.add_argument("-o", "--output", help="Output video path")
    r.add_argument("--include",
                   help="Only include segments (comma-separated, e.g. A,C,F)")
    r.add_argument("--exclude",
                   help="Exclude segments (comma-separated, e.g. B,D)")
    r.add_argument("--trim", action="append", default=[],
                   help="Trim segment: LABEL:START:END (e.g. D:0:3)")
    r.add_argument("--no-labels", action="store_true")

    # ── stats ───────────────────────────────────────────────────────────────
    s = subparsers.add_parser("stats", help="Show index statistics")
    s.add_argument("index_dir", help="Index directory")
    s.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Embedding model (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    if args.command == "index":
        index_videos(
            args.video_dir, args.output,
            model_name=args.model, use_4bit=not args.no_4bit,
            chunk_sec=args.chunk_duration, overlap_sec=args.chunk_overlap,
        )
    elif args.command == "query":
        query_index(
            args.index_dir, args.query, args.output,
            top_n=args.top_n, model_name=args.model,
            no_labels=args.no_labels, use_4bit=not args.no_4bit,
            threshold=args.threshold,
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
    elif args.command == "stats":
        show_stats(args.index_dir, model_name=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
