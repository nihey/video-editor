# Video Editor - Gameplay Highlight Detector

Automatically detects the best moments in gameplay videos using a hybrid ML approach that combines **CLIP semantic scoring** with **audio energy analysis**.

Built for PS5 gameplay clips but works with any MP4 video collection.

## How It Works

The detection pipeline has two signals that are combined into a hybrid score:

### 1. CLIP Visual Scoring (60% weight)

Samples frames at 2 FPS and scores each frame against text prompts using [OpenAI CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14). Positive prompts describe intense action (gunfights, explosions, combat) while negative prompts describe uninteresting content (menus, loading screens, idle walking). The final CLIP score is `positive_similarity - negative_similarity`.

### 2. Audio Energy Analysis (40% weight)

Extracts audio as mono 16kHz WAV via ffmpeg and computes RMS energy at 2 samples/second. Loud moments (gunfire, explosions, dramatic music) score higher.

### Hybrid Scoring

Both signals are normalized to [0, 1] per video, then combined:

```
hybrid_score = 0.6 * clip_score + 0.4 * audio_score
```

A 3-second smoothing window is applied, then peaks are detected using `scipy.signal.find_peaks` with a minimum distance of 8 seconds and a threshold at the 75th percentile. All peaks across all videos are ranked globally, and the top N are extracted as clips.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (tested on RTX 4060 8GB)
- ffmpeg
- ~2GB disk space for the CLIP model (downloaded on first run)

## Setup

```bash
conda create -n video-highlights python=3.11 -y
conda activate video-highlights

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers decord opencv-python-headless scipy
```

## Usage

```bash
conda activate video-highlights

# Basic usage - point at a folder of MP4s
python detect_highlights.py /path/to/your/videos

# With options
python detect_highlights.py /path/to/your/videos -o ./my-highlights -n 20 -w 12

# Using environment variables
VIDEO_DIR=/path/to/videos OUTPUT_DIR=./output TOP_N=15 WINDOW_SEC=10 python detect_highlights.py
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `video_dir` | Directory containing MP4 video files | `./videos` or `$VIDEO_DIR` |
| `-o, --output` | Output directory for extracted clips | `./highlights` or `$OUTPUT_DIR` |
| `-n, --top-n` | Number of top highlights to extract | `10` or `$TOP_N` |
| `-w, --window` | Duration of each clip in seconds | `8` or `$WINDOW_SEC` |

## Output

The script creates an output directory with:

- `highlight_01.mp4` ... `highlight_NN.mp4` - Extracted video clips ranked by score
- `highlights.json` - Metadata with scores, timestamps, and source video paths

Each entry in `highlights.json`:

```json
{
  "video": "/path/to/source.mp4",
  "video_name": "source_filename",
  "time_sec": 50.5,
  "score": 0.804,
  "clip_score": 0.935,
  "audio_score": 0.759
}
```

## Customizing Detection

Edit the `POSITIVE_PROMPTS` and `NEGATIVE_PROMPTS` lists in `detect_highlights.py` to change what the detector looks for. The current prompts are tuned for western game gunfight detection (Red Dead Redemption 2).

Examples for other games:

```python
# Racing game highlights
POSITIVE_PROMPTS = [
    "a dramatic car crash in a racing game",
    "a close finish at the finish line",
    "drifting at high speed around a corner",
]

# Horror game highlights
POSITIVE_PROMPTS = [
    "a jump scare in a horror game",
    "a monster attacking the player",
    "a dark scary hallway with danger",
]
```

## Performance

On an RTX 4060 (8GB VRAM) with 20 CPU cores:

- ~73 videos (3.3 hours, 15GB) processed in ~25 minutes
- ~2 FPS sampling = ~11,800 frames analyzed
- Peak GPU memory: ~3GB
- Peak RAM: ~5GB
