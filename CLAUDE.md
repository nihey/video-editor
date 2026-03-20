# Video Editor

## Project Overview

Gameplay highlight and violence detector that uses multiple ML models + audio analysis to find the best moments in video collections. Currently tuned for RDR2 PS5 gameplay clips but designed to work with any game.

## Critical Constraints

**Memory is limited. Never assume infinite RAM or VRAM.**
- Always process video frames in small batches (64 frames max) — never load all frames at once
- Free large arrays (`del frames`) as soon as they're no longer needed
- Optical flow must read frames one-at-a-time from the VideoReader, not batch-load
- VideoMAE windows must read frames individually, not via `get_batch` on large sets
- Long videos (15+ min, 1900+ frames at 2fps) are the stress test — always design for them

## Architecture

```
video-editor/
├── detect_highlights.py   # CLIP + audio hybrid highlight detection
├── detect_violence.py     # 4-signal violence detection (PANNs + CLIP + VideoMAE + optical flow)
├── condense.py            # Rapid-fire montage from highlight clips (audio transient detection)
├── detect_vlm.py          # VLM-based detection using Claude API (most accurate, slowest)
├── highlights/            # Output directory (gitignored)
├── .gitignore
├── CLAUDE.md
└── README.md
```

## Tech Stack

- **Audio Classification**: PANNs CNN14 (AudioSet 527 classes — gunshot, explosion, punch, scream)
- **Visual Scoring**: OpenAI CLIP ViT-L/14 (text-guided frame scoring)
- **Action Recognition**: VideoMAE (Kinetics-400 — punching, kicking, sword fighting, wrestling)
- **Motion Detection**: Farneback optical flow (downscaled to 240px for speed/memory)
- **Audio Transients**: scipy onset detection with adaptive median thresholding
- **VLM**: Claude API for keyframe rating (detect_vlm.py)
- **Video Decoding**: decord
- **Audio Processing**: ffmpeg + numpy
- **GPU**: CUDA via PyTorch (RTX 4060 8GB)
- **Environment**: conda (`video-highlights`)

## Scripts

### detect_highlights.py
Original highlight detector. CLIP (60%) + audio RMS energy (40%). Fast, good for gunfire.

### detect_violence.py
Full 4-signal violence detector:
- PANNs (35%): classifies audio events (gunshot, explosion, punch, scream, etc.)
- CLIP (25%): visual violence scoring via text prompts
- VideoMAE (25%): action recognition (fighting, kicking, wrestling labels)
- Optical flow (15%): fast motion detection

### condense.py
Takes highlight clips and creates a rapid-fire montage. Detects audio transients (gunshots),
extracts micro-clips (0.2-2s) around each one, merges overlapping ranges, joins into one video.

### detect_vlm.py
Sends sampled keyframes to Claude for semantic violence rating. Most accurate but requires
ANTHROPIC_API_KEY and costs ~$0.004 per batch. Has --dry-run mode to estimate cost.

## Key Design Decisions

- **Hybrid multi-signal scoring**: Each signal catches different violence types. Audio for gunshots, CLIP for visual, VideoMAE for temporal actions (fistfights), flow for fast motion.
- **Per-video normalization**: Each signal is normalized [0,1] within each video before combining, so a quiet video's best moments still surface.
- **Adaptive transient detection** (condense.py): Subtracts local median to ignore sustained noise, only detects sharp spikes. Merges overlapping time ranges to prevent duplicate/glitchy clips.
- **75th percentile threshold**: Only peaks above the 75th percentile of the smoothed hybrid score are considered candidates.

## Development

```bash
conda activate video-highlights
python detect_violence.py /path/to/videos -o ./output -n 20
python condense.py ./output -o ./output/condensed.mp4 --min-clip 0.5 --max-clip 2.0
```
