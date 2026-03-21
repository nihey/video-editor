# Video Editor

## Project Overview

Gameplay highlight and violence detector that uses multiple ML models + audio analysis to find the best moments in video collections. Currently tuned for RDR2 PS5 gameplay clips but designed to work with any game. Includes a Remotion-based renderer for cinematic output with Ken Burns zoom, transitions, SFX, color grading, action overlays, and AI voiceover. Supports both horizontal (16:9) and vertical (9:16) output for YouTube Shorts, TikTok, and Instagram Reels.

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
├── detect_highlights.py     # CLIP + audio hybrid highlight detection
├── detect_violence.py       # 4-signal violence detection (PANNs + CLIP + VideoMAE + optical flow)
├── condense.py              # Rapid-fire montage from highlight clips (audio transient detection)
├── condense_violence.py     # Best-moments condensation with labeled segments + manifest
├── detect_vlm.py            # VLM-based detection using Claude API (most accurate, slowest)
├── remove_menu.py           # Remove RDR2 radial wheel menu segments (saturation detection)
├── detect_focal_points.py   # Optical flow based focal point detection for smart vertical crop
├── generate_voiceover.py    # AI voiceover generation using Qwen3-TTS
├── analyze_batch.py         # Batch analysis: all videos → report + annotated compilation
├── build_annotated.py       # Build annotated compilation from existing report.json
├── render.sh                # Bridge: manifest JSON → Remotion render (--vertical, --no-labels)
├── renderer/                # Remotion project for cinematic video assembly
│   ├── src/
│   │   ├── index.ts                 # Entry point
│   │   ├── Root.tsx                 # Composition registration (horizontal + vertical)
│   │   ├── ViolenceHighlights.tsx   # Horizontal 16:9 composition
│   │   ├── VerticalHighlights.tsx   # Vertical 9:16 composition (smart focal crop)
│   │   ├── KenBurns.tsx             # Score-driven zoom/pan (punch-in + pull-back)
│   │   ├── ColorGrade.tsx           # CSS contrast/brightness/saturation filter
│   │   ├── ActionOverlay.tsx        # "CHAOS!", "BRUTAL!" text + score indicator
│   │   ├── ProgressBar.tsx          # Thin red progress bar (retention signal)
│   │   ├── SegmentLabel.tsx         # Label overlay (A, B, C...) with fade-in
│   │   ├── SoundEffects.tsx         # Score-triggered SFX (whoosh, impact, whip)
│   │   ├── Voiceover.tsx            # AI voiceover audio layer
│   │   └── types.ts                 # Shared types (Segment, VoiceoverEntry, etc.)
│   └── public/                      # Copied source videos + assets at render time
├── highlights/              # Output directory (gitignored)
├── highlights_violence/     # Violence detection output (gitignored)
├── highlights_batch/        # Batch analysis output (gitignored)
└── CLAUDE.md
```

## Tech Stack

- **Audio Classification**: PANNs CNN14 (AudioSet 527 classes — gunshot, explosion, punch, scream)
- **Visual Scoring**: OpenAI CLIP ViT-L/14 (text-guided frame scoring with RDR2-tuned prompts)
- **Action Recognition**: VideoMAE (Kinetics-400 — punching, kicking, sword fighting, wrestling)
- **Motion Detection**: Farneback optical flow (downscaled to 240px for speed/memory)
- **Audio Transients**: scipy onset detection with adaptive median thresholding
- **VLM**: Claude API for keyframe rating (detect_vlm.py)
- **TTS**: Qwen3-TTS 0.6B for AI voiceover narration
- **Video Rendering**: Remotion 4 (React) with TransitionSeries, Ken Burns, SFX, color grading
- **Vertical Reframing**: Optical flow focal point detection + PyAutoFlip
- **Video Decoding**: decord + OpenCV
- **Audio Processing**: ffmpeg + numpy
- **GPU**: CUDA via PyTorch (RTX 4060 8GB)
- **Environment**: conda (`video-highlights`) for Python, npm for Remotion

## Pipeline

### Single video workflow

```bash
conda activate video-highlights

# 1. Remove menu UI from source video
python remove_menu.py input.mp4 -o clean.mp4

# 2. Detect and condense best violent moments (produces manifest JSON)
python condense_violence.py clean.mp4 -o highlights/best.mp4

# 3. Detect focal points for smart vertical crop
python detect_focal_points.py highlights/best.json

# 4. Render horizontal (16:9) with Remotion
./render.sh highlights/best.json output_h.mp4 --no-labels

# 5. Render vertical (9:16) for Shorts/TikTok/Reels
./render.sh highlights/best.json output_v.mp4 --vertical --no-labels

# Dense action cut (tight clips, high threshold)
python condense_violence.py clean.mp4 -o dense.mp4 \
  --min-clip 1.5 --max-clip 5 --pad-before 0.8 --pad-after 1.5 --percentile 40
```

### Batch analysis (multiple videos)

```bash
# Analyze all videos, generate report + annotated compilation
python analyze_batch.py /path/to/videos -o ./highlights_batch -n 3

# Re-build annotated video from existing report (no re-analysis)
python build_annotated.py highlights_batch/report.json -n 50
```

### Agentic editing

The manifest JSON contains labeled segments (A, B, C...). Edit commands:
- "Remove B, D" → delete segments from array
- "Trim D to 5 seconds" → adjust start/end
- "Extend A by 3s" → widen time range
- "Merge T, U, V" → combine into one segment
- Then: `python condense_violence.py reassemble manifest.json -o edited.mp4`
- Or: `./render.sh manifest.json output.mp4` for full Remotion render

### AI Voiceover (optional)

```bash
pip install qwen-tts
python generate_voiceover.py highlights/best.json
./render.sh highlights/best.json output.mp4 --voiceover highlights/voiceover/
```

## Scripts

### condense_violence.py
Best-moments condensation with labeled segments:
- CLIP (35%) + PANNs (40%) + optical flow (25%) scoring
- Score-percentile thresholding with nearby-segment merging
- CLI args: `--min-clip`, `--max-clip`, `--pad-before`, `--pad-after`, `--merge-gap`, `--percentile`
- `reassemble` subcommand for quick re-editing from manifest

### analyze_batch.py
Batch analysis of entire video directories:
- Scores every video with CLIP + PANNs + optical flow
- Generates `report.json` (full data) and `report.md` (human-readable)
- Builds annotated compilation with score overlays explaining WHY each clip was selected
- Report includes per-video violence rating, audio events detected, signal breakdowns

### build_annotated.py
Rebuilds annotated compilation from existing `report.json` without re-running ML models.
Uses ffmpeg textfile-based drawtext to overlay score annotations.

### detect_focal_points.py
Optical flow analysis per segment to find where motion is concentrated horizontally.
Outputs `focal_x` (0.0=left, 0.5=center, 1.0=right) per segment in the manifest.
Used by VerticalHighlights for smart cropping instead of static center crop.

### remove_menu.py
Detects RDR2 radial wheel menu by HSV saturation (menu desaturates screen).
Cuts menu segments and joins remaining gameplay.

### generate_voiceover.py
Generates dramatic narration using Qwen3-TTS (0.6B for 8GB GPU).
Score-based narration templates: high=intense callouts, mid=tension, low=scene-setting.

### render.sh
Bridges Python manifests to Remotion rendering:
- `--vertical`: render 1080x1920 for Shorts/TikTok/Reels
- `--no-labels`: hide segment labels
- `--voiceover <dir>`: include AI voiceover WAV files

## Remotion Components

- **ViolenceHighlights**: Horizontal 16:9 composition with TransitionSeries
- **VerticalHighlights**: Vertical 9:16 with focal-point-aware cropping
- **KenBurns**: Score-driven zoom — spring punch-in to 1.35x on high scores, slow creep on medium, pull-back after peak
- **ColorGrade**: CSS filter — contrast(1.15) brightness(1.05) saturate(1.1)
- **ActionOverlay**: Spring-animated callouts ("CHAOS!", "BRUTAL!") with glow effect on high-score segments
- **ScoreIndicator**: Small intensity percentage + colored bar in bottom-right
- **ProgressBar**: Thin red bar at bottom showing video progress
- **SegmentLabel**: Label overlay (A, B, C...) with fade-in animation
- **SegmentSFX**: Auto-triggered whoosh/whip/impact sounds based on violence score
- **Voiceover**: AI voiceover audio layer with delay and fade-in

## Telegram Delivery

Videos can be sent to the user via Telegram bot. Credentials are in `.env` (gitignored):

```bash
source .env
# Send video (<50MB):
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendVideo" \
  -F "chat_id=${TELEGRAM_USER_ID}" -F "video=@video.mp4" \
  -F "caption=Description" -F "supports_streaming=true"

# Compress for Telegram if >50MB:
ffmpeg -i input.mp4 -c:v libx264 -crf 28 -vf "scale=1280:720" -c:a aac -b:a 128k output_tg.mp4
```

## Development

```bash
conda activate video-highlights
cd renderer && npm run studio  # Preview compositions in browser
npx tsc --noEmit               # Type-check Remotion project
```
