# Video Editor

## Project Overview

Gameplay highlight and violence detector that uses multiple ML models + audio analysis to find the best moments in video collections. Currently tuned for RDR2 PS5 gameplay clips but designed to work with any game. Includes a Remotion-based renderer for cinematic output with Ken Burns zoom, transitions, SFX, and AI voiceover.

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
├── generate_voiceover.py    # AI voiceover generation using Qwen3-TTS
├── render.sh                # Bridge: manifest JSON → Remotion render
├── renderer/                # Remotion project for cinematic video assembly
│   ├── src/
│   │   ├── index.ts               # Entry point
│   │   ├── Root.tsx               # Composition registration
│   │   ├── ViolenceHighlights.tsx # Main composition (TransitionSeries)
│   │   ├── KenBurns.tsx           # Score-driven zoom/pan effect
│   │   ├── SegmentLabel.tsx       # Label overlay with fade-in
│   │   ├── SoundEffects.tsx       # Score-triggered SFX (whoosh, impact, whip)
│   │   ├── Voiceover.tsx          # AI voiceover audio layer
│   │   └── types.ts              # Shared types
│   └── public/                    # Symlinked source videos + assets
├── highlights/              # Output directory (gitignored)
├── highlights_violence/     # Violence detection output (gitignored)
└── CLAUDE.md
```

## Tech Stack

- **Audio Classification**: PANNs CNN14 (AudioSet 527 classes — gunshot, explosion, punch, scream)
- **Visual Scoring**: OpenAI CLIP ViT-L/14 (text-guided frame scoring)
- **Action Recognition**: VideoMAE (Kinetics-400 — punching, kicking, sword fighting, wrestling)
- **Motion Detection**: Farneback optical flow (downscaled to 240px for speed/memory)
- **Audio Transients**: scipy onset detection with adaptive median thresholding
- **VLM**: Claude API for keyframe rating (detect_vlm.py)
- **TTS**: Qwen3-TTS 0.6B for AI voiceover narration
- **Video Rendering**: Remotion (React) with TransitionSeries, Ken Burns, SFX
- **Video Decoding**: decord
- **Audio Processing**: ffmpeg + numpy
- **GPU**: CUDA via PyTorch (RTX 4060 8GB)
- **Environment**: conda (`video-highlights`) for Python, npm for Remotion

## Pipeline

### Full workflow

```bash
# 1. Remove menu UI from source video
python remove_menu.py input.mp4 -o clean.mp4

# 2. Detect and condense best violent moments (produces manifest JSON)
python condense_violence.py clean.mp4 -o highlights/best_moments.mp4

# 3. (Optional) Generate AI voiceover
python generate_voiceover.py highlights/best_moments.json

# 4. Render with Remotion (Ken Burns, transitions, SFX, voiceover)
./render.sh highlights/best_moments.json output.mp4 --voiceover highlights/voiceover/

# Or for quick edits without Remotion:
python condense_violence.py reassemble highlights/best_moments.json -o edited.mp4
```

### Agentic editing

The manifest JSON contains labeled segments (A, B, C...). Edit it to:
- Remove segments: delete from array
- Trim segments: adjust start/end times
- Extend segments: widen time range (check for overlaps)
- Then re-render with `reassemble` or `render.sh`

## Scripts

### condense_violence.py
Best-moments condensation with labeled segments:
- CLIP (35%) + PANNs (40%) + optical flow (25%) scoring
- Score-percentile thresholding with nearby-segment merging
- Produces manifest JSON + labeled video for agentic editing
- `reassemble` subcommand for quick re-editing from manifest

### remove_menu.py
Detects RDR2 radial wheel menu by HSV saturation (menu desaturates screen).
Cuts menu segments and joins remaining gameplay.

### generate_voiceover.py
Generates dramatic narration for each segment using Qwen3-TTS.
Score-based narration templates (high=intense callouts, mid=tension, low=scene-setting).

### render.sh
Bridges Python manifests to Remotion rendering. Symlinks source video,
builds props JSON, invokes `npx remotion render`.

## Remotion Components

- **ViolenceHighlights**: Main composition using TransitionSeries with fade transitions
- **KenBurns**: Score-driven zoom/pan (spring punch-in for high scores, slow creep for medium)
- **SegmentLabel**: Label overlay (A, B, C...) with fade-in animation
- **SegmentSFX**: Auto-triggered sound effects based on violence score
- **Voiceover**: AI voiceover audio layer with delay and fade-in

## Key Design Decisions

- **Hybrid multi-signal scoring**: Each signal catches different violence types
- **Per-video normalization**: Each signal normalized [0,1] within each video
- **Manifest-driven editing**: JSON manifest is the single source of truth for edits
- **Remotion for rendering**: React components give flexibility for effects, transitions, overlays
- **Score-driven effects**: Ken Burns zoom, SFX volume, narration intensity all scale with violence score

## Development

```bash
conda activate video-highlights
python detect_violence.py /path/to/videos -o ./output -n 20
python condense_violence.py ./output/violence_compilation.mp4 -o ./output/best_moments.mp4
cd renderer && npm run studio  # Preview in browser
```
