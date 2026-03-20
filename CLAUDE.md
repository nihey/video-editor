# Video Editor

## Project Overview

Gameplay highlight detector that uses ML (CLIP) + audio analysis to find the best moments in video collections. Currently tuned for RDR2 PS5 gameplay clips but designed to work with any game.

## Architecture

Single-script Python project:

```
video-editor/
├── detect_highlights.py   # Main detection script (hybrid CLIP + audio)
├── highlights/            # Output directory (gitignored)
│   ├── highlight_XX.mp4   # Extracted clips
│   └── highlights.json    # Metadata with scores
├── .gitignore
├── CLAUDE.md
└── README.md
```

## Tech Stack

- **ML Model**: OpenAI CLIP ViT-L/14 via HuggingFace Transformers
- **Video Decoding**: decord (memory-efficient batched frame loading)
- **Audio Processing**: ffmpeg (WAV extraction) + numpy (RMS energy)
- **Signal Processing**: scipy (peak detection, smoothing)
- **GPU**: CUDA via PyTorch
- **Environment**: conda (`video-highlights`)

## Key Design Decisions

- **CLIP over VideoMAE**: CLIP works better on game screenshots than action recognition models trained on real-world video. It also allows customizable text prompts.
- **Hybrid scoring**: Audio catches intensity (gunshots, explosions) that CLIP might miss in dark scenes. CLIP catches visual action that might be quiet.
- **Per-video normalization**: Each signal is normalized [0,1] within each video before combining, so a quiet video's best moments still surface.
- **Memory-efficient frame loading**: Frames are loaded in batches of 64 via decord to avoid OOM on long videos (15+ min).
- **75th percentile threshold**: Only peaks above the 75th percentile of the smoothed hybrid score are considered candidates, reducing noise.

## Development

```bash
conda activate video-highlights
python detect_highlights.py /path/to/videos -n 10
```

## Extending

- Add new detection criteria by modifying `POSITIVE_PROMPTS` / `NEGATIVE_PROMPTS`
- Adjust `AUDIO_WEIGHT` / `CLIP_WEIGHT` to rebalance the hybrid score
- The `process_video()` function returns scored segments per video - can be used as a library
