#!/usr/bin/env bash
# Render a violence highlights video using Remotion.
# Usage: ./render.sh <manifest.json> [output.mp4] [--no-labels] [--voiceover <dir>]
#
# The manifest JSON is produced by condense_violence.py and contains
# the source video path + segment definitions.

set -euo pipefail

MANIFEST=""
OUTPUT=""
SHOW_LABELS=true
VOICEOVER_DIR=""
VERTICAL=false
COMPOSITION="ViolenceHighlights"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-labels) SHOW_LABELS=false; shift ;;
    --vertical) VERTICAL=true; COMPOSITION="VerticalHighlights"; shift ;;
    --voiceover) VOICEOVER_DIR="$2"; shift 2 ;;
    *)
      if [ -z "$MANIFEST" ]; then MANIFEST="$1"
      elif [ -z "$OUTPUT" ]; then OUTPUT="$1"
      fi
      shift ;;
  esac
done

if [ -z "$MANIFEST" ]; then
  echo "Usage: ./render.sh <manifest.json> [output.mp4] [--no-labels] [--vertical] [--voiceover <dir>]"
  exit 1
fi

OUTPUT="${OUTPUT:-$(dirname "$MANIFEST")/$(basename "$MANIFEST" .json)_rendered.mp4}"
RENDERER_DIR="$(cd "$(dirname "$0")/renderer" && pwd)"

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: Manifest not found: $MANIFEST"
  exit 1
fi

# Read source path from manifest and resolve to absolute
SOURCE=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['source'])")
if [ ! -f "$SOURCE" ]; then
  echo "ERROR: Source video not found: $SOURCE"
  exit 1
fi

# Copy source video into renderer/public/ so Remotion can bundle it
# (symlinks don't survive Remotion's webpack bundling to temp dir)
mkdir -p "$RENDERER_DIR/public"
cp -f "$SOURCE" "$RENDERER_DIR/public/source.mp4"

# Copy voiceover files if provided
if [ -n "$VOICEOVER_DIR" ] && [ -d "$VOICEOVER_DIR" ]; then
  mkdir -p "$RENDERER_DIR/public/voiceover"
  for wav in "$VOICEOVER_DIR"/*.wav; do
    [ -f "$wav" ] && cp -f "$wav" "$RENDERER_DIR/public/voiceover/$(basename "$wav")"
  done
fi

# Build props JSON
PROPS=$(python3 -c "
import json, sys, os

m = json.load(open('$MANIFEST'))
m['source'] = 'source.mp4'
m['showLabels'] = $( [ "$SHOW_LABELS" = "true" ] && echo "True" || echo "False" )
m['fps'] = 30

# Add voiceover entries if voiceover manifest exists
vo_dir = '$VOICEOVER_DIR'
if vo_dir:
    vo_manifest = os.path.join(vo_dir, 'voiceover.json')
    if os.path.exists(vo_manifest):
        vo = json.load(open(vo_manifest))
        m['voiceovers'] = [
            {'label': label, 'file': f'voiceover/{filename}'}
            for label, filename in vo.get('files', {}).items()
        ]

json.dump(m, sys.stdout)
")

echo "Rendering with Remotion..."
echo "  Source: $SOURCE"
echo "  Composition: $COMPOSITION"
echo "  Segments: $(python3 -c "import json; print(len(json.load(open('$MANIFEST'))['segments']))")"
echo "  Labels: $SHOW_LABELS"
echo "  Vertical: $VERTICAL"
echo "  Voiceover: ${VOICEOVER_DIR:-none}"
echo "  Output: $OUTPUT"

cd "$RENDERER_DIR"

# Install deps if needed
if [ ! -d "node_modules" ]; then
  npm install
fi

# Render
npx remotion render src/index.ts "$COMPOSITION" "$OUTPUT" \
  --props="$PROPS" \
  --codec=h264 \
  --crf=21

# Cleanup copied files
rm -f "$RENDERER_DIR/public/source.mp4"
rm -rf "$RENDERER_DIR/public/voiceover"

echo ""
echo "Done: $OUTPUT"
