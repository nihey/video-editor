#!/usr/bin/env bash
# Render a violence highlights video using Remotion.
# Usage: ./render.sh <manifest.json> [output.mp4]
#
# The manifest JSON is produced by condense_violence.py and contains
# the source video path + segment definitions.

set -euo pipefail

MANIFEST="${1:?Usage: ./render.sh <manifest.json> [output.mp4]}"
OUTPUT="${2:-$(dirname "$MANIFEST")/$(basename "$MANIFEST" .json)_rendered.mp4}"
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

# Symlink source video into renderer/public/ so Remotion can serve it
mkdir -p "$RENDERER_DIR/public"
ln -sf "$SOURCE" "$RENDERER_DIR/public/source.mp4"

# Build props JSON: replace source with staticFile reference
PROPS=$(python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
m['source'] = 'source.mp4'
m['showLabels'] = True
m['fps'] = 30
# staticFile() is called in the component, we just pass the filename
json.dump(m, sys.stdout)
")

echo "Rendering with Remotion..."
echo "  Source: $SOURCE"
echo "  Segments: $(python3 -c "import json; print(len(json.load(open('$MANIFEST'))['segments']))")"
echo "  Output: $OUTPUT"

cd "$RENDERER_DIR"

# Install deps if needed
if [ ! -d "node_modules" ]; then
  npm install
fi

# Render
npx remotion render src/index.ts ViolenceHighlights "$OUTPUT" \
  --props="$PROPS" \
  --codec=h264 \
  --crf=21

# Cleanup symlink
rm -f "$RENDERER_DIR/public/source.mp4"

echo ""
echo "Done: $OUTPUT"
