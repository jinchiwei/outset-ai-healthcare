#!/usr/bin/env bash
# Build the facilitator handout deck. Expressive mode so the slide canvas is the
# same warm "bone" as the student decks (strict mode renders on white, which
# clashes with the bone-canvas figures). --allow-composed skips per-slide
# handcrafting; the committed sidecar pins theme=bone, which expressive preserves.
# Then drop the Outset wordmark.
#
# Usage:  bash slides/build_facilitator.sh
set -euo pipefail

REPO="$HOME/arcadia/teaching/outset-ai-healthcare"
SKILL="$HOME/.claude/skills/build-pptx"
MD="$REPO/slides/facilitator.md"
OUT="$REPO/slides/build/facilitator.pptx"

( cd "$SKILL" && conda run -n outset python build.py --input "$MD" --output "$OUT" \
    --mode expressive --allow-composed )
conda run -n outset python "$REPO/slides/apply_logo.py" "$OUT"

echo "built $OUT"
