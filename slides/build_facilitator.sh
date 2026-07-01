#!/usr/bin/env bash
# Build the facilitator handout deck. Unlike the student day decks (expressive,
# handcrafted-freeform), this one uses build-pptx STRICT mode: rules-based named
# layouts, so it needs no per-slide layout file. Then drop the Outset wordmark.
#
# Usage:  bash slides/build_facilitator.sh
set -euo pipefail

REPO="$HOME/arcadia/teaching/outset-ai-healthcare"
SKILL="$HOME/.claude/skills/build-pptx"
MD="$REPO/slides/facilitator.md"
OUT="$REPO/slides/build/facilitator.pptx"

( cd "$SKILL" && conda run -n outset python build.py --input "$MD" --output "$OUT" --mode strict )
conda run -n outset python "$REPO/slides/apply_logo.py" "$OUT"

echo "built $OUT"
