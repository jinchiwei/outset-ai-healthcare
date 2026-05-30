#!/usr/bin/env bash
# Build a finalized branded deck for one day: render -> recolor section dividers
# full-bleed brand color -> drop the Outset wordmark on the cover.
#
# Usage:  bash slides/build_deck.sh day1
# Requires: conda env `outset`, the build-pptx skill at ~/.claude/skills/build-pptx
set -euo pipefail

DAY="${1:?usage: build_deck.sh <day1|day2|day3>}"
REPO="$HOME/arcadia/teaching/outset-ai-healthcare"
SKILL="$HOME/.claude/skills/build-pptx"
MD="$REPO/slides/$DAY.md"
OUT="$REPO/slides/build/$DAY.pptx"
SIDE="$REPO/slides/$DAY.md.layout.json"

# 1. render (replays the handcrafted layout sidecar deterministically)
( cd "$SKILL" && conda run -n outset python build.py --input "$MD" --output "$OUT" )

# 2. recolor section dividers full-bleed with their cycling brand accent
conda run -n outset python "$REPO/slides/apply_divider_colors.py" "$OUT" "$SIDE"

# 3. Outset wordmark (white) on the dark cover
conda run -n outset python "$REPO/slides/apply_logo.py" "$OUT"

# 4. real-image cover hero (lower-right), per day
case "$DAY" in
  day1) HERO=fundus_dr.jpg ;;
  day2) HERO=cxr_normal.png ;;
  day3) HERO=skin_melanoma.jpg ;;
  *)    HERO="" ;;
esac
if [ -n "$HERO" ]; then
  conda run -n outset python "$REPO/slides/apply_cover_hero.py" "$OUT" "$HERO" 40E0D0
fi

echo "built $OUT"
