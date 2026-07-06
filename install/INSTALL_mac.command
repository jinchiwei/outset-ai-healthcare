#!/bin/bash
# ============================================================================
#  AI in Healthcare -- complete one-file installer for macOS
#
#  You do NOT need Python, conda, git, or Jupyter first. This downloads the
#  course, installs its own private Python + all libraries, and opens the
#  notebooks in your browser. First run takes a few minutes.
#
#  HOW TO USE: double-click this file. If macOS blocks it ("unidentified
#  developer"), right-click it -> Open -> Open. You only do that once.
# ============================================================================
set -e

DEST="$HOME/Desktop/outset-ai-healthcare"

echo ""
echo "  AI in Healthcare -- installing everything (first run: a few minutes)"
echo "  -------------------------------------------------------------------"

# 1. Download the course (as a zip -- no git needed) to the Desktop.
echo "  Downloading the course..."
curl -L -o "$TMPDIR/outset.zip" \
  https://github.com/jinchiwei/outset-ai-healthcare/archive/refs/heads/main.zip
rm -rf "$DEST" "$TMPDIR/outset-ai-healthcare-main"
unzip -q "$TMPDIR/outset.zip" -d "$TMPDIR"
mv "$TMPDIR/outset-ai-healthcare-main" "$DEST"
cd "$DEST"

# 2. Install uv (a tiny tool that supplies Python + installs libraries).
if ! command -v uv >/dev/null 2>&1; then
  echo "  Installing the setup helper (uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 3. Private Python + course libraries.
echo "  Preparing Python 3.11 and installing libraries (torch, etc.)..."
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-student.txt

echo ""
echo "  Done! The course is on your Desktop in 'outset-ai-healthcare'."
echo "  Opening the notebooks in your browser now..."
echo "  Next time, just double-click START_HERE_mac.command inside that folder."
echo ""
uv run --python .venv jupyter lab notebooks
