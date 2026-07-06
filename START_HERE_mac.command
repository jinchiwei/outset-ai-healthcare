#!/bin/bash
# ============================================================================
#  AI in Healthcare -- one-click local setup for macOS
#  Double-click this file in Finder. It installs everything and opens the
#  notebooks in your browser. First run takes a few minutes (downloading
#  Python + libraries); after that it's instant.
#
#  If macOS blocks it ("unidentified developer"), right-click the file ->
#  Open -> Open. You only have to do that once.
# ============================================================================
set -e
cd "$(dirname "$0")"

echo ""
echo "  AI in Healthcare -- setting up (first run: a few minutes)"
echo "  ---------------------------------------------------------"

# 1. uv is a tiny tool that installs Python and all the libraries for us.
if ! command -v uv >/dev/null 2>&1; then
  echo "  Installing the setup helper (uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 2. Create an isolated Python and install the course libraries.
echo "  Preparing Python 3.11..."
uv venv --python 3.11 .venv
echo "  Installing course libraries (torch, etc.)..."
uv pip install --python .venv -r requirements-student.txt

# 3. Launch JupyterLab pointed at the notebooks.
echo ""
echo "  Done. Opening the notebooks in your browser..."
echo "  (Leave this window open while you work. Close it to stop.)"
echo ""
uv run --python .venv jupyter lab notebooks
