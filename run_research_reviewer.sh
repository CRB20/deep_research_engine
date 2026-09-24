#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Keep reviewer configuration separate from the other applications.
# Create a local editable env file automatically on first run, without secrets.
if [[ ! -f ".env.research_reviewer" && -f ".env.research_reviewer.example" ]]; then
  cp ".env.research_reviewer.example" ".env.research_reviewer"
  echo "[SETUP] Created .env.research_reviewer from the example template."
fi

source .venv/bin/activate
mkdir -p "review_sessions"
mkdir -p "reviewer PDFs"
python research_reviewer.py "$@"
