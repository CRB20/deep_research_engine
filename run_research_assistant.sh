#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the shared virtual environment
source .venv/bin/activate

# Create the personal document folders if they do not exist
mkdir -p documents/ROV_Control
mkdir -p documents/AEROSUB
mkdir -p documents/Thesis
mkdir -p documents/General

# Create the image folder if it does not exist
mkdir -p images

# Run the Research Assistant
python research_assistant.py "$@"
