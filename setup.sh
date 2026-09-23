#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Local Deep Research Engine + Research Assistant bootstrap
#
# This script is intended for a fresh Ubuntu machine.
# It installs:
#   - required Ubuntu packages
#   - Ollama (if missing)
#   - the local Qwen/QwQ models used by both assistants
#   - the Python virtual environment and requirements
#
# It is safe to rerun: existing packages, Ollama, models and
# .venv are detected and reused.
# ============================================================

echo "============================================================"
echo " Local Research AI setup"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Required Ubuntu packages
# ------------------------------------------------------------
if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: apt-get was not found. This setup script currently"
  echo "       targets Debian/Ubuntu Linux systems."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "ERROR: sudo was not found."
  echo "       Install the required Ubuntu packages manually or"
  echo "       rerun this script on a system with sudo available."
  exit 1
fi

echo "[1/6] Checking required Ubuntu packages..."

APT_PACKAGES=(
  python3
  python3-venv
  curl
  tesseract-ocr
  tesseract-ocr-eng
)

MISSING_PACKAGES=()

for pkg in "${APT_PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
    MISSING_PACKAGES+=("$pkg")
  fi
done

if [ "${#MISSING_PACKAGES[@]}" -gt 0 ]; then
  echo "Installing missing Ubuntu packages:"
  printf '  - %s\n' "${MISSING_PACKAGES[@]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING_PACKAGES[@]}"
else
  echo "Ubuntu prerequisites already installed."
fi

# ------------------------------------------------------------
# 2. Ollama
# ------------------------------------------------------------
echo
echo "[2/6] Checking Ollama..."

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama was not found. Installing from the official installer..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "Ollama already installed:"
  ollama --version || true
fi

# Start Ollama if it is not already reachable.
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama service..."

  if command -v systemctl >/dev/null 2>&1 \
      && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
    sudo systemctl enable --now ollama
  else
    nohup ollama serve >"${SCRIPT_DIR}/ollama.log" 2>&1 &
  fi
fi

echo "Waiting for Ollama API..."

OLLAMA_READY=false
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    OLLAMA_READY=true
    break
  fi
  sleep 1
done

if [ "$OLLAMA_READY" != "true" ]; then
  echo "ERROR: Ollama did not become available at http://127.0.0.1:11434"
  echo "       Check the Ollama service with:"
  echo "       systemctl status ollama"
  echo "       or inspect: ${SCRIPT_DIR}/ollama.log"
  exit 1
fi

echo "Ollama API is ready."

# ------------------------------------------------------------
# 3. Pull the models used by both assistants
# ------------------------------------------------------------
echo
echo "[3/6] Checking required Ollama models..."

REQUIRED_MODELS=(
  "qwq:32b"
  "qwen3.5:35b-a3b"
  "qwen3:32b"
  "qwen3.8:27b"
)

for model in "${REQUIRED_MODELS[@]}"; do
  echo
  echo "Ensuring model is available: ${model}"
  if ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$model"; then
    echo "Already installed: ${model}"
  else
    echo "Pulling: ${model}"
    ollama pull "$model"
  fi
done

# ------------------------------------------------------------
# 4. Python virtual environment
# ------------------------------------------------------------
echo
echo "[4/6] Setting up Python virtual environment..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found after Ubuntu package installation."
  exit 1
fi

PYTHON_BIN="python3"

if [ ! -d ".venv" ]; then
  echo "Creating .venv..."
  "$PYTHON_BIN" -m venv .venv
else
  echo ".venv already exists."
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# ------------------------------------------------------------
# 5. Validate the Python dependencies and create directories
# ------------------------------------------------------------
echo
echo "[5/6] Validating Python dependencies..."

python - <<'PY'
import importlib

required = [
    "yaml",
    "requests",
    "ddgs",
    "trafilatura",
    "bs4",
    "rapidfuzz",
    "dotenv",
    "pydantic",
    "reportlab",
    "pymupdf",
    "pypdf",
    "pytesseract",
    "PIL",
    "langchain",
    "langgraph",
    "langchain_ollama",
]

missing = []

for module in required:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

if missing:
    print("ERROR: Python dependency validation failed:")
    for item in missing:
        print(f"  - {item}")
    raise SystemExit(1)

print("Python dependencies OK.")
PY

mkdir -p \
  documents/ROV_Control \
  documents/AEROSUB \
  documents/Thesis \
  documents/General \
  images \
  runs \
  followup_runs \
  quick_research_runs \
  conversation_sessions \
  rag_index

# Do not automatically create/copy real .env files.
# API keys and machine-specific settings should not be committed to GitHub.
# If example files exist, tell the user where to copy them.
if [ -f ".env.deep_research_engine.example" ] || [ -f ".env.research_assistant.example" ]; then
  echo
  echo "Configuration templates detected."
  echo "Copy/edit them before running the assistants:"
  [ -f ".env.deep_research_engine.example" ] && \
    echo "  cp .env.deep_research_engine.example .env.deep_research_engine"
  [ -f ".env.research_assistant.example" ] && \
    echo "  cp .env.research_assistant.example .env.research_assistant"
fi

# ------------------------------------------------------------
# 6. Engine smoke test
# ------------------------------------------------------------
echo
echo "[6/6] Running Deep Research Engine smoke test..."
python deep_research_engine.py --smoke-test

echo
echo "============================================================"
echo " Setup completed successfully"
echo "============================================================"
echo
echo "Installed/verified:"
echo "  - Ubuntu Python/OCR prerequisites"
echo "  - Ollama"
echo "  - qwq:32b"
echo "  - qwen3.5:35b-a3b"
echo "  - qwen3:32b"
echo "  - qwen3.8:27b"
echo "  - Python .venv + requirements.txt"
echo
echo "Next steps:"
echo "  1. Configure .env.deep_research_engine"
echo "  2. Configure .env.research_assistant"
echo "  3. Run: ./run_research.sh --help"
echo "  4. Run: ./run_research_assistant.sh"
echo

