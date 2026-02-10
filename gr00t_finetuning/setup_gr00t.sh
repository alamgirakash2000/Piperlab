#!/bin/bash
# ============================================================
# GR00T N1.6 Environment Setup Script
# ============================================================
# Run this once to set up the gr00t fine-tuning environment.
#
# Usage:
#   ./setup_gr00t.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_DIR="${SCRIPT_DIR}/Isaac-GR00T"

echo "============================================================"
echo "Setting up GR00T N1.6 Environment"
echo "============================================================"

# Check if repo exists
if [ ! -d "${GROOT_DIR}" ]; then
    echo "Cloning Isaac-GR00T repository..."
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "${GROOT_DIR}"
else
    echo "Isaac-GR00T already exists, updating..."
    cd "${GROOT_DIR}"
    git pull
    git submodule update --init --recursive
fi

cd "${GROOT_DIR}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install uv
fi

echo ""
echo "Setting up Python environment with uv..."
echo "(This may take several minutes on first run)"
echo ""

# Create environment and install dependencies
uv sync --python 3.10

# Install gr00t package
uv pip install -e .

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Verify by running:"
echo "  cd Isaac-GR00T && uv run python -c 'import gr00t; print(gr00t.__version__)'"
echo ""
echo "Then start training with:"
echo "  ./train_gr00t.sh"
echo "============================================================"
