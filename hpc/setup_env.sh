#!/bin/bash
set -euo pipefail

# Resolving //home/marcelt/project to /project/6006499 right away

set -o physical

# This script sets up the AUGMENT environment on DRAC.
# It strictly follows DRAC guidelines for offline installation using --no-index.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
MODULE_STDENV="StdEnv/2023"
MODULE_PYTHON="python/3.12"
WHEELHOUSE_DIR="$PROJECT_DIR/hpc/wheelhouse"

echo "Loading modules..."
module load "$MODULE_STDENV" "$MODULE_PYTHON"

# DRAC strongly recommends scipy-stack for optimized numpy/scipy
module load scipy-stack || echo "scipy-stack module not available, continuing..."

echo "Checking for required wheels..."
if ! ls "$WHEELHOUSE_DIR"/[Vv]ienna[Rr][Nn][Aa]-*.whl 1> /dev/null 2>&1; then
    echo "ERROR: ViennaRNA wheel not found in $WHEELHOUSE_DIR."
    echo "DRAC compute nodes do not have internet access to download source packages."
    echo "Please run 'bash hpc/build_viennarna_wheel.sh' on a login node first."
    exit 1
fi

echo "Creating virtual environment at $VENV_DIR..."
# Must use --no-download on compute nodes
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Update core packaging tools offline (adding pip-tools)
pip install --no-index -U pip setuptools wheel build pip-tools

echo "Setting up Python environment..."

# Ensure the HPC SLURM log directory exists before any jobs are submitted
mkdir -p results/logs

echo "Generating requirements-drac.txt for cluster compatibility..."
# Compile requirements-drac.txt strictly against DRAC's CVMFS wheelhouse
pip-compile --no-index --find-links="$WHEELHOUSE_DIR" "$PROJECT_DIR/requirements.in" -o "$PROJECT_DIR/requirements-drac.txt"

echo "Installing project dependencies..."
# On DRAC compute nodes, use --no-index and provide wheels in a wheelhouse.
# If a package like ViennaRNA needs to be built, it should be built on a login node
# and placed in $WHEELHOUSE_DIR beforehand.
pip install --no-index --find-links="$WHEELHOUSE_DIR" -r "$PROJECT_DIR/requirements-drac.txt"

echo "Installing local package in editable mode..."
# Allows scripts to import `augmenter` directly without modifying PYTHONPATH
cd "$PROJECT_DIR"
pip install --no-index -e . --no-deps

echo "Sanity checking installation..."
python -c "import src, src.models; print('Package path:', src.__file__)"

echo "Environment setup complete!"
