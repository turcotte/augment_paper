#!/bin/bash
set -euo pipefail

# This script downloads ViennaRNA and builds it into a wheel 
# on the login node (where internet is available) so it can be 
# installed offline on compute nodes.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE_DIR="$PROJECT_DIR/hpc/wheelhouse"
BUILD_ENV_DIR="$PROJECT_DIR/hpc/buildenv_viennarna"

MODULE_STDENV="StdEnv/2023"
MODULE_PYTHON="python/3.12.4"

echo "Loading modules..."
module load "$MODULE_STDENV" "$MODULE_PYTHON"

echo "Creating temporary build environment..."
if [[ -d "$BUILD_ENV_DIR" ]]; then
    rm -rf "$BUILD_ENV_DIR"
fi

virtualenv --no-download "$BUILD_ENV_DIR"
source "$BUILD_ENV_DIR/bin/activate"

echo "Upgrading build tools..."
pip install -U pip setuptools wheel build

echo "Building ViennaRNA wheel from PyPI..."
mkdir -p "$WHEELHOUSE_DIR"
# The PyPI version of ViennaRNA contains the proper setup.py wrapper for compilation.
pip wheel --no-build-isolation -w "$WHEELHOUSE_DIR" ViennaRNA==2.7.2
# The PyPI version of ViennaRNA contains the proper setup.py wrapper for compilation.
pip wheel --no-build-isolation -w "$WHEELHOUSE_DIR" ViennaRNA==2.7.2

deactivate
rm -rf "$BUILD_ENV_DIR"

echo "Wheel build complete. It is now available in $WHEELHOUSE_DIR"
