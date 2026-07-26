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

# ViennaRNA 2.7.2 Source
VIENNARNA_URL="https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_7_x/ViennaRNA-2.7.2.tar.gz"
TARBALL_PATH="$PROJECT_DIR/hpc/ViennaRNA-2.7.2.tar.gz"

echo "Loading modules..."
module load "$MODULE_STDENV" "$MODULE_PYTHON"

echo "Downloading ViennaRNA source..."
if [[ ! -f "$TARBALL_PATH" ]]; then
    curl -o "$TARBALL_PATH" "$VIENNARNA_URL"
else
    echo "Source tarball already exists at $TARBALL_PATH"
fi

echo "Creating temporary build environment..."
if [[ -d "$BUILD_ENV_DIR" ]]; then
    rm -rf "$BUILD_ENV_DIR"
fi

virtualenv --no-download "$BUILD_ENV_DIR"
source "$BUILD_ENV_DIR/bin/activate"

echo "Upgrading build tools..."
pip install -U pip setuptools wheel build

echo "Building ViennaRNA wheel..."
mkdir -p "$WHEELHOUSE_DIR"
pip wheel --no-build-isolation -w "$WHEELHOUSE_DIR" "$TARBALL_PATH"

deactivate
rm -rf "$BUILD_ENV_DIR"

echo "Wheel build complete. It is now available in $WHEELHOUSE_DIR"
