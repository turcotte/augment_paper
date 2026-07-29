#!/bin/bash
# Script to setup a pure pip environment on DRAC (Compute Canada)
# This bypasses scipy-stack and custom DRAC PyTorch modules to ensure
# we get the official standard PyPI wheels and NVIDIA cuDNN binaries.

# 1. Purge all currently loaded modules to ensure a clean slate
module purge

# 2. Load only the bare minimum required (Python and CUDA)
# Adjust the python and cuda versions to match what is available on your cluster.
# PyTorch 2.3+ generally works well with CUDA 11.8 or 12.1
module load StdEnv/2023
module load python/3.10
module load cuda/11.8

# 3. Create a fresh virtual environment
VENV_DIR="venv_pure_pip"
echo "Creating pure pip virtual environment in $VENV_DIR..."
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install standard PyTorch with CUDA 11.8 support directly from PyPI
# This guarantees we get the official NVIDIA cuDNN binaries, not DRAC's custom build.
echo "Installing official PyTorch wheels..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. Install the rest of the project dependencies
# (We assume requirements.in contains pandas, numpy, torch_geometric, etc.)
echo "Installing project dependencies..."
pip install -r requirements.in

echo "Environment setup complete. To use it, run: source $VENV_DIR/bin/activate"
