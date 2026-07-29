#!/bin/bash

# Shared environment setup for DRAC SLURM scripts.
# Sourced by submit_gpu.sbatch and submit_cpu.sbatch

# Resolve project directory dynamically (one level up from this script)

export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Standard DRAC Python Module

export MODULE_STDENV="StdEnv/2023"
export MODULE_PYTHON="python/3.12.4"

# Load modules

module load "$MODULE_STDENV" "$MODULE_PYTHON"
module load scipy-stack || echo "scipy-stack module not available, continuing..."

# Activate the virtual environment

if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    echo "Warning: .venv not found in $PROJECT_DIR"
fi

export PYTHONUNBUFFERED=1

echo "================================================================================"
echo "SLURM Runtime Context"
echo "================================================================================"
echo "HOSTNAME: ${HOSTNAME:-unknown}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-unknown}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo
