#!/bin/bash
#SBATCH --job-name=augment_gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=results/%x.%j.out
#SBATCH --error=results/%x.%j.err

set -euo pipefail

# Ensure we have arguments
if [ "$#" -eq 0 ]; then
    echo "Usage: sbatch $0 <python_script> [args...]"
    exit 1
fi

# SLURM copies the script to a temporary spool directory during execution.
# Therefore, BASH_SOURCE[0] is unreliable. We must use SLURM_SUBMIT_DIR,
# which points to the directory where the user ran the sbatch command (project root).
source "${SLURM_SUBMIT_DIR}/hpc/common.sh"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting GPU job..."
srun python "$@"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job complete."
