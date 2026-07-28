#!/bin/bash
#SBATCH --job-name=augment_cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting CPU job..."
# We use 'env' to force PYTHONPATH explicitly on the compute node. 
# This prevents srun from stripping environment variables, which is common on HPC clusters.
srun env PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$@"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job complete."
