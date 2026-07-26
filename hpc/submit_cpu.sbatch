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

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting CPU job..."
srun python "$@"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job complete."
