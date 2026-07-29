# Compute Canada / DRAC SLURM Environment

This folder contains helper scripts to run the AUGMENT pipeline on the Digital Research Alliance of Canada (DRAC) clusters using SLURM. 

## 1. Setup

**Step 1: Build the ViennaRNA Wheel**

DRAC compute nodes have no internet access. First, jump on a login node and build the ViennaRNA wheel. The script below fetches the source and drops the built wheel into `hpc/wheelhouse/`.
```bash
bash hpc/build_viennarna_wheel.sh
```

**Step 2: Initialize the Environment**

Once the wheel is ready, the `setup_env.sh` script automates loading the required modules, creating the virtual environment (with `--no-download`), and installing the requirements offline using `--no-index`.

```bash
bash hpc/setup_env.sh
```

**Step 3: Interactive Usage (Login Nodes)**

If you are running scripts interactively on a login node (instead of submitting a SLURM job), you must manually load the same modules and activate the virtual environment so that dependencies like Pandas are correctly resolved from `scipy-stack`:

```bash
module load StdEnv/2023 python/3.12 scipy-stack
source .venv/bin/activate
```

## 2. Running Jobs

Instead of managing a separate `.sbatch` script for every experiment, we use generic wrapper scripts (`submit_gpu.sbatch` and `submit_cpu.sbatch`). These wrappers load the virtual environment and seamlessly pass all command line arguments to Python. 

This means you can simply prefix any command from the main `README.md` with `sbatch hpc/submit_gpu.sbatch`.

**Example:**
```bash
sbatch hpc/submit_gpu.sbatch scripts/train_cnn.py \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --train_set results/GSM3130435_egfp_unmod_1/read_count_standard/train.csv.gz \
    --valid_set results/GSM3130435_egfp_unmod_1/read_count_standard/valid.csv.gz \    
    --test_set results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/cnn/default_seed42
```

## 3. Overriding SLURM Defaults

If a specific job needs more time, memory, or different GPUs, you can override the defaults directly on the command line:

```bash
sbatch --time=08:00:00 --mem=32G hpc/submit_gpu.sbatch scripts/train_cnn.py ...
```
