# AUGMENT

This repository contains the codebase for the manuscript *"AUGMENT: A Graph-Based Reinforcement Learning Framework for the Targeted Optimization of 5' UTR Translation Efficiency"* by Siavash Khalaj and Marcel Turcotte.

## Overview

This codebase provides models for predicting Mean Ribosome Load (MRL) from 5' UTR sequences, reconstructing sequences via a Graph Attention Autoencoder, and fine-tuning sequence generation using REINFORCE and Difference between Augmented and Posterior (DAP).

## Repository Structure

- `src/`: Core Python modules (models, data loaders, transforms).
- `scripts/`: Executable scripts for training, evaluation, and data downloading.
- `data/`: Datasets (ignored in version control).
- `results/`: Output logs, model checkpoints, partitions, and predictions.
- `figures/`: Generated plots for the manuscript.

### Results Hierarchy

We use a "Model-Centric" hierarchy to organize artifacts across dimensions: `Dataset` -> `Partition Strategy` -> `Model` -> `Variant/Fold/Seed`.

```text
results/
└── {dataset_name}/                  # 1. Base dataset (e.g., egfp_unmod_1)
    └── {partition_name}/            # 2. Specific data split (e.g., read_count_standard)
        ├── train.csv.gz             # <- Partitions live here!
        ├── valid.csv.gz
        ├── test.csv.gz
        ├── cnn/                     # 3. Algorithm class
        │   └── default_seed42/      # 4. Variant + Replicate
        │       ├── model.pth
        │       └── scaler.json      # <- Scalers live next to the model
        ├── gat/
        │   ├── baseline_fold1/
        │   ├── baseline_fold2/
        │   └── no_struct_seed42/
        ├── autoencoder/
        │   └── default_seed42/
        └── ga/                      # <- GA baseline
            └── default_seed42/
```

## Setup Instructions

### Prerequisites

This codebase was developed and tested using the following environment:

- **Python:** `3.12`
- **PyTorch:** `2.13.0`
- **PyTorch Geometric (PyG):** `2.8.0`
- **ViennaRNA:** (Required for RNA secondary structure calculations)

### Local Environment (macOS/Linux)

**Option 1: Using standard `pip`**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Option 2: Using `uv` (used during development)**

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

*(Note: installing the project in editable mode `-e .` ensures that the `augmenter` module in `src/` is accessible to all scripts without needing to modify your `PYTHONPATH`.)*

## Downloading Datasets

Run the following script to fetch the required GEO datasets:

```bash
python scripts/download_datasets.py
```

## Creating Partitions

Before training, generate immutable data partitions using `scripts/create_partitions.py`. For example, to generate the standard read-count partition for `GSM3130435_egfp_unmod_1`:

```bash
python scripts/create_partitions.py \
    --data_path data/GSM3130435_egfp_unmod_1.csv.gz \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard \
    --strategy read_count
```

This produces `train.csv.gz`, `valid.csv.gz`, and `test.csv.gz` containing UTR sequences.

## Merging Replicates

Merging HEK293T Random End 25-nt replicates.

```bash
python scripts/merge_replicates.py --r1_path data/GSE232927_processed_random_end_hek293t_N25_r1.csv.gz --r2_path data/GSE232927_processed_random_end_hek293t_N25_r2.csv.gz --out_path data/GSE232927_processed_random_end_hek293t_N25_merged.csv.gz
```

## Training a CNN Oracle

```bash
python scripts/train_cnn.py \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --train_set results/GSM3130435_egfp_unmod_1/read_count_standard/train.csv.gz \
    --valid_set results/GSM3130435_egfp_unmod_1/read_count_standard/valid.csv.gz \
    --test_set results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/cnn/default_seed42
```

## Training a GAT Model

```bash
python scripts/train_gat.py \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --train_set results/GSM3130435_egfp_unmod_1/read_count_standard/train.csv.gz \
    --valid_set results/GSM3130435_egfp_unmod_1/read_count_standard/valid.csv.gz \
    --test_set results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/gat/default_seed42
```

## Training an Autoencoder

```bash
python scripts/train_autoencoder.py \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --train_set results/GSM3130435_egfp_unmod_1/read_count_standard/train.csv.gz \
    --valid_set results/GSM3130435_egfp_unmod_1/read_count_standard/valid.csv.gz \
    --test_set results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/autoencoder/default_seed42
```

## Fine-Tuning with REINFORCE

To fine-tune a trained autoencoder using Policy Gradients (REINFORCE) to maximize Mean Ribosome Load (MRL), use the following script. This uses the GAT as the proxy reward and the CNN as the oracle evaluator:

```bash
python scripts/finetune_reinforce.py \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --train_set results/GSM3130435_egfp_unmod_1/read_count_standard/train.csv.gz \
    --test_set results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --autoencoder_dir results/GSM3130435_egfp_unmod_1/read_count_standard/autoencoder/default_seed42 \
    --cnn_dir results/GSM3130435_egfp_unmod_1/read_count_standard/cnn/default_seed42 \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/reinforce/default_seed42 \
    --epochs 3
```

## Fine-Tuning with DAP

To fine-tune using Difference between Augmented and Posterior (DAP), which adds a regularization term to keep sequences close to the original prior model, use:

```bash
python scripts/finetune_dap.py \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --train_set results/GSM3130435_egfp_unmod_1/read_count_standard/train.csv.gz \
    --test_set results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --autoencoder_dir results/GSM3130435_egfp_unmod_1/read_count_standard/autoencoder/default_seed42 \
    --cnn_dir results/GSM3130435_egfp_unmod_1/read_count_standard/cnn/default_seed42 \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/dap/default_seed42 \
    --epochs 3 \
    --sigma 60.0
```

## Fine-Tuning with Curriculum DAP

To fine-tune using Curriculum DAP, which breaks the dataset into sequential bins and steps the optimization through increasingly difficult binary reward thresholds, use:

```bash
python scripts/finetune_curriculum_dap.py \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --train_set results/GSM3130435_egfp_unmod_1/read_count_standard/train.csv.gz \
    --test_set results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --autoencoder_dir results/GSM3130435_egfp_unmod_1/read_count_standard/autoencoder/default_seed42 \
    --cnn_dir results/GSM3130435_egfp_unmod_1/read_count_standard/cnn/default_seed42 \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/curriculum_dap/default_seed42 \
    --epochs 3 \
    --sigma 60.0 \
    --num_bins 10
```

*(Variable length inputs are supported via the `--seq_len` argument.)*

## Running the Genetic Algorithm

To optimize a specific sequence from a dataset, use the genetic algorithm script:

```bash
python scripts/run_ga.py \
    --model_dir results/GSM3130435_egfp_unmod_1/read_count_standard \
    --predictor gat \
    --proxy_predictor cnn \
    --data_path data/GSM3130435_egfp_unmod_1.csv.gz \
    --target_index 0 \
    --generations 100
```

This will automatically generate uniquely evolved sequences in `results/GSM3130435_egfp_unmod_1/read_count_standard/ga_optimization/target_0/`.

## Plotting Curves

To visualize the training loss curves or the genetic algorithm fitness history, use the `plot_curves.py` utility:

**Autoencoder Training Curves:**

```bash
python scripts/plot_curves.py \
    --input results/GSM3130435_egfp_unmod_1/read_count_standard/autoencoder/default_seed42/autoencoder_loss_curve.csv \
    --output figures/autoencoder_losses.pdf \
    --x_col epoch \
    --y_cols train_loss val_loss train_seq_loss val_seq_loss \
    --labels "Train Total" "Val Total" "Train Seq" "Val Seq" \
    --title "Multitask Autoencoder Training" \
    --ylabel "Loss"
```

**Genetic Algorithm History:**

```bash
python scripts/plot_curves.py \
    --input results/GSM3130435_egfp_unmod_1/read_count_standard/ga_optimization/target_0/history.csv \
    --output figures/ga_history.pdf \
    --x_col generation \
    --y_cols best_fitness \
    --labels "Best Sequence Fitness" \
    --title "GA Evolutionary Search" \
    --xlabel "Generation" \
    --ylabel "Fitness Score"
```

## Compute Canada / DRAC Environment

The folder `hpc` contains instructions to configure and execute jobs on the Digital Research Alliance of Canada (DRAC) clusters.

## About

`AUGMENT` is a project that focuses on sequence optimization with REINFORCE- and REINVENT-style reinforcement learning. The name also incorporates `AUG`, the canonical start codon. As an acronym, `AUGMENT` stands for `Autoencoding UTR Generative Model for Enhanced Nucleotide Translation`.

`AUGMENT-GA` is a standalone framework that implements a baseline optimization approach for MRL prediction, combining a genetic algorithm with a CNN-based oracle.

This research project was carried out by Siavash Khalaj under the supervision of Marcel Turcotte.

Khalaj, S. (2026). _Enhancing mRNA Translation Efficiency through Deep Learning_ (Master's thesis, University of Ottawa). University of Ottawa.