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
        │       ├── cnn_model.pth
        │       └── cnn_scaler.json      # <- Scalers live next to the model
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

*(Note: installing the project in editable mode `-e .` ensures that the modules in `src/` are accessible to all scripts without needing to modify your `PYTHONPATH`.)*

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
    --sigma 300.0
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
    --sigma 300.0 \
    --num_bins 10
```

*(Variable length inputs are supported via the `--seq_len` argument.)*

## Creating GA Evaluation Subsets

Before evaluating the Genetic Algorithm, you must create disjoint calibration and test subsets from the main test partition:

```bash
python scripts/create_ga_subsets.py \
    --input_file results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --n_calibration 100 \
    --n_test 1000 \
    --seed 42
```

## Calibrating the Genetic Algorithm

Before running the batch evaluation, use the calibration script to perform a parameter sweep over the edit-penalty (`lambda`) values.

```bash
python scripts/calibrate_ga.py \
    --calibration_set results/GSM3130435_egfp_unmod_1/read_count_standard/calibration-100.csv.gz \
    --predictor_dir results/GSM3130435_egfp_unmod_1/read_count_standard/gat/default_seed42 \
    --proxy_dir results/GSM3130435_egfp_unmod_1/read_count_standard/cnn/default_seed42 \
    --predictor gat \
    --proxy_predictor cnn \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/ga/calibration-100
```

This will generate a `calibration_summary.csv` containing the parsed sweep results, along with the raw data and a `calibrate_ga_summary.json` containing the metadata.

## Running the Genetic Algorithm

Once you determine the optimal lambda from the calibration step (e.g., `0.075`) by visualizing the Pareto fronts, run the batch evaluation across the large test set:

```bash
python scripts/run_ga.py \
    --input_csv results/GSM3130435_egfp_unmod_1/read_count_standard/test-1000.csv.gz \
    --predictor_dir results/GSM3130435_egfp_unmod_1/read_count_standard/gat/default_seed42 \
    --proxy_dir results/GSM3130435_egfp_unmod_1/read_count_standard/cnn/default_seed42 \
    --predictor gat \
    --proxy_predictor cnn \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/ga/test-1000 \
    --lambda_val 0.075
```

This will automatically generate uniquely evolved sequences and output them to `results/GSM3130435_egfp_unmod_1/read_count_standard/ga_optimization/default_seed42/ga_results.csv.gz`.

## Aggregating Pareto Metrics for RL/Fine-Tuning

To evaluate how generative models (e.g. DAP, REINFORCE) perform against a specific calibration set, use the `aggregate_pareto_metrics.py` script. This computes sequence divergence (edits) and fitness gain ($\Delta$MRL), creating a standard `pareto_summary.csv` suitable for plotting.

### Example 1: DAP Sigma Sweep

When evaluating a parameter sweep (e.g., DAP models trained across various $\sigma$ values), use `--param_name` to extract the parameter directly from the models' JSON metadata:
```bash
python scripts/aggregate_pareto_metrics.py \
    --calibration_set results/GSM3130435_egfp_unmod_1/read_count_standard/calibration-100.csv.gz \
    --rl_csvs results/GSM3130435_egfp_unmod_1/read_count_standard/dap/sigma*/dap_optimized_sequences.csv.gz \
    --param_name sigma \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/dap/calibration-100
```

### Example 2: REINFORCE Single Run

For a single fine-tuning run like REINFORCE where no parameter sweep occurred, simply omit `--param_name`:
```bash
python scripts/aggregate_pareto_metrics.py \
    --calibration_set results/GSM3130435_egfp_unmod_1/read_count_standard/calibration-100.csv.gz \
    --rl_csvs results/GSM3130435_egfp_unmod_1/read_count_standard/reinforce/default_seed42/reinforce_optimized_sequences.csv.gz \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/reinforce/calibration-100
```

## Plotting Pareto Curves

Once summaries are generated for the GA and generative models, you can plot their Pareto fronts using a declarative YAML configuration. This perfectly decouples data aggregation from aesthetic visualization.

First, create a configuration file (e.g., `results/GSM3130435_egfp_unmod_1/read_count_standard/figures/pareto_config.yaml`):
```yaml
output_dir: "results/GSM3130435_egfp_unmod_1/read_count_standard/figures"
plot_name: "pareto_calibration_fronts.pdf"
proxy_predictor: "cnn"
n_samples: 100
include_zero_x: true
include_zero_y: true

curves:
  - name: "Genetic Algorithm"
    file: "results/GSM3130435_egfp_unmod_1/read_count_standard/ga/calibration-100/calibration_summary.csv"
    param_col: "lambda"
    param_symbol: "$\\lambda$"
    marker: "o"
    color: "#1f77b4"
    is_front: true

  - name: "REINFORCE"
    file: "results/GSM3130435_egfp_unmod_1/read_count_standard/reinforce/calibration-100/pareto_summary.csv"
    param_col: null
    marker: "s"
    color: "#2ca02c"
    is_front: false

  - name: "DAP (Sigma Sweep)"
    file: "results/GSM3130435_egfp_unmod_1/read_count_standard/dap/calibration-100/pareto_summary.csv"
    param_col: "sigma"
    param_symbol: "$\\sigma$"
    marker: "X"
    color: "#d62728"
    is_front: true
```

Then, run the plotting script:
```bash
python scripts/plot_pareto_curves.py --config results/GSM3130435_egfp_unmod_1/read_count_standard/figures/pareto_config.yaml
```

## Plotting Training Curves

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

## Comparing Biological Metrics

To generate comprehensive biological evaluation plots comparing a fine-tuned RL model (e.g., Curriculum DAP) against the Genetic Algorithm baseline, use the comparison script. This script computes local structural accessibility (MFE), compositional shifts (GC, Entropy), and regulatory disruptions (uAUG, uORFs) without requiring PyTorch or model weights.

```bash
python scripts/compare_models.py \
    --rl_csv results/GSM3130435_egfp_unmod_1/read_count_standard/curriculum_dap/default_seed42/curriculum_dap_optimized_sequences.csv.gz \
    --ga_csv results/GSM3130435_egfp_unmod_1/read_count_standard/ga_optimization/default_seed42/ga_results.csv.gz \
    --output_dir figures/comparison_curriculum_dap_vs_ga
```

## Latent Space Visualization

You can visualize the GNN autoencoder's learned latent space using t-SNE and UMAP, colored by various biological metrics (e.g., GC content, uAUG count, MFE).

```bash
python scripts/visualize_latent_space.py \
    --data results/GSM3130435_egfp_unmod_1/read_count_standard/test.csv.gz \
    --raw_data data/GSM3130435_egfp_unmod_1.csv.gz \
    --model_dir results/GSM3130435_egfp_unmod_1/read_count_standard/autoencoder/default_seed42 \
    --output_dir results/GSM3130435_egfp_unmod_1/read_count_standard/autoencoder/default_seed42/latent_space_plots
```

> **Note**: You can set `--output_dir` to be the same as `--model_dir`. The script will output PDF plots (e.g. `tsne_gc_content.pdf`) which will not conflict with your model files. However, creating a dedicated `latent_space_plots` sub-folder keeps your results organized.

The script generates:
- t-SNE and UMAP projections (if `umap-learn` is installed).
- Heat-map scatter plots for GC Content, Total Entropy, uAUGs, uORFs, ARE Motifs, and regional/whole-sequence MFE.

## Compute Canada / DRAC Environment

The folder `hpc` contains instructions to configure and execute jobs on the Digital Research Alliance of Canada (DRAC) clusters.

## About

`AUGMENT` is a project that focuses on sequence optimization with REINFORCE- and REINVENT-style reinforcement learning. The name also incorporates `AUG`, the canonical start codon. As an acronym, `AUGMENT` stands for `Autoencoding UTR Generative Model for Enhanced Nucleotide Translation`.

`AUGMENT-GA` is a standalone framework that implements a baseline optimization approach for MRL prediction, combining a genetic algorithm with a CNN-based oracle.

This research project was carried out by Siavash Khalaj under the supervision of Marcel Turcotte.

Khalaj, S. (2026). _Enhancing mRNA Translation Efficiency through Deep Learning_ (Master's thesis, University of Ottawa). University of Ottawa.