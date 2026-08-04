#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.metrics import (
    total_sequence_entropy,
    shannon_entropy_window,
    gc_content,
    count_uaugs,
    count_uorfs,
    mfe_region,
    count_are_motifs
)


def main():
    parser = argparse.ArgumentParser(description="Generate Biological Comparisons: RL vs GA")
    parser.add_argument("--rl_csv", type=Path, required=True, help="Path to RL final results CSV (e.g. reinforce_optimized_sequences.csv.gz)")
    parser.add_argument("--ga_csv", type=Path, required=True, help="Path to GA final results CSV (e.g. ga_results.csv.gz)")
    parser.add_argument("--output_dir", type=Path, default=Path("results/comparison_plots"), help="Output directory for plots")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"Loading RL results from {args.rl_csv}...")
    df_rl = pd.read_csv(args.rl_csv)
    
    print(f"Loading GA results from {args.ga_csv}...")
    df_ga = pd.read_csv(args.ga_csv)

    # Validate columns
    required_cols = ["orig_seq", "gen_seq", "orig_cnn_mrl", "gen_cnn_mrl", "hamming_dist"]
    for col in required_cols:
        if col not in df_rl.columns:
            raise ValueError(f"RL CSV is missing required column: {col}")
        if col not in df_ga.columns:
            raise ValueError(f"GA CSV is missing required column: {col}")

    # Inner Join on orig_seq to ensure matched comparisons
    df_eval = pd.merge(
        df_rl, df_ga, 
        on=["orig_seq"], 
        suffixes=("_rl", "_ga")
    )
    
    if len(df_eval) == 0:
        raise ValueError("No matching 'orig_seq' sequences found between RL and GA files.")
        
    print(f"Successfully matched {len(df_eval)} sequences for comparison.")

    # 2. Compute Biological Metrics
    print("Computing biological metrics (this may take a moment for MFE calculation)...")
    
    # Delta MRL
    df_eval["delta_mrl_rl"] = df_eval["gen_cnn_mrl_rl"] - df_eval["orig_cnn_mrl_rl"]
    df_eval["delta_mrl_ga"] = df_eval["gen_cnn_mrl_ga"] - df_eval["orig_cnn_mrl_ga"]
    
    metrics_targets = {
        "orig": df_eval["orig_seq"],
        "rl": df_eval["gen_seq_rl"],
        "ga": df_eval["gen_seq_ga"]
    }
    
    for prefix, seq_col in metrics_targets.items():
        print(f"  Calculating for '{prefix}' sequences...")
        df_eval[f"uaug_{prefix}"] = seq_col.apply(count_uaugs)
        df_eval[f"uorf_{prefix}"] = seq_col.apply(count_uorfs)
        df_eval[f"gc_{prefix}"] = seq_col.apply(gc_content)
        df_eval[f"ent_tot_{prefix}"] = seq_col.apply(total_sequence_entropy)
        df_eval[f"ent_win_{prefix}"] = seq_col.apply(shannon_entropy_window)
        df_eval[f"mfe_cap_{prefix}"] = seq_col.apply(lambda s: mfe_region(s, 0, 30))
        df_eval[f"mfe_start_{prefix}"] = seq_col.apply(lambda s: mfe_region(s, 29, 50))
        df_eval[f"are_{prefix}"] = seq_col.apply(count_are_motifs)

    # 3. Generating Plots
    print(f"Generating Plots in {args.output_dir}/")
    sns.set_theme(style="whitegrid")
    
    # A. Global Performance: Delta MRL
    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_eval["delta_mrl_rl"].dropna(), label="RL Autoencoder", fill=True, color="blue", alpha=0.4)
    sns.kdeplot(df_eval["delta_mrl_ga"].dropna(), label="Genetic Algorithm", fill=True, color="green", alpha=0.4)
    plt.title("Distribution of Fitness Gain (CNN Proxy ΔMRL)")
    plt.xlabel("ΔMRL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "global_performance_delta_mrl.pdf")
    plt.close()
    
    # Global Performance: Edits
    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_eval["hamming_dist_rl"].dropna(), label="RL Autoencoder", fill=True, color="blue", alpha=0.4)
    sns.kdeplot(df_eval["hamming_dist_ga"].dropna(), label="Genetic Algorithm", fill=True, color="green", alpha=0.4)
    plt.title("Distribution of Edit Counts (Sequence Divergence)")
    plt.xlabel("Number of Mutations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "global_performance_edits.pdf")
    plt.close()

    # B. Regulatory Burden (uAUG)
    plt.figure(figsize=(6, 5))
    uaug_data = pd.DataFrame({
        "Model": ["Original"]*len(df_eval) + ["RL Autoencoder"]*len(df_eval) + ["Genetic Algorithm"]*len(df_eval),
        "uAUG Count": np.concatenate([df_eval["uaug_orig"], df_eval["uaug_rl"], df_eval["uaug_ga"]])
    })
    sns.boxplot(data=uaug_data, x="Model", y="uAUG Count", hue="Model", palette="Set2", legend=False)
    plt.title("Disruption of uAUG Codons")
    plt.tight_layout()
    plt.savefig(args.output_dir / "regulatory_burden_uaug.pdf")
    plt.close()
    
    # Regulatory Burden (uORF)
    plt.figure(figsize=(6, 5))
    uorf_data = pd.DataFrame({
        "Model": ["Original"]*len(df_eval) + ["RL Autoencoder"]*len(df_eval) + ["Genetic Algorithm"]*len(df_eval),
        "uORF Count": np.concatenate([df_eval["uorf_orig"], df_eval["uorf_rl"], df_eval["uorf_ga"]])
    })
    sns.boxplot(data=uorf_data, x="Model", y="uORF Count", hue="Model", palette="Set2", legend=False)
    plt.title("Disruption of uORFs")
    plt.tight_layout()
    plt.savefig(args.output_dir / "regulatory_burden_uorf.pdf")
    plt.close()
    
    # C. Compositional Entropy and GC Content
    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_eval["gc_orig"].dropna(), label="Original", fill=True, color="grey", alpha=0.3)
    sns.kdeplot(df_eval["gc_rl"].dropna(), label="RL", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df_eval["gc_ga"].dropna(), label="GA", fill=True, color="green", alpha=0.3)
    plt.title("GC Content Shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "compositional_shifts_gc.pdf")
    plt.close()
    
    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_eval["ent_win_orig"].dropna(), label="Original", fill=True, color="grey", alpha=0.3)
    sns.kdeplot(df_eval["ent_win_rl"].dropna(), label="RL", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df_eval["ent_win_ga"].dropna(), label="GA", fill=True, color="green", alpha=0.3)
    plt.title("Sliding Window Entropy (k=10)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "compositional_shifts_entropy.pdf")
    plt.close()
    
    # D. Local Structural Accessibility (MFE)
    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_eval["mfe_cap_orig"].dropna(), label="Original", fill=True, color="grey", alpha=0.3)
    sns.kdeplot(df_eval["mfe_cap_rl"].dropna(), label="RL", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df_eval["mfe_cap_ga"].dropna(), label="GA", fill=True, color="green", alpha=0.3)
    plt.title("Cap-Proximal Accessibility (Pos 1-30 MFE)")
    plt.xlabel("Minimum Free Energy (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "structural_accessibility_cap_mfe.pdf")
    plt.close()
    
    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_eval["mfe_start_orig"].dropna(), label="Original", fill=True, color="grey", alpha=0.3)
    sns.kdeplot(df_eval["mfe_start_rl"].dropna(), label="RL", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df_eval["mfe_start_ga"].dropna(), label="GA", fill=True, color="green", alpha=0.3)
    plt.title("Start-Proximal Accessibility (Pos 30-50 MFE)")
    plt.xlabel("Minimum Free Energy (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "structural_accessibility_start_mfe.pdf")
    plt.close()
    
    # E. Positional Distribution of Edits
    rl_mut_pos = np.zeros(50)
    ga_mut_pos = np.zeros(50)
    
    for _, row in df_eval.iterrows():
        orig = str(row["orig_seq"])
        rl_seq = str(row["gen_seq_rl"])
        ga_seq = str(row["gen_seq_ga"])
        
        # Guard against different lengths
        min_len_rl = min(len(orig), len(rl_seq), 50)
        min_len_ga = min(len(orig), len(ga_seq), 50)
        
        for i in range(min_len_rl):
            if orig[i] != rl_seq[i]:
                rl_mut_pos[i] += 1
                
        for i in range(min_len_ga):
            if orig[i] != ga_seq[i]:
                ga_mut_pos[i] += 1
                
    rl_mut_freq = rl_mut_pos / len(df_eval)
    ga_mut_freq = ga_mut_pos / len(df_eval)
    
    plt.figure(figsize=(12, 5))
    positions = np.arange(1, 51)
    plt.plot(positions, rl_mut_freq, label="RL Autoencoder", color="blue", marker="o", linestyle="-", linewidth=2)
    plt.plot(positions, ga_mut_freq, label="Genetic Algorithm", color="green", marker="s", linestyle="-", linewidth=2)
    plt.fill_between(positions, rl_mut_freq, alpha=0.2, color="blue")
    plt.fill_between(positions, ga_mut_freq, alpha=0.2, color="green")
    
    plt.title("Positional Distribution of Optimization Edits (Hotspots)")
    plt.xlabel("Nucleotide Position (5' to 3')")
    plt.ylabel("Mutation Frequency across Test Set")
    plt.xlim(1, 50)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "positional_edits.pdf")
    plt.close()

    print("\n================ SUMMARY STATISTICS ================")
    print(f"  Average RL CNN ΔMRL: {df_eval['delta_mrl_rl'].mean():.4f}")
    print(f"  Average GA CNN ΔMRL: {df_eval['delta_mrl_ga'].mean():.4f}")
    print(f"  Average RL Edits:    {df_eval['hamming_dist_rl'].mean():.2f}")
    print(f"  Average GA Edits:    {df_eval['hamming_dist_ga'].mean():.2f}")
    print(f"  RL ARE Insertions:   {df_eval['are_rl'].sum() - df_eval['are_orig'].sum()}")
    print(f"  GA ARE Insertions:   {df_eval['are_ga'].sum() - df_eval['are_orig'].sum()}")
    print("====================================================\n")

if __name__ == "__main__":
    main()
