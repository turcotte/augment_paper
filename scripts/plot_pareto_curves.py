#!/usr/bin/env python3

import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Allow absolute imports from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Plot Pareto curves from aggregated summary CSVs using a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(config.get("output_dir", "results/figures"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = config.get("plot_name", "pareto_front.pdf")
    proxy_predictor = config.get("proxy_predictor", "cnn").upper()
    n_samples = config.get("n_samples", 1)  # Default to 1 to avoid division by zero if not provided
    
    print("\nGenerating Pareto front plot...")
    plt.figure(figsize=(8, 6))
    
    for curve in config.get("curves", []):
        name = curve.get("name", "Unknown")
        file_path = Path(curve.get("file"))
        
        if not file_path.exists():
            print(f"Warning: File not found for '{name}': {file_path}. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Sort values by avg_edits to plot a clean line for sweeps
        df = df.sort_values(by="avg_edits")
        
        avg_edits = df["avg_edits"].values
        # Compute standard error of the mean
        std_edits_err = df["std_edits"].values / np.sqrt(n_samples)
        
        avg_delta = df["avg_delta_mrl"].values
        std_delta_err = df["std_delta_mrl"].values / np.sqrt(n_samples)
        
        marker = curve.get("marker", "o")
        color = curve.get("color", None)
        is_front = curve.get("is_front", True)
        
        linestyle = '-' if is_front else 'none'
        markersize = 8 if is_front else 10
        
        plt.errorbar(
            avg_edits, 
            avg_delta, 
            xerr=std_edits_err, 
            yerr=std_delta_err, 
            marker=marker, 
            markersize=markersize,
            linestyle=linestyle, 
            color=color,
            label=name, 
            capsize=4
        )
        
        # Annotate points if requested
        param_col = curve.get("param_col")
        param_symbol = curve.get("param_symbol", "")
        
        if param_col and param_col in df.columns:
            for idx, row in df.iterrows():
                val = row[param_col]
                # Try to format floats nicely
                if isinstance(val, (float, np.floating)):
                    # Format standard floats nicely, ignoring standard integer-like floats
                    val_str = f"{val:g}"
                else:
                    val_str = str(val)
                    
                annotation_text = f"{param_symbol}={val_str}" if param_symbol else val_str
                
                # Alternate annotation placement logic can be added here
                # Default behavior: slightly above the point for lines, below for scatters
                offset = (0, 10) if is_front else (0, -15)
                
                plt.annotate(
                    annotation_text,
                    (row["avg_edits"], row["avg_delta_mrl"]),
                    textcoords="offset points",
                    xytext=offset,
                    ha='center',
                    fontsize=9
                )

    plt.xlabel("Average Edit Count (Sequence Divergence)")
    plt.ylabel(f"Average {proxy_predictor} ΔMRL (Fitness Gain)")
    plt.title("Pareto Front Evaluation")
    
    if config.get("include_zero_x", False):
        plt.xlim(left=0)
    if config.get("include_zero_y", False):
        plt.ylim(bottom=0)
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    out_pdf = output_dir / plot_name
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Pareto plot saved to {out_pdf}")

if __name__ == "__main__":
    main()
