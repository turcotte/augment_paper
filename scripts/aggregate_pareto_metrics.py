#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys
import os

def find_summary_json(csv_path: Path):
    """Attempt to find a summary JSON file in the same directory as the CSV."""
    directory = csv_path.parent
    # Look for files ending in _summary.json or just summary.json
    json_files = list(directory.glob("*summary.json"))
    if not json_files:
        json_files = list(directory.glob("*.json"))
    return json_files[0] if json_files else None

def extract_param_from_json(json_path: Path, param_name: str):
    """Extract a parameter from a JSON file."""
    if not json_path or not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get(param_name, None)
    except Exception as e:
        print(f"Warning: Could not read {json_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Aggregate sequence generation metrics against a calibration set.")
    parser.add_argument("--calibration_set", type=Path, required=True, help="Path to the calibration dataset (CSV)")
    parser.add_argument("--rl_csvs", type=Path, nargs="+", required=True, help="List of sequence generation CSVs to evaluate")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the aggregated summary")
    parser.add_argument("--seq_col", type=str, default=None, help="Sequence column in calibration set (auto-detected if None)")
    parser.add_argument("--param_name", type=str, default=None, help="JSON key to extract parameter value from (e.g., sigma)")
    
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Calibration Dataset
    print(f"Loading calibration dataset from {args.calibration_set}...")
    df_cal = pd.read_csv(args.calibration_set)
    
    seq_col = args.seq_col
    if not seq_col:
        for col in ['utr', 'sequence', 'UTR', 'orig']:
            if col in df_cal.columns:
                seq_col = col
                break
    
    if not seq_col or seq_col not in df_cal.columns:
        print(f"Error: Could not find sequence column in calibration set. Please specify --seq_col.")
        sys.exit(1)
        
    cal_seqs = df_cal[seq_col].str.upper().str.replace("U", "T").tolist()
    n_samples = len(cal_seqs)
    print(f"Loaded {n_samples} calibration sequences.")

    results = []
    
    # 2. Process each CSV
    print(f"Aggregating metrics for {len(args.rl_csvs)} files...")
    
    # Decide column name for the parameter
    param_col_name = args.param_name if args.param_name else "variant"
    
    for rl_file in args.rl_csvs:
        print(f"\nProcessing {rl_file}...")
        df_rl = pd.read_csv(rl_file)
        
        # Determine the variant label/parameter
        param_value = None
        if args.param_name:
            json_file = find_summary_json(rl_file)
            if json_file:
                param_value = extract_param_from_json(json_file, args.param_name)
                if param_value is not None:
                    print(f"  Found {args.param_name} = {param_value} in {json_file.name}")
                else:
                    print(f"  Warning: Key '{args.param_name}' not found in {json_file.name}")
            else:
                print(f"  Warning: No JSON metadata found in {rl_file.parent}")
        
        if param_value is None:
            # Fallback to parent folder name
            param_value = rl_file.parent.name
            print(f"  Using fallback label: {param_value}")
        
        # Intersection with calibration set
        df_rl["orig_std"] = df_rl["orig_seq"].str.upper().str.replace("U", "T")
        df_rl_sampled = df_rl[df_rl["orig_std"].isin(cal_seqs)].copy()
        
        if len(df_rl_sampled) == 0:
            print(f"  Warning: No overlapping sequences found in {rl_file}. Skipping.")
            continue
            
        df_rl_sampled["rl_edits"] = df_rl_sampled.apply(lambda row: sum(c1 != c2 for c1, c2 in zip(row["orig_seq"], row["gen_seq"])), axis=1)
        df_rl_sampled["rl_delta_mrl"] = df_rl_sampled["gen_cnn_mrl"] - df_rl_sampled["orig_cnn_mrl"]
        
        mean_edits = df_rl_sampled["rl_edits"].mean()
        std_edits = df_rl_sampled["rl_edits"].std()
        mean_delta = df_rl_sampled["rl_delta_mrl"].mean()
        std_delta = df_rl_sampled["rl_delta_mrl"].std()
        
        results.append({
            param_col_name: param_value,
            "avg_edits": float(mean_edits),
            "std_edits": float(std_edits),
            "avg_delta_mrl": float(mean_delta),
            "std_delta_mrl": float(std_delta)
        })
        print(f"  (N={len(df_rl_sampled)}): Avg Edits = {mean_edits:.2f} ± {std_edits:.2f}, Avg Delta MRL = {mean_delta:.4f} ± {std_delta:.4f}")

    # 3. Save Summary CSV
    if not results:
        print("\nError: No valid results to save.")
        sys.exit(1)
        
    summary_csv_path = args.output_dir / "pareto_summary.csv"
    pd.DataFrame(results).to_csv(summary_csv_path, index=False)
    print(f"\nSaved aggregated summary to {summary_csv_path}")

if __name__ == "__main__":
    main()
