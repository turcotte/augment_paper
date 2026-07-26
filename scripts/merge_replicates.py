#!/usr/bin/env python3
import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Merge two replicate datasets and compute weighted average of MRL.")
    parser.add_argument("--r1_path", type=str, required=True, help="Path to replicate 1 CSV (e.g. data/..._r1.csv.gz)")
    parser.add_argument("--r2_path", type=str, required=True, help="Path to replicate 2 CSV (e.g. data/..._r2.csv.gz)")
    parser.add_argument("--out_path", type=str, required=True, help="Path to save merged CSV (e.g. data/..._merged.csv.gz)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.r1_path):
        print(f"Error: {args.r1_path} not found.")
        return
    if not os.path.exists(args.r2_path):
        print(f"Error: {args.r2_path} not found.")
        return
        
    print(f"Loading {args.r1_path}...")
    df_r1 = pd.read_csv(args.r1_path, usecols=["UTR", "total", "rl"])
    
    print(f"Loading {args.r2_path}...")
    df_r2 = pd.read_csv(args.r2_path, usecols=["UTR", "total", "rl"])
    
    print("Merging datasets on UTR...")
    df = pd.merge(df_r1, df_r2, how="outer", on="UTR", suffixes=("_r1", "_r2"))
    
    for c in ["total_r1", "total_r2", "rl_r1", "rl_r2"]:
        df[c] = df[c].fillna(0)
        
    df["total_reads"] = df["total_r1"] + df["total_r2"]
    df["rl"] = 0.0
    
    mask = df["total_reads"] > 0
    df.loc[mask, "rl"] = (
        df.loc[mask, "rl_r1"] * df.loc[mask, "total_r1"] +
        df.loc[mask, "rl_r2"] * df.loc[mask, "total_r2"]
    ) / df.loc[mask, "total_reads"]
    
    print(f"Total sequences after merge: {len(df):,}")
    
    # Save the necessary columns to standardized output
    df_out = df[["UTR", "total_reads", "rl"]].copy()
    
    print(f"Saving merged dataset to {args.out_path}...")
    df_out.to_csv(args.out_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
