import argparse
import os
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Create immutable dataset partitions (train/valid/test)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw dataset (e.g., data/GSM3130435_egfp_unmod_1.csv.gz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (e.g., results/GSM3130435_egfp_unmod_1/read_count_standard)")
    parser.add_argument("--strategy", type=str, choices=["read_count", "random"], default="read_count", help="Partitioning strategy")
    parser.add_argument("--test_size", type=int, default=20000, help="Number of samples for test set")
    parser.add_argument("--train_size", type=int, default=180000, help="Number of samples for train set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random strategy")
    return parser.parse_args()

def apply_read_count_strategy(df, args):
    """
    Sorts by total_reads descending.
    Splits top N into test, next M into train, and the rest into validation.
    This exactly mimics the original notebook behavior for egfp_unmod_1.
    """
    sort_col = 'total_reads' if 'total_reads' in df.columns else 'total'
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    else:
        print(f"Warning: {sort_col} not found in columns. Dataset remains unsorted.")
    
    n = len(df)
    
    if args.test_size + args.train_size >= n:
        print(f"Warning: test_size ({args.test_size}) + train_size ({args.train_size}) >= total dataset size ({n}).")
        # Adjust train size to leave at least 10% for validation if possible
        args.train_size = max(0, n - args.test_size - int(n * 0.1))
        
    test_idx = args.test_size
    train_idx = test_idx + args.train_size
    
    test_df = df.iloc[:test_idx]
    train_df = df.iloc[test_idx:train_idx]
    val_df = df.iloc[train_idx:]
    
    return train_df, val_df, test_df

def apply_random_strategy(df, args):
    """
    Randomly shuffles the dataset and splits it based on size arguments.
    """
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    
    if args.test_size + args.train_size >= n:
        print(f"Warning: test_size ({args.test_size}) + train_size ({args.train_size}) >= total dataset size ({n}).")
        args.train_size = max(0, n - args.test_size - int(n * 0.1))
        
    test_idx = args.test_size
    train_idx = test_idx + args.train_size
    
    test_df = df.iloc[:test_idx]
    train_df = df.iloc[test_idx:train_idx]
    val_df = df.iloc[train_idx:]
    
    return train_df, val_df, test_df

def main():
    args = parse_args()
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Standardize UTR column
    if 'UTR' in df.columns and 'utr' not in df.columns:
        df.rename(columns={'UTR': 'utr'}, inplace=True)
        
    if 'utr' not in df.columns:
        raise ValueError("Dataset must contain a 'utr' or 'UTR' column.")
        
    print(f"Total dataset size: {len(df)}")
    
    if args.strategy == "read_count":
        train_df, val_df, test_df = apply_read_count_strategy(df, args)
    elif args.strategy == "random":
        train_df, val_df, test_df = apply_random_strategy(df, args)
    else:
        raise NotImplementedError(f"Strategy {args.strategy} not implemented.")
        
    print(f"Partition sizes -> Train: {len(train_df)}, Valid: {len(val_df)}, Test: {len(test_df)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save partitions as compressed 1-column CSVs containing sequence strings
    train_out = os.path.join(args.output_dir, "train.csv.gz")
    val_out = os.path.join(args.output_dir, "valid.csv.gz")
    test_out = os.path.join(args.output_dir, "test.csv.gz")
    
    train_df[['utr']].to_csv(train_out, index=False, compression='gzip')
    val_df[['utr']].to_csv(val_out, index=False, compression='gzip')
    test_df[['utr']].to_csv(test_out, index=False, compression='gzip')
    
    print(f"Partitions successfully saved to {args.output_dir}")

if __name__ == "__main__":
    main()
