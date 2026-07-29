import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Split a dataset into disjoint calibration and test subsets for GA evaluation using uniform random sampling."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the input CSV file (e.g., test.csv.gz).",
    )
    parser.add_argument(
        "--n_calibration",
        type=int,
        default=100,
        help="Number of sequences to sample for the calibration set.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1000,
        help="Number of sequences to sample for the test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    print(f"Loading dataset from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    total_requested = args.n_calibration + args.n_test

    if len(df) < total_requested:
        raise ValueError(
            f"Dataset only contains {len(df)} rows, but {total_requested} were requested "
            f"({args.n_calibration} for calibration + {args.n_test} for test)."
        )

    # Sample the total required rows uniformly at random to ensure no overlap
    print(f"Sampling {total_requested} rows uniformly at random (seed={args.seed})...")
    sampled_df = df.sample(n=total_requested, random_state=args.seed)

    # Split the sampled dataframe into calibration and test disjoint subsets
    calibration_df = sampled_df.iloc[: args.n_calibration]
    test_df = sampled_df.iloc[args.n_calibration :]

    output_dir = args.input_file.parent
    
    calibration_out = output_dir / f"calibration-{args.n_calibration}.csv.gz"
    test_out = output_dir / f"test-{args.n_test}.csv.gz"

    print(f"Saving calibration set ({len(calibration_df)} rows) to {calibration_out}...")
    calibration_df.to_csv(calibration_out, index=False, compression="gzip")

    print(f"Saving test set ({len(test_df)} rows) to {test_out}...")
    test_df.to_csv(test_out, index=False, compression="gzip")
    
    print("Done!")


if __name__ == "__main__":
    main()
