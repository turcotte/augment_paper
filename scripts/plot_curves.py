import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot loss or fitness curves from CSV files.")
    parser.add_argument("--input", required=True, type=str, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, type=str, help="Path to save the output PDF.")
    parser.add_argument("--x_col", type=str, default="epoch", help="Column name for the X-axis.")
    parser.add_argument("--y_cols", required=True, nargs='+', type=str, help="Column names for the Y-axis.")
    parser.add_argument("--title", type=str, default="", help="Plot title.")
    parser.add_argument("--xlabel", type=str, default="Epoch", help="X-axis label.")
    parser.add_argument("--ylabel", type=str, default="Loss", help="Y-axis label.")
    parser.add_argument("--labels", nargs='+', type=str, help="Custom legend labels. Must match length of y_cols.")
    parser.add_argument("--ylim", nargs=2, type=float, help="Set fixed Y-axis limits (e.g. --ylim 0 1).")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling window size for smoothing curves (default: 1).")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Load data
    df = pd.read_csv(args.input)

    # Validate columns
    if args.x_col not in df.columns:
        print(f"Error: X-axis column '{args.x_col}' not found in CSV.")
        return
        
    for y_col in args.y_cols:
        if y_col not in df.columns:
            print(f"Error: Y-axis column '{y_col}' not found in CSV.")
            return

    if args.labels and len(args.labels) != len(args.y_cols):
        print("Error: The number of --labels must match the number of --y_cols.")
        return

    # Set up publication-ready aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 6))

    # Apply rolling smoothing if requested
    if args.smooth > 1:
        for y_col in args.y_cols:
            df[y_col] = df[y_col].rolling(window=args.smooth, min_periods=1).mean()

    # Determine labels
    labels = args.labels if args.labels else args.y_cols

    # Plot each column
    palette = sns.color_palette("deep", n_colors=len(args.y_cols))
    
    for y_col, label, color in zip(args.y_cols, labels, palette):
        sns.lineplot(
            data=df,
            x=args.x_col,
            y=y_col,
            label=label,
            color=color,
            linewidth=2.5
        )

    # Formatting
    plt.title(args.title, fontsize=14, pad=15)
    plt.xlabel(args.xlabel, fontsize=12)
    plt.ylabel(args.ylabel, fontsize=12)
    
    if args.ylim:
        plt.ylim(args.ylim)

    plt.legend(frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save as PDF
    plt.savefig(args.output, format='pdf', bbox_inches='tight')
    print(f"Successfully saved plot to {args.output}")

if __name__ == "__main__":
    main()
