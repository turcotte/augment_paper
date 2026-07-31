#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.manifold import TSNE

# Allow absolute imports from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gat import GATRegression
from src.models.autoencoder import ARLSTMDecoder, GATAutoEncoder
from src.data.dataset import RNAGraphSeqDataset
from src.data.transforms import add_graph_column

from src.data.metrics import (
    total_sequence_entropy,
    shannon_entropy_window,
    gc_content,
    count_uaugs,
    count_uorfs,
    mfe_region,
    count_are_motifs
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Autoencoder Latent Space")
    parser.add_argument("--data", type=str, required=True, help="Path to partition CSV file with 'utr' column")
    parser.add_argument("--raw_data", type=str, default=None, help="Path to raw dataset with 'rl' column for MRL plotting (optional)")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing autoencoder_model.pth")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--n_samples", type=int, default=None, help="Cap the number of samples to process (for speed)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu, auto)")
    parser.add_argument("--seq_len", type=int, default=50, help="Fixed length of input sequences")
    return parser.parse_args()

def generate_heatmap(x, y, c, metric_name, title, output_path):
    """Generate a heatmap scatter plot colored by a metric."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=c, cmap="viridis", s=15, alpha=0.7, edgecolors="none")
    plt.colorbar(scatter, label=metric_name)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()

    # Handle device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    if 'utr' not in df.columns:
        raise ValueError("Data CSV must contain a 'utr' column with RNA sequences.")

    if args.n_samples and len(df) > args.n_samples:
        print(f"Subsampling to {args.n_samples} records...")
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)

    if args.raw_data:
        print(f"Loading raw data for MRL from {args.raw_data}...")
        raw_df = pd.read_csv(args.raw_data)
        if 'rl' in raw_df.columns:
            raw_df = raw_df.drop_duplicates(subset=['utr'])
            df = df.merge(raw_df[['utr', 'rl']], on='utr', how='left')
        else:
            print("Warning: 'rl' column not found in raw_data. Skipping MRL.")

    print(f"Processing {len(df)} sequences...")

    # 2. Compute Biological Metrics
    print("Computing biological metrics for heat-maps...")
    metrics_data = {}
    metrics_data["GC Content"] = df["utr"].apply(gc_content).values
    metrics_data["Total Entropy"] = df["utr"].apply(total_sequence_entropy).values
    metrics_data["uAUG Count"] = df["utr"].apply(count_uaugs).values
    metrics_data["uORF Count"] = df["utr"].apply(count_uorfs).values
    metrics_data["ARE Motifs"] = df["utr"].apply(count_are_motifs).values
    
    print("  Calculating MFE (this takes a moment)...")
    metrics_data["Cap-Proximal MFE (1-30)"] = df["utr"].apply(lambda s: mfe_region(s, 0, 30)).values
    metrics_data["Start-Proximal MFE (30-50)"] = df["utr"].apply(lambda s: mfe_region(s, 29, 50)).values
    metrics_data["Whole Sequence MFE"] = df["utr"].apply(lambda s: mfe_region(s, 0, len(s))).values
    
    if 'rl' in df.columns:
        metrics_data["Mean Ribosome Load (MRL)"] = df["rl"].values

    # 3. Model Loading
    print("Loading GNN Autoencoder...")
    encoder = GATRegression(in_channels=10, edge_dim=2, hidden_channels=128)
    decoder = ARLSTMDecoder(latent_dim=128*2, hidden_dim=256, seq_len=args.seq_len)
    model = GATAutoEncoder(encoder, decoder).to(device)
    
    model_path = os.path.join(args.model_dir, "autoencoder_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 4. Feature Extraction via Graph DataLoader
    print("Building RNA graphs (this takes a moment)...")
    df = add_graph_column(df)
    
    # We supply scaled_rl as zeros because the encoder forward pass does not strictly depend on the target label.
    # The dataloader expects it to exist in the dataframe.
    if 'scaled_rl' not in df.columns:
        df['scaled_rl'] = 0.0
        
    dataset = RNAGraphSeqDataset(df, max_len=args.seq_len)
    loader = GeoDataLoader(dataset, batch_size=256, shuffle=False)

    print("Extracting latent vectors...")
    all_latents = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # The encoder returns (latent_representation, rl_prediction)
            latent, _ = model.encoder(batch)
            all_latents.append(latent.cpu().numpy())

    latent_matrix = np.vstack(all_latents)
    print(f"Extracted latent space with shape: {latent_matrix.shape}")

    # 5. Dimensionality Reduction (t-SNE)
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latent_tsne = tsne.fit_transform(latent_matrix)

    # Dimensionality Reduction (UMAP)
    has_umap = False
    latent_umap = None
    try:
        import umap
        print("Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        latent_umap = reducer.fit_transform(latent_matrix)
        has_umap = True
    except ImportError:
        print("Warning: umap-learn is not installed. Skipping UMAP projections.")
        print("To enable UMAP, run: pip install umap-learn")

    # 6. Generate Plot Heat-maps
    print(f"Generating scatter plots in {args.output_dir}...")
    sns.set_theme(style="white")

    for metric_name, values in metrics_data.items():
        # Format filename safely
        safe_name = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        # Plot t-SNE
        tsne_path = os.path.join(args.output_dir, f"tsne_{safe_name}.pdf")
        generate_heatmap(
            latent_tsne[:, 0], latent_tsne[:, 1], values, 
            metric_name, f"t-SNE Latent Space ({metric_name})", tsne_path
        )
        
        # Plot UMAP if available
        if has_umap:
            umap_path = os.path.join(args.output_dir, f"umap_{safe_name}.pdf")
            generate_heatmap(
                latent_umap[:, 0], latent_umap[:, 1], values, 
                metric_name, f"UMAP Latent Space ({metric_name})", umap_path
            )
            
    print("All visualizations complete!")

if __name__ == "__main__":
    main()
