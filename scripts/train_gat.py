import argparse
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from src.models.gat import GATRegression
from src.data.dataset import RNAGraphSeqDataset
from src.data.transforms import add_graph_column, standardize_dataframe

def parse_args():
    parser = argparse.ArgumentParser(description="Train GAT model for MRL prediction")
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--train_set", type=str, required=True, help="Path to train set partition")
    parser.add_argument("--valid_set", type=str, required=True, help="Path to valid set partition")
    parser.add_argument("--test_set", type=str, default=None, help="Path to test set partition (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seq_len", type=int, default=50, help="Fixed length of input sequences")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu, auto)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    if not os.path.exists(args.raw_data):
        print(f"Dataset not found at {args.raw_data}. Please run download_datasets.py first.")
        return
        
    print(f"Loading raw data from {args.raw_data}...")
    df = pd.read_csv(args.raw_data)
    df = standardize_dataframe(df)
    
    print("Loading partitions...")
    train_utrs = pd.read_csv(args.train_set)['utr']
    valid_utrs = pd.read_csv(args.valid_set)['utr']
    
    train_df = df[df['utr'].isin(train_utrs)].reset_index(drop=True)
    val_df = df[df['utr'].isin(valid_utrs)].reset_index(drop=True)
    
    if args.test_set:
        test_utrs = pd.read_csv(args.test_set)['utr']
        test_df = df[df['utr'].isin(test_utrs)].reset_index(drop=True)
    else:
        test_df = None
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if 'rl' in train_df.columns:
        mean_rl = train_df['rl'].mean()
        std_rl = train_df['rl'].std()
        
        train_df = train_df.copy()
        val_df = val_df.copy()
        if test_df is not None:
            test_df = test_df.copy()
        
        train_df.loc[:, 'scaled_rl'] = (train_df['rl'] - mean_rl) / std_rl
        val_df.loc[:, 'scaled_rl'] = (val_df['rl'] - mean_rl) / std_rl
        if test_df is not None:
            test_df.loc[:, 'scaled_rl'] = (test_df['rl'] - mean_rl) / std_rl
        
        with open(os.path.join(output_dir, "gat_scaler.json"), "w") as f:
            json.dump({"mean": float(mean_rl), "std": float(std_rl)}, f)

    print("Precomputing RNA secondary structure graphs...")
    train_df = add_graph_column(train_df)
    val_df = add_graph_column(val_df)
    
    if test_df is not None:
        print("Adding graph representation to test set (this takes a moment)...")
        test_df = add_graph_column(test_df)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    else:
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    train_dataset = RNAGraphSeqDataset(train_df, max_len=args.seq_len)
    val_dataset = RNAGraphSeqDataset(val_df, max_len=args.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    if test_df is not None:
        test_dataset = RNAGraphSeqDataset(test_df, max_len=args.seq_len)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = GATRegression(in_channels=10, edge_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    start_time = time.time()
    history = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            rl_true = batch.y.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            _, pred = model(batch)
            loss = criterion(pred, rl_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                rl_true = batch.y.to(device).view(-1, 1)
                
                _, pred = model(batch)
                loss = criterion(pred, rl_true)
                
                val_loss += loss.item()
                
        t_loss = train_loss/len(train_loader)
        v_loss = val_loss/len(val_loader)
        history.append({"epoch": epoch+1, "train_loss": t_loss, "val_loss": v_loss})
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {t_loss:.4f} - Val Loss: {v_loss:.4f}")
        
    training_time = time.time() - start_time
    
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "gat_loss_curve.csv"), index=False)
        
    model_out_path = os.path.join(output_dir, "gat_model.pth")
    torch.save(model.state_dict(), model_out_path)
    print(f"Training complete. Model saved to {model_out_path}")
    
    if test_df is not None:
        print("Evaluating on test set...")
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                rl_true = batch.y.to(device).view(-1, 1)
                _, pred = model(batch)
                predictions.extend(pred.cpu().numpy().flatten())
                actuals.extend(rl_true.cpu().numpy().flatten())
                
        pearson_corr, _ = pearsonr(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        print(f"Test Pearson r: {pearson_corr:.4f}")
        print(f"Test R^2 Score: {r2:.4f}")
        
        summary = {
            "model": "GATRegression",
            "data_path": args.raw_data,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
            "training_time_seconds": training_time,
            "test_pearson_r": float(pearson_corr),
            "test_r2": float(r2),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df)
        }
    else:
        print("No test set provided; skipping final evaluation.")
        summary = {
            "model": "GATRegression",
            "data_path": args.raw_data,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
            "training_time_seconds": training_time,
            "train_size": len(train_df),
            "val_size": len(val_df)
        }
        
    with open(os.path.join(output_dir, "gat_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
