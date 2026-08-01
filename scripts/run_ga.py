#!/usr/bin/env python3

import argparse
import json

import torch
import numpy as np
import pandas as pd

from src.models.cnn import TangCNNRegressor
from src.models.gat import GATRegression
from src.models.ga import run_genetic_algorithm, cnn_predict_mrl_unscaled, gat_predict_mrl_unscaled

def main():
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for MRL optimization over a dataset")
    parser.add_argument("--predictor_dir", type=str, required=True, help="Directory containing the predictor model and scaler")
    parser.add_argument("--proxy_dir", type=str, required=True, help="Directory containing the proxy model and scaler")
    parser.add_argument("--predictor", type=str, choices=["cnn", "gat"], required=True, help="Which model architecture to use as optimizer")
    parser.add_argument("--proxy_predictor", type=str, choices=["cnn", "gat"], required=True, help="Proxy model architecture for evaluation")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the dataset CSV (e.g., test.csv.gz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final ga_results.csv.gz")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length of the model")
    parser.add_argument("--population_size", type=int, default=256, help="GA population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to run")
    parser.add_argument("--lambda_val", type=float, required=True, help="Penalty multiplier for edits (calibrated from calibrate_ga.py)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
        
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
        
    # Load Predictor Model
    model_path = os.path.join(args.predictor_dir, f"{args.predictor}_model.pth")
    scaler_path = os.path.join(args.predictor_dir, f"{args.predictor}_scaler.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler JSON not found at {scaler_path}")
        
    with open(scaler_path, "r") as f:
        scaler_dict = json.load(f)
        
    print(f"Loading {args.predictor.upper()} Optimizer...")
    if args.predictor == "cnn":
        model = TangCNNRegressor(sequence_length=args.seq_len).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        predict_wrapper = lambda seqs: cnn_predict_mrl_unscaled(model, seqs, device, scaler_dict)
    elif args.predictor == "gat":
        model = GATRegression(in_channels=10, edge_dim=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        predict_wrapper = lambda seqs: gat_predict_mrl_unscaled(model, seqs, device, scaler_dict)

    # Load Proxy Model
    proxy_model_path = os.path.join(args.proxy_dir, f"{args.proxy_predictor}_model.pth")
    proxy_scaler_path = os.path.join(args.proxy_dir, f"{args.proxy_predictor}_scaler.json")
    
    if not os.path.exists(proxy_model_path):
        raise FileNotFoundError(f"Proxy model checkpoint not found at {proxy_model_path}")
    if not os.path.exists(proxy_scaler_path):
        raise FileNotFoundError(f"Proxy scaler JSON not found at {proxy_scaler_path}")
        
    with open(proxy_scaler_path, "r") as f:
        proxy_scaler_dict = json.load(f)
        
    print(f"Loading {args.proxy_predictor.upper()} Proxy Evaluator...")
    if args.proxy_predictor == "cnn":
        proxy_model = TangCNNRegressor(sequence_length=args.seq_len).to(device)
        proxy_model.load_state_dict(torch.load(proxy_model_path, map_location=device, weights_only=True))
        proxy_wrapper = lambda seqs: cnn_predict_mrl_unscaled(proxy_model, seqs, device, proxy_scaler_dict)
    elif args.proxy_predictor == "gat":
        proxy_model = GATRegression(in_channels=10, edge_dim=2).to(device)
        proxy_model.load_state_dict(torch.load(proxy_model_path, map_location=device, weights_only=True))
        proxy_wrapper = lambda seqs: gat_predict_mrl_unscaled(proxy_model, seqs, device, proxy_scaler_dict)
        
    # Load Input Dataset
    print(f"Loading input dataset from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    seq_col = 'utr' if 'utr' in df.columns else 'sequence'
    if seq_col not in df.columns:
        seq_col = 'UTR'
    if seq_col not in df.columns and 'orig' in df.columns:
        seq_col = 'orig'
        
    print(f"Running Genetic Algorithm on {len(df)} sequences (lambda={args.lambda_val})...")
    
    records = []
    output_csv = os.path.join(args.output_dir, "ga_results.csv.gz")
    
    for idx, row in df.iterrows():
        if idx > 0 and idx % 10 == 0:
            print(f"Processed {idx}/{len(df)} sequences...")
        orig_seq = str(row[seq_col]).upper().replace("U", "T")
        target_pred_raw = float(predict_wrapper([orig_seq])[0])
        
        ga_df, _ = run_genetic_algorithm(
            predict_fn=predict_wrapper,
            target_seq=orig_seq,
            target_pred_raw=target_pred_raw,
            population_size=args.population_size,
            generations=args.generations,
            fitness_lambda=args.lambda_val
        )
        
        # Take the best candidate
        best_seq = ga_df.iloc[0]["sequence"]
        best_edits = ga_df.iloc[0]["edit_count"]
        
        # Evaluate using the Proxy Evaluator
        orig_mrl = float(proxy_wrapper([orig_seq])[0])
        gen_mrl = float(proxy_wrapper([best_seq])[0])
        
        records.append({
            "orig_seq": orig_seq,
            "orig_cnn_mrl": orig_mrl,
            "gen_seq": best_seq,
            "gen_cnn_mrl": gen_mrl,
            "hamming_dist": best_edits
        })
        
        # Periodic checkpointing every 10 sequences
        if idx > 0 and idx % 10 == 0:
            pd.DataFrame(records).to_csv(output_csv, index=False, compression="gzip")
            
    # Final save
    final_df = pd.DataFrame(records)
    final_df.to_csv(output_csv, index=False, compression="gzip")
    print(f"\nOptimization complete! Processed {len(final_df)} sequences.")
    print(f"Saved GA results to: {output_csv}")

if __name__ == "__main__":
    main()
