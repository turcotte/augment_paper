#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
import os
import sys

import time
import torch
import numpy as np
import pandas as pd

# Allow absolute imports from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import TangCNNRegressor
from src.models.gat import GATRegression
from src.models.ga import run_genetic_algorithm, cnn_predict_mrl_unscaled, gat_predict_mrl_unscaled


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Calibrate lambda for Genetic Algorithm vs RL")
    parser.add_argument("--calibration_set", type=Path, required=True, help="Path to the calibration dataset (CSV)")
    parser.add_argument("--predictor_dir", type=str, required=True, help="Directory containing the predictor model and scaler")
    parser.add_argument("--proxy_dir", type=str, required=True, help="Directory containing the proxy model and scaler")
    parser.add_argument("--predictor", type=str, choices=["cnn", "gat"], required=True, help="Which model architecture to use as optimizer")
    parser.add_argument("--proxy_predictor", type=str, choices=["cnn", "gat"], required=True, help="Proxy model architecture for evaluation")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save calibration results (e.g. results/GSM3130435_egfp_unmod_1/read_count_standard/ga/calibration-100)")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length of the model")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.001875, 0.01875, 0.0375, 0.075, 0.15, 0.3, 0.6], help="Lambda values to sweep")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    # 1. Load Calibration Dataset
    print(f"Loading calibration dataset from {args.calibration_set}...")
    df_cal = pd.read_csv(args.calibration_set)
    seq_col = 'utr' if 'utr' in df_cal.columns else 'sequence'
    if seq_col not in df_cal.columns:
        seq_col = 'UTR'
    if seq_col not in df_cal.columns and 'orig' in df_cal.columns:
        seq_col = 'orig'
    
    n_samples = len(df_cal)

    # 2. (Removed RL metrics computation)

    # 3. Load Models
    print(f"Loading {args.predictor.upper()} Optimizer...")
    model_path = os.path.join(args.predictor_dir, f"{args.predictor}_model.pth")
    scaler_path = os.path.join(args.predictor_dir, f"{args.predictor}_scaler.json")
    
    with open(scaler_path, "r") as f:
        scaler_dict = json.load(f)
        
    if args.predictor == "cnn":
        model = TangCNNRegressor(sequence_length=args.seq_len).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        predict_wrapper = lambda seqs: cnn_predict_mrl_unscaled(model, seqs, device, scaler_dict)
    elif args.predictor == "gat":
        model = GATRegression(in_channels=10, edge_dim=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        predict_wrapper = lambda seqs: gat_predict_mrl_unscaled(model, seqs, device, scaler_dict)

    print(f"Loading {args.proxy_predictor.upper()} Proxy Evaluator...")
    proxy_model_path = os.path.join(args.proxy_dir, f"{args.proxy_predictor}_model.pth")
    proxy_scaler_path = os.path.join(args.proxy_dir, f"{args.proxy_predictor}_scaler.json")
    
    with open(proxy_scaler_path, "r") as f:
        proxy_scaler_dict = json.load(f)
        
    if args.proxy_predictor == "cnn":
        proxy_model = TangCNNRegressor(sequence_length=args.seq_len).to(device)
        proxy_model.load_state_dict(torch.load(proxy_model_path, map_location=device, weights_only=True))
        proxy_wrapper = lambda seqs: cnn_predict_mrl_unscaled(proxy_model, seqs, device, proxy_scaler_dict)
    elif args.proxy_predictor == "gat":
        proxy_model = GATRegression(in_channels=10, edge_dim=2).to(device)
        proxy_model.load_state_dict(torch.load(proxy_model_path, map_location=device, weights_only=True))
        proxy_wrapper = lambda seqs: gat_predict_mrl_unscaled(proxy_model, seqs, device, proxy_scaler_dict)

    # 4. Sweep lambdas
    results = []
    raw_data_records = []

    print(f"Starting GA sweep over lambdas: {args.lambdas}")
    
    for lmbda in args.lambdas:
        print(f"\n--- Testing lambda = {lmbda} ---")
        lmbda_edits = []
        lmbda_deltas = []
        
        for idx, row in df_cal.iterrows():
            if idx > 0 and idx % 10 == 0:
                print(f"  Processed {idx}/{len(df_cal)} sequences...")
            orig_seq = str(row[seq_col]).upper().replace("U", "T")
            target_pred_raw = float(predict_wrapper([orig_seq])[0])
            
            ga_df, _ = run_genetic_algorithm(
                predict_fn=predict_wrapper,
                target_seq=orig_seq,
                target_pred_raw=target_pred_raw,
                population_size=256,
                generations=100,
                fitness_lambda=lmbda
            )
            
            best_seq = ga_df.iloc[0]["sequence"]
            best_edits = ga_df.iloc[0]["edit_count"]
            
            # Evaluate with proxy
            orig_mrl_proxy = float(proxy_wrapper([orig_seq])[0])
            best_mrl_proxy = float(proxy_wrapper([best_seq])[0])
            delta_mrl = best_mrl_proxy - orig_mrl_proxy
            
            lmbda_edits.append(best_edits)
            lmbda_deltas.append(delta_mrl)
            
            raw_data_records.append({
                "lambda": lmbda,
                "orig_seq": orig_seq,
                "gen_seq": best_seq,
                "orig_mrl_proxy": orig_mrl_proxy,
                "gen_mrl_proxy": best_mrl_proxy,
                "delta_mrl": delta_mrl,
                "edit_count": best_edits
            })
            
            pd.DataFrame(raw_data_records).to_csv(args.output_dir / "calibration_raw_data.csv", index=False)
            
        avg_edits = np.mean(lmbda_edits)
        std_edits = np.std(lmbda_edits)
        avg_delta = np.mean(lmbda_deltas)
        std_delta = np.std(lmbda_deltas)
        print(f"Result for lambda={lmbda}: Avg Edits={avg_edits:.2f} ± {std_edits:.2f}, Avg Delta MRL={avg_delta:.4f} ± {std_delta:.4f}")
        
        results.append({
            "lambda": lmbda,
            "avg_edits": float(avg_edits),
            "std_edits": float(std_edits),
            "avg_delta_mrl": float(avg_delta),
            "std_delta_mrl": float(std_delta)
        })

    summary_csv_path = args.output_dir / "calibration_summary.csv"
    pd.DataFrame(results).to_csv(summary_csv_path, index=False)
    print(f"\nSaved GA calibration summary to {summary_csv_path}")
    
    # 6. Save JSON Metadata
    execution_time = time.time() - start_time
    json_output = {
        "execution_time_seconds": execution_time,
        "calibration_set": str(args.calibration_set),
        "predictor": args.predictor,
        "proxy_predictor": args.proxy_predictor,
        "seq_len": args.seq_len,
        "lambdas": args.lambdas,
        "device": str(device)
    }
    summary_json_path = args.output_dir / "calibrate_ga_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved metadata summary to {summary_json_path}")

if __name__ == "__main__":
    main()
