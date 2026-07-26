#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Tuple, List, Dict, Any, Callable

# Allow absolute imports from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.models.cnn import TangCNNRegressor
from src.models.gat import GATRegression
from src.data.dataset import seq_to_idx
from src.data.transforms import sequence_to_graph, get_structure_and_mfe
from torch_geometric.data import Batch

NUCLEOTIDE_ALPHABET = "ACGT"
NUCLEOTIDES = np.array(list(NUCLEOTIDE_ALPHABET))

# ---------------------------------------------------------
# Evolutionary Operators
# ---------------------------------------------------------

def compute_fitness(
    population: np.ndarray, 
    target_seq_array: np.ndarray, 
    pred_raw: np.ndarray, 
    target_pred_raw: float, 
    fitness_lambda: float
) -> Tuple[np.ndarray, np.ndarray]:
    edit_counts = np.count_nonzero(population != target_seq_array, axis=1)
    delta_mrl_raw = pred_raw - target_pred_raw
    fitness = delta_mrl_raw - (fitness_lambda * edit_counts)
    return fitness.astype(np.float32), edit_counts.astype(np.int32)

def select_parents_tournament(population: np.ndarray, fitness: np.ndarray, tournament_size: int) -> np.ndarray:
    pop_size = population.shape[0]
    selected_indices = np.empty(pop_size, dtype=np.int32)
    for i in range(pop_size):
        participants = np.random.choice(pop_size, size=tournament_size, replace=False)
        selected_indices[i] = participants[np.argmax(fitness[participants])]
    return population[selected_indices]

def uniform_crossover(parents: np.ndarray, crossover_rate: float) -> np.ndarray:
    np.random.shuffle(parents)
    offspring = []
    num_parents, seq_length = parents.shape
    for i in range(0, num_parents, 2):
        p1, p2 = parents[i], parents[(i + 1) % num_parents]
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < crossover_rate:
            mask = np.random.randint(0, 2, size=seq_length).astype(bool)
            c1[mask], c2[mask] = p2[mask], p1[mask]
        offspring.extend([c1, c2])
    return np.asarray(offspring[:num_parents], dtype=np.uint8)

def mutate_population(offspring: np.ndarray, mutation_rate: float) -> np.ndarray:
    mask = np.random.rand(*offspring.shape) < mutation_rate
    offsets = np.random.randint(1, 4, size=offspring.shape, dtype=np.uint8)
    mutated = offspring.copy()
    mutated[mask] = (mutated[mask] + offsets[mask]) % len(NUCLEOTIDES)
    return mutated

def get_unique_elites(population: np.ndarray, fitness: np.ndarray, count: int) -> np.ndarray:
    ranked_idx = np.argsort(fitness)[::-1]
    elites, seen = [], set()
    for idx in ranked_idx:
        candidate = population[idx]
        key = candidate.tobytes()
        if key not in seen:
            elites.append(candidate.copy())
            seen.add(key)
        if len(elites) == count: 
            break
    return np.asarray(elites, dtype=np.uint8)

# ---------------------------------------------------------
# Inference Wrappers
# ---------------------------------------------------------

def cnn_predict_mrl_unscaled(model, seqs: List[str], device, scaler_dict, batch_size=256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            seq_len = len(batch_seqs[0])
            idxs = torch.stack([seq_to_idx(s, seq_len) for s in batch_seqs])
            x = torch.nn.functional.one_hot(idxs, num_classes=4).float().to(device)
            _, pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            
    preds = np.array(preds)
    preds = (preds * scaler_dict["std"]) + scaler_dict["mean"]
    return preds

def gat_predict_mrl_unscaled(model, seqs: List[str], device, scaler_dict, batch_size=64) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            data_list = []
            for s in batch_seqs:
                data_list.append(sequence_to_graph(s))
            batch = Batch.from_data_list(data_list).to(device)
            _, pred = model(batch)
            preds.extend(pred.cpu().numpy().flatten())
            
    preds = np.array(preds)
    preds = (preds * scaler_dict["std"]) + scaler_dict["mean"]
    return preds

# ---------------------------------------------------------
# Core Algorithm Logic
# ---------------------------------------------------------

def run_genetic_algorithm(
    predict_fn: Callable[[List[str]], np.ndarray],
    target_seq: str,
    target_pred_raw: float,
    population_size: int,
    generations: int,
    fitness_lambda: float
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:

    target_array = np.array([NUCLEOTIDE_ALPHABET.index(n) for n in target_seq], dtype=np.uint8)
    
    population = np.random.randint(0, 4, size=(population_size, len(target_seq)), dtype=np.uint8)
    population[0] = target_array
    
    history = []
    
    for gen in range(1, generations + 1):
        seqs = ["".join(NUCLEOTIDES[row]) for row in population]
        pred_raw = predict_fn(seqs)
        fitness, edit_counts = compute_fitness(
            population, target_array, pred_raw, target_pred_raw, fitness_lambda
        )
        
        best_idx = int(np.argmax(fitness))
        history.append({
            "generation": gen, 
            "best_fitness": float(fitness[best_idx]),
            "best_raw_mrl": float(pred_raw[best_idx])
        })

        elites = get_unique_elites(population, fitness, count=16)
        parents = select_parents_tournament(population, fitness, tournament_size=3)
        offspring = uniform_crossover(parents, crossover_rate=0.8)
        offspring = mutate_population(offspring, mutation_rate=0.02)
        
        population = np.vstack((elites, offspring))[:population_size]

    final_seqs = ["".join(NUCLEOTIDES[row]) for row in population]
    final_preds_raw = predict_fn(final_seqs)
    final_fitness, final_edits = compute_fitness(
        population, target_array, final_preds_raw, target_pred_raw, fitness_lambda
    )
    
    results_df = pd.DataFrame({
        "sequence": final_seqs,
        "fitness": final_fitness,
        "mrl_opt": final_preds_raw,
        "edit_count": final_edits
    }).drop_duplicates(subset=["sequence"]).sort_values(by="fitness", ascending=False)
    
    return results_df, history

# ---------------------------------------------------------
# CLI Router
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for MRL optimization")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model and scaler")
    parser.add_argument("--predictor", type=str, choices=["cnn", "gat"], required=True, help="Which model architecture to use")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument("--target_index", type=int, required=True, help="Row index of the sequence to optimize")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length of the model")
    parser.add_argument("--population_size", type=int, default=256, help="GA population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to run")
    parser.add_argument("--fitness_lambda", type=float, default=0.02, help="Penalty multiplier for edits")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--proxy_predictor", type=str, choices=["cnn", "gat"], required=True, help="Proxy model architecture")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
        
    model_path = os.path.join(args.model_dir, f"{args.predictor}_model.pth")
    scaler_path = os.path.join(args.model_dir, f"{args.predictor}_scaler.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler JSON not found at {scaler_path}")
        
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

    # Load proxy model (assumed to be in the same model_dir)
    proxy_model_path = os.path.join(args.model_dir, f"{args.proxy_predictor}_model.pth")
    proxy_scaler_path = os.path.join(args.model_dir, f"{args.proxy_predictor}_scaler.json")
    
    if not os.path.exists(proxy_model_path):
        raise FileNotFoundError(f"Proxy model checkpoint not found at {proxy_model_path}")
    if not os.path.exists(proxy_scaler_path):
        raise FileNotFoundError(f"Proxy scaler JSON not found at {proxy_scaler_path}")
        
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
        
    df = pd.read_csv(args.data_path)
    seq_col = 'utr' if 'utr' in df.columns else 'sequence'
    if seq_col not in df.columns:
        seq_col = 'UTR'
        
    target_seq = str(df.iloc[args.target_index][seq_col]).upper().replace("U", "T")
    target_pred_raw = float(predict_wrapper([target_seq])[0])
    
    print(f"Optimizing Target {args.target_index} using {args.predictor.upper()}...")
    print(f"Target Sequence: {target_seq}")
    print(f"Initial Raw MRL: {target_pred_raw:.4f}")
    
    results_df, history = run_genetic_algorithm(
        predict_fn=predict_wrapper,
        target_seq=target_seq,
        target_pred_raw=target_pred_raw,
        population_size=args.population_size,
        generations=args.generations,
        fitness_lambda=args.fitness_lambda
    )
    
    print(f"Running Proxy Evaluation using {args.proxy_predictor.upper()}...")
    eval_seqs = results_df["sequence"].tolist()
    eval_preds_raw = proxy_wrapper(eval_seqs)
    results_df["mrl_eval"] = eval_preds_raw
    
    output_dir = os.path.join(args.model_dir, "ga_optimization", f"target_{args.target_index}")
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)
    results_df.head(10).to_csv(os.path.join(output_dir, "top_candidates.csv"), index=False)
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        summary_dict = {
            "target_sequence": target_seq, 
            "predictor": args.predictor,
            "proxy_predictor": args.proxy_predictor,
            "generations": args.generations,
            "initial_raw_mrl": target_pred_raw,
            "best_mrl_opt": float(results_df.iloc[0]["mrl_opt"]),
            "best_mrl_eval": float(results_df.iloc[0]["mrl_eval"]),
            "best_fitness": float(results_df.iloc[0]["fitness"])
        }
        json.dump(summary_dict, f, indent=2)
        
    print(f"Optimization complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
