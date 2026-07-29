import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Dict, Any, Callable

from src.data.dataset import seq_to_idx
from src.data.transforms import sequence_to_graph
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
    elites_list = []
    seen = set()
    for idx in ranked_idx:
        candidate = population[idx]
        key = candidate.tobytes()
        if key not in seen:
            elites_list.append(candidate.copy())
            seen.add(key)
        if len(elites_list) == count: 
            break
    return np.asarray(elites_list, dtype=np.uint8)


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
