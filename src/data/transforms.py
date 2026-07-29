import math
import torch
from torch_geometric.data import Data
try:
    import RNA
except ImportError:
    print("ViennaRNA python bindings not found. Structure prediction will fail.")

def get_structure_and_mfe(seq):
    structure, mfe = RNA.fold(seq)
    return structure, mfe

def structure_to_edges_with_attrs(structure, sequence):
    pair_table = RNA.ptable(structure)
    edges = []
    edge_attrs = []
    n = len(structure)
    for i in range(1, n + 1):
        j = pair_table[i]
        if j > i:
            edges.append((i - 1, j - 1))
            edge_attrs.append([0, 1])
            edges.append((j - 1, i - 1))
            edge_attrs.append([0, 1])
    for i in range(n - 1):
        edges.append((i, i + 1))
        edge_attrs.append([1, 0])
        edges.append((i + 1, i))
        edge_attrs.append([1, 0])
    return edges, edge_attrs

def sinusoidal_positional_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def one_hot_encode_with_positional(seq, pos_dim=6):
    vocab = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_len = len(seq)
    x = torch.zeros((seq_len, 4))
    for i, nt in enumerate(seq):
        if nt in vocab: 
            x[i, vocab[nt]] = 1.0
    pos_enc = sinusoidal_positional_encoding(seq_len, pos_dim)
    return torch.cat([x, pos_enc], dim=1)

def sequence_to_graph(utr, structure=None):
    if structure is None:
        structure, _ = get_structure_and_mfe(utr)
    edge_list, edge_attrs = structure_to_edges_with_attrs(structure, utr)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    x = one_hot_encode_with_positional(utr)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, utr=utr, structure=structure)

def add_graph_column(df):
    structures, graphs = [], []
    for utr in df["utr"]:
        graph = sequence_to_graph(utr)
        structures.append(graph.structure)
        graphs.append(graph)
    df = df.copy()
    df["structure"] = structures
    df["graph"] = graphs
    return df

def standardize_dataframe(df):
    """
    Standardize the dataframe columns to ensure 'utr', 'rl', and 'scaled_rl' exist.
    Handles inconsistent capitalization (e.g., 'UTR' vs 'utr').
    """
    df = df.copy()
    if 'UTR' in df.columns and 'utr' not in df.columns:
        df.rename(columns={'UTR': 'utr'}, inplace=True)
    return df

