import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def seq_to_idx(seq, max_len):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    idxs = torch.full((max_len,), fill_value=0, dtype=torch.long)
    for i, nt in enumerate(seq[:max_len]):
        if nt in mapping: 
            idxs[i] = mapping[nt]
    return idxs

class RNAGraphSeqDataset(Dataset):
    """
    PyTorch Geometric Dataset for RNA sequences and their secondary structure graphs.
    """
    def __init__(self, df, max_len=50):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.targets = torch.tensor(self.df["scaled_rl"].values, dtype=torch.float32).view(-1, 1)
        self.sequences = self.df["utr"].tolist()
        self.graphs = self.df["graph"].tolist()
        
    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        data = self.graphs[idx].clone()
        data.y = self.targets[idx]
        data.target_seq = seq_to_idx(self.sequences[idx], self.max_len)
        return data

class RNATensorDataset(Dataset):
    """
    Standard PyTorch Dataset for RNA sequences without graph structure (e.g. for CNN).
    """
    def __init__(self, df, max_len=50):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.targets = torch.tensor(self.df["scaled_rl"].values, dtype=torch.float32).view(-1, 1)
        self.sequences = self.df["utr"].tolist()
        
    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        target = self.targets[idx]
        # One-hot encoding logic should ideally be applied here or passed in pre-computed
        seq_idx = seq_to_idx(self.sequences[idx], self.max_len)
        return seq_idx, target
