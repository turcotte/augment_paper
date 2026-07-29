import math
import numpy as np
from ViennaRNA import RNA

# =================================================================
# Biological Metrics (from Siavash Thesis)
# =================================================================

def total_sequence_entropy(seq):
    if not isinstance(seq, str): return np.nan
    probs = [seq.count(base) / len(seq) for base in "ACGT"]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def shannon_entropy_window(seq, window_size=10):
    if not isinstance(seq, str): return np.nan
    seq = seq.upper()
    if len(seq) < window_size:
        return np.nan
    entropies = []
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i+window_size]
        probs = np.array([window.count(nt) / window_size for nt in "ACGT"])
        entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
        entropies.append(entropy)
    return np.mean(entropies)

def gc_content(seq):
    if not isinstance(seq, str): return np.nan
    return (seq.count('G') + seq.count('C')) / len(seq)

def count_uaugs(seq):
    if not isinstance(seq, str): return 0
    seq = seq.upper()
    count = 0
    for i in range(len(seq) - 2):
        if seq[i:i+3] == "ATG":
            count += 1
    return count

def count_uorfs(seq):
    if not isinstance(seq, str): return 0
    seq = seq.upper()
    stop_codons = {"TAA", "TAG", "TGA"}
    count = 0
    for i in range(len(seq) - 2):
        if seq[i:i+3] == "ATG":
            for j in range(i + 3, len(seq) - 2, 3):
                if seq[j:j+3] in stop_codons:
                    count += 1
                    break
    return count

def mfe_region(seq, start_idx, end_idx):
    """Calculate Minimum Free Energy for a specific sequence region."""
    if not isinstance(seq, str): return np.nan
    subseq = seq[start_idx:end_idx]
    if len(subseq) == 0:
        return 0.0
    _, mfe = RNA.fold(subseq)
    return float(mfe)

def count_are_motifs(seq):
    """Count canonical ARE motifs inserted by the optimization."""
    if not isinstance(seq, str): return 0
    motifs = ["ATTTA", "TTATT", "ATTTAA"]
    return sum(seq.count(m) for m in motifs)
