import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from src.models.gat import GATRegression
from src.models.autoencoder import ARLSTMDecoder, GATAutoEncoder
from src.data.transforms import sequence_to_graph

class TestDataset(Dataset):
    def __init__(self, latents, mrls, seqs):
        self.latents = latents
        self.mrls = mrls
        self.seqs = seqs
    def __len__(self):
        return len(self.latents)
    def __getitem__(self, idx):
        return self.latents[idx], self.mrls[idx], self.seqs[idx]

def load_agent_and_prior(autoencoder_dir, seq_len, device):
    """Loads the pre-trained autoencoder to act as the RL agent, and its decoder as the prior."""
    encoder = GATRegression(in_channels=10, edge_dim=2, hidden_channels=128)
    decoder = ARLSTMDecoder(latent_dim=128*2, hidden_dim=256, seq_len=seq_len)
    model = GATAutoEncoder(encoder, decoder).to(device)
    
    prior_decoder = ARLSTMDecoder(latent_dim=128*2, hidden_dim=256, seq_len=seq_len).to(device)
    
    model_path = os.path.join(autoencoder_dir, "autoencoder_model.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    decoder_state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}
    prior_decoder.load_state_dict(decoder_state_dict)
    
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    
    for param in prior_decoder.parameters():
        param.requires_grad = False
    prior_decoder.eval()
    
    return model, prior_decoder

def get_latents_mrls(model, df_split, batch_size, device, desc):
    """Precomputes graph latents and MRLs using the frozen encoder."""
    latents = []
    mrls = []
    seqs = []
    
    print(f"{desc}...")
    for i in range(0, len(df_split), batch_size):
        batch_seqs = df_split["utr"].iloc[i:i+batch_size].values
        graphs = Batch.from_data_list(
            [sequence_to_graph(s) for s in batch_seqs]
        ).to(device)
        
        with torch.no_grad():
            z, m = model.encoder(graphs)
        
        latents.append(z.cpu())
        mrls.append(m.cpu())
        seqs.extend(batch_seqs)
        
    return torch.cat(latents), torch.cat(mrls), seqs

def evaluate_cnn_oracle(model, cnn_model, dataloader, device, seq_len):
    """Evaluate MRL shift and Hamming distance using CNN Oracle on the test set."""
    model.eval()
    cnn_model.eval()
    
    orig_mrls = []
    gen_mrls = []
    hamming_dists = []
    all_orig_seqs = []
    all_gen_seqs = []
    
    print("Evaluating with CNN Oracle...")
    with torch.no_grad():
        for latents, mrls, orig_seqs in dataloader:
            latents = latents.to(device)
            mrls = mrls.to(device)
            
            # Generate new sequences
            logits = model.decoder(latents, mrls)
            tokens = logits.argmax(dim=-1)
            gen_seqs = [''.join(['ACGT'[idx.item()] for idx in row]) for row in tokens]
            
            # Predict MRL for generated sequences using CNN
            # CNN expects (batch, seq_len, 4) one-hot
            arr = np.zeros((len(gen_seqs), seq_len, 4), dtype=np.float32)
            vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            for i, s in enumerate(gen_seqs):
                for j, nt in enumerate(s[:seq_len]):
                    if nt in vocab:
                        arr[i, j, vocab[nt]] = 1.0
            
            _, out_mrl = cnn_model(torch.tensor(arr).to(device))
            gen_mrls.extend(out_mrl.cpu().numpy().flatten())
            
            # Predict MRL for original sequences using CNN
            arr_orig = np.zeros((len(orig_seqs), seq_len, 4), dtype=np.float32)
            for i, s in enumerate(orig_seqs):
                for j, nt in enumerate(s[:seq_len]):
                    if nt in vocab:
                        arr_orig[i, j, vocab[nt]] = 1.0
            _, out_orig_mrl = cnn_model(torch.tensor(arr_orig).to(device))
            orig_mrls.extend(out_orig_mrl.cpu().numpy().flatten())
            
            # Calculate Hamming and track sequences
            for o, g in zip(orig_seqs, gen_seqs):
                hamming_dists.append(sum(c1 != c2 for c1, c2 in zip(o, g)))
                all_orig_seqs.append(o)
                all_gen_seqs.append(g)
                
    df_results = pd.DataFrame({
        "orig_seq": all_orig_seqs,
        "gen_seq": all_gen_seqs,
        "orig_cnn_mrl": orig_mrls,
        "gen_cnn_mrl": gen_mrls,
        "hamming_dist": hamming_dists
    })
    return np.mean(orig_mrls), np.mean(gen_mrls), np.mean(hamming_dists), df_results
