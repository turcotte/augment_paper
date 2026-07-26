import argparse
import os
import json
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.gat import GATRegression
from src.models.autoencoder import ARLSTMDecoder, GATAutoEncoder
from src.models.cnn import TangCNNRegressor
from src.data.transforms import sequence_to_graph

def parse_args():
    parser = argparse.ArgumentParser(description="REINFORCE Fine-tuning of Autoencoder")
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--train_set", type=str, required=True, help="Path to train set partition")
    parser.add_argument("--test_set", type=str, required=True, help="Path to test set partition")
    parser.add_argument("--autoencoder_dir", type=str, default="results/GSM3130435_egfp_unmod_1", help="Dir with autoencoder_model.pth")
    parser.add_argument("--cnn_dir", type=str, default="results/GSM3130435_egfp_unmod_1", help="Dir with cnn_model.pth")
    parser.add_argument("--output_dir", type=str, default="results/GSM3130435_egfp_unmod_1/reinforce", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=50, help="Fixed length of input sequences")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Decoder")
    parser.add_argument("--train_subset_size", type=int, default=50000, help="Size of stratified training subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu, auto)")
    return parser.parse_args()

def evaluate_cnn_mrl(model, cnn_model, dataloader, device):
    """Evaluate MRL shift and Hamming distance using CNN Oracle on the test set."""
    model.eval()
    cnn_model.eval()
    
    orig_mrls = []
    gen_mrls = []
    hamming_dists = []
    all_orig_seqs = []
    all_gen_seqs = []
    
    with torch.no_grad():
        for latents, mrls, orig_seqs in tqdm(dataloader, desc="Evaluating with CNN Oracle"):
            latents = latents.to(device)
            mrls = mrls.to(device)
            
            # Generate new sequences
            logits = model.decoder(latents, mrls)
            tokens = logits.argmax(dim=-1)
            gen_seqs = [''.join(['ACGT'[idx.item()] for idx in row]) for row in tokens]
            
            # Predict MRL for generated sequences using CNN
            # CNN expects (batch, seq_len, 4) one-hot
            arr = np.zeros((len(gen_seqs), 50, 4), dtype=np.float32)
            vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            for i, s in enumerate(gen_seqs):
                for j, nt in enumerate(s[:50]):
                    if nt in vocab:
                        arr[i, j, vocab[nt]] = 1.0
            
            _, out_mrl = cnn_model(torch.tensor(arr).to(device))
            gen_mrls.extend(out_mrl.cpu().numpy().flatten())
            
            # Predict MRL for original sequences using CNN
            arr_orig = np.zeros((len(orig_seqs), 50, 4), dtype=np.float32)
            for i, s in enumerate(orig_seqs):
                for j, nt in enumerate(s[:50]):
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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Load Data and stratify
    print(f"Loading raw data from {args.raw_data}...")
    df = pd.read_csv(args.raw_data)
    
    print("Loading partitions...")
    train_utrs = pd.read_csv(args.train_set)['utr']
    test_utrs = pd.read_csv(args.test_set)['utr']
    
    train_df = df[df['utr'].isin(train_utrs)].reset_index(drop=True)
    test_df = df[df['utr'].isin(test_utrs)].reset_index(drop=True)
    
    print(f"Original Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Subsetting Train to 50k stratified by MRL
    train_df = train_df.copy()
    train_df["mrl_bin"] = pd.qcut(train_df["rl"], q=10, labels=False, duplicates="drop")
    train_subset_df = (
        train_df
        .groupby("mrl_bin", group_keys=False)
        .apply(lambda x: x.sample(frac=min(1.0, args.train_subset_size / len(train_df)), random_state=args.seed))
        .reset_index(drop=True)
    )
    print(f"Stratified Train Subset: {len(train_subset_df)}")
    
    # 2. Load Autoencoder
    print("Loading pre-trained Autoencoder...")
    encoder = GATRegression(in_channels=10, edge_dim=2, hidden_channels=128)
    decoder = ARLSTMDecoder(latent_dim=128*2, hidden_dim=256, seq_len=args.seq_len)
    model = GATAutoEncoder(encoder, decoder).to(device)
    model.load_state_dict(torch.load(os.path.join(args.autoencoder_dir, "autoencoder_model.pth"), map_location=device))
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    
    # 3. Precompute Latents and predicted MRLs for the subsets
    def get_latents_mrls(df_split, desc):
        latents = []
        mrls = []
        seqs = []
        
        for i in tqdm(range(0, len(df_split), args.batch_size), desc=desc):
            batch_seqs = df_split["utr"].iloc[i:i+args.batch_size].values
            graphs = Batch.from_data_list(
                [sequence_to_graph(s) for s in batch_seqs]
            ).to(device)
            
            with torch.no_grad():
                z, m = model.encoder(graphs)
            
            latents.append(z.cpu())
            mrls.append(m.cpu())
            seqs.extend(batch_seqs)
            
        return torch.cat(latents), torch.cat(mrls), seqs

    l_train, m_train, _ = get_latents_mrls(train_subset_df, "Precomputing Train Subset Latents")
    train_loader = DataLoader(TensorDataset(l_train, m_train), batch_size=args.batch_size, shuffle=True)
    
    # For evaluation, we pass the original sequences as well
    l_test, m_test, s_test = get_latents_mrls(test_df, "Precomputing Test Set Latents")
    
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, latents, mrls, seqs):
            self.latents = latents
            self.mrls = mrls
            self.seqs = seqs
        def __len__(self):
            return len(self.latents)
        def __getitem__(self, idx):
            return self.latents[idx], self.mrls[idx], self.seqs[idx]
            
    test_loader = DataLoader(TestDataset(l_test, m_test, s_test), batch_size=args.batch_size, shuffle=False)
    
    # 4. Load CNN Oracle
    print("Loading pre-trained CNN Oracle...")
    cnn_model = TangCNNRegressor(sequence_length=args.seq_len).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(args.cnn_dir, "cnn_model.pth"), map_location=device))
    cnn_model.eval()
    
    # Evaluate Baseline Before Fine-tuning
    print("\nEvaluating Zero-Shot Autoencoder (Before REINFORCE)...")
    orig_mrl, gen_mrl, avg_edits, _ = evaluate_cnn_mrl(model, cnn_model, test_loader, device)
    print(f"Before - Orig MRL: {orig_mrl:.4f}, Gen MRL: {gen_mrl:.4f}, Avg Edits: {avg_edits:.2f}")

    # 5. REINFORCE Fine-Tuning
    print("\nStarting REINFORCE Fine-tuning...")
    opt = optim.Adam(model.decoder.parameters(), lr=args.lr)
    baseline = 0.0
    history = []
    
    start_time = time.time()
    for epoch in range(args.epochs):
        model.decoder.train()
        epoch_rewards = []
        epoch_loss = []
        
        for latents, mrls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            latents = latents.to(device)
            mrls = mrls.to(device)
            
            opt.zero_grad()
            
            tokens, log_probs = model.decoder.sample_with_log_probs(latents, mrls)
            seqs = [''.join(['ACGT'[idx.item()] for idx in row]) for row in tokens]
            
            graphs = Batch.from_data_list(
                [sequence_to_graph(s) for s in seqs]
            ).to(device)
            
            with torch.no_grad():
                _, rewards = model.encoder(graphs)
                
            loss = (-log_probs.sum(dim=1) * (rewards.squeeze() - baseline).detach()).mean()
            loss.backward()
            opt.step()
            
            batch_reward = rewards.mean().item()
            baseline = 0.9 * baseline + 0.1 * batch_reward
            
            epoch_rewards.append(batch_reward)
            epoch_loss.append(loss.item())
            
        print(f"Epoch {epoch+1} - Avg Reward: {np.mean(epoch_rewards):.4f} - Baseline: {baseline:.4f}")
        history.append({
            "epoch": epoch + 1,
            "avg_reward": float(np.mean(epoch_rewards)),
            "baseline": float(baseline),
            "loss": float(np.mean(epoch_loss))
        })
        
    training_time = time.time() - start_time
    
    # 6. Final Evaluation
    print("\nEvaluating Fine-tuned Autoencoder (After REINFORCE)...")
    orig_mrl, post_gen_mrl, post_avg_edits, df_results = evaluate_cnn_mrl(model, cnn_model, test_loader, device)
    print(f"After - Orig MRL: {orig_mrl:.4f}, Gen MRL: {post_gen_mrl:.4f}, Avg Edits: {post_avg_edits:.2f}")
    
    # 7. Serialization
    df_results.to_csv(os.path.join(args.output_dir, "reinforce_optimized_sequences.csv.gz"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "reinforce_loss_curve.csv"), index=False)
    
    model_out_path = os.path.join(args.output_dir, "reinforce_autoencoder.pth")
    torch.save(model.state_dict(), model_out_path)
    
    summary = {
        "model": "REINFORCE_Autoencoder",
        "epochs": args.epochs,
        "train_subset_size": args.train_subset_size,
        "training_time_seconds": training_time,
        "test_orig_cnn_mrl": float(orig_mrl),
        "test_gen_cnn_mrl_before": float(gen_mrl),
        "test_gen_cnn_mrl_after": float(post_gen_mrl),
        "mrl_shift": float(post_gen_mrl - orig_mrl),
        "avg_hamming_edits": float(post_avg_edits)
    }
    with open(os.path.join(args.output_dir, "reinforce_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Training complete. Model saved to {model_out_path}")

if __name__ == "__main__":
    main()
