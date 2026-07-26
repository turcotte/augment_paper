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
    parser = argparse.ArgumentParser(description="Curriculum DAP Fine-tuning of Autoencoder")
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--train_set", type=str, required=True, help="Path to train set partition")
    parser.add_argument("--test_set", type=str, required=True, help="Path to test set partition")
    parser.add_argument("--autoencoder_dir", type=str, default="results/GSM3130435_egfp_unmod_1", help="Dir with autoencoder_model.pth")
    parser.add_argument("--cnn_dir", type=str, default="results/GSM3130435_egfp_unmod_1", help="Dir with cnn_model.pth")
    parser.add_argument("--output_dir", type=str, default="results/GSM3130435_egfp_unmod_1/curriculum_dap", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=50, help="Fixed length of input sequences")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per bin")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Decoder")
    parser.add_argument("--sigma", type=float, default=60.0, help="Weighting factor for the reward against the prior")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of quantile bins for the curriculum")
    parser.add_argument("--train_subset_size", type=int, default=None, help="Size of stratified training subset (None to use full train split)")
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
            
            logits = model.decoder(latents, mrls)
            tokens = logits.argmax(dim=-1)
            gen_seqs = [''.join(['ACGT'[idx.item()] for idx in row]) for row in tokens]
            
            arr = np.zeros((len(gen_seqs), 50, 4), dtype=np.float32)
            vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            for i, s in enumerate(gen_seqs):
                for j, nt in enumerate(s[:50]):
                    if nt in vocab:
                        arr[i, j, vocab[nt]] = 1.0
            
            _, out_mrl = cnn_model(torch.tensor(arr).to(device))
            gen_mrls.extend(out_mrl.cpu().numpy().flatten())
            
            arr_orig = np.zeros((len(orig_seqs), 50, 4), dtype=np.float32)
            for i, s in enumerate(orig_seqs):
                for j, nt in enumerate(s[:50]):
                    if nt in vocab:
                        arr_orig[i, j, vocab[nt]] = 1.0
            _, out_orig_mrl = cnn_model(torch.tensor(arr_orig).to(device))
            orig_mrls.extend(out_orig_mrl.cpu().numpy().flatten())
            
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
    
    print(f"Loading raw data from {args.raw_data}...")
    df = pd.read_csv(args.raw_data)
    
    print("Loading partitions...")
    train_utrs = pd.read_csv(args.train_set)['utr']
    test_utrs = pd.read_csv(args.test_set)['utr']
    
    train_df = df[df['utr'].isin(train_utrs)].reset_index(drop=True)
    test_df = df[df['utr'].isin(test_utrs)].reset_index(drop=True)
    
    train_df = train_df.copy()
    train_df["mrl_bin"] = pd.qcut(train_df["rl"], q=args.num_bins, labels=False, duplicates="drop")
    
    if args.train_subset_size is not None and args.train_subset_size < len(train_df):
        train_df = (
            train_df
            .groupby("mrl_bin")
            .sample(frac=args.train_subset_size / len(train_df), random_state=args.seed)
            .reset_index(drop=True)
        )
    print(f"Train Size: {len(train_df)}")
    
    print("Loading pre-trained Autoencoder (Agent and Prior)...")
    encoder = GATRegression(in_channels=10, edge_dim=2, hidden_channels=128)
    decoder = ARLSTMDecoder(latent_dim=128*2, hidden_dim=256, seq_len=args.seq_len)
    model = GATAutoEncoder(encoder, decoder).to(device)
    
    prior_decoder = ARLSTMDecoder(latent_dim=128*2, hidden_dim=256, seq_len=args.seq_len).to(device)
    
    model_path = os.path.join(args.autoencoder_dir, "autoencoder_model.pth")
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

    # In Curriculum DAP, we precompute latents for the entire train set up front
    print("Precomputing Train Latents...")
    l_train, m_train, _ = get_latents_mrls(train_df, "Precomputing Train Subset Latents")
    
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
    
    cnn_model = TangCNNRegressor(sequence_length=args.seq_len).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(args.cnn_dir, "cnn_model.pth"), map_location=device))
    # Evaluate Baseline Before Fine-tuning
    print("\nEvaluating Zero-Shot Autoencoder (Before Curriculum DAP)...")
    orig_mrl, gen_mrl, avg_edits, _ = evaluate_cnn_mrl(model, cnn_model, test_loader, device)
    print(f"Before - Orig MRL: {orig_mrl:.4f}, Gen MRL: {gen_mrl:.4f}, Avg Edits: {avg_edits:.2f}")

    print("\nStarting Curriculum DAP Fine-tuning...")
    opt = optim.Adam(model.decoder.parameters(), lr=args.lr)
    history = []
    
    bin_values = sorted(train_df["mrl_bin"].dropna().unique())
    
    start_time = time.time()
    for bin_position, current_bin in enumerate(bin_values):
        # Subset indices for current bin
        bin_indices = train_df[train_df["mrl_bin"] == current_bin].index
        l_bin = l_train[bin_indices]
        m_bin = m_train[bin_indices]
        
        current_train_loader = DataLoader(
            TensorDataset(l_bin, m_bin),
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Calculate threshold
        if bin_position < len(bin_values) - 1:
            next_bin = bin_values[bin_position + 1]
            next_bin_indices = train_df[train_df["mrl_bin"] == next_bin].index
            m_next_bin = m_train[next_bin_indices]
            threshold = np.percentile(m_next_bin.numpy(), 80)
        else:
            threshold = np.percentile(m_bin.numpy(), 90)
            
        print(f"\n===== Curriculum Bin {current_bin} | Threshold: {threshold:.4f} =====")
        
        for epoch in range(args.epochs):
            model.decoder.train()
            epoch_rewards = []
            epoch_loss = []
            
            for latents, mrls in tqdm(current_train_loader, desc=f"Bin {current_bin} Epoch {epoch+1}"):
                latents = latents.to(device)
                mrls = mrls.to(device)
                
                opt.zero_grad()
                
                sampled_tokens, log_probs = model.decoder.sample_with_log_probs(latents, mrls)
                logP_agent = log_probs.sum(dim=1)
                
                seqs = [''.join(['ACGT'[idx.item()] for idx in row]) for row in sampled_tokens]
                
                graphs = Batch.from_data_list(
                    [sequence_to_graph(s) for s in seqs]
                ).to(device)
                
                with torch.no_grad():
                    _, predicted_mrls = model.encoder(graphs)
                    
                predicted_mrls = predicted_mrls.squeeze()
                
                # Binary reward based on threshold
                reward = (predicted_mrls.detach() >= threshold).float()
                
                with torch.no_grad():
                    log_probs_prior = prior_decoder.evaluate_log_probs(latents, mrls, sampled_tokens)
                    logP_prior = log_probs_prior.sum(dim=1)
                    
                logP_augmented = logP_prior + args.sigma * reward
                
                loss = F.mse_loss(logP_agent, logP_augmented.detach())
                loss.backward()
                opt.step()
                
                epoch_rewards.append(reward.mean().item())
                epoch_loss.append(loss.item())
                
            print(f"Bin {current_bin} Epoch {epoch+1} - Success Rate: {np.mean(epoch_rewards):.4f} - Loss: {np.mean(epoch_loss):.4f}")
            history.append({
                "bin": int(current_bin),
                "epoch": epoch + 1,
                "threshold": float(threshold),
                "success_rate": float(np.mean(epoch_rewards)),
                "loss": float(np.mean(epoch_loss))
            })
            
    training_time = time.time() - start_time
    # 6. Final Evaluation
    print("\nEvaluating Fine-tuned Autoencoder (After Curriculum DAP)...")
    orig_mrl, post_gen_mrl, post_avg_edits, df_results = evaluate_cnn_mrl(model, cnn_model, test_loader, device)
    print(f"After - Orig MRL: {orig_mrl:.4f}, Gen MRL: {post_gen_mrl:.4f}, Avg Edits: {post_avg_edits:.2f}")
    
    # 7. Serialization
    df_results.to_csv(os.path.join(args.output_dir, "curriculum_dap_optimized_sequences.csv.gz"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "curriculum_dap_loss_curve.csv"), index=False)
    
    model_out_path = os.path.join(args.output_dir, "curriculum_dap_autoencoder.pth")
    torch.save(model.state_dict(), model_out_path)
    
    summary = {
        "model": "Curriculum_DAP_Autoencoder",
        "epochs_per_bin": args.epochs,
        "num_bins": args.num_bins,
        "sigma": args.sigma,
        "train_subset_size": len(train_df),
        "training_time_seconds": training_time,
        "test_orig_cnn_mrl": float(orig_mrl),
        "test_gen_cnn_mrl_before": float(gen_mrl),
        "test_gen_cnn_mrl_after": float(post_gen_mrl),
        "mrl_shift": float(post_gen_mrl - orig_mrl),
        "avg_hamming_edits": float(post_avg_edits)
    }
    with open(os.path.join(args.output_dir, "curriculum_dap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Training complete. Model saved to {model_out_path}")

if __name__ == "__main__":
    main()
