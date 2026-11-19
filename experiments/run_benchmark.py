import os
import time
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any

from blockvga.config import BlockGridConfig
from blockvga.model import BlockGridModel
from baselines import (
    ISABBaseline,
    PerceiverBaseline,
    AxialAttentionBaseline,
    LinearAttentionBaseline,
    FullAttentionBaseline,
    LocalConvBaseline
)
from experiments.common import set_seed, count_parameters
from experiments.dataset import SyntheticBlockTransportDataset

def get_stratified_eval_data(num_episodes: int, task: str, H: int, W: int, P: int, Q: int, F_dim: int, seed: int):
    """
    Generate fixed evaluation dataset with stratified distance bins:
    bins: {1-2, 3-5, 6-10, 11-20, 21-30}
    """
    bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 30)]
    episodes_per_bin = num_episodes // len(bins)
    
    datasets = []
    for b_idx, (min_d, max_d) in enumerate(bins):
        ds = SyntheticBlockTransportDataset(
            num_episodes=episodes_per_bin,
            task=task,
            min_dist=min_d,
            max_dist=max_d,
            seed=seed + b_idx * 1000,
            H=H, W=W, P=P, Q=Q, F_dim=F_dim
        )
        datasets.append((f"{min_d}-{max_d}", ds))
    return datasets

def evaluate_on_bins(model, bin_datasets, device="cuda", batch_size=32):
    model.eval()
    criterion = nn.MSELoss()
    bin_results = {}
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for bin_name, ds in bin_datasets:
            bin_loss = 0.0
            n = len(ds)
            for i in range(0, n, batch_size):
                cur_b = min(batch_size, n - i)
                X_list, rf_list, dst_list, target_list = [], [], [], []
                for k in range(cur_b):
                    X, rf, (di, dj), target, _ = ds[i + k]
                    X_list.append(X)
                    rf_list.append(rf)
                    dst_list.append((di, dj))
                    target_list.append(target)
                
                X_b = torch.stack(X_list).to(device)
                rf_b = torch.stack(rf_list).to(device)
                target_b = torch.stack(target_list).to(device)
                
                out = model(X_b, routing_features=rf_b)
                dst_preds = torch.stack([out[b, dst_list[b][0], dst_list[b][1]] for b in range(cur_b)])
                loss = criterion(dst_preds, target_b)
                bin_loss += loss.item() * cur_b
                
            avg_bin_loss = bin_loss / n
            bin_results[bin_name] = avg_bin_loss
            total_loss += bin_loss
            total_samples += n
            
    bin_results["overall"] = total_loss / total_samples
    return bin_results

def train_and_eval_model(name: str, model: nn.Module, train_ds, val_bins, steps=1000, lr=3e-4, wd=0.01, clip=1.0, batch_size=32, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    
    params = count_parameters(model)
    print(f"\n--- Training {name} (Params: {params:,}) ---")
    start_t = time.time()
    
    model.train()
    n_train = len(train_ds)
    for step in range(steps):
        optimizer.zero_grad()
        X_list, rf_list, dst_list, target_list = [], [], [], []
        # Random batch sample
        indices = torch.randint(0, n_train, (batch_size,))
        for idx in indices:
            X, rf, (di, dj), target, _ = train_ds[idx.item()]
            X_list.append(X)
            rf_list.append(rf)
            dst_list.append((di, dj))
            target_list.append(target)
            
        X_b = torch.stack(X_list).to(device)
        rf_b = torch.stack(rf_list).to(device)
        target_b = torch.stack(target_list).to(device)
        
        out = model(X_b, routing_features=rf_b)
        dst_preds = torch.stack([out[b, dst_list[b][0], dst_list[b][1]] for b in range(batch_size)])
        loss = criterion(dst_preds, target_b)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if (step + 1) % 250 == 0 or step == steps - 1:
            print(f"Step {step+1}/{steps} - Train MSE: {loss.item():.4f}")
            
    dur = time.time() - start_t
    print(f"Training completed in {dur:.2f}s. Evaluating on fixed validation bins...")
    results = evaluate_on_bins(model, val_bins, device=device, batch_size=batch_size)
    return results, params

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Benchmark on: {device}")
    
    H, W, P, Q, F_dim, R = 16, 16, 8, 8, 4, 2
    steps = 1000
    batch_size = 32
    
    # 1. Training stream (bins 1-20 only)
    print("Generating training dataset (bins 1-20)...")
    train_ds = SyntheticBlockTransportDataset(
        num_episodes=5000,
        task="1a",
        min_dist=1,
        max_dist=20,
        seed=100,
        H=H, W=W, P=P, Q=Q, F_dim=F_dim
    )
    
    # 2. Fixed validation bins (100 episodes per bin for fast pilot benchmark)
    print("Generating validation stratified bins...")
    val_bins = get_stratified_eval_data(
        num_episodes=500,
        task="1a",
        H=H, W=W, P=P, Q=Q, F_dim=F_dim,
        seed=999
    )
    
    all_results = {}
    
    # Arm 1: BlockGrid (Default untied readback, free G)
    cfg_free = BlockGridConfig(H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=32, d=128, tied_readback=False, slot_mix="free")
    model_free = BlockGridModel(cfg_free, num_layers=2)
    all_results["BlockGrid (Free G)"], _ = train_and_eval_model(
        "BlockGrid (Free G)", model_free, train_ds, val_bins, steps=steps, batch_size=batch_size, device=device
    )
    
    # Arm 2: BlockGrid (Planar Uncrossed G = I - UV^T)
    cfg_planar = BlockGridConfig(H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=32, d=128, tied_readback=False, slot_mix="planar_uncrossed")
    model_planar = BlockGridModel(cfg_planar, num_layers=2)
    all_results["BlockGrid (Planar Uncrossed)"], _ = train_and_eval_model(
        "BlockGrid (Planar Uncrossed)", model_planar, train_ds, val_bins, steps=steps, batch_size=batch_size, device=device
    )
    
    # Baseline B1: ISAB
    b1_isab = ISABBaseline(H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R, K=32, num_layers=2)
    all_results["B1 ISAB"], _ = train_and_eval_model(
        "B1 ISAB", b1_isab, train_ds, val_bins, steps=steps, batch_size=batch_size, device=device
    )
    
    # Baseline B2: Perceiver
    b2_perceiver = PerceiverBaseline(H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R, K=32, num_layers=2)
    all_results["B2 Perceiver"], _ = train_and_eval_model(
        "B2 Perceiver", b2_perceiver, train_ds, val_bins, steps=steps, batch_size=batch_size, device=device
    )
    
    # Baseline B5: Full Attention
    b5_full = FullAttentionBaseline(H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R, num_layers=2)
    all_results["B5 Full Attention"], _ = train_and_eval_model(
        "B5 Full Attention", b5_full, train_ds, val_bins, steps=steps, batch_size=batch_size, device=device
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT BENCHMARK SUMMARY (Validation MSE by Distance Bin)")
    print("="*80)
    header = f"{'Model':<30} | {'1-2':<8} | {'3-5':<8} | {'6-10':<8} | {'11-20':<8} | {'21-30 (Held-out)':<16} | {'Overall':<8}"
    print(header)
    print("-" * len(header))
    
    for name, res in all_results.items():
        row = f"{name:<30} | {res['1-2']:.4f}   | {res['3-5']:.4f}   | {res['6-10']:.4f}   | {res['11-20']:.4f}    | {res['21-30']:.4f}           | {res['overall']:.4f}"
        print(row)
    print("="*80)

if __name__ == "__main__":
    main()
