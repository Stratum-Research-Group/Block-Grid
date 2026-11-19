import os
import time
import torch
import torch.nn as nn
from blockvga.config import BlockGridConfig
from blockvga.model import BlockGridModel
from experiments.common import set_seed, count_parameters
from experiments.dataset import SyntheticBlockTransportDataset
from experiments.run_benchmark import get_stratified_eval_data, train_and_eval_model

def run_task_1b_benchmark():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Task 1b Benchmark: Inner Operator Transformation ===")
    print(f"Device: {device}")
    
    H, W, P, Q, F_dim, R = 16, 16, 8, 8, 4, 2
    steps = 1000
    batch_size = 32
    
    # Generate fixed, frozen full-rank random matrices A* and B*
    torch.manual_seed(1234)
    A_star = torch.randn(P, P)
    B_star = torch.randn(Q, Q)
    # Ensure well-conditioned / full rank via QR
    q_a, _ = torch.linalg.qr(A_star)
    q_b, _ = torch.linalg.qr(B_star)
    A_star = q_a
    B_star = q_b
    
    # 1. Training stream (bins 1-20 only)
    print("Generating Task 1b training dataset (bins 1-20)...")
    train_ds = SyntheticBlockTransportDataset(
        num_episodes=5000,
        task="1b",
        A_star=A_star,
        B_star=B_star,
        min_dist=1,
        max_dist=20,
        seed=200,
        H=H, W=W, P=P, Q=Q, F_dim=F_dim
    )
    
    # 2. Validation stratified bins
    print("Generating Task 1b validation bins...")
    val_bins = get_stratified_eval_data(
        num_episodes=500,
        task="1b",
        H=H, W=W, P=P, Q=Q, F_dim=F_dim,
        seed=888
    )
    # Inject A_star and B_star into val_bins datasets
    for _, ds in val_bins:
        ds.A_star = A_star
        ds.B_star = B_star
        
    results_1b = {}
    
    # Arm 1: BlockGrid with inner_op=False (Transport only, no internal transform)
    cfg_no_inner = BlockGridConfig(
        H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=32, d=128,
        tied_readback=False, slot_mix="planar_uncrossed",
        inner_op=False
    )
    model_no_inner = BlockGridModel(cfg_no_inner, num_layers=2)
    results_1b["BlockGrid (inner_op=False)"], _ = train_and_eval_model(
        "BlockGrid (inner_op=False)", model_no_inner, train_ds, val_bins,
        steps=steps, batch_size=batch_size, device=device
    )
    
    # Arm 2: BlockGrid with inner_op=True (Two-sided factorized L_k = A_k Z B_k^T)
    cfg_inner = BlockGridConfig(
        H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=32, d=128,
        tied_readback=False, slot_mix="planar_uncrossed",
        inner_op=True
    )
    model_inner = BlockGridModel(cfg_inner, num_layers=2)
    results_1b["BlockGrid (inner_op=True)"], _ = train_and_eval_model(
        "BlockGrid (inner_op=True)", model_inner, train_ds, val_bins,
        steps=steps, batch_size=batch_size, device=device
    )
    
    print("\n" + "="*80)
    print("TASK 1b BENCHMARK SUMMARY (Inner Operator Hypothesis Test)")
    print("="*80)
    header = f"{'Model':<30} | {'1-2':<8} | {'3-5':<8} | {'6-10':<8} | {'11-20':<8} | {'21-30 (Held-out)':<16} | {'Overall':<8}"
    print(header)
    print("-" * len(header))
    
    for name, res in results_1b.items():
        row = f"{name:<30} | {res['1-2']:.4f}   | {res['3-5']:.4f}   | {res['6-10']:.4f}   | {res['11-20']:.4f}    | {res['21-30']:.4f}           | {res['overall']:.4f}"
        print(row)
    print("="*80)

if __name__ == "__main__":
    run_task_1b_benchmark()
