import os
import time
import torch
import torch.nn as nn
from blockvga.config import BlockGridConfig
from blockvga.model import BlockGridModel
from baselines.full_attn import FullAttentionBaseline
from baselines.perceiver import PerceiverBaseline
from baselines.isab import ISABBaseline
from experiments.common import set_seed, count_parameters

def benchmark_scale(H: int, W: int, P: int = 4, Q: int = 4, F_dim: int = 4, R: int = 0, batch_size: int = 1, device: str = "cuda"):
    N = H * W
    print(f"\n=======================================================")
    print(f"STRESS TEST: Virtual Grid {H}x{W} (N = {N:,} Cells)")
    print(f"Total Block Dimension per Cell: {P}x{Q}x{F_dim} = {P*Q*F_dim} scalars")
    print(f"Total Grid Payload: {N * P * Q * F_dim:,} elements")
    print(f"=======================================================")
    
    models = [
        ("BlockGrid (Linear O(N*K))", lambda: BlockGridModel(
            BlockGridConfig(H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=32, d=64, slot_mix="planar_uncrossed"),
            num_layers=1
        )),
        ("ISAB (Linear O(N*K))", lambda: ISABBaseline(
            H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R, K=32, num_layers=1
        )),
        ("Perceiver IO (Linear O(N*K))", lambda: PerceiverBaseline(
            H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R, K=32, num_layers=1
        )),
        ("Full Attention (Quadratic O(N^2))", lambda: FullAttentionBaseline(
            H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R, num_layers=1
        )),
    ]
    
    results = []
    
    for name, model_fn in models:
        # Clear cache before each test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        try:
            model = model_fn().to(device)
            x = torch.randn(batch_size, H, W, P, Q, F_dim, device=device, requires_grad=True)
            
            # Warmup
            t0 = time.time()
            out = model(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) * 1000 # ms
            
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB
            print(f"  {name:<35}: PASSED | Peak VRAM: {peak_vram:>8.2f} MB | Latency: {elapsed:>8.2f} ms")
            results.append((name, f"{peak_vram:.2f} MB", f"{elapsed:.2f} ms", "PASSED"))
            
            # Clean up
            del model, x, out, loss
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  {name:<35}: OOM CRASH (Out of Memory)!")
            results.append((name, "OOM CRASH", "FAILED", "OOM"))
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:<35}: ERROR: {str(e)[:40]}")
            results.append((name, "ERROR", "FAILED", str(e)[:20]))
            torch.cuda.empty_cache()
            
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Massive Scaling Stress Test on: {device}")
    
    # Test scales:
    # Scale 1: 32x32 (1,024 cells)
    # Scale 2: 64x64 (4,096 cells)
    # Scale 3: 100x100 (10,000 cells)
    # Scale 4: 128x128 (16,384 cells)
    
    scales = [(32, 32), (64, 64), (100, 100), (128, 128)]
    all_scale_results = {}
    
    for H, W in scales:
        all_scale_results[f"{H}x{W}"] = benchmark_scale(H=H, W=W, device=device)
        
    print("\n" + "="*85)
    print(f"{'SCALING SUMMARY TABLE':^85}")
    print("="*85)
    header = f"{'Grid Size':<12} | {'Model':<30} | {'Status':<10} | {'Peak VRAM':<12} | {'Latency':<12}"
    print(header)
    print("-" * len(header))
    
    for grid_sz, res_list in all_scale_results.items():
        for name, vram, lat, status in res_list:
            print(f"{grid_sz:<12} | {name:<30} | {status:<10} | {vram:<12} | {lat:<12}")
        print("-" * len(header))

if __name__ == "__main__":
    main()
