import os
import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model: nn.Module, sample_x: torch.Tensor, rf=None, pos=None) -> int:
    """
    Simplified dense FLOPs estimator based on spec Section 1.
    """
    params = count_parameters(model)
    # 2 ops per param * batch * grid cells
    B, H, W, P, Q, F_dim = sample_x.shape
    N = H * W
    return 2 * params * N

def memory_preflight(model: nn.Module, batch_size: int, H: int, W: int, P: int, Q: int, F_dim: int, R: int, device: str = "cpu"):
    """
    Section 4 Memory preflight:
    (a) run ONE allocated forward/backward at tiny batch;
    (b) measure peak memory.
    """
    model.to(device)
    x = torch.randn(batch_size, H, W, P, Q, F_dim, device=device, requires_grad=True)
    rf = torch.randn(batch_size, H, W, R, device=device) if R > 0 else None
    
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        
    out = model(x, routing_features=rf)
    loss = out.sum()
    loss.backward()
    
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = peak_bytes / (1024 * 1024)
    else:
        peak_mb = 0.0 # CPU
        
    model.zero_grad()
    return peak_mb
