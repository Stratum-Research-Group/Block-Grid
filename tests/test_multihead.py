import pytest
import torch
from blockgrid import BlockGridConfig, BlockGridLayer

def test_multihead_routing_simplex():
    # 4 heads, K=32 -> 8 slots per head
    num_heads = 4
    K = 32
    cfg = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, K=K, num_heads=num_heads, d=32)
    layer = BlockGridLayer(cfg)
    
    X = torch.randn(2, 4, 4, 2, 2, 4)
    w_write, w_read = layer.assignment(X)
    
    # Check that each head sums to 1.0
    w_heads = w_write.view(2, 4, 4, num_heads, 8)
    head_sums = w_heads.sum(dim=-1)
    
    diff = (head_sums - 1.0).abs().max().item()
    assert diff < 1e-5, f"Multi-head simplex failed: head sums differ from 1.0 by {diff}"

def test_multihead_forward_backward():
    cfg = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, K=32, num_heads=4, d=32, inner_op=True)
    layer = BlockGridLayer(cfg)
    
    X = torch.randn(2, 4, 4, 2, 2, 4, requires_grad=True)
    out = layer(X)
    
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()
    assert X.grad is not None and X.grad.norm().item() > 0
