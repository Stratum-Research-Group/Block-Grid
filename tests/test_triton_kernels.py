import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton not installed")
from blockgrid.pooling import BlockPooling
from blockgrid.kernels import triton_pool, triton_readback

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_triton_pool_equivalence():
    B, H, W, P, Q, F_dim, K = 2, 8, 8, 4, 4, 4, 16
    V = torch.randn(B, H, W, P, Q, F_dim, device="cuda", dtype=torch.float32)
    w = torch.softmax(torch.randn(B, H, W, K, device="cuda", dtype=torch.float32), dim=-1)

    pooling = BlockPooling(normalize_slots=False)
    z_torch = pooling.pool(V, w)
    z_triton = triton_pool(V, w)

    diff = (z_torch - z_triton).abs().max().item()
    assert diff < 1e-4, f"Triton pool mismatch: max diff {diff} >= 1e-4"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_triton_readback_equivalence():
    B, H, W, P, Q, F_dim, K = 2, 8, 8, 4, 4, 4, 16
    Z = torch.randn(B, K, P, Q, F_dim, device="cuda", dtype=torch.float32)
    r = torch.softmax(torch.randn(B, H, W, K, device="cuda", dtype=torch.float32), dim=-1)

    pooling = BlockPooling(normalize_slots=False)
    y_torch = pooling.readback(Z, r)
    y_triton = triton_readback(Z, r)

    diff = (y_torch - y_triton).abs().max().item()
    assert diff < 1e-4, f"Triton readback mismatch: max diff {diff} >= 1e-4"
