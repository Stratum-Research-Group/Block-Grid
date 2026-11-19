import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton not installed")
from blockvga.autograd_kernel import triton_fused_autograd_pool

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_fused_autograd_pool_gradients():
    B, N, K, C = 2, 64, 8, 16
    V = torch.randn(B, N, C, device="cuda", dtype=torch.float32, requires_grad=True)
    W = torch.randn(B, N, K, device="cuda", dtype=torch.float32, requires_grad=True)

    # 1. PyTorch native baseline
    Z_ref = torch.einsum('bnc,bnk->bkc', V, W)
    loss_ref = Z_ref.sum()
    loss_ref.backward()
    grad_v_ref = V.grad.clone()
    grad_w_ref = W.grad.clone()

    V.grad.zero_()
    W.grad.zero_()

    # 2. Triton fused autograd operator
    Z_triton = triton_fused_autograd_pool(V, W)
    loss_triton = Z_triton.sum()
    loss_triton.backward()

    # Verify forward match
    diff_fwd = (Z_ref - Z_triton).abs().max().item()
    assert diff_fwd < 1e-4, f"Forward mismatch: {diff_fwd}"

    # Verify backward gradient dV match
    diff_dv = (grad_v_ref - V.grad).abs().max().item()
    assert diff_dv < 1e-4, f"Backward dV mismatch: {diff_dv}"

    # Verify backward gradient dW match
    diff_dw = (grad_w_ref - W.grad).abs().max().item()
    assert diff_dw < 1e-4, f"Backward dW mismatch: {diff_dw}"
