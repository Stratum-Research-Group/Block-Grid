import pytest
import torch
from blockgrid.sparse_hyper_layer import SparseHyperGridConfig, SparseHyperBlockGrid

def test_sparse_hyper_33d_binary():
    """
    Test routing 1,024 active entities sampled across a 33-dimensional Boolean hypercube:
    Total coordinate locations: 2^33 = 8,589,934,592 cells.
    Cell payload: (4, 4, 4) multi-layer tensor.
    """
    D = 33
    M = 1024
    B = 2
    cfg = SparseHyperGridConfig(
        D=D,
        axis_radix=2,
        cell_shape=(4, 4, 4),
        K=16,
        d=32,
        inner_op=True
    )
    layer = SparseHyperBlockGrid(cfg)

    # Active 33D coordinates
    positions = torch.randint(0, 2, (B, M, D))
    # Active 3D tensor cell payloads
    X = torch.randn(B, M, 4, 4, 4)

    out = layer(X, positions)
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()

def test_sparse_hyper_22d_ternary():
    """
    Test routing 512 active entities across a 22-dimensional ternary space {-1, 0, +1}^22:
    Total coordinate locations: 3^22 = 31,381,059,609 cells!
    Cell payload: (8, 8) matrix.
    """
    D = 22
    M = 512
    B = 2
    cfg = SparseHyperGridConfig(
        D=D,
        axis_radix=3, # Ternary states 0, 1, 2
        cell_shape=(8, 8),
        K=16,
        d=32,
        inner_op=True
    )
    layer = SparseHyperBlockGrid(cfg)

    positions = torch.randint(0, 3, (B, M, D))
    X = torch.randn(B, M, 8, 8)

    out = layer(X, positions)
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU timing")
def test_sparse_hyper_64d_cuda():
    """
    Test extreme 64-dimensional space (2^64 locations) on GPU!
    """
    D = 64
    M = 2048
    B = 1
    cfg = SparseHyperGridConfig(
        D=D,
        axis_radix=2,
        cell_shape=(4, 4, 4),
        K=16,
        d=32,
        inner_op=True
    )
    layer = SparseHyperBlockGrid(cfg).cuda()

    positions = torch.randint(0, 2, (B, M, D), device="cuda")
    X = torch.randn(B, M, 4, 4, 4, device="cuda")

    out = layer(X, positions)
    assert out.shape == (B, M, 4, 4, 4)
    loss = out.sum()
    loss.backward()
