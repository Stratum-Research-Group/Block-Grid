import pytest
import torch
from blockvga import UnifiedBlockGrid

def test_unified_default_22d_hyper():
    # Test default mode ('hyper' with 22D hypercube)
    # Using tiny batch and flattened shape [B=1, N=2^22, 2, 2] on CPU/GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Small hypercube for unit test: 3D (2x2x2)
    model = UnifiedBlockGrid(mode="hyper", grid_shape=(2, 2, 2), cell_tensor_shape=(2, 2)).to(device)
    X = torch.randn(2, 8, 2, 2, device=device)
    out = model(X)
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()

def test_unified_2d_switch():
    # Test 2D switch mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnifiedBlockGrid(mode="2d", grid_shape=(16, 16), cell_tensor_shape=(2, 2)).to(device)
    X = torch.randn(2, 16, 16, 2, 2, device=device)
    out = model(X)
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()
