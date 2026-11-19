import pytest
import torch
from blockvga.unified import UnifiedBlockGrid

def test_unified_hypercube_with_neighborhood():
    """
    Verify UnifiedBlockGrid in 4D hypercube mode with 1-bit Hamming neighborhood operator.
    """
    model = UnifiedBlockGrid(
        mode="hyper",
        grid_shape=(2, 2, 2, 2), # 4D hypercube, 16 vertices
        cell_tensor_shape=(2, 2), # 2x2 matrix cell
        K=4,
        d=16,
        use_neighborhood=True
    )

    X = torch.randn(2, 2, 2, 2, 2, 2, 2) # [B=2, 2, 2, 2, 2, P=2, Q=2]
    out = model(X)
    assert out.shape == X.shape
    assert not torch.isnan(out).any()

    # Backprop test
    loss = out.sum()
    loss.backward()
    assert model.neighborhood_op.w_self.weight.grad is not None
