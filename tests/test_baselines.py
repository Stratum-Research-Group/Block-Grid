import pytest
import torch
from baselines import (
    ISABBaseline,
    PerceiverBaseline,
    AxialAttentionBaseline,
    LinearAttentionBaseline,
    FullAttentionBaseline,
    LocalConvBaseline
)

@pytest.mark.parametrize("baseline_cls", [
    ISABBaseline,
    PerceiverBaseline,
    AxialAttentionBaseline,
    LinearAttentionBaseline,
    FullAttentionBaseline,
    LocalConvBaseline
])
def test_baseline_io_shapes(baseline_cls):
    H, W, P, Q, F_dim, R = 4, 4, 2, 2, 4, 2
    model = baseline_cls(H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R)
    
    X = torch.randn(2, H, W, P, Q, F_dim)
    rf = torch.randn(2, H, W, R)
    out = model(X, routing_features=rf)
    
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()
