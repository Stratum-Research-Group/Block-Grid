import pytest
import torch
from blockvga.hyper_config import HyperGridConfig
from blockvga.hyper_layer import HyperGridLayer

def test_hypercube_d3():
    # 2 x 2 x 2 hypercube (D=3, N=8 vertices)
    cfg = HyperGridConfig(
        grid_shape=(2, 2, 2),
        P=4, Q=4, F=2,
        K=8, d=16,
        inner_op=True
    )
    layer = HyperGridLayer(cfg)
    X = torch.randn(2, 2, 2, 2, 4, 4, 2)
    out = layer(X)
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()

def test_hypercube_d10():
    # 10-dimensional hypercube: (2,)*10 = 2^10 = 1,024 vertices
    cfg = HyperGridConfig(
        grid_shape=(2,) * 10,
        P=2, Q=2, F=2,
        K=16, d=32,
        inner_op=True
    )
    layer = HyperGridLayer(cfg)
    X = torch.randn(1, 1024, 2, 2, 2) # flattened input format
    out = layer(X)
    assert out.shape == (1, 1024, 2, 2, 2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for D=20")
def test_hypercube_d20_cuda():
    # 20-dimensional hypercube: 2^20 = 1,048,576 vertices on GPU
    cfg = HyperGridConfig(
        grid_shape=(2,) * 20,
        P=2, Q=2, F=2,
        K=16, d=32,
        inner_op=False
    )
    layer = HyperGridLayer(cfg).cuda()
    X = torch.randn(1, 1048576, 2, 2, 2, device="cuda")
    out = layer(X)
    assert out.shape == (1, 1048576, 2, 2, 2)
