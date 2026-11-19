import pytest
import torch
from blockgrid import BlockGridConfig, BlockGridLayer

def test_shapes_basic():
    # P == Q
    cfg = BlockGridConfig(H=4, W=4, P=3, Q=3, F=8, K=16, d=32)
    layer = BlockGridLayer(cfg)
    x = torch.randn(2, 4, 4, 3, 3, 8)
    out = layer(x)
    assert out.shape == x.shape

def test_shapes_asymmetric_pq():
    # P != Q
    cfg = BlockGridConfig(H=4, W=4, P=2, Q=5, F=6, K=16, d=32, inner_op=True)
    layer = BlockGridLayer(cfg)
    x = torch.randn(2, 4, 4, 2, 5, 6)
    out = layer(x)
    assert out.shape == x.shape

def test_shapes_routing_features():
    # R > 0
    cfg = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, R=3, K=16, d=32)
    layer = BlockGridLayer(cfg)
    x = torch.randn(2, 4, 4, 2, 2, 4)
    rf = torch.randn(2, 4, 4, 3)
    out = layer(x, routing_features=rf)
    assert out.shape == x.shape

def test_shapes_explicit_positions():
    cfg = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, K=16, d=32)
    layer = BlockGridLayer(cfg)
    x = torch.randn(2, 4, 4, 2, 2, 4)
    grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
    positions = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).expand(2, -1, -1, -1)
    out = layer(x, positions=positions)
    assert out.shape == x.shape

def test_shapes_invalid_float_positions():
    cfg = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, K=16, d=32)
    layer = BlockGridLayer(cfg)
    x = torch.randn(2, 4, 4, 2, 2, 4)
    positions = torch.randn(2, 4, 4, 2)
    with pytest.raises(ValueError, match="integer coordinate indices"):
        layer(x, positions=positions)
