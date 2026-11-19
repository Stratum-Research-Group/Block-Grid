import pytest
import torch
from blockvga.neighborhood import HypercubeNeighborhoodOperator

def test_hypercube_neighborhood_operator_shapes():
    # Test D=4 hypercube (16 vertices), C=8 features
    D = 4
    C = 8
    B = 2
    N = 2 ** D

    op_iso = HypercubeNeighborhoodOperator(D=D, cell_features=C, isotropic=True)
    op_aniso = HypercubeNeighborhoodOperator(D=D, cell_features=C, isotropic=False)

    # Test flat shape [B, N, C]
    X_flat = torch.randn(B, N, C)
    out_iso = op_iso(X_flat)
    out_aniso = op_aniso(X_flat)

    assert out_iso.shape == (B, N, C)
    assert out_aniso.shape == (B, N, C)

    # Test hypercube shape [B, 2, 2, 2, 2, C]
    X_cube = torch.randn(B, 2, 2, 2, 2, C)
    out_cube_iso = op_iso(X_cube)
    assert out_cube_iso.shape == (B, 2, 2, 2, 2, C)

def test_hypercube_1_bit_communication():
    """
    Verify that an impulse at vertex (0, 0, 0) propagates strictly to its
    1-bit Hamming neighbors: (1, 0, 0), (0, 1, 0), and (0, 0, 1).
    """
    D = 3
    C = 1
    op = HypercubeNeighborhoodOperator(D=D, cell_features=C, isotropic=True, bias=False)
    # Set identity weights
    op.w_self.weight.data.fill_(1.0)
    op.w_neighbor.weight.data.fill_(1.0)

    # Impulse at (0, 0, 0)
    X = torch.zeros(1, 2, 2, 2, 1)
    X[0, 0, 0, 0, 0] = 1.0

    out = op(X)

    # Vertex (0, 0, 0) gets self weight 1.0
    assert torch.isclose(out[0, 0, 0, 0, 0], torch.tensor(1.0))

    # 1-bit neighbors: (1, 0, 0), (0, 1, 0), (0, 0, 1) get neighbor weight 1.0
    assert torch.isclose(out[0, 1, 0, 0, 0], torch.tensor(1.0))
    assert torch.isclose(out[0, 0, 1, 0, 0], torch.tensor(1.0))
    assert torch.isclose(out[0, 0, 0, 1, 0], torch.tensor(1.0))

    # 2-bit neighbors: (1, 1, 0), (1, 0, 1), (0, 1, 1) must be 0.0
    assert torch.isclose(out[0, 1, 1, 0, 0], torch.tensor(0.0))
    assert torch.isclose(out[0, 1, 0, 1, 0], torch.tensor(0.0))
    assert torch.isclose(out[0, 0, 1, 1, 0], torch.tensor(0.0))

    # 3-bit neighbor: (1, 1, 1) must be 0.0
    assert torch.isclose(out[0, 1, 1, 1, 0], torch.tensor(0.0))
