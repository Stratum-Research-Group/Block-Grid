import pytest
import torch
from blockvga.tensor_cell_config import GeneralizedHyperTensorConfig
from blockvga.hyper_tensor_layer import GeneralizedHyperTensorLayer

def test_hyper_tensor_3d_grid_with_3d_tensors():
    """
    Outer grid: 2 x 2 x 2 hypercube (D=3, N=8 cells)
    Inside EACH cell: a 3D tensor of shape (2, 2, 2)
    Total tensor shape: [B=2, 2, 2, 2, 2, 2, 2]
    """
    cfg = GeneralizedHyperTensorConfig(
        outer_grid=(2, 2, 2),
        cell_tensor_shape=(2, 2, 2),
        K=4, d=16,
        inner_transform=True
    )
    layer = GeneralizedHyperTensorLayer(cfg)
    X = torch.randn(2, 2, 2, 2, 2, 2, 2) # [B, S1, S2, S3, T1, T2, T3]
    out = layer(X)
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()

def test_hyper_tensor_4d_grid_with_matrix_cells():
    """
    Outer grid: 2 x 2 x 2 x 2 hypercube (D=4, N=16 cells)
    Inside EACH cell: a (4, 4) matrix tensor
    """
    cfg = GeneralizedHyperTensorConfig(
        outer_grid=(2, 2, 2, 2),
        cell_tensor_shape=(4, 4),
        K=8, d=16,
        inner_transform=True
    )
    layer = GeneralizedHyperTensorLayer(cfg)
    X = torch.randn(2, 2, 2, 2, 2, 4, 4)
    out = layer(X)
    assert out.shape == X.shape
    loss = out.sum()
    loss.backward()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 2^20 hypercube")
def test_hyper_tensor_d20_grid_with_matrix_cells_cuda():
    """
    Outer grid: 20-dimensional hypercube (2^20 = 1,048,576 cells)
    Inside EACH cell: a (2, 2) tensor
    Running live on GPU.
    """
    cfg = GeneralizedHyperTensorConfig(
        outer_grid=(2,) * 20,
        cell_tensor_shape=(2, 2),
        K=16, d=32,
        inner_transform=True
    )
    layer = GeneralizedHyperTensorLayer(cfg).cuda()
    # Flattened format: [B=1, N=1048576, T1=2, T2=2]
    X = torch.randn(1, 1048576, 2, 2, device="cuda")
    out = layer(X)
    assert out.shape == (1, 1048576, 2, 2)
