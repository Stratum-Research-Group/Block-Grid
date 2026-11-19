import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from blockvga.config import BlockGridConfig
from blockvga.layer import BlockGridLayer
from blockvga.tensor_cell_config import GeneralizedHyperTensorConfig
from blockvga.hyper_tensor_layer import GeneralizedHyperTensorLayer
from blockvga.neighborhood import HypercubeNeighborhoodOperator

class UnifiedBlockGrid(nn.Module):
    """
    Unified BlockGrid Switch Interface:
    Supports:
      - mode='hyper' (DEFAULT): Arbitrary D-dimensional hypercube/hyper-torus
                                (e.g. 2x2x2 up to 2^22) with intact multi-linear
                                cell tensors at each vertex.
      - mode='2d': Ultra-fast planar grid (H x W) for 2D spatial layouts.
    """
    def __init__(
        self,
        mode: str = "hyper",                                # "hyper" (default) or "2d"
        grid_shape: Tuple[int, ...] = (2,) * 22,            # default 22D hypercube (2^22 = 4,194,304 vertices)
        cell_tensor_shape: Tuple[int, ...] = (2, 2),        # default (2, 2) matrix inside each cell
        K: int = 16,                                        # landmark slots
        num_heads: int = 1,                                 # multi-head routing
        d: int = 32,                                        # coordinate embedding dimension
        slot_mix: str = "planar_uncrossed",                 # "planar_uncrossed", "free", "symmetric"
        inner_transform: bool = True,                       # factorized tensor contractions inside cells
        tied_readback: bool = False,                        # untied read/write routing
        use_neighborhood: bool = False,                     # 1-bit Hamming distance neighborhood operator
    ):
        super().__init__()
        self.mode = mode.lower()
        self.grid_shape = grid_shape
        self.cell_tensor_shape = cell_tensor_shape
        self.use_neighborhood = use_neighborhood

        if self.mode == "hyper":
            self.hyper_config = GeneralizedHyperTensorConfig(
                outer_grid=grid_shape,
                cell_tensor_shape=cell_tensor_shape,
                K=K,
                num_heads=num_heads,
                d=d,
                slot_mix=slot_mix,
                inner_transform=inner_transform,
                tied_readback=tied_readback,
            )
            self.core = GeneralizedHyperTensorLayer(self.hyper_config)
            if self.use_neighborhood and all(s == 2 for s in grid_shape):
                self.neighborhood_op = HypercubeNeighborhoodOperator(
                    D=len(grid_shape),
                    cell_features=self.hyper_config.cell_volume,
                    isotropic=True
                )
            else:
                self.neighborhood_op = None
            self._2d_layer = None

        elif self.mode == "2d":
            if len(grid_shape) != 2:
                raise ValueError(f"mode='2d' requires a 2D grid_shape (H, W), got {grid_shape}")
            if len(cell_tensor_shape) != 2:
                # Interpret as (P, Q) with F=1
                P, Q = cell_tensor_shape[0], cell_tensor_shape[1] if len(cell_tensor_shape) > 1 else 1
                F_dim = 1
            else:
                P, Q = cell_tensor_shape[0], cell_tensor_shape[1]
                F_dim = 1

            self._2d_config = BlockGridConfig(
                H=grid_shape[0],
                W=grid_shape[1],
                P=P,
                Q=Q,
                F=F_dim,
                K=K,
                num_heads=num_heads,
                d=d,
                slot_mix=slot_mix,
                inner_op=inner_transform,
                tied_readback=tied_readback,
            )
            self._2d_layer = BlockGridLayer(self._2d_config)
            self.core = None
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'hyper' or '2d'.")

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        if self.mode == "hyper":
            out = self.core(X, positions=positions)
            if self.neighborhood_op is not None:
                # out is [B, S_1, ..., S_D, T_1, ..., T_M]
                # reshape cell tensor to flat feature for neighborhood operator
                orig_shape = out.shape
                B = orig_shape[0]
                D = len(self.grid_shape)
                cell_shape = orig_shape[1 + D:]
                V_cell = self.hyper_config.cell_volume
                
                # [B, 2, ..., 2, V_cell]
                out_flat_cells = out.view(B, *self.grid_shape, V_cell)
                out_neigh = self.neighborhood_op(out_flat_cells)
                out = out_neigh.view(*orig_shape)
            return out
        else:
            # 2D layer expects [B, H, W, P, Q, F]
            # Ensure input matches 2D layer dims
            if X.dim() == 4: # [B, H, W, P*Q]
                B, H, W, _ = X.shape
                X = X.view(B, H, W, self._2d_config.P, self._2d_config.Q, self._2d_config.F)
            elif X.dim() == 5: # [B, H, W, P, Q]
                X = X.unsqueeze(-1)
            out = self._2d_layer(X, routing_features=routing_features, positions=positions)
            return out.squeeze(-1) if self.cell_tensor_shape == (self._2d_config.P, self._2d_config.Q) else out
