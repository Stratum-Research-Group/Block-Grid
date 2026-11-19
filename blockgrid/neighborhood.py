import torch
import torch.nn as nn
from typing import Tuple, Optional

class HypercubeNeighborhoodOperator(nn.Module):
    r"""
    Hypercube Graph Neighborhood Operator:
    Performs nearest-neighbor 1-bit Hamming distance message passing across
    the D-dimensional hypercube $\{0, 1\}^D$.
    
    For vertex $v \in \{0, 1\}^D$, its degree is $D$, and its neighbors are
    $v \oplus 2^i$ for $i \in \{0, \dots, D-1\}$ (single bit flips).
    
    The operator computes:
        $$\mathcal{X}'_v = W_{self} \mathcal{X}_v + \sum_{i=0}^{D-1} W_i \mathcal{X}_{v \oplus 2^i}$$
    
    If parameter sharing is enabled (isotropic diffusion):
        $$\mathcal{X}'_v = W_{self} \mathcal{X}_v + W_{neighbor} \sum_{i=0}^{D-1} \mathcal{X}_{v \oplus 2^i}$$
    
    This provides exact local communication across hypercube edges in $O(D \cdot N)$
    time, complementing global landmark slot-mixing with high-resolution local message passing.
    """
    def __init__(
        self,
        D: int,
        cell_features: int,
        isotropic: bool = True,
        bias: bool = True
    ):
        super().__init__()
        self.D = D
        self.cell_features = cell_features
        self.isotropic = isotropic

        self.w_self = nn.Linear(cell_features, cell_features, bias=bias)
        if isotropic:
            self.w_neighbor = nn.Linear(cell_features, cell_features, bias=False)
        else:
            self.w_neighbors = nn.Parameter(torch.randn(D, cell_features, cell_features) * (1.0 / (cell_features ** 0.5)))
            if bias:
                self.neighbor_bias = nn.Parameter(torch.zeros(cell_features))
            else:
                self.register_parameter('neighbor_bias', None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, 2, 2, ..., 2, C] with D binary dimensions and C feature channels,
           OR [B, N, C] where N = 2^D.
        Returns:
            X_out: tensor of matching shape after hypercube neighborhood message passing.
        """
        is_flat = (X.dim() == 3)
        B = X.shape[0]
        C = X.shape[-1]
        N = 2 ** self.D

        if is_flat:
            assert X.shape[1] == N, f"Expected N={N} for D={self.D}, got {X.shape[1]}"
            # Reshape to [B, 2, 2, ..., 2, C]
            X_cube = X.view(B, *([2] * self.D), C)
        else:
            X_cube = X

        # 1. Self transformation
        out = self.w_self(X_cube)

        # 2. Hypercube bit-flip neighbor aggregation
        # In a hypercube tensor [B, 2_0, 2_1, ..., 2_{D-1}, C], flipping the i-th bit
        # corresponds to torch.flip(X_cube, dims=[1 + i]).
        if self.isotropic:
            neighbor_sum = torch.zeros_like(X_cube)
            for i in range(self.D):
                # Dimension 1 + i corresponds to hypercube axis i
                flipped = torch.flip(X_cube, dims=[1 + i])
                neighbor_sum += flipped
            out += self.w_neighbor(neighbor_sum)
        else:
            for i in range(self.D):
                flipped = torch.flip(X_cube, dims=[1 + i]) # [B, 2, ..., 2, C]
                # Contract with W_i: [C, C]
                W_i = self.w_neighbors[i] # [C, C]
                out += torch.matmul(flipped, W_i.T)
            if self.neighbor_bias is not None:
                out += self.neighbor_bias

        if is_flat:
            return out.view(B, N, C)
        return out
