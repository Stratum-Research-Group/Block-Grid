import torch
import torch.nn as nn
from baselines.common import BaselineBase

class ISABBlock(nn.Module):
    """
    Induced Set Attention Block (Set Transformer, Lee et al. 2019).
    N cells -> K inducing points -> N cells.
    """
    def __init__(self, dim: int, num_heads: int = 4, num_inducing: int = 32):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mab2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B = x.shape[0]
        I = self.inducing_points.expand(B, -1, -1) # [B, K, C]
        
        # MAB1: I queries x
        h, _ = self.mab1(I, x, x)
        H = self.norm1(I + h) # [B, K, C]
        
        # MAB2: x queries H
        out, _ = self.mab2(x, H, H)
        out = self.norm2(x + out)
        
        out = out + self.mlp(out)
        return out

class ISABBaseline(BaselineBase):
    def __init__(self, H: int, W: int, P: int, Q: int, F_dim: int, R: int = 0, K: int = 32, num_layers: int = 2):
        super().__init__(H, W, P, Q, F_dim, R)
        dim = self.payload_dim
        self.blocks = nn.ModuleList([
            ISABBlock(dim=dim, num_inducing=K) for _ in range(num_layers)
        ])

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        h = self.prepare_input(X, routing_features=routing_features, positions=positions)
        for block in self.blocks:
            h = block(h)
        return self.restore_output(h)
