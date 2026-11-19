import torch
import torch.nn as nn
from baselines.common import BaselineBase

class FullAttentionBlock(nn.Module):
    """
    Standard full pairwise self-attention over N cells (O(N^2) complexity).
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.self_attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.mlp(x))
        return x

class FullAttentionBaseline(BaselineBase):
    def __init__(self, H: int, W: int, P: int, Q: int, F_dim: int, R: int = 0, num_layers: int = 2):
        super().__init__(H, W, P, Q, F_dim, R)
        dim = self.payload_dim
        self.blocks = nn.ModuleList([
            FullAttentionBlock(dim=dim) for _ in range(num_layers)
        ])

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        h = self.prepare_input(X, routing_features=routing_features, positions=positions)
        for block in self.blocks:
            h = block(h)
        return self.restore_output(h)
