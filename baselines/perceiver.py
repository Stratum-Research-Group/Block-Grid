import torch
import torch.nn as nn
from baselines.common import BaselineBase

class PerceiverBlock(nn.Module):
    """
    Perceiver IO architecture (Jaegle et al. 2021).
    Cross-attention into latents, latent self-attention, cross-attention back to outputs.
    """
    def __init__(self, dim: int, num_latents: int = 32, num_heads: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.cross_attn_in = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.latent_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_out = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B = x.shape[0]
        L = self.latents.expand(B, -1, -1) # [B, K, C]
        
        # 1. Encode: latents query inputs
        h, _ = self.cross_attn_in(L, x, x)
        L = self.norm1(L + h)
        
        # 2. Process: latent self-attention
        h_self, _ = self.latent_self_attn(L, L, L)
        L = self.norm2(L + h_self)
        
        # 3. Decode: inputs query latents
        out, _ = self.cross_attn_out(x, L, L)
        out = self.norm3(x + out)
        out = out + self.mlp(out)
        return out

class PerceiverBaseline(BaselineBase):
    def __init__(self, H: int, W: int, P: int, Q: int, F_dim: int, R: int = 0, K: int = 32, num_layers: int = 2):
        super().__init__(H, W, P, Q, F_dim, R)
        dim = self.payload_dim
        self.blocks = nn.ModuleList([
            PerceiverBlock(dim=dim, num_latents=K) for _ in range(num_layers)
        ])

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        h = self.prepare_input(X, routing_features=routing_features, positions=positions)
        for block in self.blocks:
            h = block(h)
        return self.restore_output(h)
