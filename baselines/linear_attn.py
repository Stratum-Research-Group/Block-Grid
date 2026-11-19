import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.common import BaselineBase

class LinearAttentionBlock(nn.Module):
    """
    Linear Attention (Katharopoulos et al. 2020) via kernel feature map phi(x) = elu(x) + 1.
    Computes attention in O(N * d^2) rather than O(N^2 * d).
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        q = F.elu(self.q_proj(x).view(B, N, H, D)) + 1.0 # [B, N, H, D]
        k = F.elu(self.k_proj(x).view(B, N, H, D)) + 1.0 # [B, N, H, D]
        v = self.v_proj(x).view(B, N, H, D)              # [B, N, H, D]
        
        # KV product: sum over N -> [B, H, D_k, D_v]
        kv = torch.einsum('bnhd,bnhm->bhdm', k, v)
        
        # Denominator: q * sum(k)
        z = 1.0 / (torch.einsum('bnhd,bhd->bnh', q, k.sum(dim=1)) + 1e-6) # [B, N, H]
        
        # Output: q * KV * z
        num = torch.einsum('bnhd,bhdm->bnhm', q, kv)
        out = (num * z.unsqueeze(-1)).reshape(B, N, C)
        out = self.out_proj(out)
        
        x = self.norm1(x + out)
        x = self.norm2(x + self.mlp(x))
        return x

class LinearAttentionBaseline(BaselineBase):
    def __init__(self, H: int, W: int, P: int, Q: int, F_dim: int, R: int = 0, num_layers: int = 2):
        super().__init__(H, W, P, Q, F_dim, R)
        dim = self.payload_dim
        self.blocks = nn.ModuleList([
            LinearAttentionBlock(dim=dim) for _ in range(num_layers)
        ])

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        h = self.prepare_input(X, routing_features=routing_features, positions=positions)
        for block in self.blocks:
            h = block(h)
        return self.restore_output(h)
