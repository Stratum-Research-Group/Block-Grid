import torch
import torch.nn as nn
from baselines.common import BaselineBase

class AxialAttentionBlock(nn.Module):
    """
    Axial Attention (Ho et al. 2019): Factorized attention across Row (W) then Column (H).
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, H*W, C]
        B, N, C = x.shape
        grid = x.view(B, H, W, C)
        
        # 1. Row attention: attend across W for each row H
        # Reshape: [B*H, W, C]
        row_in = grid.view(B * H, W, C)
        h_row, _ = self.row_attn(row_in, row_in, row_in)
        grid = self.norm1(row_in + h_row).view(B, H, W, C)
        
        # 2. Col attention: attend across H for each col W
        # Permute: [B, W, H, C] -> [B*W, H, C]
        col_in = grid.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)
        h_col, _ = self.col_attn(col_in, col_in, col_in)
        grid = self.norm2(col_in + h_col).view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        
        out = grid.view(B, N, C)
        out = out + self.mlp(out)
        return out

class AxialAttentionBaseline(BaselineBase):
    def __init__(self, H: int, W: int, P: int, Q: int, F_dim: int, R: int = 0, num_layers: int = 2):
        super().__init__(H, W, P, Q, F_dim, R)
        dim = self.payload_dim
        self.blocks = nn.ModuleList([
            AxialAttentionBlock(dim=dim) for _ in range(num_layers)
        ])

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        h = self.prepare_input(X, routing_features=routing_features, positions=positions)
        for block in self.blocks:
            h = block(h, self.H, self.W)
        return self.restore_output(h)
