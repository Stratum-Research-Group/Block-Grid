import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineBase(nn.Module):
    """
    Common wrapper for flattening cell blocks R^{P x Q x F} into R^C and back,
    and injecting routing_features / positions as per section 1 / 5.
    """
    def __init__(self, H: int, W: int, P: int, Q: int, F_dim: int, R: int = 0):
        super().__init__()
        self.H = H
        self.W = W
        self.P = P
        self.Q = Q
        self.F_dim = F_dim
        self.R = R
        self.payload_dim = P * Q * F_dim
        
        # Position embeddings
        self.row_embed = nn.Embedding(H, 16)
        self.col_embed = nn.Embedding(W, 16)
        self.pos_dim = 32
        
        # Input projection combining payload + routing_features + pos
        in_dim = self.payload_dim + R + self.pos_dim
        self.in_proj = nn.Linear(in_dim, self.payload_dim)
        self.out_proj = nn.Linear(self.payload_dim, self.payload_dim)

    def prepare_input(self, X: torch.Tensor, routing_features=None, positions=None):
        B, H, W, P, Q, F_dim = X.shape
        device = X.device
        
        # Positions
        if positions is None:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            row_idx = grid_y.unsqueeze(0).expand(B, -1, -1)
            col_idx = grid_x.unsqueeze(0).expand(B, -1, -1)
        else:
            row_idx = positions[..., 0]
            col_idx = positions[..., 1]
            
        pos_feat = torch.cat([self.row_embed(row_idx), self.col_embed(col_idx)], dim=-1) # [B, H, W, 32]
        
        # Flatten payload
        flat_x = X.flatten(start_dim=-3) # [B, H, W, P*Q*F]
        
        pieces = [flat_x]
        if self.R > 0:
            if routing_features is None:
                raise ValueError(f"routing_features required for R={self.R}")
            pieces.append(routing_features)
        pieces.append(pos_feat)
        
        combined = torch.cat(pieces, dim=-1) # [B, H, W, total_in]
        x_proj = self.in_proj(combined)      # [B, H, W, C]
        return x_proj.flatten(start_dim=1, end_dim=2) # [B, N, C]

    def restore_output(self, x_out: torch.Tensor):
        # x_out: [B, N, C]
        B = x_out.shape[0]
        x_proj = self.out_proj(x_out) # [B, N, C]
        return x_proj.view(B, self.H, self.W, self.P, self.Q, self.F_dim)
