import torch
import torch.nn as nn
from blockgrid.config import BlockGridConfig

class InnerOperator(nn.Module):
    """
    Step 5: Optional per-slot inner operator (the structural hypothesis).
    For each feature f: Z'_k[:, :, f] <- A_k Z'_k[:, :, f] B_k^T
    Contraction: 'bkpqf, kpi, kqj -> bkijf'
    A_k in R^{P x P}, B_k in R^{Q x Q}, init identity + N(0, 0.01)
    """
    def __init__(self, config: BlockGridConfig):
        super().__init__()
        self.config = config
        K, P, Q = config.K, config.P, config.Q
        
        # Init A_k: [K, P, P], identity + N(0, 0.01)
        A_init = torch.eye(P).unsqueeze(0).repeat(K, 1, 1) + torch.randn(K, P, P) * 0.01
        self.A = nn.Parameter(A_init)
        
        # Init B_k: [K, Q, Q], identity + N(0, 0.01)
        B_init = torch.eye(Q).unsqueeze(0).repeat(K, 1, 1) + torch.randn(K, Q, Q) * 0.01
        self.B = nn.Parameter(B_init)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [B, K, P, Q, F]
        Returns:
            Z_out: [B, K, P, Q, F]
        """
        # Contraction:
        # Z_{b, k, p, q, f} * A_{k, i, p} * B_{k, j, q}
        # In terms of (A Z B^T)_{i, j} = \sum_{p, q} A_{i, p} Z_{p, q} B_{j, q}:
        # 'kip, kjq, bkpqf -> bkijf'
        Z_out = torch.einsum('kip,kjq,bkpqf->bkijf', self.A, self.B, Z)
        return Z_out
