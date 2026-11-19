import torch
import torch.nn as nn
from blockgrid.config import BlockGridConfig

class SlotMixing(nn.Module):
    """
    Step 4: Slot mixing (signed linear map over slots).
    Z'_l = sum_k G_{lk} * Z_k
    """
    def __init__(self, config: BlockGridConfig):
        super().__init__()
        self.config = config
        K = config.K
        
        if config.slot_mix == "free":
            # Free K x K matrix init N(0, 0.02)
            self.G = nn.Parameter(torch.randn(K, K) * 0.02)
        elif config.slot_mix == "symmetric":
            # Constrained symmetric matrix from free M
            self.M = nn.Parameter(torch.randn(K, K) * 0.02)
        elif config.slot_mix == "planar_uncrossed":
            # Derived from singlet uncrossing: G = I - U V^T (rank-R correction)
            rank = min(8, K)
            self.U = nn.Parameter(torch.randn(K, rank) * 0.02)
            self.V = nn.Parameter(torch.randn(K, rank) * 0.02)
            self.register_buffer("eye", torch.eye(K))
        else:
            raise ValueError(f"Unknown slot_mix mode: {config.slot_mix}")

    def get_mixing_matrix(self) -> torch.Tensor:
        if self.config.slot_mix == "free":
            return self.G
        elif self.config.slot_mix == "symmetric":
            return 0.5 * (self.M + self.M.t())
        elif self.config.slot_mix == "planar_uncrossed":
            return self.eye - torch.matmul(self.U, self.V.t())
        raise RuntimeError("Unreachable")

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [B, K, P, Q, F]
        Returns:
            Z_prime: [B, K, P, Q, F]
        """
        G = self.get_mixing_matrix()
        # Z'_l = sum_k G_{lk} * Z_k
        # einsum: 'lk, bkpqf -> blpqf'
        Z_prime = torch.einsum('lk,bkpqf->blpqf', G, Z)
        return Z_prime
