import torch
import torch.nn as nn

class BlockPooling(nn.Module):
    """
    Step 3 & Step 6:
      Step 3: Write / pool: Z_k = sum_a w_{a,k} * V_a
      Step 6: Read back:    Y_a = sum_k r_{a,k} * Z'_k
    """
    def __init__(self, normalize_slots: bool = False, eps: float = 1e-6):
        super().__init__()
        self.normalize_slots = normalize_slots
        self.eps = eps

    def pool(self, V: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        V: [B, H, W, P, Q, F]
        w: [B, H, W, K]
        Returns:
            Z: [B, K, P, Q, F]
        """
        # Einsum contraction: 'bijpqf,bijk->bkpqf'
        Z = torch.einsum('bijpqf,bijk->bkpqf', V, w)
        
        if self.normalize_slots:
            # m_k: [B, K]
            m_k = w.sum(dim=(1, 2))
            denom = (m_k + self.eps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1, 1]
            Z = Z / denom
            
        return Z

    def readback(self, Z_prime: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Z_prime: [B, K, P, Q, F]
        r:       [B, H, W, K]
        Returns:
            Y: [B, H, W, P, Q, F]
        """
        # Einsum contraction: 'bkpqf,bijk->bijpqf'
        Y = torch.einsum('bkpqf,bijk->bijpqf', Z_prime, r)
        return Y
