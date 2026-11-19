import torch
import torch.nn as nn

try:
    from .autograd_kernel import triton_fused_autograd_pool
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

class BlockPooling(nn.Module):
    """
    Step 3 & Step 6:
      Step 3: Write / pool: Z_k = sum_a w_{a,k} * V_a
      Step 6: Read back:    Y_a = sum_k r_{a,k} * Z'_k
    """
    def __init__(self, normalize_slots: bool = False, eps: float = 1e-6, use_triton: bool = False):
        super().__init__()
        self.normalize_slots = normalize_slots
        self.eps = eps
        self.use_triton = use_triton

    def pool(self, V: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        V: [B, H, W, P, Q, F]
        w: [B, H, W, K]
        Returns:
            Z: [B, K, P, Q, F]
        """
        B, H, W, P, Q, F = V.shape
        K = w.shape[-1]

        if self.use_triton and HAS_TRITON and V.is_cuda and not self.normalize_slots:
            # Flatten spatial into N, channel into C = P*Q*F
            C = P * Q * F
            V_reshaped = V.reshape(B, H * W, C)
            w_reshaped = w.reshape(B, H * W, K)
            Z_flat = triton_fused_autograd_pool(V_reshaped, w_reshaped)  # [B, K, C]
            return Z_flat.reshape(B, K, P, Q, F)

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
