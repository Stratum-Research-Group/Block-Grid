import torch
import torch.nn as nn
from blockgrid.config import BlockGridConfig
from blockgrid.assignment import LandmarkAssignment
from blockgrid.pooling import BlockPooling
from blockgrid.slotmix import SlotMixing
from blockgrid.inner import InnerOperator

class BlockGridLayer(nn.Module):
    """
    Interface contract:
        class BlockGridLayer(nn.Module):
            def forward(self, X, routing_features=None, positions=None):
                returns: [B, H, W, P, Q, F], same shape as X
    """
    def __init__(self, config: BlockGridConfig):
        super().__init__()
        self.config = config
        
        self.assignment = LandmarkAssignment(config)
        self.pooling = BlockPooling(
            normalize_slots=config.normalize_slots,
            use_triton=config.use_triton
        )
        self.slotmix = SlotMixing(config)
        
        if config.inner_op:
            self.inner = InnerOperator(config)
        else:
            self.inner = None
            
        # Step 7: Residual combine
        # proj: per-(P,Q)-position Linear(F, F) shared across cells
        self.proj = nn.Linear(config.F, config.F)
        
        # gamma: learnable scalar per layer, init 0.1
        self.gamma = nn.Parameter(torch.tensor(float(config.gamma_init)))
        
        if config.layernorm_cells:
            self.ln = nn.LayerNorm(config.F)
        else:
            self.ln = None

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        """
        X: [B, H, W, P, Q, F]
        routing_features: [B, H, W, R] or None
        positions: [B, H, W, 2] or None
        """
        B, H, W, P, Q, F_dim = X.shape
        assert H == self.config.H and W == self.config.W
        assert P == self.config.P and Q == self.config.Q
        assert F_dim == self.config.F
        
        # Step 1 & 2: Landmark assignment
        w_write, w_read = self.assignment(X, routing_features=routing_features, positions=positions)
        
        # Step 3: Write / pool
        Z = self.pooling.pool(X, w_write)  # [B, K, P, Q, F]
        
        # Step 4: Slot mixing
        Z_prime = self.slotmix(Z)  # [B, K, P, Q, F]
        
        # Step 5: Optional inner operator
        if self.inner is not None:
            Z_prime = self.inner(Z_prime)
            
        # Step 6: Read back
        Y = self.pooling.readback(Z_prime, w_read)  # [B, H, W, P, Q, F]
        
        # Step 7: Residual combine
        out = X + self.gamma * self.proj(Y)
        
        if self.ln is not None:
            out = self.ln(out)
            
        return out
