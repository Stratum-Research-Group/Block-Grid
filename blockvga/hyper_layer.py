import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blockvga.hyper_config import HyperGridConfig
from blockvga.slotmix import SlotMixing
from blockvga.inner import InnerOperator
from blockvga.config import BlockGridConfig

class HyperLandmarkAssignment(nn.Module):
    """
    Landmark routing for D-dimensional hypercube lattices {0, 1}^D or arbitrary discrete grids.
    Coordinate p(x) = sum_{i=1}^D E_i(x_i)
    """
    def __init__(self, config: HyperGridConfig):
        super().__init__()
        self.config = config
        d = config.d
        K = config.K
        self.D = config.D

        # One embedding table per hyper-axis: E_i in R^{S_i x d}
        # For a binary hypercube, S_i = 2, so each axis has 2 embedding vectors.
        self.axis_embeds = nn.ModuleList([
            nn.Embedding(s, d) for s in config.grid_shape
        ])

        # Summary MLP: over local block fiber R^{P x Q x F}
        self.summary_mlp = nn.Sequential(
            nn.Linear(config.F, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        if config.R > 0:
            self.linear_r = nn.Linear(config.R, d)
        else:
            self.linear_r = None

        # Multi-head simplex projections
        self.num_heads = config.num_heads
        if K % self.num_heads != 0:
            raise ValueError(f"K ({K}) must be divisible by num_heads ({self.num_heads})")
        self.slots_per_head = K // self.num_heads

        self.W_p = nn.Linear(d, K, bias=False)
        self.W_c = nn.Linear(d, K, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(config.tau_init, 0.05))))

        if not config.tied_readback:
            self.W_p_read = nn.Linear(d, K, bias=False)
            self.W_c_read = nn.Linear(d, K, bias=False)
        else:
            self.W_p_read = None
            self.W_c_read = None

    def get_temperature(self) -> torch.Tensor:
        tau = torch.exp(self.log_tau)
        return torch.clamp(tau, min=0.05)

    def _compute_simplex(self, logits: torch.Tensor, V_dtype: torch.dtype) -> torch.Tensor:
        # logits: [B, N, K] -> [B, N, num_heads, slots_per_head]
        B, N, K = logits.shape
        tau = self.get_temperature()
        logits_reshaped = logits.view(B, N, self.num_heads, self.slots_per_head)
        if V_dtype in (torch.float16, torch.bfloat16):
            w = F.softmax((logits_reshaped / tau).float(), dim=-1).to(V_dtype)
        else:
            w = F.softmax(logits_reshaped / tau, dim=-1)
        return w.view(B, N, K)

    def forward(self, V_flat, routing_features=None, positions=None):
        """
        V_flat: [B, N, P, Q, F] where N = S_1 * ... * S_D
        routing_features: [B, N, R] or None
        positions: [B, N, D] integer coordinates or None
        """
        B, N, P, Q, F_dim = V_flat.shape
        d = self.config.d
        device = V_flat.device

        # If positions not explicitly provided, generate canonical hyper-mesh coordinates
        if positions is None:
            # Build meshgrid across all D axes
            grids = torch.meshgrid(
                *[torch.arange(s, device=device) for s in self.config.grid_shape],
                indexing='ij'
            )
            # Stack into [N, D]
            pos = torch.stack([g.flatten() for g in grids], dim=-1) # [N, D]
            pos = pos.unsqueeze(0).expand(B, -1, -1) # [B, N, D]
        else:
            pos = positions

        # Hyper-coordinate embedding: sum of axis embeddings p(x) = sum_i E_i(x_i)
        p_embed = 0.0
        for i, embed_layer in enumerate(self.axis_embeds):
            axis_coord = pos[..., i] # [B, N]
            p_embed = p_embed + embed_layer(axis_coord) # [B, N, d]

        # Cell summary over local block (P, Q)
        mean_v = V_flat.mean(dim=(-3, -2)) # [B, N, F]
        s_embed = self.summary_mlp(mean_v) # [B, N, d]

        if self.linear_r is not None:
            if routing_features is None:
                raise ValueError(f"routing_features must be provided when R={self.config.R} > 0")
            s_embed = s_embed + self.linear_r(routing_features)

        logits_write = self.W_p(p_embed) + self.W_c(s_embed)
        w_write = self._compute_simplex(logits_write, V_flat.dtype)

        if self.config.tied_readback:
            w_read = w_write
        else:
            logits_read = self.W_p_read(p_embed) + self.W_c_read(s_embed)
            w_read = self._compute_simplex(logits_read, V_flat.dtype)

        return w_write, w_read

class HyperGridLayer(nn.Module):
    """
    Hyperdimensional BlockGrid Layer operating on arbitrary hypercube arrays:
    X: [B, S_1, S_2, ..., S_D, P, Q, F]
    """
    def __init__(self, config: HyperGridConfig):
        super().__init__()
        self.config = config
        self.assignment = HyperLandmarkAssignment(config)

        # Slot mixing & inner operators
        dummy_bg_cfg = BlockGridConfig(
            H=1, W=config.N, P=config.P, Q=config.Q, F=config.F,
            K=config.K, slot_mix=config.slot_mix, inner_op=config.inner_op
        )
        self.slotmix = SlotMixing(dummy_bg_cfg)
        if config.inner_op:
            self.inner = InnerOperator(dummy_bg_cfg)
        else:
            self.inner = None

        self.proj = nn.Linear(config.F, config.F)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        """
        X: [B, S_1, ..., S_D, P, Q, F] or already flattened [B, N, P, Q, F]
        """
        B = X.shape[0]
        P, Q, F_dim = self.config.P, self.config.Q, self.config.F
        orig_shape = X.shape

        # Flatten hyper-axes S_1 ... S_D into N
        if X.dim() == self.config.D + 4: # [B, S_1, ..., S_D, P, Q, F]
            V_flat = X.view(B, self.config.N, P, Q, F_dim)
        elif X.dim() == 5:
            V_flat = X
        else:
            raise ValueError(f"Expected tensor with {self.config.D + 4} dims or 5 dims, got {X.dim()}")

        # 1. Landmark Assignment across hypercube coordinates
        w_write, w_read = self.assignment(V_flat, routing_features=routing_features, positions=positions)

        # 2. Pool whole blocks: Z_k = sum_n w_{n, k} * V_n
        # einsum: 'bnc, bnk -> bkc' where c = P*Q*F
        C = P * Q * F_dim
        V_c = V_flat.view(B, self.config.N, C)
        Z = torch.einsum('bnc,bnk->bkc', V_c, w_write).view(B, self.config.K, P, Q, F_dim)

        # 3. Slot mixing
        Z_prime = self.slotmix(Z)

        # 4. Optional factorized two-sided inner transform
        if self.inner is not None:
            Z_prime = self.inner(Z_prime)

        # 5. Read back to hypercube coordinates: Y_n = sum_k r_{n, k} * Z'_k
        Z_c = Z_prime.reshape(B, self.config.K, C)
        Y_c = torch.einsum('bkc,bnk->bnc', Z_c, w_read)
        Y = Y_c.view(B, self.config.N, P, Q, F_dim)

        # 6. Residual combine
        out_flat = V_flat + self.gamma * self.proj(Y)
        return out_flat.view(orig_shape)
