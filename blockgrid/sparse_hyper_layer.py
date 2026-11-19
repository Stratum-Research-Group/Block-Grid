import math
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from blockgrid.config import BlockGridConfig
from blockgrid.slotmix import SlotMixing

@dataclass
class SparseHyperGridConfig:
    """
    Configuration for Sparse / Active Hyperdimensional Grid Routing:
    - D: Number of hyper-axes (e.g. 11, 22, 33, 64, or any arbitrary dimension)
    - axis_radix: Number of discrete states per axis (default 2 for binary hypercube {0, 1}^D,
                  or 3 for ternary {-1, 0, +1}^D, or arbitrary integer)
    - cell_shape: Inner tensor shape per cell (e.g. (P, Q, F) or arbitrary (T_1, ..., T_M))
    - K: Number of global landmark slots (default 16 or 32)
    - d: Coordinate embedding dimension (default 32 or 64)
    """
    D: int                                          # Number of hyperdimensional axes (e.g. 33)
    axis_radix: Union[int, Tuple[int, ...]] = 2     # Radix per axis (2 = binary, 3 = ternary, or tuple)
    cell_shape: Tuple[int, ...] = (4, 4, 4)         # Payload shape inside each active cell
    K: int = 16                                     # Number of landmark slots
    num_heads: int = 1                              # Multi-head slots
    d: int = 32                                     # Coordinate embedding dimension
    tau_init: float = 1.0                           # Initial routing temperature
    slot_mix: str = "planar_uncrossed"              # "planar_uncrossed", "free", "symmetric"
    inner_op: bool = True                           # Factorized intra-cell transformation
    tied_readback: bool = False                     # Separate write and read routing simplices

    @property
    def cell_volume(self) -> int:
        vol = 1
        for s in self.cell_shape:
            vol *= s
        return vol

    def get_radix_for_axis(self, axis_idx: int) -> int:
        if isinstance(self.axis_radix, int):
            return self.axis_radix
        return self.axis_radix[axis_idx]

class SparseHyperLandmarkAssignment(nn.Module):
    """
    Maps M active points from an arbitrary D-dimensional coordinate space into K landmark slots.
    Complexity is strictly O(M * K), completely independent of 2^D total universe size!
    """
    def __init__(self, config: SparseHyperGridConfig):
        super().__init__()
        self.config = config
        self.D = config.D
        d = config.d
        K = config.K

        # Additive embedding per hyper-axis: E_i in R^{radix_i x d}
        self.axis_embeds = nn.ModuleList([
            nn.Embedding(config.get_radix_for_axis(i), d) for i in range(config.D)
        ])

        # Cell summary MLP over flattened cell volume
        self.summary_mlp = nn.Sequential(
            nn.Linear(config.cell_volume, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

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

    def _compute_simplex(self, logits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        B, M, K = logits.shape
        tau = self.get_temperature()
        logits_reshaped = logits.view(B, M, self.num_heads, self.slots_per_head)
        if dtype in (torch.float16, torch.bfloat16):
            w = F.softmax((logits_reshaped / tau).float(), dim=-1).to(dtype)
        else:
            w = F.softmax(logits_reshaped / tau, dim=-1)
        return w.view(B, M, K)

    def forward(self, V_active: torch.Tensor, positions: torch.Tensor):
        """
        V_active: [B, M, *cell_shape] active cell payload tensors
        positions: [B, M, D] integer coordinates in {0, ..., radix-1}^D
        """
        B, M = positions.shape[0], positions.shape[1]
        assert positions.shape[-1] == self.D, f"Expected {self.D}-dim coordinates, got {positions.shape[-1]}"

        # 1. Additive Hyper-Coordinate Embedding: sum_i E_i(p_i)
        p_embed = 0.0
        for i, embed_layer in enumerate(self.axis_embeds):
            axis_coord = positions[..., i] # [B, M]
            p_embed = p_embed + embed_layer(axis_coord) # [B, M, d]

        # 2. Content Summary Embedding
        V_flat = V_active.reshape(B, M, self.config.cell_volume)
        s_embed = self.summary_mlp(V_flat) # [B, M, d]

        # 3. Landmark Write Simplex
        logits_write = self.W_p(p_embed) + self.W_c(s_embed)
        w_write = self._compute_simplex(logits_write, V_active.dtype) # [B, M, K]

        # 4. Landmark Read Simplex
        if self.config.tied_readback:
            w_read = w_write
        else:
            logits_read = self.W_p_read(p_embed) + self.W_c_read(s_embed)
            w_read = self._compute_simplex(logits_read, V_active.dtype) # [B, M, K]

        return w_write, w_read

class SparseHyperBlockGrid(nn.Module):
    """
    Sparse Hyperdimensional Grid Transport Layer:
    Routes an active batch of M cells residing anywhere in an arbitrary D-dimensional coordinate space
    (e.g., D=11, 22, 33, 64) through K global landmark hubs in O(M * K) time and memory.
    """
    def __init__(self, config: SparseHyperGridConfig):
        super().__init__()
        self.config = config
        self.assignment = SparseHyperLandmarkAssignment(config)

        # Slot mixing across the K landmark hubs
        dummy_bg_cfg = BlockGridConfig(
            H=1, W=config.K, P=1, Q=1, F=config.cell_volume,
            K=config.K, slot_mix=config.slot_mix, inner_op=False
        )
        self.slotmix = SlotMixing(dummy_bg_cfg)

        # Mode-wise factorized multi-linear inner operator over cell dimensions
        if config.inner_op:
            self.mode_transforms = nn.ModuleList([
                nn.Linear(dim, dim, bias=False) for dim in config.cell_shape
            ])
        else:
            self.mode_transforms = None

        self.proj = nn.Linear(config.cell_volume, config.cell_volume)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def _apply_inner_op(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Apply mode-wise multi-linear contraction across each inner tensor dimension:
        Z: [B, K, T_1, T_2, ..., T_M]
        """
        if self.mode_transforms is None:
            return Z

        curr = Z
        for m, transform in enumerate(self.mode_transforms):
            # Move mode m to the last dimension, apply linear transform, move back
            # Z has dims: B(0), K(1), T_1(2), ..., T_M(M+1)
            axis = m + 2
            curr = curr.transpose(axis, -1)
            curr = transform(curr)
            curr = curr.transpose(axis, -1)
        return curr

    def forward(self, X: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        X: [B, M, *cell_shape] Active entity payloads
        positions: [B, M, D] Coordinates of each active entity in the D-dimensional universe
        Returns: [B, M, *cell_shape] Transformed payloads with global landmark communication
        """
        B, M = X.shape[0], X.shape[1]
        C = self.config.cell_volume

        # 1. Landmark Assignment
        w_write, w_read = self.assignment(X, positions) # [B, M, K]

        # 2. Pool active entities into K landmark hubs: Z_k = sum_{m=1}^M w_{m, k} * X_m
        X_flat = X.reshape(B, M, C)
        Z_flat = torch.einsum('bmc,bmk->bkc', X_flat, w_write) # [B, K, C]
        Z = Z_flat.reshape(B, self.config.K, *self.config.cell_shape)

        # 3. Global slot mixing among the K hubs
        Z_mixed = self.slotmix(Z.reshape(B, self.config.K, 1, 1, C)).reshape(B, self.config.K, *self.config.cell_shape)

        # 4. Multi-linear inner transformations on each mode
        Z_prime = self._apply_inner_op(Z_mixed)

        # 5. Read back from K landmark hubs to active entities
        Z_prime_flat = Z_prime.reshape(B, self.config.K, C)
        Y_flat = torch.einsum('bkc,bmk->bmc', Z_prime_flat, w_read) # [B, M, C]

        # 6. Residual combine
        out_flat = X_flat + self.gamma * self.proj(Y_flat)
        return out_flat.reshape(X.shape)
