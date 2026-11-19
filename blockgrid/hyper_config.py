from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class HyperGridConfig:
    grid_shape: Tuple[int, ...]         # e.g. (2, 2, 2) for D=3, or (2,)*22 for D=22
    P: int                              # inner block row dimension
    Q: int                              # inner block col dimension
    F: int                              # inner block channels
    R: int = 0                          # routing marker channels
    K: int = 32                         # number of landmark slots
    num_heads: int = 1                  # multi-head routing slots
    d: int = 64                         # coordinate embedding dimension
    tau_init: float = 1.0               # routing temperature (floored at 0.05)
    slot_mix: str = "planar_uncrossed"  # "planar_uncrossed", "free", or "symmetric"
    inner_op: bool = True               # factorized two-sided operators L_k = A_k Z B_k^T
    tied_readback: bool = False         # untied read/write simplex (recommended for transport)
    normalize_slots: bool = False       # slot normalization

    @property
    def D(self) -> int:
        """Number of hyperdimensional axes."""
        return len(self.grid_shape)

    @property
    def N(self) -> int:
        """Total number of hypercube vertices / grid sites."""
        n = 1
        for s in self.grid_shape:
            n *= s
        return n

    @property
    def payload_dim(self) -> int:
        """Scalars per hypercube site."""
        return self.P * self.Q * self.F
