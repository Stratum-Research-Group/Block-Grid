from dataclasses import dataclass
from typing import Tuple

@dataclass
class GeneralizedHyperTensorConfig:
    """
    Generalized Hyperdimensional Grid of Arbitrary Tensors:
    - outer_grid: D-dimensional hypercube/hyper-torus (S_1, S_2, ..., S_D), e.g. (2, 2, 2) or (2,)*22
    - cell_tensor_shape: arbitrary inner tensor (T_1, T_2, ..., T_M), e.g. (2, 2) matrix, (3, 3, 3) 3D tensor, etc.
    """
    outer_grid: Tuple[int, ...]                 # e.g. (2, 2, 2) or (2,)*D
    cell_tensor_shape: Tuple[int, ...]          # arbitrary inner tensor shape at EACH cell
    R: int = 0                                  # routing marker channels
    K: int = 32                                 # number of landmark slots
    num_heads: int = 1                          # multi-head routing slots
    d: int = 64                                 # coordinate embedding dimension
    tau_init: float = 1.0                       # routing temperature
    slot_mix: str = "planar_uncrossed"          # "planar_uncrossed", "free", "symmetric"
    inner_transform: bool = True                # apply factorized contractions on cell tensors
    tied_readback: bool = False                 # untied read/write simplex

    @property
    def D(self) -> int:
        return len(self.outer_grid)

    @property
    def N(self) -> int:
        """Total number of hypercube vertices / cells."""
        n = 1
        for s in self.outer_grid:
            n *= s
        return n

    @property
    def cell_volume(self) -> int:
        """Total number of scalar values inside each cell tensor."""
        v = 1
        for t in self.cell_tensor_shape:
            v *= t
        return v
