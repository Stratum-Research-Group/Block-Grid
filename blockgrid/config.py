from dataclasses import dataclass

@dataclass
class BlockGridConfig:
    H: int
    W: int
    P: int
    Q: int
    F: int
    R: int = 0                          # routing-feature width; 0 = unused
    K: int = 32                         # number of landmark slots
    num_heads: int = 1                  # number of multi-head routing slots (K must be divisible by num_heads)
    d: int = 128                        # routing embedding dimension
    tau_init: float = 1.0               # routing softmax temperature (floored at 0.05)
    normalize_slots: bool = False       # slot-mass normalization (mean vs sum semantics)
    assignment: str = "pos+content"     # "pos" | "content" | "pos+content"
    summary: str = "mean"               # "mean" | "structured"
    slot_mix: str = "free"              # "free" | "symmetric" | "planar_uncrossed"
    tied_readback: bool = True          # reuse assignment weights for readback
    inner_op: bool = False              # apply per-slot 2-sided transforms L_k
    use_triton: bool = False            # use fused OpenAI Triton kernel for pooling on CUDA
    gamma_init: float = 0.1             # residual scale initialization
    layernorm_cells: bool = False       # optional layer norm
