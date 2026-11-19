# Quickstart & API Reference: BlockGrid (`blockvga`)

`blockvga` is a PyTorch layer for transporting and transforming structured tensor blocks on large virtual grids in linear time $\mathcal{O}(N \cdot K)$.

---

## 1. Installation

```bash
pip install -e .
```

Requires: `torch>=2.0.0`, `numpy>=1.20.0`, `pyyaml>=6.0`.

---

## 2. Basic Usage

```python
import torch
from blockvga import BlockGridConfig, BlockGridLayer

# 1. Configure the virtual grid layer
config = BlockGridConfig(
    H=64, W=64,       # Virtual Grid size: 64x64 (4,096 cells)
    P=4, Q=4,         # Inner Block dimensions (4x4 spatial block)
    F=8,              # Feature channels per block coordinate
    K=32,             # Number of landmark communication slots
    d=128,            # Routing embedding dimension
    slot_mix="planar_uncrossed", # G = I - UV^T (from planar uncrossing identity)
    inner_op=True,    # Enable two-sided factorized transforms L_k = A_k Z B_k^T
    tied_readback=False # Independent read/write simplex (recommended for transport)
)

# 2. Instantiate layer
layer = BlockGridLayer(config).cuda()

# 3. Input tensor: [Batch, H, W, P, Q, F]
X = torch.randn(2, 64, 64, 4, 4, 8, device="cuda")

# 4. Optional routing features / markers: [Batch, H, W, R]
# (e.g. indicating target source and destination cells)
out = layer(X)
print("Output shape:", out.shape) # [2, 64, 64, 4, 4, 8]
```

---

## 3. Configuration Parameters (`BlockGridConfig`)

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `H, W` | `int` | *Required* | Outer virtual grid height and width ($N = H \times W$). |
| `P, Q` | `int` | *Required* | Inner block matrix dimensions ($P \times Q$). |
| `F` | `int` | *Required* | Feature channels per inner coordinate. |
| `R` | `int` | `0` | Number of non-payload routing marker channels (e.g., coordinates, flags). |
| `K` | `int` | `32` | Number of landmark transport slots ($\mathcal{O}(N \cdot K)$ linear bottleneck). |
| `d` | `int` | `128` | Dimension of summary and positional routing embeddings. |
| `tau_init` | `float` | `1.0` | Routing temperature (floored at $0.05$ for numerical stability). |
| `slot_mix` | `str` | `"free"` | Slot mixing map: `"free"`, `"symmetric"`, or `"planar_uncrossed"`. |
| `inner_op` | `bool` | `False` | Apply per-slot two-sided inner transformations $A_k Z B_k^T$. |
| `tied_readback`| `bool` | `True` | If `False`, readback uses an independent routing simplex. |
| `normalize_slots`| `bool`| `False` | Mean vs. sum pooling semantics across slot assignments. |

---

## 4. Multi-Layer Models (`BlockGridModel`)

```python
from blockvga import BlockGridModel

# Stack 4 BlockGrid layers with an optional classification head
model = BlockGridModel(config, num_layers=4, num_classes=10).cuda()

logits = model(X) # Automatically pools across spatial & block dimensions -> [Batch, 10]
```

---

## 5. Running the Tests

```bash
pytest tests/ -v
```
All 18 tests cover:
- Shape validation ($P=Q$, $P \neq Q$, $R > 0$, coordinate validation)
- Permutation equivariance across grid sites
- Exact analytical `torch.autograd.gradcheck` on float64
- Stability at the temperature floor ($\tau = 0.05$) without NaN/Inf
- The Planar Uncrossing parameterization ($G = I - U V^T$)
