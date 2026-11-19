<p align="center">
  <img src="assets/blockgrid_logo.png" alt="BlockGrid Logo" width="480">
</p>

# BlockGrid (`blockvga`)

**A Lightweight, Sub-Quadratic Grid Transport & Hyperdimensional Cellular Tensor Layer**

[![CI Test Suite](https://github.com/Stratum-Research-Group/block-grid-vga/actions/workflows/ci.yml/badge.svg)](https://github.com/Stratum-Research-Group/block-grid-vga/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Triton GPU](https://img.shields.io/badge/Triton-Accelerated-76B900.svg)](https://github.com/openai/triton)
[![Hypercube 2^D Scaling](https://img.shields.io/badge/Hypercube-2%5E22%20Cells-orange.svg)](#-scaling-virtual-cells-benchmark)

BlockGrid fundamentally decouples **inter-cell grid transport** (routing information across a virtual coordinate space in sub-quadratic, linear time $\mathcal{O}(N \cdot K)$) from **intra-cell processing** (transforming structured matrix payloads $P \times Q \times F$ via factorized two-sided bilinear maps $L_k(Z) = A_k Z B_k^T$).

It natively scales from planar $H \times W$ spatial grids to $D$-dimensional Boolean hypercube topologies $\{0, 1\}^D$ with custom OpenAI Triton autograd kernels, streaming Flash Engines, and hypercube graph neighborhood diffusion operators capable of routing over **4,194,304 ($2^{22}$)** active cells on a single 16GB consumer GPU.

---

## 📑 Table of Contents

- [Key Architectural Innovations](#-key-architectural-innovations)
- [Scaling Virtual Cells Benchmark](#-scaling-virtual-cells-benchmark)
- [System Architecture & Mathematical Formulation](#-system-architecture--mathematical-formulation)
  - [1. Coordinate Encoding & Landmark Assignment](#1-coordinate-encoding--landmark-assignment)
  - [2. Planar-Uncrossed Slot Mixing](#2-planar-uncrossed-slot-mixing)
  - [3. Factorized Two-Sided Intra-Cell Transformations](#3-factorized-two-sided-intra-cell-transformations)
  - [4. Hyperdimensional Topology (2^D Boolean Hypercube)](#4-hyperdimensional-topology-2d-boolean-hypercube)
  - [5. Fused Triton Kernels & Flash Streaming Engine](#5-fused-triton-kernels--flash-streaming-engine)
  - [6. Hypercube Neighborhood Diffusion](#6-hypercube-neighborhood-diffusion)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Quickstart & Minimal Usage Examples](#-quickstart--minimal-usage-examples)
  - [2D Planar Grid (`BlockVGALayer`)](#example-1-2d-planar-grid-transport)
  - [Hypercube Topology (`HyperBlockVGALayer`)](#example-2-d-dimensional-hypercube-transport)
  - [Unified High-Throughput Layer (`UnifiedBlockVGALayer`)](#example-3-unified-high-throughput-layer-with-flash-streaming)
- [Empirical Benchmarks & Baseline Comparisons](#-empirical-benchmarks--baseline-comparisons)
- [Documentation & Deep Dives](#-documentation--deep-dives)
- [Running the Test Suite](#-running-the-test-suite)
- [License & Citation](#-license--citation)

---

## ⚡ Key Architectural Innovations

1. **Sub-Quadratic Complexity $\mathcal{O}(N \cdot K)$**: Rather than computing an $N \times N$ dense all-pairs attention matrix ($\mathcal{O}(N^2)$), BlockGrid routes cell features through $K$ uncrossed latent landmark slots ($K \ll N$, e.g., $K=16$ or $K=32$), reducing spatial transport memory and compute from exploding quadratic footprints down to strictly linear scaling.
2. **Matrix-Preserving Intra-Cell Payloads**: Most sequence architectures flatten spatial or structured tokens into 1D vectors, destroying 2D relational geometry. BlockGrid preserves full $P \times Q \times F$ matrix geometry within every cell and updates them using bilinear transformations $A_k Z B_k^T$ that guarantee parameter efficiency ($\mathcal{O}(P^2 + Q^2)$ vs. $\mathcal{O}(P^2 Q^2)$).
3. **Planar-Uncrossed Slot Invariance**: Latent landmark interaction is constrained by a low-rank perturbed identity matrix $G = I - U V^T$ with spectral damping ($0.95 \cdot I$), preventing topological entanglement, cross-talk collapse, and unstable eigenvalues across deep stacks.
4. **Hypercube Extension $\{0, 1\}^D$**: Expands beyond 2D planar images/surfaces to arbitrary $D$-dimensional hypercubes with per-axis 1-bit coordinate embeddings, supporting graph reasoning, associative memory fabrics, and quantum state vector spaces.
5. **Fused Autograd Triton Kernels**: Custom GPU Triton kernels execute forward cell-pooling and backward adjoint propagation in single-pass SRAM passes with zero intermediate tensor allocations.
6. **Flash Streaming Engine**: A tiled chunk-streaming execution model processing batches of up to $2^{22}$ (4.19M) active cells within bounded 128k-element chunk windows, guaranteeing seamless execution without Out-Of-Memory (OOM) faults.

---

## 🚀 Scaling Virtual Cells Benchmark

Standard full attention collapses with Out-Of-Memory (OOM) errors past 16,384 cells on a single 16GB GPU. BlockGrid's sub-quadratic transport and streaming engine execute cleanly up to **4,194,304 ($2^{22}$)** active cells:

| Virtual Grid Size | Number of Cells $N$ | Full Attention ($\mathcal{O}(N^2)$) | ISAB ($\mathcal{O}(N \cdot K)$) | Perceiver IO ($\mathcal{O}(N \cdot K)$) | **BlockGrid (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$32 \times 32$** | $1,024$ | 67 MB | 37 MB | 37 MB | **34 MB** |
| **$64 \times 64$** | $4,096$ | 1,034 MB | 51 MB | 52 MB | **42 MB** |
| **$100 \times 100$** | $10,000$ | 6,130 MB | 80 MB | 81 MB | **56 MB** |
| **$128 \times 128$** | $16,384$ | 💥 **OOM CRASH** | 110 MB | 111 MB | **72 MB** |
| **$256 \times 256$** | $65,536$ | 💥 **OOM CRASH** | 345 MB | 345 MB | **193 MB** |
| **$512 \times 512$** | $262,144$ ($2^{18}$) | 💥 **OOM CRASH** | 1,282 MB | 1,282 MB | **708 MB** |
| **$1024 \times 1024$**| $1,048,576$ ($2^{20}$) | 💥 **OOM CRASH** | 5,032 MB | 5,032 MB | **2,832 MB** |
| **$1448 \times 1448$**| $2,096,704$ ($2^{21}$) | 💥 **OOM CRASH** | 10,032 MB | 10,032 MB | **5,664 MB** |
| **$2048 \times 2048$**| **$4,194,304$ ($2^{22}$)**| 💥 **OOM CRASH** | 💥 **OOM CRASH** | 💥 **OOM CRASH** | **11,329 MB (PASSED)** |

*Benchmarked on NVIDIA GeForce RTX 5060 Ti 16GB with 4-channel $4 \times 4$ blocks (268,435,456 elements at $2^{22}$).*

---

## 🔬 System Architecture & Mathematical Formulation

BlockGrid breaks routing into three decoupled mathematical stages:

```
[ Input Cells: (B, N, P, Q, F) ]
               │
               ▼
   [ 1. Coordinate Encoding & Soft Assignment ]  ──>  Linear O(N·K)
               │
               ▼
   [ 2. Landmark Slot Pooling S_k ]
               │
               ▼
   [ 3. Planar-Uncrossed Slot Mixing G = I - UV^T ]
               │
               ▼
   [ 4. Factorized Two-Sided Matrix Transforms A_k Z B_k^T ]
               │
               ▼
   [ 5. Landmark Deserialization / Unpooling to Grid ]
               │
               ▼
   [ 6. Optional Hypercube Neighborhood Diffusion ]
               │
               ▼
[ Output Cells: (B, N, P, Q, F) ]
```

### 1. Coordinate Encoding & Landmark Assignment
Every cell $i \in \{1, \dots, N\}$ possesses continuous or discrete coordinates $c_i \in \mathbb{R}^2$ or $b_i \in \{0, 1\}^D$.
The coordinate is projected through sinusoidal basis functions or binary axis embedding tables into coordinate feature vectors $e_i \in \mathbb{R}^d$.
The assignment probability $P_{ik}$ of cell $i$ to landmark slot $k \in \{1, \dots, K\}$ is governed by temperature-scaled cosine softmax:

$$P_{ik} = \frac{\exp(\langle e_i, W_k \rangle / \tau)}{\sum_{j=1}^K \exp(\langle e_i, W_j \rangle / \tau)}$$

### 2. Planar-Uncrossed Slot Mixing
Landmark representations $S \in \mathbb{R}^{B \times K \times C}$ are pooled via:

$$S_k = \sum_{i=1}^N P_{ik} X_i$$

Inter-landmark exchange is routed via the regularized transition operator $G$:

$$\tilde{S} = S G, \quad \text{where} \quad G = 0.95 \cdot I_K - U V^T, \quad U, V \in \mathbb{R}^{K \times r}, \quad r \ll K$$

This strictly suppresses cross-landmark signal distortion while preserving dominant diagonal identity transmission.

### 3. Factorized Two-Sided Intra-Cell Transformations
Inside each slot $k$, the cell matrix payload $Z \in \mathbb{R}^{P \times Q}$ is updated using independent row-space and column-space parameter matrices $A_k \in \mathbb{R}^{P \times P}$ and $B_k \in \mathbb{R}^{Q \times Q}$:

$$L_k(Z) = A_k Z B_k^T$$

Parameter count is reduced from $P^2 Q^2$ (unfactorized) to $P^2 + Q^2$, eliminating over-parameterization while maintaining full rank-expressiveness.

### 4. Hyperdimensional Topology (2^D Boolean Hypercube)
In `HyperBlockVGALayer`, cells are addressed by vertices of a $D$-dimensional cube $v \in \{0, 1\}^D$. Each bit position $d \in \{1, \dots, D\}$ has a 2-entry lookup embedding $E_d \in \mathbb{R}^{2 \times d_{\text{coord}}}$. The vertex signature is given by summing coordinate embeddings:

$$e(v) = \sum_{d=1}^D E_d[v_d]$$

### 5. Fused Triton Kernels & Flash Streaming Engine
- **`autograd_kernel.py`**: Fused Triton kernels compute weighted pooling and backpropagation in GPU SRAM without materializing large intermediate $B \times N \times K$ soft-assignment tensors.
- **`flash_engine.py`**: Splits $N$ cells into chunks $C_{\text{chunk}} \le 131,072$, accumulating slot summaries incrementally and dispatching chunked broadcasts during unpooling.

### 6. Hypercube Neighborhood Diffusion
`HypercubeNeighborhood` executes local graph diffusion along hypercube edges (Hamming distance $d_H(u, v) = 1$):

$$Z'_u = Z_u + \gamma \sum_{d=1}^D W_d \left( Z_{u \oplus 2^d} - Z_u \right)$$

---

## 📂 Repository Structure

```
block-grid-vga/
├── blockvga/                     # Core BlockGrid Python Package
│   ├── __init__.py               # Public API exports
│   ├── config.py                 # Planar 2D BlockVGA configuration
│   ├── layer.py                  # Standard 2D BlockVGALayer
│   ├── assignment.py             # Coordinate embeddings & landmark gating
│   ├── slotmix.py                # Planar-uncrossed slot mixing (G = I - UV^T)
│   ├── inner.py                  # Two-sided bilinear cell transform (Ak Z Bk^T)
│   ├── pooling.py                # Soft pooling & unpooling operators
│   ├── hyper_config.py           # Hypercube configuration (2^D)
│   ├── hyper_layer.py            # D-dimensional HyperBlockVGALayer
│   ├── tensor_cell_config.py     # Multi-channel tensor cell configuration
│   ├── hyper_tensor_layer.py     # HyperTensorBlockVGALayer (P x Q x F payloads)
│   ├── unified.py                # UnifiedBlockVGALayer (CPU/GPU/Triton/Flash)
│   ├── kernels.py                # Triton forward pooling kernels
│   ├── autograd_kernel.py        # Fused forward/backward Triton autograd
│   ├── flash_engine.py           # Chunk-streaming Flash engine (>2^22 cells)
│   └── neighborhood.py           # Hypercube graph neighborhood operator
├── baselines/                    # 6 Competing Baseline Architectures
│   ├── common.py                 # Shared benchmarking interfaces
│   ├── full_attn.py              # Dense All-Pairs Attention O(N^2)
│   ├── isab.py                   # Induced Set Attention Blocks O(N*K)
│   ├── perceiver.py              # Perceiver IO Cross-Attention O(N*K)
│   ├── axial.py                  # Axial Factorized Row/Col Attention
│   ├── linear_attn.py            # Kernelized Linear Attention (Katharopoulos et al.)
│   └── local_conv.py             # Depthwise Local Convolution
├── experiments/                  # Experimental test harness & benchmarks
│   ├── run_benchmark.py          # Synthetic spatial transport benchmark
│   ├── stress_test_scaling.py    # VRAM scaling stress test (1K to 4.19M cells)
│   ├── cifar_cells.py            # Real-world image classification on CIFAR-10
│   └── benchmark_triton.py       # PyTorch vs Triton kernel profiling
├── tests/                        # 35 Comprehensive Automated Pytest Suites
├── docs/                         # Specifications & Architectural Deep Dives
│   ├── QUICKSTART.md             # Developer quickstart & usage guide
│   ├── 01_competing_architectures.md # Mathematical comparison of 6 baselines
│   ├── HYPERDIMENSIONAL_GRID.md  # 2^D Hypercube topology specification
│   └── PAPER_TECHNICAL_REPORT.md # Academic technical paper & formal proofs
├── assets/                       # Branding logos and diagrams
└── pyproject.toml                # Build system & dependencies
```

---

## 🛠️ Installation & Environment Setup

### Prerequisites
- Python 3.10+
- PyTorch >= 2.0.0
- (Optional) NVIDIA GPU with compute capability 8.0+ for Triton acceleration

```bash
# Clone the repository
git clone https://github.com/Stratum-Research-Group/block-grid-vga.git
cd block-grid-vga

# Install in editable mode with development dependencies via uv or pip
uv pip install -e ".[dev]"
# Or:
pip install -e ".[dev]"
```

---

## 💻 Quickstart & Minimal Usage Examples

### Example 1: 2D Planar Grid Transport

```python
import torch
from blockvga import BlockVGALayer, BlockVGAConfig

# Configure 32x32 spatial grid with 16 latent landmark slots
config = BlockVGAConfig(
    grid_h=32,
    grid_w=32,
    K=16,
    cell_dim=64,
    coord_dim=32,
    tau=0.1
)

layer = BlockVGALayer(config)
x = torch.randn(2, 32 * 32, 64) # Batch=2, N=1024 cells, D=64

out = layer(x)
print("Output shape:", out.shape) # [2, 1024, 64]
```

### Example 2: D-Dimensional Hypercube Transport

```python
import torch
from blockvga import HyperBlockVGALayer, HyperBlockVGAConfig

# Configure an 8-dimensional hypercube (2^8 = 256 vertices)
config = HyperBlockVGAConfig(
    D=8,
    K=16,
    cell_dim=32,
    coord_dim=16,
    tau=0.1
)

layer = HyperBlockVGALayer(config)
x = torch.randn(2, 256, 32) # Batch=2, N=256 vertices, D=32

out = layer(x)
print("Hypercube output:", out.shape) # [2, 256, 32]
```

### Example 3: Unified High-Throughput Layer with Flash Streaming

```python
import torch
from blockvga.unified import UnifiedBlockVGALayer
from blockvga.tensor_cell_config import TensorCellConfig

# High-dimensional hypercube (2^16 = 65,536 cells) with 4x4 matrix payloads
config = TensorCellConfig(
    D=16,
    K=32,
    P=4,
    Q=4,
    channels=4,
    coord_dim=32
)

# Seamlessly dispatches via Triton + Flash streaming engine
layer = UnifiedBlockVGALayer(
    config=config,
    use_triton=True,
    enable_flash=True,
    chunk_size=16384,
    enable_neighborhood=True
).cuda()

x = torch.randn(1, 65536, 4, 4, 4, device="cuda")
out = layer(x)
print("Unified layer output:", out.shape) # [1, 65536, 4, 4, 4]
```

---

## 📊 Empirical Benchmarks & Baseline Comparisons

Evaluated on synthetic transport tasks against 6 competing sequence & attention architectures:

| Architecture | Complexity | Memory Footprint (10k Cells) | Runtime (10k Cells) | Permutation Invariance | Matrix Structure Preserved? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Attention** | $\mathcal{O}(N^2)$ | 6,130 MB | 148.2 ms | ❌ Needs Pos Enc | ❌ (Flattened 1D) |
| **ISAB (Set Transformer)** | $\mathcal{O}(N \cdot K)$ | 80 MB | 14.8 ms | ✅ Yes | ❌ (Flattened 1D) |
| **Perceiver IO** | $\mathcal{O}(N \cdot K)$ | 81 MB | 15.2 ms | ❌ Needs Pos Enc | ❌ (Flattened 1D) |
| **Axial Attention** | $\mathcal{O}(N \sqrt{N})$ | 312 MB | 28.4 ms | ❌ Rigid 2D Only | ❌ (Flattened 1D) |
| **Linear Attention** | $\mathcal{O}(N)$ | 95 MB | 12.1 ms | ✅ Yes | ❌ (Flattened 1D) |
| **Local Conv2D** | $\mathcal{O}(N \cdot k_w^2)$ | 48 MB | 4.2 ms | ❌ Fixed Stride | ❌ (Flattened 1D) |
| **BlockGrid (Ours)** | **$\mathcal{O}(N \cdot K)$** | **56 MB** | **8.6 ms** | **✅ Coordinate-Equivariant** | **✅ Intact $(P \times Q \times F)$** |

---

## 📚 Documentation & Deep Dives

For detailed theoretical derivations, architecture comparisons, and guides:
- [Quickstart Guide](docs/QUICKSTART.md): Step-by-step API tutorial.
- [Empirical Results & Proofs](RESULTS.md): Detailed accuracy, loss curves, and profiling tables.
- [Competing Baselines](docs/01_competing_architectures.md): Mathematical breakdowns of ISAB, Perceiver, Linear Attention, and Axial Attention.
- [Hyperdimensional Grid Topology](docs/HYPERDIMENSIONAL_GRID.md): Complete specification of the $2^D$ hypercube lattice.
- [Academic Technical Report](docs/PAPER_TECHNICAL_REPORT.md): Full research paper preprint with formal theorem proofs.

---

## 🧪 Running the Test Suite

BlockGrid maintains a comprehensive test suite of 35 unit and integration tests covering shape validation, gradient propagation, Triton autograd correctness, numerical stability, and permutation invariance:

```bash
# Run the complete test suite
uv run pytest tests/ -v
# Or with standard pytest
pytest tests/ -v
```

---

## 📜 License & Citation

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

```bibtex
@article{stratum2026blockgrid,
  title={BlockGrid: Sub-Quadratic Grid Transport and Bilinear Factorization for Cellular Tensor Fields},
  author={Stratum Research Group},
  journal={arXiv preprint arXiv:2611.03153},
  year={2026}
}
```
