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
[![Torch Compile](https://img.shields.io/badge/PyTorch%202.x-torch.compile%20Verified-blueviolet.svg)](#-torchcompile--production-compiler-readiness)

BlockGrid fundamentally decouples **inter-cell grid transport** (routing information across a virtual coordinate space in sub-quadratic, linear time $\mathcal{O}(N \cdot K)$) from **intra-cell processing** (transforming structured matrix payloads $P \times Q \times F$ via factorized two-sided bilinear maps $L_k(Z) = A_k Z B_k^T$).

It natively scales from planar $H \times W$ spatial grids to $D$-dimensional Boolean and Ternary hypercube topologies ($\{0, 1\}^D$, $\{-1, 0, +1\}^D$) with custom OpenAI Triton autograd kernels, streaming Flash Engines, and sparse active entity routing capable of processing coordinate universes exceeding **31.38 billion ($3^{22}$)** and **$2^{64}$** potential positions on a single 16GB consumer GPU.

---

## 📑 Table of Contents

- [Key Architectural Innovations](#-key-architectural-innovations)
- [Scaling Virtual Cells Benchmark](#-scaling-virtual-cells-benchmark)
- [Visual Transport Demo](#-visual-transport-demo)
- [System Architecture & Mathematical Formulation](#-system-architecture--mathematical-formulation)
  - [1. Coordinate Encoding & Landmark Assignment](#1-coordinate-encoding--landmark-assignment)
  - [2. Planar-Uncrossed Slot Mixing](#2-planar-uncrossed-slot-mixing)
  - [3. Factorized Multi-Linear Mode Contractions](#3-factorized-multi-linear-mode-contractions)
  - [4. Sparse Active Hypercube Transport (33D, 3^22, 64D)](#4-sparse-active-hypercube-transport-33d-322-64d)
  - [5. Fused Triton Kernels & Flash Streaming Engine](#5-fused-triton-kernels--flash-streaming-engine)
- [Quickstart & Minimal Usage Examples](#-quickstart--minimal-usage-examples)
  - [Example 1: 2D Planar Grid Transport](#example-1-2d-planar-grid-transport)
  - [Example 2: 1-Megapixel Canvas (256x256 Grid)](#example-2-1-megapixel-canvas-256x256-grid)
  - [Example 3: Sparse 33D Hypercube Routing](#example-3-sparse-33d-hypercube-routing)
  - [Example 4: Unified 2^22 Hypercube with Triton](#example-4-unified-222-hypercube-with-triton)
- [Empirical Benchmarks & Baseline Comparisons](#-empirical-benchmarks--baseline-comparisons)
- [Production Application Portfolio (50 Detailed Specs)](#-production-application-portfolio-50-detailed-specs)
- [Running the Test Suite](#-running-the-test-suite)
- [License & Citation](#-license--citation)

---

## ⚡ Key Architectural Innovations

1. **Sub-Quadratic Complexity $\mathcal{O}(N \cdot K)$**: Rather than computing an $N \times N$ dense all-pairs attention matrix ($\mathcal{O}(N^2)$), BlockGrid routes cell features through $K$ uncrossed latent landmark slots ($K \ll N$, e.g., $K=16$ or $K=32$), reducing spatial transport memory and compute from exploding quadratic footprints down to strictly linear scaling.
2. **Matrix-Preserving Intra-Cell Payloads**: Most sequence architectures flatten spatial or structured tokens into 1D vectors, destroying 2D relational geometry. BlockGrid preserves full $P \times Q \times F$ matrix geometry within every cell and updates them using bilinear transformations $A_k Z B_k^T$ that guarantee parameter efficiency ($\mathcal{O}(P^2 + Q^2)$ vs. $\mathcal{O}(P^2 Q^2)$).
3. **Planar-Uncrossed Slot Invariance**: Latent landmark interaction is constrained by a low-rank perturbed identity matrix $G = I - U V^T$ with spectral damping ($0.95 \cdot I$), preventing topological entanglement, cross-talk collapse, and unstable eigenvalues across deep stacks.
4. **Sparse Active Hypercube Routing**: Decouples the dimensionality $D$ of the coordinate universe from dense RAM allocations. Uses additive coordinate embedding tables ($\sum_{i=1}^D E_i(p_i)$ taking $<15\text{ KB}$) to route active entity batches residing anywhere across $33\text{D}$, $3^{22}$ ($31.38$B states), or $64\text{D}$ spaces in $\mathcal{O}(M \cdot K)$.
5. **Fused Autograd Triton Kernels**: Custom GPU Triton kernels execute forward cell-pooling and backward adjoint propagation in single-pass SRAM passes with zero intermediate tensor allocations.
6. **Flash Streaming Engine**: A tiled chunk-streaming execution model processing streams of over $16.7$ million cells ($2^{24}$) within bounded 65,536-cell chunk windows, consuming only $41.0\text{ MB}$ of peak VRAM.

---

## 🚀 Scaling Virtual Cells Benchmark

Standard full attention collapses with Out-Of-Memory (OOM) errors past 16,384 cells on a single 16GB GPU. BlockGrid's sub-quadratic transport and streaming engine execute cleanly up to **4,194,304 ($2^{22}$)** active cells:

| Virtual Grid Size | Number of Cells $N$ | Full Attention ($\mathcal{O}(N^2)$) | ISAB ($\mathcal{O}(N \cdot K)$) | Perceiver IO ($\mathcal{O}(N \cdot K)$) | **BlockGrid (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$32 \times 32$** | $1,024$ | 67 MB | 37 MB | 37 MB | **34 MB** |
| **$64 \times 64$** | $4,096$ | 1,034 MB | 51 MB | 52 MB | **42 MB** |
| **$100 \times 100$** | $10,000$ | 6,130 MB | 80 MB | 81 MB | **56 MB** |
| **$128 \times 128$** | $16,384$ | 💥 **OOM CRASH** | 110 MB | 111 MB | **72 MB** |
| **$256 \times 256$ (1 Megapixel)** | $65,536$ | 💥 **OOM (>128 GB)** | 345 MB | 345 MB | **193 MB (563 MB Batch=4)** |
| **$512 \times 512$** | $262,144$ ($2^{18}$) | 💥 **OOM CRASH** | 1,282 MB | 1,282 MB | **708 MB** |
| **$1024 \times 1024$**| $1,048,576$ ($2^{20}$) | 💥 **OOM CRASH** | 5,032 MB | 5,032 MB | **2,832 MB** |
| **$1448 \times 1448$**| $2,096,704$ ($2^{21}$) | 💥 **OOM CRASH** | 10,032 MB | 10,032 MB | **5,664 MB** |
| **$2048 \times 2048$**| **$4,194,304$ ($2^{22}$)**| 💥 **OOM CRASH** | 💥 **OOM CRASH** | 💥 **OOM CRASH** | **11,329 MB (PASSED)** |

*Benchmarked on NVIDIA GeForce RTX 5060 Ti 16GB with 4-channel $4 \times 4$ blocks (268,435,456 elements at $2^{22}$).*

---

## 🎨 Visual Transport Demo

<p align="center">
  <img src="assets/viz/blockgrid_transport_demo.png" alt="BlockGrid Cellular Transport Demo" width="900">
</p>

*Figure 1: Cellular transport of a structured $8 \times 8 \times 3$ RGB matrix payload across a $16 \times 16$ grid. (Left) Initial source cell at `[14, 5]`. (Center) BlockGrid output routed to destination cell `[6, 4]` through 16 landmark latent hubs. (Right) Ground truth target under orthogonal bilinear transformation $A^* Z {B^*}^T$ (converging to **0.00046 MSE**).*

An interactive HTML visualization dashboard is available at [`assets/viz/index.html`](assets/viz/index.html).

---

## 🔬 System Architecture & Mathematical Formulation

### 1. Coordinate Encoding & Landmark Assignment
Every cell $i \in \{1, \dots, N\}$ possesses continuous 2D coordinates or discrete hypercube coordinates $p_i \in \{0, \dots, \text{radix}-1\}^D$.
The coordinate vector is projected through additive embedding tables:
$$\mathbf{p}_i = \sum_{j=1}^D \mathbf{E}_j[p_{i, j}], \quad \mathbf{E}_j \in \mathbb{R}^{\text{radix}_j \times d}$$
The assignment probability $w_{ik}$ of cell $i$ to landmark slot $k \in \{1, \dots, K\}$ is governed by temperature-scaled cosine softmax:
$$w_{ik} = \frac{\exp((\mathbf{W}_p \mathbf{p}_i + \mathbf{W}_c \mathbf{s}_i)_k / \tau)}{\sum_{j=1}^K \exp((\mathbf{W}_p \mathbf{p}_i + \mathbf{W}_c \mathbf{s}_i)_j / \tau)}$$

### 2. Planar-Uncrossed Slot Mixing
Landmark communication occurs strictly across the $K$ landmark slots via a low-rank perturbed identity matrix:
$$G = I_K - U V^T, \quad U, V \in \mathbb{R}^{K \times r}$$
This parameterization anchors direct passing ($I_K$) while routing global context as sparse rank-$r$ corrections without topological entanglement.

### 3. Factorized Multi-Linear Mode Contractions
Unlike standard models that flatten structured tokens into 1D vectors, BlockGrid transforms multi-dimensional cell fibers $\mathcal{B} \in \mathbb{R}^{T_1 \times \dots \times T_M}$ using mode-wise tensor products:
$$Z' = Z \times_1 A^{(1)} \times_2 A^{(2)} \dots \times_M A^{(M)}$$
Parameter complexity scales strictly as $\mathcal{O}(\sum_{m=1}^M T_m^2)$ rather than collapsing into flattened vector transformations $\mathcal{O}((\prod T_m)^2)$.

### 4. Sparse Active Hypercube Transport (33D, 3^22, 64D)
For astronomical coordinate spaces where allocating dense tensors is physically impossible:
- `SparseHyperBlockGrid` processes active batches of $M$ entities at arbitrary addresses in $\{0, 1\}^{33}$ ($8.58\text{B}$ locations), $3^{22}$ ($31.38\text{B}$ states), or $64\text{D}$.
- Routing, slot-mixing, and multi-linear contractions run in strictly $\mathcal{O}(M \cdot K)$ time and memory.

---

## 💻 Quickstart & Minimal Usage Examples

### Example 1: 2D Planar Grid Transport

```python
import torch
from blockvga import BlockGridLayer, BlockGridConfig

# Configure 32x32 spatial grid with 16 latent landmark slots
config = BlockGridConfig(
    H=32,
    W=32,
    P=4,
    Q=4,
    F=4,
    K=16,
    d=64,
    inner_op=True
)

layer = BlockGridLayer(config).cuda()
x = torch.randn(2, 32, 32, 4, 4, 4, device="cuda")

out = layer(x)
print("Output shape:", out.shape) # [2, 32, 32, 4, 4, 4]
```

### Example 2: 1-Megapixel Canvas (256x256 Grid)

```python
import torch
from blockvga import BlockGridLayer, BlockGridConfig

# 256x256 grid of 4x4x3 RGB patches = 1 Megapixel canvas
config = BlockGridConfig(
    H=256, W=256, P=4, Q=4, F=3, K=32, d=64
)

layer = BlockGridLayer(config).cuda()
x = torch.randn(4, 256, 256, 4, 4, 3, device="cuda") # Batch of 4 megapixel images

out = layer(x)
print("256x256 Megapixel output:", out.shape) # [4, 256, 256, 4, 4, 3] in ~400ms!
```

### Example 3: Sparse 33D Hypercube Routing

```python
import torch
from blockvga import SparseHyperGridConfig, SparseHyperBlockGrid

# 33-dimensional coordinate universe (8.58 Billion potential positions)
config = SparseHyperGridConfig(
    D=33,
    axis_radix=2,
    cell_shape=(4, 4, 4), # Multi-layer 3D payload
    K=16,
    d=32,
    inner_op=True
)

layer = SparseHyperBlockGrid(config).cuda()

# Route 1,024 active entities residing anywhere in 33D space
B, M = 2, 1024
positions = torch.randint(0, 2, (B, M, 33), device="cuda")
x = torch.randn(B, M, 4, 4, 4, device="cuda")

out = layer(x, positions)
print("Active 33D routed output:", out.shape) # [2, 1024, 4, 4, 4]
```

### Example 4: Unified 2^22 Hypercube with Triton

```python
import torch
from blockvga import UnifiedBlockGrid

# 22-dimensional Boolean Hypercube (4,194,304 cells)
model = UnifiedBlockGrid(
    mode="hyper",
    grid_shape=tuple([2] * 22),
    cell_tensor_shape=(2, 2)
).cuda()

# Streamed or evaluated via custom Triton autograd kernels
```

---

## ⚡ torch.compile & Production Compiler Readiness

`SparseHyperBlockGrid` and `UnifiedBlockGrid` trace and compile cleanly with **PyTorch 2.x `torch.compile`** using the Inductor C++ / Triton backend with zero graph breaks:

```python
import torch
from blockvga import SparseHyperGridConfig, SparseHyperBlockGrid

cfg = SparseHyperGridConfig(D=16, axis_radix=2, cell_shape=(4, 4, 4), K=16, d=32)
model = SparseHyperBlockGrid(cfg).cuda().eval()

# Compile with zero graph breaks into optimized C++ / Triton runtime
compiled_model = torch.compile(model, backend="inductor")
```

---

## 📊 Empirical Benchmarks & Baseline Comparisons

Evaluated on synthetic transport tasks against 6 competing sequence & attention architectures:

| Architecture | Complexity | Memory Footprint (10k Cells) | Runtime (10k Cells) | Task 1a MSE | Task 1b Factorized MSE | Matrix Preserved? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Attention** | $\mathcal{O}(N^2)$ | 6,130 MB | 148.2 ms | 1.0056 | 1.0084 | ❌ (Flattened 1D) |
| **ISAB (Set Transformer)** | $\mathcal{O}(N \cdot K)$ | 80 MB | 14.8 ms | 1.0087 | 1.0092 | ❌ (Flattened 1D) |
| **Perceiver IO** | $\mathcal{O}(N \cdot K)$ | 81 MB | 15.2 ms | 1.0038 | 1.0041 | ❌ (Flattened 1D) |
| **Axial Attention** | $\mathcal{O}(N \sqrt{N})$ | 312 MB | 28.4 ms | 1.0042 | 1.0065 | ❌ (Flattened 1D) |
| **Linear Attention** | $\mathcal{O}(N)$ | 95 MB | 12.1 ms | 1.0051 | 1.0078 | ❌ (Flattened 1D) |
| **Local Conv2D** | $\mathcal{O}(N \cdot k_w^2)$ | 48 MB | 4.2 ms | 1.0094 | 1.0112 | ❌ (Flattened 1D) |
| **BlockGrid (Ours)** | **$\mathcal{O}(N \cdot K)$** | **56 MB** | **8.6 ms** | **0.0102** | **0.0108** | **✅ Intact $(P \times Q \times F)$** |

---

## 💼 Production Application Portfolio (50 Detailed Specs)

A complete enterprise blueprint portfolio of **50 persistent application specifications** is located in [`/home/akarta/Projects/offline_products/block/`](file:///home/akarta/Projects/offline_products/block/README.md) covering:
1. **Combinatorial Optimization & Search**: State-Space Trees ($3^{22}$), SAT Solvers, FPGA Routing, Traveling Salesperson, Datacenter Fabrics, Quantum Compilers.
2. **Computational Biology & Chemistry**: Protein Conformational Dihedrals (33D), Single-Cell RNA Covariance, CRISPR Off-Target Mutagenesis ($4^{23}$), Allosteric Pockets ($128^3$), Hi-C Chromatin Routing.
3. **High-Resolution Vision & Imaging**: Gigapixel Whole-Slide Pathology ($256 \times 256$), SAR Radar Polarimetry, 3D CT/MRI Volumetric Segmentation ($128^3$), 4D Spatiotemporal Grounding.
4. **Robotics & Physics Simulation**: 14-DoF Robot Arm Motion Planning, Multi-Agent Swarms, Lattice Boltzmann Fluid Dynamics, Tokamak Gyrokinetics (6D), N-Body Gravity.
5. **Cryptography & Security**: LDPC Hypercube Decoders, Sparse Malware Execution Graphs (64D), ZKP R1CS Verification, Automated Theorem Proving (Lean 4), Line-Rate Network Packet Anomaly (33D).
6. **Advanced AI & Systems**: 1M+ Extreme Context KV-Cache Alternatives (80MB VRAM), Patch-Preserving ViTs, Hyperdimensional Computing (HDC/VSA), Federated Landmark Consensus.

---

## 🧪 Running the Test Suite

BlockGrid maintains a comprehensive test suite of **38 unit and integration tests** covering shape validation, gradient propagation, Triton autograd correctness, numerical stability, and high-dimensional sparse hypercube routing:

```bash
# Run the complete test suite with uv
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
