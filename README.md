# BlockGrid (`blockvga`)

**A Lightweight, Sub-Quadratic Grid Transport Layer for Structured Tensor Payloads**

[![CI Test Suite](https://github.com/Stratum-Research-Group/block-grid-vga/actions/workflows/ci.yml/badge.svg)](https://github.com/Stratum-Research-Group/block-grid-vga/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

BlockGrid separates **inter-cell grid transport** (moving information across a virtual coordinate space $H \times W$ in linear time $\mathcal{O}(N \cdot K)$) from **intra-cell processing** (transforming structured $P \times Q \times F$ matrix blocks via factorized two-sided operators $L_k(Z) = A_k Z B_k^T$).

---

## 🚀 Quick Benchmark: 65,536 Virtual Cells on a Single GPU

| Architecture | Complexity | 1,024 Cells | 4,096 Cells | 10,000 Cells | 65,536 Cells |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Attention** | $\mathcal{O}(N^2)$ | 67 MB | 1,034 MB | 6,130 MB | 💥 **OOM CRASH** |
| **ISAB (Set Transformer)** | $\mathcal{O}(N \cdot K)$ | 37 MB | 51 MB | 80 MB | 344 MB |
| **Perceiver IO** | $\mathcal{O}(N \cdot K)$ | 37 MB | 51 MB | 80 MB | 345 MB |
| **BlockGrid (Ours)** | **$\mathcal{O}(N \cdot K)$** | **34 MB** | **42 MB** | **56 MB** | **193 MB** |

*Benchmarked on NVIDIA GeForce RTX 5060 Ti with 4-channel $4 \times 4$ tensor blocks per cell.*

---

## 📖 Documentation
- [Quickstart & API Reference](docs/QUICKSTART.md)
- [Empirical Benchmark Results & Proofs](RESULTS.md)
- [Explanation of the 6 Competing Baseline Architectures](docs/01_competing_architectures.md)
- [Original Research Specification](block-grid-vga-project-spec-36d5ffde.txt)

### Commercial Product Blueprints
1. [TensorRAM: Neural Virtual Memory Engine](docs/products/01_tensor_ram.md)
2. [SimGrid Neural: Multi-Physics PDE Simulator](docs/products/02_simgrid_neural.md)
3. [PlanarContract: Quantum Circuit & Tensor Network Optimizer](docs/products/03_planar_contract.md)
4. [SiliconFlow: Autonomous VLSI Floorplanning](docs/products/04_silicon_flow.md)
5. [CellMIM: Spatial Biology & Oncology AI](docs/products/05_cell_mim.md)
6. [SwarmGrid: Warehouse Robotics Coordination](docs/products/06_swarm_grid.md)

---

## 🛠️ Installation & Tests

```bash
git clone https://github.com/Stratum-Research-Group/block-grid-vga.git
cd block-grid-vga
pip install -e ".[dev]"
pytest tests/ -v
```

