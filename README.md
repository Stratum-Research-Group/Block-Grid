# BlockGrid (`blockvga`)

**A Lightweight, Sub-Quadratic Grid Transport Layer for Structured Tensor Payloads**

[![CI Test Suite](https://github.com/Stratum-Research-Group/block-grid-vga/actions/workflows/ci.yml/badge.svg)](https://github.com/Stratum-Research-Group/block-grid-vga/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

BlockGrid separates **inter-cell grid transport** (moving information across a virtual coordinate space $H \times W$ in linear time $\mathcal{O}(N \cdot K)$) from **intra-cell processing** (transforming structured $P \times Q \times F$ matrix blocks via factorized two-sided operators $L_k(Z) = A_k Z B_k^T$).

---

## 🚀 Quick Benchmark: Scaling Virtual Cells on a Single 16GB GPU

| Virtual Grid Size | Number of Cells | Full Attention ($\mathcal{O}(N^2)$) | ISAB ($\mathcal{O}(N \cdot K)$) | Perceiver IO ($\mathcal{O}(N \cdot K)$) | **BlockGrid (Ours)** |
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

## 📖 Documentation
- [Quickstart & API Reference](docs/QUICKSTART.md)
- [Empirical Benchmark Results & Proofs](RESULTS.md)
- [Explanation of the 6 Competing Baseline Architectures](docs/01_competing_architectures.md)
- [Hyperdimensional Grid Topology (2^D)](docs/HYPERDIMENSIONAL_GRID.md)
- [Academic Technical Report & Proofs](docs/PAPER_TECHNICAL_REPORT.md)
- [Original Research Specification](block-grid-vga-project-spec-36d5ffde.txt)

---

## 🛠️ Installation & Tests

```bash
git clone https://github.com/Stratum-Research-Group/block-grid-vga.git
cd block-grid-vga
pip install -e ".[dev]"
pytest tests/ -v
```

