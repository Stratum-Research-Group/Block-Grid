# Hyperdimensional Grid Architecture: D-Dimensional Coordinate Scaling ($2^D$)

## 1. Abstract
The traditional 2D formulation ($H \times W$) of BlockGrid is a special case ($D=2$) of an arbitrary **Hyperdimensional Grid Array**:

$$\mathcal{X} \in \mathbb{R}^{S_1 \times S_2 \times \dots \times S_D \times \mathcal{B}}$$

where:
- The **outer topology** is a $D$-dimensional hypercube or hyper-torus: $S = (2, 2, \dots, 2) \in \{0, 1\}^D$ scaling from $D=1$ to $D=22$ ($N = 2^{22} = 4,194,304$ sites).
- The **inner fiber / payload** $\mathcal{B} \in \mathbb{R}^{P \times Q \times F}$ is a local tensor living at each hyperdimensional coordinate.

---

## 2. Why Hyperdimensional Grids Break Standard Transformers
In a $D$-dimensional binary hypercube $\{0, 1\}^D$:
1. **Curse of Dimensionality in Pairwise Attention**:
   Full attention computes an $N \times N = 2^D \times 2^D = 2^{2D}$ matrix.
   - For $D=8$ ($256$ vertices): $65,536$ pairs.
   - For $D=14$ ($16,384$ vertices): $268$ million pairs (crashes GPU VRAM).
   - For $D=22$ ($4,194,304$ vertices): **$17.6 \times 10^{12}$ pairs** (physically impossible on any modern supercomputer).
2. **Loss of Metric Topology**:
   Standard Transformers treat all tokens as an unstructured 1D sequence, destroying the hypercube's Hamming metric:
   $$d_{\text{Hamming}}(x, y) = \sum_{i=1}^D |x_i - y_i|$$

---

## 3. How BlockGrid Solves Hyperdimensional Array Transport ($\mathcal{O}(D \cdot K)$)

### A. Coordinate Factorization via Linear Embeddings
Instead of $N = 2^D$ separate positional embeddings, a hypercube coordinate $\mathbf{x} = (x_1, x_2, \dots, x_D) \in \{0, 1\}^D$ is encoded via $D$ coordinate bit projections:
$$p(\mathbf{x}) = \sum_{i=1}^D E_i(x_i), \quad E_i \in \mathbb{R}^{2 \times d}$$
- **Parameter Cost**: Only $2 \cdot D \cdot d$ parameters (for $D=22$ and $d=64$, this is just **$2,816$ floats**).
- **Metric Fidelity**: The dot product between coordinate embeddings directly reflects the exact Hamming distance and bit-flip parity on the hypercube.

### B. Landmark Hyper-Buses ($K$ Latent Portals)
- Communication between distant vertices across the hypercube (e.g. from $(0, 0, \dots, 0)$ to $(1, 1, \dots, 1)$ with Hamming distance $D$) routes through $K$ landmark slots in $\mathcal{O}(2^D \cdot K)$ operations.
- Information traverses diameter $D$ in a **single layer** without needing $D$ separate graph convolution steps.

### C. Multi-Linear Fiber Transformations ($L_k$)
The local tensor fiber $\mathcal{B} = P \times Q \times F$ at each hypercube node transforms under two-sided factorized operations:
$$L_k(Z) = A_k Z B_k^T$$
preserving internal gauge symmetries, Lie group actions, or quantum spin matrices associated with each hypercube site.
