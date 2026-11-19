# The 6 Competing Standard Architectures Explained

To understand why **BlockGrid** behaves the way it does, we must understand how the deep learning world traditionally processes grids and sets of tokens, and why each of these 6 architectures struggles when scaling structured tensor arrays.

---

## 1. Full Pairwise Attention (`FullAttentionBaseline`)
* **Found in**: Standard Transformers (BERT, GPT, ViT - Vision Transformer).
* **How it works**:
  Every cell on the grid looks at every other cell on the grid. If you have $N$ cells, it creates an $N \times N$ matrix of attention scores:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
* **Pros**: Maximum expressive power. Any cell can talk directly to any other cell in a single step.
* **Cons (The Quadratic Wall)**:
  - **Memory & Compute**: Scales as $\mathcal{O}(N^2)$. If your grid is $16 \times 16$ ($N=256$), the matrix has $65,536$ entries. If your grid scales to $256 \times 256$ ($N=65,536$), the attention matrix requires **4.29 billion entries per head, per layer**. It immediately causes GPU Out-of-Memory crashes.
  - **The Flattening Problem**: It must flatten the inner block $P \times Q \times F$ into a flat 1D vector, destroying the spatial/algebraic relationships inside the block.

---

## 2. ISAB - Induced Set Attention Block (`ISABBaseline`)
* **Found in**: Set Transformer (Lee et al., 2019).
* **How it works**:
  Instead of all $N$ cells talking to all $N$ cells, ISAB introduces a small number of $K$ learned "virtual points" (inducing points, e.g., $K=32$):
  1. The $K$ inducing points query the $N$ cells: $\mathcal{O}(N \cdot K)$.
  2. The $N$ cells query the updated $K$ inducing points: $\mathcal{O}(N \cdot K)$.
* **Pros**: Reduces complexity from quadratic $\mathcal{O}(N^2)$ to linear $\mathcal{O}(N \cdot K)$.
* **Cons**:
  - The inducing points are static, learned global parameters, not conditioned dynamically on coordinates.
  - It treats cells as an unordered set, ignoring 2D grid coordinates unless positional vectors are concatenated.
  - Still flattens the internal tensor block into an unstructured 1D vector.

---

## 3. Perceiver / Perceiver IO (`PerceiverBaseline`)
* **Found in**: DeepMind's Perceiver & Perceiver IO (Jaegle et al., 2021).
* **How it works**:
  Uses an "information bottleneck" of latent vectors (e.g., $K=32$ or $K=64$ latents):
  1. **Cross-Attention In**: The $K$ latents query the entire input grid $N$ to absorb data.
  2. **Latent Transformer**: The $K$ latents perform self-attention amongst themselves.
  3. **Cross-Attention Out**: The output queries read back from the latents.
* **Pros**: Completely decouples model depth from input resolution. Can handle millions of inputs by compressing into latents.
* **Cons**:
  - Extremely parameter-heavy (needs $Q, K, V$ projections for both cross-attentions and latent self-attention; in our test, Perceiver had over 2,000,000 parameters vs BlockGrid's 77,000).
  - Struggles to learn precise point-to-point routing (like moving a block from $(2, 3)$ to $(14, 11)$) because the latents act as a global fuzzy summarizer rather than a coordinate routing switch.

---

## 4. Axial Attention (`AxialAttentionBaseline`)
* **Found in**: Axial Transformers (Ho et al., 2019), CCNet.
* **How it works**:
  Instead of 2D attention over $H \times W$, it factors attention into two 1D steps:
  1. **Row Attention**: Every cell attends only to other cells in its own row ($\mathcal{O}(H \cdot W^2)$).
  2. **Column Attention**: Every cell attends only to other cells in its own column ($\mathcal{O}(W \cdot H^2)$).
* **Pros**: Much faster than full 2D attention ($\mathcal{O}(N \sqrt{N})$ instead of $\mathcal{O}(N^2)$). Preserves row and column structure.
* **Cons**:
  - Information cannot travel diagonally in one step; it takes multiple layers for information to zigzag across the grid.
  - Rigidly tied to rectangular grids—cannot easily route through arbitrary coordinate addresses or topological graphs.

---

## 5. Linear Attention (`LinearAttentionBaseline`)
* **Found in**: "Transformers are RNNs" (Katharopoulos et al., 2020), Performer, Fastformer.
* **How it works**:
  Uses the associative property of matrix multiplication to avoid the $N \times N$ matrix. By applying a kernel feature map $\phi(x) = \text{ELU}(x) + 1$:
  $$\text{Standard: } (Q K^T) V \quad [\mathcal{O}(N^2 \cdot d)]$$
  $$\text{Linear: } Q (K^T V) \quad [\mathcal{O}(N \cdot d^2)]$$
* **Pros**: Strictly linear $\mathcal{O}(N)$ in time and memory. Very fast for long sequences.
* **Cons**:
  - Loses the sharp, peaked attention distribution of softmax. It acts like an unnormalized low-frequency filter.
  - Incapable of sharp, discrete point-to-point routing (cannot isolate a single source cell from noisy background cells).

---

## 6. Local Convolution Stack (`LocalConvBaseline`)
* **Found in**: Standard CNNs (ResNet, ConvNeXt).
* **How it works**:
  Applies small sliding window filters (e.g., $3 \times 3$ convolutions) repeatedly.
* **Pros**:
  - Strict translation equivariance.
  - Extremely fast on GPUs with low memory footprint ($\mathcal{O}(N)$).
* **Cons**:
  - **No Global Reach**: A $3 \times 3$ kernel can only see 1 pixel away. To move information from coordinate $(0, 0)$ to $(15, 15)$, you need a deep stack of at least 15–20 consecutive convolutional layers.
  - Cannot jump directly across coordinates in a single layer.

---

## Summary Comparison Matrix

| Architecture | Complexity | Preserves Block Tensor? | Point-to-Point Routing Precision | Scaling to $N=100,000$ |
| :--- | :---: | :---: | :---: | :---: |
| **Full Attention** | $\mathcal{O}(N^2)$ | ❌ (Flattens) | Medium (High params) | ❌ Crashes OOM |
| **ISAB (Set Transformer)** | $\mathcal{O}(N \cdot K)$ | ❌ (Flattens) | Low (Global latents) | ✅ Linear |
| **Perceiver IO** | $\mathcal{O}(N \cdot K)$ | ❌ (Flattens) | Low (Fuzzy summary) | ✅ Linear |
| **Axial Attention** | $\mathcal{O}(N \sqrt{N})$ | ❌ (Flattens) | Medium (Row/Col only) | ⚠️ Quadratic along axes |
| **Linear Attention** | $\mathcal{O}(N)$ | ❌ (Flattens) | ❌ Very Low (Kernel blur) | ✅ Linear |
| **Local Convolutions** | $\mathcal{O}(N)$ | ❌ (Flattens) | ❌ Fails (Receptive field limit) | ✅ Linear |
| **BlockGrid (Ours)** | **$\mathcal{O}(N \cdot K)$** | **✅ YES ($A Z B^T$)** | **✅ High ($99\%$ in Task 1a/1b)** | **✅ Linear & Low VRAM** |
