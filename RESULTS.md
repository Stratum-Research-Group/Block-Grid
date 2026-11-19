# Empirical Findings & Benchmark Report: BlockGrid Transport

## 1. Overview & Setup
- **Task**: Synthetic Block Transport (Task 1a) on an $H = W = 16$ virtual coordinate grid ($N = 256$ cells).
- **Payload**: Each virtual cell holds an inner block $P = Q = 8$ with $F = 4$ channels ($256$ scalars per cell).
- **Goal**: Transport a randomly initialized target block pattern ($\mathcal{N}(0, 1)$) from a source cell to a destination cell given only coordinate routing markers ($R = 2$). Background cells contain noise ($\mathcal{N}(0, 0.1)$).
- **Training**: 1,000 steps with AdamW, lr $3\times 10^{-4}$, weight decay $0.01$. Training episodes restricted to distance bins $1\text{--}20$.
- **Validation**: Stratified across distance bins $\{1\text{--}2, 3\text{--}5, 6\text{--}10, 11\text{--}20, 21\text{--}30\}$, including the completely held-out long-distance bin $21\text{--}30$.

---

## 2. Quantitative Results

| Model Architecture | Parameters | FLOP Scaling | Dist 1–2 | Dist 3–5 | Dist 6–10 | Dist 11–20 | Dist 21–30 (Held-Out) | Overall MSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BlockGrid (Planar Uncrossed)** | **77,100** | $\mathcal{O}(N)$ | **0.0104** | **0.0102** | **0.0103** | **0.0101** | **0.0103** | **0.0102** |
| **BlockGrid (Free $G$)** | **78,124** | $\mathcal{O}(N)$ | 0.0105 | 0.0103 | 0.0105 | 0.0104 | 0.0105 | 0.0105 |
| **B1 ISAB (Set Transformer)** | 1,475,072 | $\mathcal{O}(N)$ | 1.0047 | 1.0082 | 1.0138 | 1.0107 | 1.0060 | 1.0087 |
| **B2 Perceiver IO** | 2,002,432 | $\mathcal{O}(N)$ | 0.9998 | 1.0088 | 1.0017 | 1.0076 | 1.0010 | 1.0038 |
| **B5 Full Attention** | 932,352 | $\mathcal{O}(N^2)$ | 0.9963 | 1.0047 | 1.0057 | 1.0027 | 1.0188 | 1.0056 |

---

## 3. Analysis & Key Insights

### A. The "Flattening Collapse" of Standard Baselines
- **ISAB, Perceiver, and Full Attention failed completely ($\text{MSE} \approx 1.00$)** despite having between **$12\times$ and $26\times$ more parameters**.
- **Root Cause**: When baselines flatten each cell's block into a 256-dimensional vector and project it with large linear matrices, the model faces a catastrophic signal-to-noise problem. The network must simultaneously learn to extract the routing marker from the concatenated vector, maintain 256 independent payload channels through dense Multi-Head Attention, and reconstruct the block at the destination without leaking background noise. At 1,000 steps, they remain trapped in the mean-prediction floor ($\text{MSE} \approx 1.0$).

### B. BlockGrid Disentanglement & Sample Efficiency
- Both BlockGrid variants converged from $\text{MSE} \approx 1.0$ down to $\text{MSE} \approx 0.010$ in under 1,000 steps ($18.5$ seconds on GPU).
- By explicitly keeping **inter-cell routing** (the $K$-slot simplex) separate from **intra-cell block payload preservation** (whole-block einsum contraction), the routing path learns *where* to send data immediately, while the payload passes through undistorted.

### C. Zero Generalization Gap on Held-Out Distances
- In the held-out bin **$21\text{--}30$** (distances never seen during training), BlockGrid maintained an identical $\text{MSE} = 0.0103$.
- This confirms that landmark routing learns coordinate-conditioned transport rather than memorizing training distance combinations.

### D. The Residual Noise Floor ($\approx 0.010$)
- In Section 5 of the original spec, the theoretical target MSE was hypothesized to be 0.
- As predicted in our initial mathematical review, the soft softmax routing ($w_{a,k} > 0$ for all cells) causes background noise ($\mathcal{N}(0, 0.1)$) from non-source cells to leak into the destination, creating an irreducible floor of $\approx \sigma^2 = (0.1)^2 = 0.010$. The model achieved this mathematical limit.

### E. Planar Uncrossing Parameterization ($G = I - UV^T$)
- The planar uncrossed formulation derived from the `eth` tensor uncrossing mathematics outperformed free matrix mixing in both convergence speed (Step 250 MSE: $0.6916$ vs $0.9169$) and final validation loss ($0.0102$ vs $0.0105$), while requiring fewer parameters.
