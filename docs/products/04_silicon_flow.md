# Product 4: SiliconFlow — Autonomous VLSI Floorplanning & Macro Placement

## 1. Executive Summary
**SiliconFlow** is a deep reinforcement learning chip design software for next-generation semiconductor floorplanning, macro placement, and wire routing. It maps electronic design automation (EDA) netlists onto virtual 2D silicon grids, optimizing timing, power, and thermal hotspots simultaneously.

---

## 2. The Core Problem It Solves
Modern system-on-chip (SoC) floorplanning (e.g., Apple M-series, Nvidia GPUs) involves placing thousands of macro blocks across billions of transistors:
- Graph neural networks (GNNs) suffer from over-smoothing over long wirelength hops.
- Standard vision models (CNNs) cannot model non-local wire connections between opposite corners of the chip.
- Human engineering teams spend months manually tuning macro locations to avoid wiring congestion and clock skew.

---

## 3. How BlockGrid Powers SiliconFlow
1. **Long-Range Signal Routing Without Congestion**:
   BlockGrid routes wire connectivity across the silicon grid ($H \times W$) using landmark buses ($K$ global routing highways), allowing macro blocks at coordinate $(0, 0)$ to communicate directly with $(H, W)$ without routing congestion.
2. **Local Geometry Preservation**:
   Each grid cell holds a macro's physical properties ($P \times Q$ aspect ratio, pin distribution, capacitive load, leakage power) which are modified via two-sided transformations during placement optimization.
3. **Equivariant Placement Updates**:
   Permutation equivariance ensures that identical macro clusters receive identical placement rewards regardless of storage array ordering.

---

## 4. Product Features & Architecture
- **Interactive Floorplan Explorer**: Real-time wirelength estimation and thermal hotspot visualization.
- **Timing Closure Engine**: Directly predicts parasitics and clock jitter before physical synthesis.
- **PPA Optimization**: Achieves 10–15% better Power, Performance, and Area (PPA) compared to traditional simulated annealing tools.

---

## 5. Target Market & Monetization
- **Primary Market**: Semiconductor design companies (Qualcomm, AMD, Intel, Arm, RISC-V startups), EDA vendors (Synopsys, Cadence).
- **Business Model**: Annual enterprise tool license per seat/tape-out; hybrid SaaS with high-throughput cloud placement runs.
